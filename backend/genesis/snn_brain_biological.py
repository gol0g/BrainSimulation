"""
Biological SNN Brain - Bottom-Up Approach
==========================================

FEP 수학 없이, 생물학적 구조만으로 구현:
1. 실제 뉴런 역학 (LIF with adaptation)
2. 실제 학습 규칙 (STDP, homeostatic plasticity)
3. 실제 뇌 구조 (감각 → 연합 → 운동)
4. 희소 연결 (생물학적으로 현실적)

목표: 인간 뇌처럼 작동하는 시스템
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


@dataclass
class BiologicalConfig:
    """생물학적으로 현실적인 뇌 설정"""

    # === 뉴런 수 (스케일업) ===
    # 목표: 하드웨어가 버틸 수 있는 최대

    # 감각 영역
    visual_v1: int = 50_000      # 1차 시각 피질
    visual_v2: int = 30_000      # 2차 시각 피질
    auditory_a1: int = 20_000    # 1차 청각 피질

    # 연합 영역
    temporal: int = 30_000       # 측두엽 (객체 인식)
    parietal: int = 20_000       # 두정엽 (공간)
    prefrontal: int = 30_000     # 전전두엽 (계획, 작업기억)

    # 기억 영역
    hippocampus: int = 20_000    # 해마 (에피소드 기억)

    # 운동 영역
    motor: int = 10_000          # 운동 피질

    # 총 뉴런: ~210,000

    # === 연결 희소성 (생물학적) ===
    # 실제 뇌는 ~1-10% 연결
    intra_region_sparsity: float = 0.05   # 영역 내 5%
    inter_region_sparsity: float = 0.01   # 영역 간 1%

    # === LIF 뉴런 파라미터 ===
    tau_mem: float = 20.0        # 막전위 시상수 (ms)
    tau_syn: float = 5.0         # 시냅스 시상수 (ms)
    v_th: float = 1.0            # 발화 임계값
    v_reset: float = 0.0         # 리셋 전위
    v_rest: float = 0.0          # 휴지 전위

    # === 적응 (Adaptation) ===
    # 실제 뉴런은 지속 자극에 적응
    tau_adapt: float = 100.0     # 적응 시상수
    adapt_strength: float = 0.1  # 적응 강도

    # === STDP 파라미터 ===
    tau_plus: float = 20.0       # LTP 시상수
    tau_minus: float = 20.0      # LTD 시상수
    a_plus: float = 0.005        # LTP 강도 (스케일에 맞게 조정)
    a_minus: float = 0.006       # LTD 강도
    w_max: float = 1.0           # 최대 가중치
    w_min: float = 0.0           # 최소 가중치

    # === 항상성 가소성 ===
    # 뉴런 활동을 목표 범위로 유지
    target_rate: float = 0.05    # 목표 발화율 5%
    homeostatic_tau: float = 1000.0  # 느린 조정

    # === 시뮬레이션 ===
    dt: float = 1.0              # 시간 스텝 (ms)

    def total_neurons(self) -> int:
        return (self.visual_v1 + self.visual_v2 + self.auditory_a1 +
                self.temporal + self.parietal + self.prefrontal +
                self.hippocampus + self.motor)


class SparseSTDPConnection(nn.Module):
    """
    희소 STDP 시냅스 연결

    생물학적 특징:
    - 희소 연결 (dense 아님)
    - Spike-timing dependent plasticity
    - 가중치 제한 (soft bounds)
    """

    def __init__(self,
                 pre_size: int,
                 post_size: int,
                 sparsity: float,
                 config: BiologicalConfig):
        super().__init__()
        self.pre_size = pre_size
        self.post_size = post_size
        self.config = config

        # 희소 연결 생성
        n_connections = int(pre_size * post_size * sparsity)

        # 랜덤 연결 인덱스
        pre_idx = torch.randint(0, pre_size, (n_connections,))
        post_idx = torch.randint(0, post_size, (n_connections,))

        self.register_buffer('pre_idx', pre_idx)
        self.register_buffer('post_idx', post_idx)

        # 가중치 (학습 가능)
        # 초기화: 활동 전파 가능하도록 적절한 크기
        weights = torch.rand(n_connections) * 0.05 + 0.01  # 0.01 ~ 0.06
        self.weights = nn.Parameter(weights)

        # 흥분성/억제성 구분 (80% 흥분, 20% 억제 - Dale's law)
        n_inhibitory = int(n_connections * 0.2)
        inhibitory_mask = torch.zeros(n_connections, dtype=torch.bool)
        inhibitory_mask[:n_inhibitory] = True
        self.register_buffer('inhibitory_mask', inhibitory_mask)

        # STDP trace
        self.register_buffer('pre_trace', torch.zeros(pre_size))
        self.register_buffer('post_trace', torch.zeros(post_size))

        # === 도파민-조절 STDP (3-factor rule) ===
        # Eligibility trace: STDP 신호를 일시적으로 저장
        # 도파민이 오면 이 trace를 가중치에 적용
        self.register_buffer('eligibility_trace', torch.zeros(n_connections))
        self.tau_eligibility = 500.0  # 500ms 동안 유지 (생물학적으로 현실적)

        print(f"  Connection: {pre_size} -> {post_size}, {n_connections:,} synapses ({sparsity*100:.1f}%)")

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """
        전파: pre_spikes -> post_input
        흥분성(+)과 억제성(-) 시냅스 구분
        """
        # 발화한 pre 뉴런의 가중치 합산
        # pre_spikes: (pre_size,) binary

        # 활성화된 시냅스 찾기
        active_mask = pre_spikes[self.pre_idx] > 0.5

        # post 뉴런별 입력 합산
        post_input = torch.zeros(self.post_size, device=pre_spikes.device)

        if active_mask.any():
            active_weights = self.weights[active_mask].clone()
            # 억제성 시냅스는 음수로 (억제 4배 강화 - E/I balance)
            active_inhibitory = self.inhibitory_mask[active_mask]
            active_weights[active_inhibitory] = -active_weights[active_inhibitory] * 4.0

            active_post = self.post_idx[active_mask]
            post_input.scatter_add_(0, active_post, active_weights)

        return post_input * 0.5  # 전체 입력 스케일링

    def update_stdp(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                    dopamine: float = 0.0, use_eligibility: bool = True):
        """
        도파민-조절 STDP (3-factor learning rule)

        생물학적 근거:
        - Eligibility trace: 시냅스가 "학습 준비" 상태를 유지
        - 도파민이 오면: eligibility trace → 실제 가중치 변화
        - 도파민 없으면: 약한 기본 STDP만 적용

        Args:
            pre_spikes: pre-synaptic spikes
            post_spikes: post-synaptic spikes
            dopamine: 도파민 수준 (-1 ~ 2)
            use_eligibility: True면 3-factor rule, False면 기본 STDP
        """
        cfg = self.config

        # Trace 업데이트 (지수 감쇠)
        decay_pre = np.exp(-cfg.dt / cfg.tau_plus)
        decay_post = np.exp(-cfg.dt / cfg.tau_minus)

        self.pre_trace = self.pre_trace * decay_pre + pre_spikes
        self.post_trace = self.post_trace * decay_post + post_spikes

        # Eligibility trace 감쇠
        decay_elig = np.exp(-cfg.dt / self.tau_eligibility)
        self.eligibility_trace = self.eligibility_trace * decay_elig

        with torch.no_grad():
            # === STDP 신호 계산 ===

            # LTP: post 발화 시, pre trace에 비례
            post_fired = post_spikes[self.post_idx] > 0.5
            ltp_signal = torch.zeros_like(self.eligibility_trace)
            if post_fired.any():
                pre_traces = self.pre_trace[self.pre_idx[post_fired]]
                ltp_signal[post_fired] = cfg.a_plus * pre_traces

            # LTD: pre 발화 시, post trace에 비례
            pre_fired = pre_spikes[self.pre_idx] > 0.5
            ltd_signal = torch.zeros_like(self.eligibility_trace)
            if pre_fired.any():
                post_traces = self.post_trace[self.post_idx[pre_fired]]
                ltd_signal[pre_fired] = cfg.a_minus * post_traces

            # === 3-factor rule 적용 ===
            if use_eligibility:
                # STDP 신호를 eligibility trace에 축적
                self.eligibility_trace += ltp_signal - ltd_signal

                # 도파민이 있으면 eligibility trace를 가중치에 적용
                if abs(dopamine) > 0.1:
                    # dopamine > 0: 양의 eligibility → 강화
                    # dopamine < 0: 양의 eligibility → 약화 (punishment)
                    delta_w = dopamine * self.eligibility_trace * 0.1

                    # Soft bounds
                    delta_w = delta_w * torch.where(
                        delta_w > 0,
                        cfg.w_max - self.weights,
                        self.weights - cfg.w_min
                    )

                    self.weights += delta_w

                    # Eligibility 소비 (일부만)
                    self.eligibility_trace *= 0.5

                # 기본 STDP도 약하게 적용 (unsupervised learning)
                base_rate = 0.1  # 10%만
                self.weights += base_rate * (ltp_signal * (cfg.w_max - self.weights) -
                                              ltd_signal * (self.weights - cfg.w_min))
            else:
                # 기존 STDP (도파민 조절 없음)
                self.weights += ltp_signal * (cfg.w_max - self.weights)
                self.weights -= ltd_signal * (self.weights - cfg.w_min)

            # 가중치 제한
            self.weights.clamp_(cfg.w_min, cfg.w_max)


class AdaptiveLIFLayer(nn.Module):
    """
    적응형 LIF 뉴런 레이어

    생물학적 특징:
    - Leaky Integrate-and-Fire
    - Spike-rate adaptation (지속 자극에 둔감해짐)
    - Homeostatic plasticity (발화율 조절)
    """

    def __init__(self, size: int, config: BiologicalConfig, name: str = ""):
        super().__init__()
        self.size = size
        self.config = config
        self.name = name

        # 뉴런 상태
        self.register_buffer('v_mem', torch.zeros(size))      # 막전위
        self.register_buffer('adaptation', torch.zeros(size)) # 적응 변수
        self.register_buffer('threshold', torch.ones(size) * config.v_th)  # 가변 임계값

        # 발화율 추적 (항상성용)
        self.register_buffer('firing_rate_avg', torch.ones(size) * config.target_rate)

        print(f"  Layer '{name}': {size:,} neurons")

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        한 시간 스텝 시뮬레이션

        Returns: 발화 여부 (binary tensor)
        """
        cfg = self.config

        # 막전위 누출
        leak = np.exp(-cfg.dt / cfg.tau_mem)
        self.v_mem = self.v_mem * leak + input_current

        # 적응 적용 (발화율 높으면 임계값 상승)
        effective_threshold = self.threshold + self.adaptation

        # 발화 판정
        spikes = (self.v_mem >= effective_threshold).float()

        # 발화 후 리셋
        self.v_mem = torch.where(spikes > 0.5,
                                  torch.full_like(self.v_mem, cfg.v_reset),
                                  self.v_mem)

        # 적응 업데이트
        adapt_decay = np.exp(-cfg.dt / cfg.tau_adapt)
        self.adaptation = self.adaptation * adapt_decay + spikes * cfg.adapt_strength

        # 발화율 추적 (느린 이동 평균)
        rate_decay = np.exp(-cfg.dt / cfg.homeostatic_tau)
        self.firing_rate_avg = self.firing_rate_avg * rate_decay + spikes * (1 - rate_decay)

        return spikes

    def apply_homeostasis(self):
        """
        항상성 가소성: 발화율을 목표 범위로 조정
        """
        cfg = self.config

        # 목표 발화율과의 차이
        rate_error = self.firing_rate_avg - cfg.target_rate

        # 임계값 조정 (발화 많으면 임계값 올림) - 더 강하게
        self.threshold += rate_error * 0.1
        self.threshold.clamp_(cfg.v_th * 0.5, cfg.v_th * 3.0)

    def reset(self):
        """상태 초기화"""
        self.v_mem.zero_()
        self.adaptation.zero_()
        self.firing_rate_avg.fill_(self.config.target_rate)


class BiologicalBrain(nn.Module):
    """
    생물학적 뇌 - Bottom-Up 구현

    구조:
    - 감각 입력 → V1 → V2 → Temporal/Parietal → Prefrontal → Motor
    - Hippocampus: 모든 영역과 연결 (기억 허브)
    - 피드백 연결 (top-down)

    학습:
    - STDP (시냅스)
    - Homeostatic plasticity (뉴런)
    - 명시적 보상/FEP 없음
    """

    def __init__(self, config: BiologicalConfig = None):
        super().__init__()
        self.config = config or BiologicalConfig()
        cfg = self.config

        print(f"\n{'='*60}")
        print("Building Biological Brain")
        print(f"{'='*60}")
        print(f"Total neurons: {cfg.total_neurons():,}")
        print(f"Device: {DEVICE}")
        print(f"\nCreating layers...")

        # === 뉴런 레이어 ===
        self.v1 = AdaptiveLIFLayer(cfg.visual_v1, cfg, "V1")
        self.v2 = AdaptiveLIFLayer(cfg.visual_v2, cfg, "V2")
        self.a1 = AdaptiveLIFLayer(cfg.auditory_a1, cfg, "A1")
        self.temporal = AdaptiveLIFLayer(cfg.temporal, cfg, "Temporal")
        self.parietal = AdaptiveLIFLayer(cfg.parietal, cfg, "Parietal")
        self.prefrontal = AdaptiveLIFLayer(cfg.prefrontal, cfg, "Prefrontal")
        self.hippocampus = AdaptiveLIFLayer(cfg.hippocampus, cfg, "Hippocampus")
        self.motor = AdaptiveLIFLayer(cfg.motor, cfg, "Motor")

        print(f"\nCreating connections...")

        # === Feedforward 연결 ===
        self.conn_v1_v2 = SparseSTDPConnection(
            cfg.visual_v1, cfg.visual_v2, cfg.inter_region_sparsity, cfg)
        self.conn_v2_temporal = SparseSTDPConnection(
            cfg.visual_v2, cfg.temporal, cfg.inter_region_sparsity, cfg)
        self.conn_v2_parietal = SparseSTDPConnection(
            cfg.visual_v2, cfg.parietal, cfg.inter_region_sparsity, cfg)
        self.conn_temporal_prefrontal = SparseSTDPConnection(
            cfg.temporal, cfg.prefrontal, cfg.inter_region_sparsity, cfg)
        self.conn_parietal_prefrontal = SparseSTDPConnection(
            cfg.parietal, cfg.prefrontal, cfg.inter_region_sparsity, cfg)
        self.conn_prefrontal_motor = SparseSTDPConnection(
            cfg.prefrontal, cfg.motor, cfg.inter_region_sparsity, cfg)

        # === 청각 경로 ===
        self.conn_a1_temporal = SparseSTDPConnection(
            cfg.auditory_a1, cfg.temporal, cfg.inter_region_sparsity, cfg)

        # === 해마 연결 (기억 허브) ===
        self.conn_temporal_hippo = SparseSTDPConnection(
            cfg.temporal, cfg.hippocampus, cfg.inter_region_sparsity, cfg)
        self.conn_parietal_hippo = SparseSTDPConnection(
            cfg.parietal, cfg.hippocampus, cfg.inter_region_sparsity, cfg)
        self.conn_prefrontal_hippo = SparseSTDPConnection(
            cfg.prefrontal, cfg.hippocampus, cfg.inter_region_sparsity, cfg)
        self.conn_hippo_prefrontal = SparseSTDPConnection(
            cfg.hippocampus, cfg.prefrontal, cfg.inter_region_sparsity, cfg)

        # === Feedback 연결 (top-down) ===
        self.conn_v2_v1 = SparseSTDPConnection(
            cfg.visual_v2, cfg.visual_v1, cfg.inter_region_sparsity * 0.5, cfg)
        self.conn_prefrontal_temporal = SparseSTDPConnection(
            cfg.prefrontal, cfg.temporal, cfg.inter_region_sparsity * 0.5, cfg)

        # === 영역 내 연결 (recurrent) ===
        self.conn_v1_v1 = SparseSTDPConnection(
            cfg.visual_v1, cfg.visual_v1, cfg.intra_region_sparsity, cfg)
        self.conn_prefrontal_prefrontal = SparseSTDPConnection(
            cfg.prefrontal, cfg.prefrontal, cfg.intra_region_sparsity, cfg)

        # 모든 연결 리스트 (STDP 업데이트용)
        self.all_connections = [
            (self.conn_v1_v2, 'v1', 'v2'),
            (self.conn_v2_temporal, 'v2', 'temporal'),
            (self.conn_v2_parietal, 'v2', 'parietal'),
            (self.conn_temporal_prefrontal, 'temporal', 'prefrontal'),
            (self.conn_parietal_prefrontal, 'parietal', 'prefrontal'),
            (self.conn_prefrontal_motor, 'prefrontal', 'motor'),
            (self.conn_a1_temporal, 'a1', 'temporal'),
            (self.conn_temporal_hippo, 'temporal', 'hippocampus'),
            (self.conn_parietal_hippo, 'parietal', 'hippocampus'),
            (self.conn_prefrontal_hippo, 'prefrontal', 'hippocampus'),
            (self.conn_hippo_prefrontal, 'hippocampus', 'prefrontal'),
            (self.conn_v2_v1, 'v2', 'v1'),
            (self.conn_prefrontal_temporal, 'prefrontal', 'temporal'),
            (self.conn_v1_v1, 'v1', 'v1'),
            (self.conn_prefrontal_prefrontal, 'prefrontal', 'prefrontal'),
        ]

        # 입력 인코더 (외부 입력 → V1 스파이크)
        self.visual_encoder = nn.Linear(64*64, cfg.visual_v1)  # 64x64 이미지 가정
        self.auditory_encoder = nn.Linear(256, cfg.auditory_a1)  # 256 오디오 특징 가정

        # 통계
        self.step_count = 0
        self.spike_history = deque(maxlen=1000)

        # GPU로 이동
        self.to(DEVICE)

        # 메모리 사용량 확인
        self._check_memory()

        print(f"\n{'='*60}")
        print("Brain construction complete!")
        print(f"{'='*60}\n")

    def _check_memory(self):
        """GPU 메모리 사용량 확인"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"\nGPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def encode_visual(self, image: torch.Tensor) -> torch.Tensor:
        """
        시각 입력 → V1 스파이크율로 변환
        image: (64, 64) or (batch, 64, 64)
        """
        if image.dim() == 2:
            image = image.view(-1)
        elif image.dim() == 3:
            image = image.view(image.size(0), -1)

        # 연속값 → 발화 확률
        rates = torch.sigmoid(self.visual_encoder(image))
        # 확률적 발화
        spikes = (torch.rand_like(rates) < rates).float()
        return spikes

    def encode_auditory(self, audio: torch.Tensor) -> torch.Tensor:
        """
        청각 입력 → A1 스파이크율로 변환
        """
        rates = torch.sigmoid(self.auditory_encoder(audio))
        spikes = (torch.rand_like(rates) < rates).float()
        return spikes

    def step(self,
             visual_input: Optional[torch.Tensor] = None,
             auditory_input: Optional[torch.Tensor] = None,
             learn: bool = True,
             dopamine: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        한 시간 스텝 시뮬레이션

        Args:
            visual_input: 시각 입력 (64x64 이미지)
            auditory_input: 청각 입력 (256 특징)
            learn: STDP 학습 여부
            dopamine: 도파민 수준 (-1 ~ 2), 3-factor learning에 사용

        Returns:
            각 영역의 스파이크
        """
        self.step_count += 1
        spikes = {}

        # === 감각 입력 처리 ===
        if visual_input is not None:
            v1_input = self.encode_visual(visual_input.to(DEVICE))
        else:
            v1_input = torch.zeros(self.config.visual_v1, device=DEVICE)

        if auditory_input is not None:
            a1_input = self.encode_auditory(auditory_input.to(DEVICE))
        else:
            a1_input = torch.zeros(self.config.auditory_a1, device=DEVICE)

        # === V1 (+ recurrent + feedback) ===
        v1_recurrent = self.conn_v1_v1(self.v1.v_mem > 0.5)
        v1_feedback = self.conn_v2_v1(self.v2.v_mem > 0.5) if hasattr(self, '_prev_v2') else 0
        v1_total = v1_input + v1_recurrent * 0.5 + v1_feedback * 0.3
        spikes['v1'] = self.v1(v1_total)

        # === A1 ===
        spikes['a1'] = self.a1(a1_input)

        # === V2 ===
        v2_input = self.conn_v1_v2(spikes['v1'])
        spikes['v2'] = self.v2(v2_input)
        self._prev_v2 = spikes['v2']

        # === Temporal (시각 + 청각 통합) ===
        temporal_input = (self.conn_v2_temporal(spikes['v2']) +
                         self.conn_a1_temporal(spikes['a1']) +
                         self.conn_prefrontal_temporal(spikes.get('prefrontal', torch.zeros(self.config.prefrontal, device=DEVICE))) * 0.3)
        spikes['temporal'] = self.temporal(temporal_input)

        # === Parietal ===
        parietal_input = self.conn_v2_parietal(spikes['v2'])
        spikes['parietal'] = self.parietal(parietal_input)

        # === Hippocampus (기억 허브) ===
        hippo_input = (self.conn_temporal_hippo(spikes['temporal']) +
                       self.conn_parietal_hippo(spikes['parietal']) +
                       self.conn_prefrontal_hippo(spikes.get('prefrontal', torch.zeros(self.config.prefrontal, device=DEVICE))))
        spikes['hippocampus'] = self.hippocampus(hippo_input)

        # === Prefrontal (+ recurrent + hippocampus) ===
        pfc_input = (self.conn_temporal_prefrontal(spikes['temporal']) +
                     self.conn_parietal_prefrontal(spikes['parietal']) +
                     self.conn_hippo_prefrontal(spikes['hippocampus']) +
                     self.conn_prefrontal_prefrontal(self.prefrontal.v_mem > 0.5) * 0.5)
        spikes['prefrontal'] = self.prefrontal(pfc_input)

        # === Motor ===
        motor_input = self.conn_prefrontal_motor(spikes['prefrontal'])
        spikes['motor'] = self.motor(motor_input)

        # === STDP 학습 (도파민 조절) ===
        if learn:
            self._apply_stdp(spikes, dopamine)

        # === 통계 기록 ===
        total_spikes = sum(s.sum().item() for s in spikes.values())
        total_neurons = self.config.total_neurons()
        self.spike_history.append(total_spikes / total_neurons)

        # 주기적 항상성 조정
        if self.step_count % 100 == 0:
            self._apply_homeostasis()

        return spikes

    def _apply_stdp(self, spikes: Dict[str, torch.Tensor], dopamine: float = 0.0):
        """모든 연결에 도파민-조절 STDP 적용"""
        for conn, pre_name, post_name in self.all_connections:
            pre_spikes = spikes.get(pre_name, torch.zeros(conn.pre_size, device=DEVICE))
            post_spikes = spikes.get(post_name, torch.zeros(conn.post_size, device=DEVICE))
            conn.update_stdp(pre_spikes, post_spikes, dopamine=dopamine, use_eligibility=True)

    def _apply_homeostasis(self):
        """모든 레이어에 항상성 가소성 적용"""
        for layer in [self.v1, self.v2, self.a1, self.temporal,
                      self.parietal, self.prefrontal, self.hippocampus, self.motor]:
            layer.apply_homeostasis()

    def get_motor_action(self, spikes: Dict[str, torch.Tensor], n_actions: int = 4) -> int:
        """
        운동 피질 스파이크 → 행동 선택
        간단한 population coding
        """
        motor_spikes = spikes['motor']

        # 뉴런을 n_actions 그룹으로 나눔
        group_size = len(motor_spikes) // n_actions

        group_rates = []
        for i in range(n_actions):
            start = i * group_size
            end = start + group_size
            rate = motor_spikes[start:end].mean().item()
            group_rates.append(rate)

        # 가장 활성화된 그룹 선택
        return int(np.argmax(group_rates))

    def reset(self):
        """뇌 상태 초기화"""
        for layer in [self.v1, self.v2, self.a1, self.temporal,
                      self.parietal, self.prefrontal, self.hippocampus, self.motor]:
            layer.reset()
        self.step_count = 0

    def get_stats(self) -> Dict:
        """뇌 통계"""
        return {
            'step_count': self.step_count,
            'avg_firing_rate': np.mean(list(self.spike_history)) if self.spike_history else 0,
            'total_neurons': self.config.total_neurons(),
            'layers': {
                'v1': self.v1.firing_rate_avg.mean().item(),
                'v2': self.v2.firing_rate_avg.mean().item(),
                'temporal': self.temporal.firing_rate_avg.mean().item(),
                'prefrontal': self.prefrontal.firing_rate_avg.mean().item(),
                'motor': self.motor.firing_rate_avg.mean().item(),
            }
        }


def test_biological_brain(scale: str = "medium"):
    """생물학적 뇌 테스트"""
    print("\n" + "="*60)
    print(f"Testing Biological Brain (scale={scale})")
    print("="*60)

    if scale == "small":
        config = BiologicalConfig(
            visual_v1=10_000,
            visual_v2=5_000,
            auditory_a1=3_000,
            temporal=5_000,
            parietal=3_000,
            prefrontal=5_000,
            hippocampus=3_000,
            motor=2_000,
        )
    elif scale == "medium":
        # 8GB GPU에 맞게 조정 (135K neurons, ~1.4GB)
        config = BiologicalConfig(
            visual_v1=30_000,
            visual_v2=20_000,
            auditory_a1=10_000,
            temporal=20_000,
            parietal=15_000,
            prefrontal=20_000,
            hippocampus=15_000,
            motor=5_000,
            # 희소성 낮춤
            intra_region_sparsity=0.02,  # 5% → 2%
            inter_region_sparsity=0.005,  # 1% → 0.5%
        )
    elif scale == "optimal":
        # 8GB GPU 최적화: 100K neurons
        config = BiologicalConfig(
            visual_v1=25_000,
            visual_v2=15_000,
            auditory_a1=8_000,
            temporal=15_000,
            parietal=10_000,
            prefrontal=15_000,
            hippocampus=10_000,
            motor=4_000,
            intra_region_sparsity=0.01,  # 1%
            inter_region_sparsity=0.005,  # 0.5%
        )
    elif scale == "large":
        config = BiologicalConfig(
            visual_v1=100_000,
            visual_v2=60_000,
            auditory_a1=40_000,
            temporal=60_000,
            parietal=40_000,
            prefrontal=60_000,
            hippocampus=40_000,
            motor=20_000,
        )
    else:
        config = BiologicalConfig()

    brain = BiologicalBrain(config)

    print(f"\nRunning 100 timesteps...")
    start_time = time.time()

    for t in range(100):
        # 랜덤 시각 입력
        visual = torch.rand(64, 64) * 0.3  # 낮은 입력 강도
        spikes = brain.step(visual_input=visual, learn=True)

        if t % 20 == 0:
            rates = {k: f"{v.mean().item():.3f}" for k, v in spikes.items()}
            print(f"  Step {t}: {rates}")

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f}s ({100/elapsed:.1f} steps/sec)")

    stats = brain.get_stats()
    print(f"\nFinal stats:")
    print(f"  Total neurons: {stats['total_neurons']:,}")
    print(f"  Avg firing rate: {stats['avg_firing_rate']:.4f}")

    # GPU 메모리 확인
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"  GPU memory used: {allocated:.2f} GB")

    return brain


def find_max_scale():
    """하드웨어가 버틸 수 있는 최대 스케일 찾기"""
    print("\n" + "="*60)
    print("Finding maximum scale...")
    print("="*60)

    scales = [
        ("small", 36_000),
        ("medium", 210_000),
        ("large", 420_000),
    ]

    for name, expected in scales:
        try:
            print(f"\nTrying {name} ({expected:,} neurons)...")
            torch.cuda.empty_cache()
            brain = test_biological_brain(name)
            print(f"SUCCESS: {name} works!")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"FAILED: {name} - Out of GPU memory")
                break
            else:
                raise

    return brain


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "max":
        brain = find_max_scale()
    elif len(sys.argv) > 1:
        brain = test_biological_brain(sys.argv[1])
    else:
        brain = test_biological_brain("optimal")
