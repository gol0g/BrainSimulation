"""
Spiking Neural Network Brain

실제 뇌처럼 작동하는 인공 뇌

핵심 원리:
1. LIF (Leaky Integrate-and-Fire) 뉴런 - 실제 뉴런처럼 발화
2. STDP (Spike-Timing Dependent Plasticity) - 헤비안 학습
3. 희소 활성화 - 동시에 소수만 발화
4. 계층적 예측 코딩 - 오차만 전파
5. 시간적 동역학 - 스파이크 타이밍이 정보

목표: GPU 브루트포스가 아닌, 뇌 원리로 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time

# Norse - Spiking Neural Network library
import norse.torch as norse
from norse.torch.module.lif import LIFCell, LIFParameters
from norse.torch.functional.lif import LIFState

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class BrainConfig:
    """뇌 구성 설정"""
    # 뉴런 수 (실제 뇌의 축소판)
    sensory_neurons: int = 1024      # 감각 입력
    v1_neurons: int = 2048           # 1차 시각
    v2_neurons: int = 1024           # 2차 시각
    association_neurons: int = 512   # 연합 영역
    prefrontal_neurons: int = 256    # 전전두엽 (계획/추론)
    motor_neurons: int = 64          # 운동 출력

    # LIF 뉴런 파라미터
    tau_mem: float = 20.0    # 막전위 시상수 (ms)
    tau_syn: float = 5.0     # 시냅스 시상수 (ms)
    v_th: float = 1.0        # 발화 임계값
    v_reset: float = 0.0     # 리셋 전위

    # STDP 파라미터
    tau_plus: float = 20.0   # LTP 시상수
    tau_minus: float = 20.0  # LTD 시상수
    a_plus: float = 0.01     # LTP 강도
    a_minus: float = 0.012   # LTD 강도 (약간 더 큼 = 안정성)
    w_max: float = 1.0       # 최대 시냅스 가중치
    w_min: float = 0.0       # 최소 시냅스 가중치

    # 희소성
    target_sparsity: float = 0.05  # 목표: 5% 뉴런만 활성

    # 시간 스텝
    dt: float = 1.0          # 시뮬레이션 시간 단위 (ms)
    time_steps: int = 50     # 입력당 시간 스텝


class STDPSynapse(nn.Module):
    """
    STDP 시냅스 - 헤비안 학습

    "Neurons that fire together wire together"
    - pre before post: LTP (강화)
    - post before pre: LTD (약화)
    """

    def __init__(self, in_features: int, out_features: int, config: BrainConfig):
        super().__init__()
        self.config = config

        # 시냅스 가중치
        self.weight = nn.Parameter(
            torch.rand(out_features, in_features) * 0.3
        )

        # STDP 추적 변수 (학습 시 업데이트)
        self.register_buffer('pre_trace', torch.zeros(in_features))
        self.register_buffer('post_trace', torch.zeros(out_features))

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """시냅스 전달"""
        # pre_spikes: (batch, in_features)
        return F.linear(pre_spikes, self.weight)

    def update_stdp(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        STDP 가중치 업데이트

        Local learning rule - 역전파 없음!
        """
        with torch.no_grad():
            # Trace 업데이트 (지수 감쇠)
            decay_pre = np.exp(-self.config.dt / self.config.tau_plus)
            decay_post = np.exp(-self.config.dt / self.config.tau_minus)

            self.pre_trace = self.pre_trace * decay_pre + pre_spikes.mean(0)
            self.post_trace = self.post_trace * decay_post + post_spikes.mean(0)

            # STDP 규칙
            # LTP: post spike → pre trace 기반 강화
            # LTD: pre spike → post trace 기반 약화

            # (out, in) 형태로 맞춤
            ltp = self.config.a_plus * torch.outer(post_spikes.mean(0), self.pre_trace)
            ltd = self.config.a_minus * torch.outer(self.post_trace, pre_spikes.mean(0))

            # 가중치 업데이트
            self.weight.data += ltp - ltd

            # 범위 제한
            self.weight.data.clamp_(self.config.w_min, self.config.w_max)

    def reset_traces(self):
        """Trace 리셋"""
        self.pre_trace.zero_()
        self.post_trace.zero_()


class SpikingLayer(nn.Module):
    """
    스파이킹 뉴런 레이어

    LIF 뉴런 + STDP 시냅스
    """

    def __init__(self, in_features: int, out_features: int, config: BrainConfig):
        super().__init__()
        self.config = config
        self.out_features = out_features

        # STDP 시냅스
        self.synapse = STDPSynapse(in_features, out_features, config)

        # LIF 뉴런 파라미터
        self.lif_params = LIFParameters(
            tau_mem_inv=1.0 / config.tau_mem,
            tau_syn_inv=1.0 / config.tau_syn,
            v_th=torch.tensor(config.v_th),
            v_reset=torch.tensor(config.v_reset),
        )

        # LIF 셀
        self.lif = LIFCell(p=self.lif_params)

        # 뉴런 상태
        self.state = None

    def forward(self, input_spikes: torch.Tensor,
                learn: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파

        Returns:
            output_spikes: 출력 스파이크
            membrane: 막전위 (디버깅/분석용)
        """
        # 시냅스 전달
        synaptic_input = self.synapse(input_spikes)

        # 상태 초기화 (Norse API)
        if self.state is None:
            self.state = self.lif.initial_state(synaptic_input)

        # LIF 뉴런 업데이트
        output_spikes, self.state = self.lif(synaptic_input, self.state)

        # STDP 학습
        if learn:
            self.synapse.update_stdp(input_spikes, output_spikes)

        return output_spikes, self.state.v

    def reset_state(self):
        """뉴런 상태 리셋"""
        self.state = None
        self.synapse.reset_traces()

    def get_sparsity(self) -> float:
        """현재 희소성 계산"""
        if self.state is None:
            return 0.0
        # 막전위가 임계값의 50% 이상인 뉴런 비율
        active = (self.state.v > self.config.v_th * 0.5).float().mean()
        return active.item()


class PredictiveCodingLayer(nn.Module):
    """
    예측 코딩 레이어 (스파이킹 버전)

    - 상위 레이어가 하위 레이어를 예측
    - 예측 오차만 상위로 전파
    - 뇌의 핵심 계산 원리
    """

    def __init__(self, lower_dim: int, higher_dim: int, config: BrainConfig):
        super().__init__()
        self.config = config

        # 상향 경로 (오차 전파)
        self.error_pathway = SpikingLayer(lower_dim, higher_dim, config)

        # 하향 경로 (예측 생성)
        self.prediction_pathway = SpikingLayer(higher_dim, lower_dim, config)

        # 예측 오차 버퍼
        self.register_buffer('prediction_error', torch.zeros(1, lower_dim))

    def forward(self, lower_activity: torch.Tensor,
                higher_activity: Optional[torch.Tensor] = None,
                learn: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        예측 코딩 순전파

        Args:
            lower_activity: 하위 레이어 활동
            higher_activity: 상위 레이어 활동 (있으면 예측 생성)

        Returns:
            error_signal: 상위로 전파될 오차
            prediction: 생성된 예측
        """
        # 예측 생성 (상위 → 하위)
        if higher_activity is not None:
            prediction, _ = self.prediction_pathway(higher_activity, learn=learn)
        else:
            prediction = torch.zeros_like(lower_activity)

        # 예측 오차 계산
        self.prediction_error = lower_activity - prediction

        # 오차를 상위로 전파
        error_signal, _ = self.error_pathway(
            torch.relu(self.prediction_error),  # 양의 오차만
            learn=learn
        )

        return error_signal, prediction

    def reset_state(self):
        self.error_pathway.reset_state()
        self.prediction_pathway.reset_state()


class WorkingMemory(nn.Module):
    """
    작업 기억 - 지속적 활동을 통한 정보 유지

    전전두엽의 지속적 발화 패턴 모사
    """

    def __init__(self, dim: int, config: BrainConfig):
        super().__init__()
        self.dim = dim
        self.config = config

        # 자기 순환 연결 (정보 유지)
        self.recurrent = SpikingLayer(dim, dim, config)

        # 게이팅 (무엇을 기억할지)
        self.gate = nn.Linear(dim, dim)

        # 기억 상태
        self.register_buffer('memory_state', torch.zeros(1, dim))

    def forward(self, input_spikes: torch.Tensor,
                learn: bool = True) -> torch.Tensor:
        """
        작업 기억 업데이트
        """
        # 게이트 계산 (어떤 정보를 저장할지)
        gate = torch.sigmoid(self.gate(input_spikes))

        # 순환 연결로 기존 기억 유지
        recurrent_activity, _ = self.recurrent(self.memory_state, learn=learn)

        # 새 입력과 기존 기억 결합
        self.memory_state = gate * input_spikes + (1 - gate) * recurrent_activity

        return self.memory_state

    def reset(self):
        self.memory_state.zero_()
        self.recurrent.reset_state()


class SpikingBrain(nn.Module):
    """
    스파이킹 신경망 기반 인공 뇌

    구조:
    - 감각 입력 → V1 → V2 → 연합 영역 → 전전두엽 → 운동 출력
    - 각 단계에서 예측 코딩
    - STDP로 연속 학습
    - 작업 기억으로 정보 유지
    """

    def __init__(self, config: Optional[BrainConfig] = None):
        super().__init__()
        self.config = config or BrainConfig()

        # 감각 인코딩 (픽셀 → 스파이크)
        self.sensory_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 63 * 63, self.config.sensory_neurons)
        )

        # 시각 경로 (계층적)
        self.v1 = SpikingLayer(
            self.config.sensory_neurons,
            self.config.v1_neurons,
            self.config
        )
        self.v2 = SpikingLayer(
            self.config.v1_neurons,
            self.config.v2_neurons,
            self.config
        )

        # 예측 코딩 연결
        self.pc_v1_v2 = PredictiveCodingLayer(
            self.config.v1_neurons,
            self.config.v2_neurons,
            self.config
        )

        # 연합 영역
        self.association = SpikingLayer(
            self.config.v2_neurons,
            self.config.association_neurons,
            self.config
        )

        # 전전두엽 (계획/추론)
        self.prefrontal = SpikingLayer(
            self.config.association_neurons,
            self.config.prefrontal_neurons,
            self.config
        )

        # 작업 기억
        self.working_memory = WorkingMemory(
            self.config.prefrontal_neurons,
            self.config
        )

        # 운동 출력
        self.motor = SpikingLayer(
            self.config.prefrontal_neurons,
            self.config.motor_neurons,
            self.config
        )

        # 상태 추적
        self.total_spikes = 0
        self.time_step = 0

    def encode_sensory(self, image: torch.Tensor) -> torch.Tensor:
        """
        이미지를 스파이크 트레인으로 인코딩

        Rate coding: 밝기 → 발화율
        """
        # CNN으로 특징 추출
        features = self.sensory_encoder(image)

        # 포아송 스파이크 생성 (rate coding)
        rates = torch.sigmoid(features)  # 0-1 발화율
        spikes = torch.bernoulli(rates)  # 확률적 스파이크

        return spikes

    def forward(self, image: torch.Tensor,
                learn: bool = True) -> Dict[str, torch.Tensor]:
        """
        순전파 - 전체 처리 파이프라인
        """
        batch_size = image.shape[0]

        # 감각 인코딩
        sensory_spikes = self.encode_sensory(image)

        # 시간에 따른 스파이크 누적
        v1_spikes_acc = torch.zeros(batch_size, self.config.v1_neurons, device=image.device)
        v2_spikes_acc = torch.zeros(batch_size, self.config.v2_neurons, device=image.device)
        motor_spikes_acc = torch.zeros(batch_size, self.config.motor_neurons, device=image.device)

        # 시간 스텝 시뮬레이션
        for t in range(self.config.time_steps):
            # V1 처리
            v1_spikes, v1_membrane = self.v1(sensory_spikes, learn=learn)

            # V1 → V2 예측 코딩
            v2_error, v2_pred = self.pc_v1_v2(v1_spikes, None, learn=learn)

            # V2 처리
            v2_spikes, v2_membrane = self.v2(v1_spikes, learn=learn)

            # 연합 영역
            assoc_spikes, _ = self.association(v2_spikes, learn=learn)

            # 전전두엽
            pfc_spikes, _ = self.prefrontal(assoc_spikes, learn=learn)

            # 작업 기억 업데이트
            wm_state = self.working_memory(pfc_spikes, learn=learn)

            # 운동 출력
            motor_spikes, _ = self.motor(wm_state, learn=learn)

            # 누적
            v1_spikes_acc += v1_spikes
            v2_spikes_acc += v2_spikes
            motor_spikes_acc += motor_spikes

            self.time_step += 1

        # 스파이크 카운트
        self.total_spikes += (v1_spikes_acc.sum() + v2_spikes_acc.sum() +
                             motor_spikes_acc.sum()).item()

        return {
            'sensory': sensory_spikes,
            'v1': v1_spikes_acc / self.config.time_steps,
            'v2': v2_spikes_acc / self.config.time_steps,
            'motor': motor_spikes_acc / self.config.time_steps,
            'motor_raw': motor_spikes_acc,
        }

    def get_action(self, motor_spikes: torch.Tensor) -> int:
        """운동 스파이크 → 행동 변환"""
        # 가장 많이 발화한 뉴런 = 선택된 행동
        spike_counts = motor_spikes.sum(dim=0)
        action = spike_counts.argmax().item()
        return action

    def reset_state(self):
        """모든 상태 리셋"""
        self.v1.reset_state()
        self.v2.reset_state()
        self.pc_v1_v2.reset_state()
        self.association.reset_state()
        self.prefrontal.reset_state()
        self.working_memory.reset()
        self.motor.reset_state()

    def get_statistics(self) -> Dict:
        """뇌 통계"""
        return {
            'total_spikes': self.total_spikes,
            'time_steps': self.time_step,
            'avg_spikes_per_step': self.total_spikes / max(self.time_step, 1),
            'v1_sparsity': self.v1.get_sparsity(),
            'v2_sparsity': self.v2.get_sparsity(),
        }


def test_spiking_brain():
    """스파이킹 뇌 테스트"""
    print("=" * 60)
    print("Spiking Neural Network Brain Test")
    print("=" * 60)

    # 설정
    config = BrainConfig(
        sensory_neurons=512,
        v1_neurons=1024,
        v2_neurons=512,
        association_neurons=256,
        prefrontal_neurons=128,
        motor_neurons=16,
        time_steps=20,
    )

    # 뇌 생성
    brain = SpikingBrain(config).to(DEVICE)

    # 파라미터 수
    total_params = sum(p.numel() for p in brain.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Device: {DEVICE}")

    # 테스트 이미지 (256x256 RGB)
    print("\n[1] Processing test image...")
    test_image = torch.rand(1, 3, 256, 256).to(DEVICE)

    start_time = time.time()
    output = brain(test_image, learn=True)
    elapsed = time.time() - start_time

    print(f"    Processing time: {elapsed*1000:.1f} ms")
    print(f"    Sensory spikes: {output['sensory'].sum().item():.0f}")
    print(f"    V1 activity: {output['v1'].mean().item():.4f}")
    print(f"    V2 activity: {output['v2'].mean().item():.4f}")
    print(f"    Motor activity: {output['motor'].mean().item():.4f}")

    # 행동 선택
    action = brain.get_action(output['motor_raw'])
    print(f"    Selected action: {action}")

    # 통계
    print("\n[2] Brain statistics:")
    stats = brain.get_statistics()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    # 연속 처리 테스트
    print("\n[3] Continuous processing (10 frames)...")
    for i in range(10):
        test_image = torch.rand(1, 3, 256, 256).to(DEVICE)
        output = brain(test_image, learn=True)

    print("    Done!")
    stats = brain.get_statistics()
    print(f"    Total spikes after 10 frames: {stats['total_spikes']:.0f}")

    # STDP 학습 확인
    print("\n[4] STDP learning check:")
    w_before = brain.v1.synapse.weight.clone()

    # 같은 이미지 반복 (패턴 학습)
    pattern = torch.rand(1, 3, 256, 256).to(DEVICE)
    for _ in range(20):
        brain(pattern, learn=True)

    w_after = brain.v1.synapse.weight
    w_change = (w_after - w_before).abs().mean().item()
    print(f"    Weight change (V1): {w_change:.6f}")
    print(f"    STDP learning: {'Active' if w_change > 0 else 'Inactive'}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == '__main__':
    test_spiking_brain()
