"""
Scalable SNN - snnTorch 기반 (100만 뉴런 목표)

snnTorch 사용:
- 최적화된 LIF 뉴런 구현
- GPU 가속 자동 지원
- Surrogate gradient 지원

RTX 3070 8GB에서 100만 뉴런을 목표로 설계:
- 희소 연결 (0.1% connectivity)
- Float16 가중치
- DA-STDP 지역 학습 (backprop 없음)

생물학적 제약 준수:
- Dale's Law: 흥분/억제 뉴런 분리
- STDP: 스파이크 타이밍 기반 가소성
- 도파민 조절: 3-factor learning rule
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class ScalableSNNConfig:
    """스케일러블 SNN 설정"""
    # 뉴런 수 (스케일별)
    n_sensory: int = 10000      # 감각 뉴런
    n_hidden: int = 100000      # 은닉 뉴런 (핵심)
    n_motor: int = 1000         # 운동 뉴런

    # 희소 연결
    sparsity: float = 0.001     # 0.1% 연결

    # LIF 파라미터 (snnTorch)
    beta: float = 0.9           # 막전위 감쇠 (beta = exp(-dt/tau))
    threshold: float = 1.0      # 발화 역치

    # STDP 파라미터
    tau_plus: float = 20.0      # LTP 시상수
    tau_minus: float = 20.0     # LTD 시상수
    a_plus: float = 0.01        # LTP 강도
    a_minus: float = 0.012      # LTD 강도 (약간 더 강함)

    # 도파민 파라미터
    da_baseline: float = 0.5    # 기저 도파민
    tau_eligibility: float = 1000.0  # 적격성 흔적 시상수 (ms)

    # 시뮬레이션
    dt: float = 1.0             # 시간 스텝 (ms)

    # Dale's Law
    excitatory_ratio: float = 0.8   # 흥분 뉴런 비율

    # 메모리 최적화
    use_float16: bool = True    # Float16 사용


class SparseSynapses:
    """
    희소 시냅스 구현 (PyTorch Sparse Tensor 사용)

    COO 형식으로 저장:
    - indices: (2, n_connections) - 연결 좌표
    - values: (n_connections,) - 가중치
    """

    def __init__(self, n_pre: int, n_post: int, sparsity: float,
                 excitatory_ratio: float = 0.8, use_float16: bool = True):
        self.n_pre = n_pre
        self.n_post = n_post
        self.sparsity = sparsity
        self.excitatory_ratio = excitatory_ratio
        self.dtype = torch.float16 if use_float16 else torch.float32

        # 연결 수 계산
        self.n_connections = int(n_pre * n_post * sparsity)

        # Dale's Law: 각 프리시냅스 뉴런은 흥분 또는 억제
        self.is_excitatory = torch.rand(n_pre, device=DEVICE) < excitatory_ratio

        # 랜덤 연결 생성
        self._create_random_connections()

        # 적격성 흔적 (sparse하게 저장)
        self.eligibility = torch.zeros(self.n_connections, dtype=self.dtype, device=DEVICE)

        # 통계
        self.total_potentiation = 0.0
        self.total_depression = 0.0

    def _create_random_connections(self):
        """랜덤 희소 연결 생성"""
        # 랜덤 연결 인덱스
        pre_indices = torch.randint(0, self.n_pre, (self.n_connections,), device=DEVICE)
        post_indices = torch.randint(0, self.n_post, (self.n_connections,), device=DEVICE)

        self.indices = torch.stack([pre_indices, post_indices])

        # 가중치 초기화 - 더 강한 연결
        # sparse 보상: sqrt(fan_in) 스케일링
        fan_in = self.n_pre * self.sparsity
        std = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.1
        self.weights = torch.randn(self.n_connections, dtype=self.dtype, device=DEVICE) * std

        # Dale's Law 적용 (흥분/억제 분리)
        signs = torch.where(self.is_excitatory[pre_indices],
                           torch.ones(self.n_connections, device=DEVICE),
                           -0.5 * torch.ones(self.n_connections, device=DEVICE))  # 억제 약하게
        self.weights = self.weights.abs() * signs.to(self.dtype)

        # Sparse tensor 생성
        self._rebuild_sparse()

    def _rebuild_sparse(self):
        """Sparse tensor 재구성 (lazy rebuild for performance)"""
        self.sparse_weights = torch.sparse_coo_tensor(
            self.indices.long(),
            self.weights,
            size=(self.n_pre, self.n_post),
            dtype=self.dtype,
            device=DEVICE
        ).coalesce()
        # Cache transposed float32 version for fast forward pass
        self._sparse_t_f32 = self.sparse_weights.float().t()
        self._needs_rebuild = False

    def _ensure_sparse(self):
        """Lazy rebuild: only rebuild when needed"""
        if getattr(self, '_needs_rebuild', True):
            self._rebuild_sparse()

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """
        스파이크 전파

        Args:
            pre_spikes: (n_pre,) 프리시냅스 스파이크

        Returns:
            (n_post,) 포스트시냅스 입력 전류
        """
        # Ensure sparse tensor is up to date
        self._ensure_sparse()

        # Sparse matrix-vector multiplication (use cached transposed float32)
        result = torch.sparse.mm(self._sparse_t_f32, pre_spikes.unsqueeze(1).float()).squeeze()
        return result

    def update_eligibility(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                           tau: float, dt: float):
        """
        적격성 흔적 업데이트

        STDP 기반:
        - Pre before Post: LTP eligible (trace-based)
        - Post before Pre: LTD eligible (weaker)
        """
        # 감쇠
        decay = np.exp(-dt / tau)
        self.eligibility *= decay

        # Get spikes at connection indices
        pre_active = pre_spikes[self.indices[0]]
        post_active = post_spikes[self.indices[1]]

        # Maintain spike traces for proper STDP timing
        if not hasattr(self, 'pre_trace'):
            self.pre_trace = torch.zeros(self.n_connections, dtype=self.dtype, device=DEVICE)
            self.post_trace = torch.zeros(self.n_connections, dtype=self.dtype, device=DEVICE)

        # Update traces (decay + spike)
        trace_decay = np.exp(-dt / 20.0)  # 20ms trace decay
        self.pre_trace = self.pre_trace * trace_decay + pre_active.to(self.dtype)
        self.post_trace = self.post_trace * trace_decay + post_active.to(self.dtype)

        # STDP: post spike with pre trace → LTP
        ltp = post_active.to(self.dtype) * self.pre_trace
        # STDP: pre spike with post trace → LTD (weaker)
        ltd = pre_active.to(self.dtype) * self.post_trace * 0.3

        # Update eligibility (biased towards potentiation)
        self.eligibility += (ltp - ltd)

    def apply_dopamine(self, dopamine: float, a_plus: float, a_minus: float):
        """
        도파민 조절 학습

        3-factor rule: Δw = DA × eligibility
        """
        # 도파민이 높으면 강화, 낮으면 약화
        da_signal = dopamine - 0.5  # 기저선 대비

        # 가중치 업데이트
        delta = da_signal * self.eligibility

        # Dale's Law 유지
        signs = torch.where(self.is_excitatory[self.indices[0]],
                           torch.ones(self.n_connections, device=DEVICE),
                           -torch.ones(self.n_connections, device=DEVICE)).to(self.dtype)

        self.weights = self.weights + delta * 0.01
        self.weights = self.weights.abs() * signs  # 부호 유지

        # 가중치 클리핑
        self.weights = torch.clamp(self.weights, -1.0, 1.0)

        # 통계
        self.total_potentiation += (delta > 0).sum().item()
        self.total_depression += (delta < 0).sum().item()

        # Mark for lazy rebuild (will rebuild on next forward pass)
        self._needs_rebuild = True


class SNNTorchLayer(nn.Module):
    """
    snnTorch 기반 LIF 레이어

    snnTorch.Leaky 사용:
    - 자동 GPU 최적화
    - Surrogate gradient 지원
    - 더 효율적인 메모리 사용
    """

    def __init__(self, n_neurons: int, config: ScalableSNNConfig):
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config

        # snnTorch Leaky 뉴런 (surrogate gradient 포함)
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif = snn.Leaky(
            beta=config.beta,
            threshold=config.threshold,
            spike_grad=spike_grad,
            init_hidden=False,  # 수동으로 mem 관리
            reset_mechanism='zero'
        )

        # 상태 저장 (외부 접근용)
        self.spikes = torch.zeros(n_neurons, device=DEVICE)
        self.mem = torch.zeros(n_neurons, device=DEVICE)
        self.trace = torch.zeros(n_neurons, device=DEVICE)

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        LIF 업데이트 (snnTorch)
        """
        # snnTorch forward (mem 명시적 전달)
        self.spikes, self.mem = self.lif(input_current, self.mem)

        # 스파이크 흔적 업데이트
        trace_decay = np.exp(-self.config.dt / self.config.tau_plus)
        self.trace = self.trace * trace_decay + self.spikes

        return self.spikes

    def reset(self):
        """상태 초기화"""
        self.spikes = torch.zeros(self.n_neurons, device=DEVICE)
        self.mem = torch.zeros(self.n_neurons, device=DEVICE)
        self.trace = torch.zeros(self.n_neurons, device=DEVICE)


class DopamineSystem:
    """
    도파민 시스템 (VTA/SNc 모델)

    Novelty 기반 도파민 분비:
    - 새로운 패턴 → 도파민 증가
    - 익숙한 패턴 → 도파민 감소 (습관화)
    """

    def __init__(self, config: ScalableSNNConfig):
        self.config = config
        self.baseline = config.da_baseline
        self.level = self.baseline

        # 익숙함 추적
        self.activity_history = deque(maxlen=100)
        self.habituation = 0.0

        # 보상 예측 오차
        self.predicted_reward = 0.0

    def update(self, activity: torch.Tensor, reward: float = 0.0):
        """
        도파민 레벨 업데이트

        Args:
            activity: 현재 뉴런 활동 패턴
            reward: 외부 보상 (선택적)
        """
        # 활동 요약
        activity_summary = activity.mean().item()

        # Novelty 계산
        if len(self.activity_history) > 0:
            recent = np.mean(list(self.activity_history)[-10:])
            novelty = abs(activity_summary - recent)
        else:
            novelty = 1.0

        self.activity_history.append(activity_summary)

        # 습관화 (반복 자극에 novelty 감소)
        self.habituation = 0.95 * self.habituation + 0.05 * (1 - novelty)
        novelty_adjusted = novelty * (1 - 0.5 * self.habituation)

        # 보상 예측 오차
        rpe = reward - self.predicted_reward
        self.predicted_reward = 0.9 * self.predicted_reward + 0.1 * reward

        # 도파민 계산
        self.level = self.baseline + 0.3 * novelty_adjusted + 0.5 * rpe
        self.level = np.clip(self.level, 0.0, 1.0)

        return self.level


class ScalableSNN(nn.Module):
    """
    스케일러블 SNN - snnTorch 기반 100만 뉴런 목표

    아키텍처:
    Sensory (10K) → Hidden (100K~1M) → Motor (1K)

    모든 연결은 0.1% 희소
    """

    def __init__(self, config: Optional[ScalableSNNConfig] = None):
        super().__init__()
        self.config = config or ScalableSNNConfig()

        # snnTorch 레이어
        self.sensory = SNNTorchLayer(self.config.n_sensory, self.config)
        self.hidden = SNNTorchLayer(self.config.n_hidden, self.config)
        self.motor = SNNTorchLayer(self.config.n_motor, self.config)

        # 시냅스 (희소)
        self.syn_sensory_hidden = SparseSynapses(
            self.config.n_sensory,
            self.config.n_hidden,
            self.config.sparsity,
            self.config.excitatory_ratio,
            self.config.use_float16
        )

        self.syn_hidden_motor = SparseSynapses(
            self.config.n_hidden,
            self.config.n_motor,
            self.config.sparsity,
            self.config.excitatory_ratio,
            self.config.use_float16
        )

        # Recurrent (hidden-hidden) - 매우 희소하게
        self.syn_hidden_hidden = SparseSynapses(
            self.config.n_hidden,
            self.config.n_hidden,
            self.config.sparsity * 0.1,  # 0.01% recurrent
            self.config.excitatory_ratio,
            self.config.use_float16
        )

        # 도파민 시스템
        self.dopamine = DopamineSystem(self.config)

        # 통계
        self.step_count = 0
        self.spike_counts = {'sensory': 0, 'hidden': 0, 'motor': 0}

    def forward(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            sensory_input: (n_sensory,) 감각 입력

        Returns:
            (n_motor,) 운동 출력 스파이크
        """
        # Sensory layer
        sensory_spikes = self.sensory(sensory_input * 5.0)
        hidden_input = self.syn_sensory_hidden.forward(sensory_spikes)

        # Recurrent in Hidden
        if self.step_count > 0:
            recurrent_input = self.syn_hidden_hidden.forward(self.hidden.spikes)
            hidden_input = hidden_input + 0.5 * recurrent_input

        # Hidden processing
        hidden_spikes = self.hidden(hidden_input * 5.0)

        # Hidden → Motor
        motor_input = self.syn_hidden_motor.forward(hidden_spikes)
        motor_spikes = self.motor(motor_input * 20.0)

        # 통계 업데이트
        self.spike_counts['sensory'] += sensory_spikes.sum().item()
        self.spike_counts['hidden'] += hidden_spikes.sum().item()
        self.spike_counts['motor'] += motor_spikes.sum().item()
        self.step_count += 1

        return motor_spikes

    def learn(self, reward: float = 0.0):
        """
        DA-STDP 학습

        3-factor rule:
        1. Pre-post spike timing → Eligibility
        2. Dopamine → Gate
        3. Δw = DA × eligibility
        """
        # 도파민 업데이트
        da = self.dopamine.update(self.hidden.spikes, reward)

        # 적격성 업데이트
        self.syn_sensory_hidden.update_eligibility(
            self.sensory.spikes,
            self.hidden.spikes,
            self.config.tau_eligibility,
            self.config.dt
        )

        self.syn_hidden_motor.update_eligibility(
            self.hidden.spikes,
            self.motor.spikes,
            self.config.tau_eligibility,
            self.config.dt
        )

        self.syn_hidden_hidden.update_eligibility(
            self.hidden.spikes,
            self.hidden.spikes,
            self.config.tau_eligibility,
            self.config.dt
        )

        # 도파민 조절 가중치 업데이트
        self.syn_sensory_hidden.apply_dopamine(da, self.config.a_plus, self.config.a_minus)
        self.syn_hidden_motor.apply_dopamine(da, self.config.a_plus, self.config.a_minus)
        self.syn_hidden_hidden.apply_dopamine(da * 0.5, self.config.a_plus, self.config.a_minus)

    def reset(self):
        """상태 초기화"""
        self.sensory.reset()
        self.hidden.reset()
        self.motor.reset()

    def get_statistics(self) -> Dict:
        """통계"""
        total_neurons = self.config.n_sensory + self.config.n_hidden + self.config.n_motor
        total_synapses = (self.syn_sensory_hidden.n_connections +
                        self.syn_hidden_motor.n_connections +
                        self.syn_hidden_hidden.n_connections)

        return {
            'total_neurons': total_neurons,
            'total_synapses': total_synapses,
            'steps': self.step_count,
            'spike_counts': self.spike_counts.copy(),
            'dopamine_level': self.dopamine.level,
            'memory_mb': self._estimate_memory(),
        }

    def _estimate_memory(self) -> float:
        """메모리 사용량 추정 (MB)"""
        bytes_per_element = 2 if self.config.use_float16 else 4

        # 시냅스 가중치
        syn_memory = (self.syn_sensory_hidden.n_connections +
                     self.syn_hidden_motor.n_connections +
                     self.syn_hidden_hidden.n_connections) * bytes_per_element

        # 뉴런 상태
        neuron_memory = (self.config.n_sensory + self.config.n_hidden + self.config.n_motor) * 4 * 3

        # 인덱스 (int64)
        index_memory = (self.syn_sensory_hidden.n_connections +
                       self.syn_hidden_motor.n_connections +
                       self.syn_hidden_hidden.n_connections) * 8 * 2

        total = syn_memory + neuron_memory + index_memory
        return total / (1024 * 1024)


# ============================================================================
# Adaptive LIF Layer - Neural Adaptation (Short-term Synaptic Depression)
# ============================================================================
class AdaptiveLIFLayer(nn.Module):
    """
    적응형 LIF 뉴런 - 신경 순응 (Neural Adaptation) 구현

    생물학적 원리:
    - 뉴런이 반복 발화하면 "피로"가 쌓여 발화가 어려워짐
    - 시간이 지나면 피로가 회복됨
    - 이로 인해 같은 행동을 계속 반복하지 않고 자연스럽게 행동 전환

    수식:
    - effective_threshold = base_threshold + adaptation
    - 발화 시: adaptation += adapt_beta
    - 매 스텝: adaptation *= exp(-dt/tau_adapt)
    """

    def __init__(self, n_neurons: int, config: ScalableSNNConfig,
                 tau_adapt: float = 100.0, adapt_beta: float = 0.5):
        """
        Args:
            n_neurons: 뉴런 수
            config: SNN 설정
            tau_adapt: 피로 회복 시상수 (ms) - 클수록 천천히 회복
            adapt_beta: 발화당 피로 증가량 - 클수록 빨리 피로해짐
        """
        super().__init__()
        self.n_neurons = n_neurons
        self.config = config
        self.tau_adapt = tau_adapt
        self.adapt_beta = adapt_beta

        # snnTorch Leaky 뉴런 (기본)
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif = snn.Leaky(
            beta=config.beta,
            threshold=config.threshold,
            spike_grad=spike_grad,
            init_hidden=False,
            reset_mechanism='zero'
        )

        # 상태
        self.spikes = torch.zeros(n_neurons, device=DEVICE)
        self.mem = torch.zeros(n_neurons, device=DEVICE)
        self.trace = torch.zeros(n_neurons, device=DEVICE)

        # 적응 상태 (피로도)
        self.adaptation = torch.zeros(n_neurons, device=DEVICE)

    def forward(self, input_current: torch.Tensor) -> torch.Tensor:
        """
        적응형 LIF 업데이트

        1. 피로도만큼 입력 전류 감소 (effective threshold 증가 효과)
        2. 발화 시 피로도 증가
        3. 시간에 따라 피로도 자연 감소
        """
        # 피로도에 따라 입력 감소 (임계값 증가와 동일 효과)
        # adaptation이 높으면 발화하기 어려워짐
        effective_input = input_current - self.adaptation

        # snnTorch forward
        self.spikes, self.mem = self.lif(effective_input, self.mem)

        # 발화한 뉴런의 피로도 증가
        self.adaptation = self.adaptation + self.spikes * self.adapt_beta

        # 시간에 따른 피로도 자연 감소 (회복)
        decay = np.exp(-self.config.dt / self.tau_adapt)
        self.adaptation = self.adaptation * decay

        # 스파이크 흔적 업데이트
        trace_decay = np.exp(-self.config.dt / self.config.tau_plus)
        self.trace = self.trace * trace_decay + self.spikes

        return self.spikes

    def reset(self):
        """상태 초기화"""
        self.spikes = torch.zeros(self.n_neurons, device=DEVICE)
        self.mem = torch.zeros(self.n_neurons, device=DEVICE)
        self.trace = torch.zeros(self.n_neurons, device=DEVICE)
        self.adaptation = torch.zeros(self.n_neurons, device=DEVICE)

    def get_fatigue_level(self) -> float:
        """평균 피로도 반환 (디버깅용)"""
        return self.adaptation.mean().item()


# ============================================================================
# Backward Compatibility: SparseLIFLayer alias
# ============================================================================
SparseLIFLayer = SNNTorchLayer


def benchmark_scalable_snn():
    """스케일러블 SNN 벤치마크"""
    print("=" * 70)
    print("Scalable SNN Benchmark (snnTorch) - 꿀벌 스케일 목표")
    print("=" * 70)

    # GPU 정보
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWARNING: CUDA not available, using CPU")

    # 스케일별 테스트
    scales = [
        ("C. elegans", ScalableSNNConfig(n_sensory=50, n_hidden=200, n_motor=50)),
        ("Fruit fly (1%)", ScalableSNNConfig(n_sensory=500, n_hidden=5000, n_motor=500)),
        ("Fruit fly (10%)", ScalableSNNConfig(n_sensory=2000, n_hidden=20000, n_motor=2000)),
        ("Honeybee (1%)", ScalableSNNConfig(n_sensory=5000, n_hidden=50000, n_motor=5000)),
        ("Honeybee (10%)", ScalableSNNConfig(n_sensory=10000, n_hidden=100000, n_motor=10000)),
    ]

    results = []

    for name, config in scales:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")

        try:
            # SNN 생성
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            start_time = time.time()
            snn = ScalableSNN(config).to(DEVICE)
            init_time = time.time() - start_time

            stats = snn.get_statistics()
            print(f"  Neurons: {stats['total_neurons']:,}")
            print(f"  Synapses: {stats['total_synapses']:,}")
            print(f"  Est. Memory: {stats['memory_mb']:.1f} MB")
            print(f"  Init Time: {init_time:.2f}s")

            # 실행 테스트 (100 스텝)
            start_time = time.time()
            for _ in range(100):
                sensory = torch.rand(config.n_sensory, device=DEVICE)
                output = snn.forward(sensory)
                snn.learn(reward=0.1)
            run_time = time.time() - start_time

            print(f"  Run Time (100 steps): {run_time:.2f}s")
            print(f"  Steps/sec: {100/run_time:.1f}")

            # 메모리 사용량
            if torch.cuda.is_available():
                mem_used = torch.cuda.max_memory_allocated() / 1e9
                print(f"  Actual VRAM: {mem_used:.2f} GB")

            results.append({
                'name': name,
                'neurons': stats['total_neurons'],
                'synapses': stats['total_synapses'],
                'memory_mb': stats['memory_mb'],
                'steps_per_sec': 100/run_time,
                'success': True
            })

            del snn
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'name': name,
                'success': False,
                'error': str(e)
            })

    # 요약
    print(f"\n{'='*70}")
    print("SUMMARY (snnTorch Backend)")
    print(f"{'='*70}")

    max_neurons = max([r.get('neurons', 0) for r in results if r.get('success', False)], default=0)
    print(f"\nMaximum achievable scale: ~{max_neurons:,} neurons")

    if max_neurons >= 1000000:
        print("SUCCESS: Bee-scale (1M) achievable!")
    elif max_neurons >= 100000:
        print("Fruit fly scale (100K) achievable")
    else:
        print("Limited to C. elegans scale")

    return results


def test_learning():
    """학습 테스트"""
    print("=" * 70)
    print("DA-STDP Learning Test (snnTorch)")
    print("=" * 70)

    config = ScalableSNNConfig(
        n_sensory=1000,
        n_hidden=10000,
        n_motor=100,
        sparsity=0.01,
    )

    snn = ScalableSNN(config).to(DEVICE)

    # 연속 입력 패턴
    sensory_pattern = torch.rand(config.n_sensory, device=DEVICE) * 0.8 + 0.2

    # 학습 전 반응
    print("\n[Before Learning]")
    initial_response = []
    for trial in range(5):
        snn.reset()
        total_motor = 0
        for _ in range(50):
            output = snn.forward(sensory_pattern)
            total_motor += output.sum().item()
        initial_response.append(total_motor)
        print(f"  Trial {trial+1}: motor spikes = {total_motor:.0f}")

    print(f"  Average: {np.mean(initial_response):.1f} ± {np.std(initial_response):.1f}")

    # 학습 (보상 있음)
    print("\n[Learning with reward]")
    for ep in range(10):
        snn.reset()
        episode_spikes = 0
        for _ in range(100):
            output = snn.forward(sensory_pattern)
            snn.learn(reward=0.8)
            episode_spikes += output.sum().item()
        if ep % 2 == 0:
            print(f"  Episode {ep+1}: motor spikes = {episode_spikes:.0f}, DA = {snn.dopamine.level:.3f}")

    # 학습 후 반응
    print("\n[After Learning]")
    final_response = []
    for trial in range(5):
        snn.reset()
        total_motor = 0
        for _ in range(50):
            output = snn.forward(sensory_pattern)
            total_motor += output.sum().item()
        final_response.append(total_motor)
        print(f"  Trial {trial+1}: motor spikes = {total_motor:.0f}")

    print(f"  Average: {np.mean(final_response):.1f} ± {np.std(final_response):.1f}")

    # 변화량
    delta = np.mean(final_response) - np.mean(initial_response)
    print(f"\n  Delta: {delta:+.1f} spikes")

    stats = snn.get_statistics()
    print(f"\n  Potentiation events: {snn.syn_sensory_hidden.total_potentiation + snn.syn_hidden_motor.total_potentiation:,.0f}")
    print(f"  Depression events: {snn.syn_sensory_hidden.total_depression + snn.syn_hidden_motor.total_depression:,.0f}")
    print(f"  Final dopamine: {stats['dopamine_level']:.3f}")


if __name__ == '__main__':
    benchmark_scalable_snn()
    print("\n")
    test_learning()
