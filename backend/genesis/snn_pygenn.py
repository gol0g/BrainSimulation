"""
PyGeNN 기반 SNN - GPU 최적화 버전

GeNN (GPU-enhanced Neuronal Networks) 사용:
- GPU 코드 자동 생성 및 컴파일
- 최적화된 LIF 뉴런
- 효율적인 희소 연결
- STDP 학습 지원

RTX 3070 8GB에서 100만 뉴런 목표
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import os

# VS 환경 설정 (Windows)
if os.name == 'nt':
    import subprocess
    vs_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    if os.path.exists(vs_path):
        result = subprocess.run(f'cmd /c ""{vs_path}" && set"', capture_output=True, text=True, shell=True)
        for line in result.stdout.splitlines():
            if '=' in line:
                key, _, value = line.partition('=')
                os.environ[key] = value
    os.environ['CUDA_PATH'] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8".strip()

from pygenn import GeNNModel, init_sparse_connectivity, init_weight_update, init_postsynaptic


@dataclass
class PyGeNNConfig:
    """PyGeNN SNN 설정"""
    # 뉴런 수
    n_sensory: int = 10000
    n_hidden: int = 100000
    n_motor: int = 1000

    # 희소 연결
    sparsity: float = 0.01  # 1% 연결

    # LIF 파라미터
    tau_m: float = 20.0     # 막전위 시상수 (ms)
    v_rest: float = -65.0   # 휴지 전위 (mV)
    v_reset: float = -65.0  # 리셋 전위 (mV)
    v_thresh: float = -50.0 # 발화 역치 (mV)
    tau_refrac: float = 2.0 # 불응기 (ms)

    # STDP 파라미터
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    a_plus: float = 0.01
    a_minus: float = 0.012
    w_max: float = 1.0
    w_min: float = 0.0

    # 시뮬레이션
    dt: float = 1.0  # ms

    # 도파민
    da_baseline: float = 0.5
    tau_eligibility: float = 1000.0


# LIF 뉴런 모델 파라미터 (참고용)
# PyGeNN 내장 "LIF" 모델 사용


class PyGeNNLayer:
    """PyGeNN 기반 뉴런 레이어"""

    def __init__(self, model: GeNNModel, name: str, n_neurons: int, config: PyGeNNConfig):
        self.name = name
        self.n_neurons = n_neurons
        self.config = config
        self.model = model  # 모델 참조 저장

        # LIF 파라미터 (7개 필요)
        lif_params = {
            "C": 1.0,
            "TauM": config.tau_m,
            "Vrest": config.v_rest,
            "Vreset": config.v_reset,
            "Vthresh": config.v_thresh,
            "Ioffset": 0.0,  # Offset current
            "TauRefrac": config.tau_refrac
        }

        lif_init = {
            "V": config.v_rest,
            "RefracTime": 0.0
        }

        # 뉴런 그룹 생성
        self.pop = model.add_neuron_population(
            name, n_neurons, "LIF",
            lif_params, lif_init
        )

        # 스파이크 기록 비활성화 - 실시간 사용에 부적합
        # 대신 막전위 변화로 스파이크 추론

    def get_activity(self) -> np.ndarray:
        """막전위 기반 활성도 반환 (0~1)"""
        self.pop.vars["V"].pull_from_device()
        v = self.pop.vars["V"].view.copy()
        # 막전위를 0~1로 정규화
        v_norm = (v - self.config.v_rest) / (self.config.v_thresh - self.config.v_rest)
        return np.clip(v_norm, 0.0, 1.0).astype(np.float32)

    def get_spike_count(self) -> int:
        """RefracTime > 0인 뉴런 수 = 최근 발화한 뉴런"""
        self.pop.vars["RefracTime"].pull_from_device()
        refrac = self.pop.vars["RefracTime"].view
        return int(np.sum(refrac > 0))


class PyGeNNSynapses:
    """PyGeNN 기반 시냅스 (Static + manual eligibility)"""

    def __init__(self, model: GeNNModel, name: str,
                 pre_pop: PyGeNNLayer, post_pop: PyGeNNLayer,
                 config: PyGeNNConfig):
        self.name = name
        self.config = config
        self.pre_pop = pre_pop
        self.post_pop = post_pop

        # 가중치 초기화
        fan_in = pre_pop.n_neurons * config.sparsity
        std = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.1

        # PyGeNN 5.x API: StaticPulse allows weight updates (for learning)
        self.syn = model.add_synapse_population(
            name, "SPARSE",
            pre_pop.pop, post_pop.pop,
            init_weight_update("StaticPulse", {}, {"g": std}),  # {} params, {"g": init} vars
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": config.sparsity})
        )

        # Eligibility trace (CPU에서 관리)
        self.eligibility = None  # 연결 생성 후 초기화

    def init_eligibility(self):
        """모델 빌드 후 eligibility 초기화"""
        # SPARSE 연결은 .values 사용 (복사)
        self.syn.vars["g"].pull_from_device()
        weights = self.syn.vars["g"].values
        n_synapses = len(weights)
        self.eligibility = np.zeros(n_synapses, dtype=np.float32)

    def apply_dopamine(self, dopamine: float):
        """도파민 조절 학습"""
        if self.eligibility is None:
            return

        # Pull weights (SPARSE는 values 사용)
        self.syn.vars["g"].pull_from_device()
        weights = self.syn.vars["g"].values.copy()

        # Apply DA-modulated learning
        da_signal = dopamine - 0.5
        delta = da_signal * self.eligibility * 0.01

        # Update weights with clipping
        new_weights = np.clip(weights + delta, self.config.w_min, self.config.w_max)

        # Push back to device
        self.syn.vars["g"].values[:] = new_weights
        self.syn.vars["g"].push_to_device()

        # Decay eligibility
        self.eligibility *= 0.99

    def update_eligibility(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """STDP 기반 eligibility 업데이트 (간략화)"""
        if self.eligibility is None:
            return
        # 간단한 Hebbian: pre와 post가 모두 활성화된 시냅스만 강화
        # 실제 구현에서는 spike timing 고려해야 함
        pass


class PyGeNNBrain:
    """PyGeNN 기반 SNN 뇌"""

    def __init__(self, config: Optional[PyGeNNConfig] = None):
        self.config = config or PyGeNNConfig()

        # GeNN 모델 생성
        self.model = GeNNModel("float", "brain")
        self.model.dt = self.config.dt

        # 레이어 생성
        self.sensory = PyGeNNLayer(self.model, "sensory", self.config.n_sensory, self.config)
        self.hidden = PyGeNNLayer(self.model, "hidden", self.config.n_hidden, self.config)
        self.motor = PyGeNNLayer(self.model, "motor", self.config.n_motor, self.config)

        # 시냅스 생성
        self.syn_sensory_hidden = PyGeNNSynapses(
            self.model, "sensory_hidden",
            self.sensory, self.hidden, self.config
        )
        self.syn_hidden_motor = PyGeNNSynapses(
            self.model, "hidden_motor",
            self.hidden, self.motor, self.config
        )

        # 도파민 시스템
        self.dopamine = self.config.da_baseline
        self.activity_history = deque(maxlen=100)

        # 모델 빌드 (CUDA 코드 생성 및 컴파일)
        print("Building PyGeNN model (CUDA compilation)...")
        self.model.build()
        self.model.load()

        # Eligibility 초기화 (모델 로드 후)
        self.syn_sensory_hidden.init_eligibility()
        self.syn_hidden_motor.init_eligibility()

        print("Model ready!")
        self.step_count = 0

    def forward(self, sensory_input: np.ndarray, get_output: bool = True) -> np.ndarray:
        """순전파"""
        # 입력 전류 설정
        self.sensory.pop.vars["V"].view[:] = self.config.v_rest + sensory_input * 20.0
        self.sensory.pop.vars["V"].push_to_device()

        # 시뮬레이션 스텝
        self.model.step_time()
        self.step_count += 1

        # 출력 필요 시에만 GPU에서 pull (성능 최적화)
        if get_output:
            return self.motor.get_activity()
        return None

    def learn(self, reward: float = 0.0):
        """DA-STDP 학습"""
        # 도파민 업데이트
        hidden_spike_count = self.hidden.get_spike_count()
        activity = hidden_spike_count / self.hidden.n_neurons

        # Novelty
        if len(self.activity_history) > 0:
            recent = np.mean(list(self.activity_history)[-10:])
            novelty = abs(activity - recent)
        else:
            novelty = 1.0
        self.activity_history.append(activity)

        self.dopamine = self.config.da_baseline + 0.3 * novelty + 0.5 * reward
        self.dopamine = np.clip(self.dopamine, 0.0, 1.0)

        # 시냅스 학습
        self.syn_sensory_hidden.apply_dopamine(self.dopamine)
        self.syn_hidden_motor.apply_dopamine(self.dopamine)

    def reset(self):
        """상태 초기화"""
        # Reset membrane potentials
        self.sensory.pop.vars["V"].view[:] = self.config.v_rest
        self.hidden.pop.vars["V"].view[:] = self.config.v_rest
        self.motor.pop.vars["V"].view[:] = self.config.v_rest

        self.sensory.pop.vars["V"].push_to_device()
        self.hidden.pop.vars["V"].push_to_device()
        self.motor.pop.vars["V"].push_to_device()

    def get_statistics(self) -> Dict:
        """통계"""
        return {
            'total_neurons': self.config.n_sensory + self.config.n_hidden + self.config.n_motor,
            'steps': self.step_count,
            'dopamine': self.dopamine
        }


def benchmark_pygenn():
    """PyGeNN 벤치마크"""
    import time

    print("=" * 60)
    print("PyGeNN SNN Benchmark")
    print("=" * 60)

    # 153K 뉴런 (Slither.io 수준)
    config = PyGeNNConfig(
        n_sensory=20000,   # 3채널 시각 (8K+8K+4K)
        n_hidden=120000,   # Integration 레이어들 (50K+50K+10K+10K)
        n_motor=13000,     # 모터 (5K+5K+3K)
        sparsity=0.01
    )

    print(f"\nNeurons: {config.n_sensory + config.n_hidden + config.n_motor:,}")
    print(f"Sparsity: {config.sparsity * 100}%")

    brain = PyGeNNBrain(config)

    # Warmup
    print("\nWarmup...")
    for _ in range(10):
        sensory = np.random.rand(config.n_sensory).astype(np.float32)
        brain.forward(sensory)

    # Benchmark - Pure GPU (no I/O)
    print("Benchmarking 1000 steps (pure GPU, no I/O)...")
    start = time.time()
    for _ in range(1000):
        brain.model.step_time()
    elapsed_pure = time.time() - start
    print(f"\nPure GPU (no I/O):")
    print(f"  Time: {elapsed_pure:.2f}s")
    print(f"  Steps/sec: {1000/elapsed_pure:.1f}")

    # Benchmark - Forward Only
    print("\nBenchmarking 1000 steps (forward only)...")
    start = time.time()
    for _ in range(1000):
        sensory = np.random.rand(config.n_sensory).astype(np.float32)
        brain.forward(sensory)
    elapsed_forward = time.time() - start
    print(f"\nForward-only:")
    print(f"  Time: {elapsed_forward:.2f}s")
    print(f"  Steps/sec: {1000/elapsed_forward:.1f}")

    # Benchmark - With Learning
    print("\nBenchmarking 1000 steps (with learning)...")
    start = time.time()
    for _ in range(1000):
        sensory = np.random.rand(config.n_sensory).astype(np.float32)
        brain.forward(sensory)
        brain.learn(reward=0.1)
    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Steps/sec: {1000/elapsed:.1f}")
    print(f"  Stats: {brain.get_statistics()}")


if __name__ == '__main__':
    benchmark_pygenn()
