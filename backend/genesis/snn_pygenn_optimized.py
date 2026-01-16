"""
PyGeNN 기반 SNN - GPU 최적화 버전 (STDP on GPU)

핵심 최적화:
1. GPU에서 STDP 학습 실행 (CustomUpdate)
2. 배치 처리로 I/O 최소화
3. 도파민 조절도 GPU에서 처리
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from collections import deque
import os
import time

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

from pygenn import (GeNNModel, init_sparse_connectivity, init_weight_update,
                    init_postsynaptic, create_weight_update_model)


@dataclass
class PyGeNNOptConfig:
    """PyGeNN 최적화 설정"""
    n_sensory: int = 20000
    n_hidden: int = 120000
    n_motor: int = 13000
    sparsity: float = 0.01

    # LIF 파라미터
    tau_m: float = 20.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_thresh: float = -50.0
    tau_refrac: float = 2.0

    # STDP 파라미터
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    a_plus: float = 0.01
    a_minus: float = 0.012
    w_max: float = 1.0
    w_min: float = 0.0

    dt: float = 1.0
    da_baseline: float = 0.5


# DA-modulated STDP Weight Update Model (GPU에서 실행)
# PyGeNN 5.x: dopamine은 일반 param으로 선언 후 set_param_dynamic()으로 동적화
da_stdp_model = create_weight_update_model(
    "DA_STDP",
    params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax", "dopamine"],
    vars=[("g", "scalar"), ("eligibility", "scalar")],

    # Pre-synaptic spike: eligibility trace 업데이트
    pre_spike_syn_code="""
        // Decay eligibility and add LTD component
        eligibility = eligibility * exp(-dt / tauMinus) - aMinus;
    """,

    # Post-synaptic spike: eligibility trace 업데이트 + weight update
    post_spike_syn_code="""
        // Decay eligibility and add LTP component
        eligibility = eligibility * exp(-dt / tauPlus) + aPlus;

        // DA-modulated weight update
        scalar da_signal = dopamine - 0.5;
        g = fmin(wMax, fmax(wMin, g + da_signal * eligibility * 0.01));
    """,
)


class PyGeNNOptBrain:
    """GPU 최적화된 PyGeNN 기반 SNN"""

    def __init__(self, config: Optional[PyGeNNOptConfig] = None):
        self.config = config or PyGeNNOptConfig()
        self.step_count = 0
        self.dopamine = self.config.da_baseline

        # GeNN 모델 생성
        self.model = GeNNModel("float", "brain_opt")
        self.model.dt = self.config.dt

        # LIF 파라미터
        lif_params = {
            "C": 1.0,
            "TauM": self.config.tau_m,
            "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset,
            "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0}

        # 뉴런 그룹 생성
        self.sensory = self.model.add_neuron_population(
            "sensory", self.config.n_sensory, "LIF", lif_params, lif_init
        )
        self.hidden = self.model.add_neuron_population(
            "hidden", self.config.n_hidden, "LIF", lif_params, lif_init
        )
        self.motor = self.model.add_neuron_population(
            "motor", self.config.n_motor, "LIF", lif_params, lif_init
        )

        # STDP 시냅스 파라미터 (dopamine 포함)
        stdp_params = {
            "tauPlus": self.config.tau_plus,
            "tauMinus": self.config.tau_minus,
            "aPlus": self.config.a_plus,
            "aMinus": self.config.a_minus,
            "wMin": self.config.w_min,
            "wMax": self.config.w_max,
            "dopamine": self.config.da_baseline,
        }

        # 가중치 초기화
        fan_in_sh = self.config.n_sensory * self.config.sparsity
        fan_in_hm = self.config.n_hidden * self.config.sparsity
        std_sh = 1.0 / np.sqrt(fan_in_sh) if fan_in_sh > 0 else 0.1
        std_hm = 1.0 / np.sqrt(fan_in_hm) if fan_in_hm > 0 else 0.1

        stdp_init_sh = {"g": std_sh, "eligibility": 0.0}
        stdp_init_hm = {"g": std_hm, "eligibility": 0.0}

        # DA-STDP 시냅스 (GPU에서 학습)
        self.syn_sh = self.model.add_synapse_population(
            "sensory_hidden", "SPARSE",
            self.sensory, self.hidden,
            init_weight_update(da_stdp_model, stdp_params, stdp_init_sh),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": self.config.sparsity})
        )

        self.syn_hm = self.model.add_synapse_population(
            "hidden_motor", "SPARSE",
            self.hidden, self.motor,
            init_weight_update(da_stdp_model, stdp_params, stdp_init_hm),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": self.config.sparsity})
        )

        # dopamine을 dynamic param으로 설정 (weight update model param)
        self.syn_sh.set_wu_param_dynamic("dopamine")
        self.syn_hm.set_wu_param_dynamic("dopamine")

        # 빌드 및 로드
        print("Building GPU-optimized PyGeNN model...")
        self.model.build()
        self.model.load()
        print(f"Model ready! Neurons: {self.config.n_sensory + self.config.n_hidden + self.config.n_motor:,}")

    def forward(self, sensory_input: np.ndarray, get_output: bool = True) -> Optional[np.ndarray]:
        """순전파 (GPU에서 실행)"""
        # 입력 설정
        self.sensory.vars["V"].view[:] = self.config.v_rest + sensory_input * 20.0
        self.sensory.vars["V"].push_to_device()

        # GPU 시뮬레이션 스텝
        self.model.step_time()
        self.step_count += 1

        if get_output:
            self.motor.vars["V"].pull_from_device()
            v = self.motor.vars["V"].view.copy()
            v_norm = (v - self.config.v_rest) / (self.config.v_thresh - self.config.v_rest)
            return np.clip(v_norm, 0.0, 1.0).astype(np.float32)
        return None

    def set_dopamine(self, dopamine: float):
        """도파민 레벨 설정 (GPU로 전송)"""
        self.dopamine = np.clip(dopamine, 0.0, 1.0)
        # Dynamic param 업데이트
        self.syn_sh.set_dynamic_param_value("dopamine", self.dopamine)
        self.syn_hm.set_dynamic_param_value("dopamine", self.dopamine)

    def update_dopamine_from_activity(self, reward: float = 0.0):
        """활동 기반 도파민 업데이트"""
        # RefracTime으로 최근 스파이크 수 추정
        self.hidden.vars["RefracTime"].pull_from_device()
        refrac = self.hidden.vars["RefracTime"].view
        spike_count = np.sum(refrac > 0)
        activity = spike_count / self.config.n_hidden

        # Novelty 계산 (간단한 방식)
        novelty = min(activity * 10, 1.0)  # 활동이 많으면 novelty

        # 도파민 계산
        new_da = self.config.da_baseline + 0.3 * novelty + 0.5 * reward
        self.set_dopamine(new_da)

    def reset(self):
        """상태 초기화"""
        for pop in [self.sensory, self.hidden, self.motor]:
            pop.vars["V"].view[:] = self.config.v_rest
            pop.vars["V"].push_to_device()

    def get_statistics(self) -> Dict:
        """통계"""
        return {
            'total_neurons': self.config.n_sensory + self.config.n_hidden + self.config.n_motor,
            'steps': self.step_count,
            'dopamine': float(self.dopamine)
        }


def benchmark_optimized():
    """최적화된 PyGeNN 벤치마크"""
    from gpu_monitor import start_monitoring, stop_monitoring

    print("=" * 60)
    print("PyGeNN Optimized SNN Benchmark (GPU STDP)")
    print("=" * 60)

    # GPU 모니터링 시작
    monitor = start_monitoring(interval=0.5)

    config = PyGeNNOptConfig(
        n_sensory=20000,
        n_hidden=120000,
        n_motor=13000,
        sparsity=0.01
    )

    print(f"\nNeurons: {config.n_sensory + config.n_hidden + config.n_motor:,}")
    print(f"Sparsity: {config.sparsity * 100}%")

    brain = PyGeNNOptBrain(config)

    # Warmup
    print("\nWarmup...")
    for _ in range(10):
        sensory = np.random.rand(config.n_sensory).astype(np.float32)
        brain.forward(sensory)

    # Benchmark 1: Pure GPU
    print("\n[1] Pure GPU (no I/O)...")
    start = time.time()
    for _ in range(1000):
        brain.model.step_time()
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s | Steps/sec: {1000/elapsed:.1f}")
    monitor.print_status()

    # Benchmark 2: Forward only
    print("\n[2] Forward only (with I/O)...")
    start = time.time()
    for _ in range(1000):
        sensory = np.random.rand(config.n_sensory).astype(np.float32)
        brain.forward(sensory)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s | Steps/sec: {1000/elapsed:.1f}")
    monitor.print_status()

    # Benchmark 3: With GPU STDP learning
    print("\n[3] With GPU STDP learning...")
    brain.set_dopamine(0.7)  # 높은 도파민으로 학습 활성화
    start = time.time()
    for i in range(1000):
        sensory = np.random.rand(config.n_sensory).astype(np.float32)
        brain.forward(sensory)
        if i % 100 == 0:
            brain.update_dopamine_from_activity(reward=0.1)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s | Steps/sec: {1000/elapsed:.1f}")
    monitor.print_status()

    print(f"\nFinal stats: {brain.get_statistics()}")

    # GPU 모니터링 요약
    stop_monitoring()


if __name__ == '__main__':
    benchmark_optimized()
