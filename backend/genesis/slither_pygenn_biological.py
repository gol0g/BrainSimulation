"""
Slither.io PyGeNN Agent - Biological Architecture

snnTorch의 생물학적 회로를 PyGeNN GPU 가속으로 이식:
1. 감각 분리 (Sensory Segregation): Food Eye / Enemy Eye / Body Eye
2. 선천적 본능 (Innate Reflex): 적 회피 시냅스 3x 부스트
3. 억제 회로 (Lateral Inhibition): Fear --| Hunger

Target: Best 57+ (snnTorch 기록 돌파)
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
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
                    init_postsynaptic, create_weight_update_model, create_neuron_model)
from slither_gym import SlitherGym, SlitherConfig

# Checkpoint directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "slither_pygenn_bio"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# === R-STDP (Reward-Modulated STDP) with Long Eligibility Trace ===
# 핵심: 3초 전 행동도 기억하는 "화학적 흔적"
#
# 2-Trace System:
#   1. stdp_trace (τ=20ms): spike timing 감지 (LTP/LTD 부호 결정)
#   2. eligibility (τ=3000ms): 보상까지 기억 유지 (3초간 생존)
#
# 동작:
#   Pre-spike → stdp_trace 감소 (LTD 준비)
#   Post-spike → stdp_trace 증가 (LTP)
#   매 스텝 → stdp_trace를 eligibility로 누적, 둘 다 감쇠
#   보상 → eligibility 기반으로 가중치 업데이트 (Dopamine Shower)
#
r_stdp_model = create_weight_update_model(
    "R_STDP_LONG_TRACE",
    params=["tauStdp", "tauElig", "aPlus", "aMinus", "wMin", "wMax", "dopamine", "eta"],
    vars=[("g", "scalar"), ("stdp_trace", "scalar"), ("eligibility", "scalar")],
    pre_spike_syn_code="""
        // Pre-spike: 전류 전달 + LTD 준비
        addToPost(g);
        stdp_trace -= aMinus;
    """,
    post_spike_syn_code="""
        // Post-spike: LTP (양의 흔적)
        stdp_trace += aPlus;
    """,
    # 매 시뮬레이션 스텝마다 실행 (synapse dynamics)
    synapse_dynamics_code="""
        // 1. STDP trace 감쇠 (빠름: 20ms)
        stdp_trace *= exp(-dt / tauStdp);

        // 2. STDP → Eligibility 누적 (흔적 전달)
        eligibility += stdp_trace * dt * 0.01;

        // 3. Eligibility trace 감쇠 (느림: 3000ms)
        eligibility *= exp(-dt / tauElig);

        // 4. 도파민 조절 학습 (보상 시에만 유의미)
        scalar da_signal = dopamine - 0.5;
        if (fabs(da_signal) > 0.1) {
            // 보상/벌칙이 있을 때만 가중치 업데이트
            g = fmin(wMax, fmax(wMin, g + eta * da_signal * eligibility));
        }
    """,
)

# Legacy: 이전 DA-STDP 모델 (비교용으로 유지)
da_stdp_model = create_weight_update_model(
    "DA_STDP_BIO",
    params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax", "dopamine"],
    vars=[("g", "scalar"), ("eligibility", "scalar")],
    pre_spike_syn_code="""
        eligibility = eligibility * exp(-dt / tauMinus) - aMinus;
    """,
    post_spike_syn_code="""
        eligibility = eligibility * exp(-dt / tauPlus) + aPlus;
        scalar da_signal = dopamine - 0.5;
        g = fmin(wMax, fmax(wMin, g + da_signal * eligibility * 0.01));
    """,
)

# === Adaptive LIF Model (Motor neurons only) ===
# 승자 독식 방지: 많이 발화할수록 임계값 상승 → 피로 → 다른 뉴런에게 기회
adaptive_lif_model = create_neuron_model(
    "AdaptiveLIF",
    params=["C", "TauM", "Vrest", "Vreset", "VthreshBase", "TauAdapt", "Beta", "Ioffset", "TauRefrac"],
    vars=[("V", "scalar"), ("Vthresh", "scalar"), ("RefracTime", "scalar")],
    sim_code="""
        // Refractory period check
        if (RefracTime > 0.0) {
            RefracTime -= dt;
        } else {
            // Threshold adaptation decay (towards baseline)
            Vthresh += (VthreshBase - Vthresh) * (dt / TauAdapt);
            // Standard LIF dynamics
            V += (-(V - Vrest) + Ioffset) * (dt / TauM);
        }
    """,
    threshold_condition_code="""
        RefracTime <= 0.0 && V >= Vthresh
    """,
    reset_code="""
        V = Vreset;
        Vthresh += Beta;  // Threshold increases on spike (fatigue)
        RefracTime = TauRefrac;
    """
)


@dataclass
class BiologicalConfig:
    """생물학적 PyGeNN 설정"""
    n_rays: int = 32

    # === SENSORY (분리된 채널) ===
    n_food_eye: int = 8000       # Food detection only
    n_enemy_eye: int = 8000      # Enemy detection only
    n_body_eye: int = 4000       # Self-body detection

    # === SPECIALIZED CIRCUITS ===
    n_hunger_circuit: int = 10000   # Food seeking drive
    n_fear_circuit: int = 10000     # Danger avoidance drive
    n_attack_circuit: int = 5000    # Predator attack drive (적 추적)

    # === INTEGRATION (Mushroom Body) ===
    n_integration_1: int = 50000    # First integration
    n_integration_2: int = 50000    # Second integration

    # === MOTOR ===
    n_motor_left: int = 5000     # Turn left
    n_motor_right: int = 5000    # Turn right
    n_motor_boost: int = 3000    # Emergency boost

    @classmethod
    def lite(cls) -> "BiologicalConfig":
        """경량 설정 - GPU 메모리 절약용 (50K neurons)"""
        return cls(
            n_food_eye=2000,
            n_enemy_eye=2000,
            n_body_eye=1000,
            n_hunger_circuit=3000,
            n_fear_circuit=3000,
            n_attack_circuit=1500,
            n_integration_1=15000,
            n_integration_2=15000,
            n_motor_left=1500,
            n_motor_right=1500,
            n_motor_boost=1000,
            sparsity=0.01,  # 약간 더 조밀한 연결
        )

    @classmethod
    def dev(cls) -> "BiologicalConfig":
        """개발/디버깅용 초경량 설정 (15K neurons)"""
        return cls(
            n_food_eye=800,
            n_enemy_eye=800,
            n_body_eye=400,
            n_hunger_circuit=1000,
            n_fear_circuit=1000,
            n_attack_circuit=500,
            n_integration_1=4000,
            n_integration_2=4000,
            n_motor_left=500,
            n_motor_right=500,
            n_motor_boost=300,
            sparsity=0.02,  # 더 조밀한 연결로 보상
        )

    # Network parameters
    sparsity: float = 0.005      # 0.5% connectivity

    # LIF parameters
    tau_m: float = 20.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_thresh: float = -50.0
    tau_refrac: float = 2.0

    # === R-STDP parameters (Long Eligibility Trace) ===
    # 핵심: 3초 전 행동도 기억하는 "화학적 흔적"
    tau_stdp: float = 20.0       # STDP 타이밍 윈도우 (빠름)
    tau_eligibility: float = 3000.0  # 자격 흔적 시간 (3초 = 3000ms)
    a_plus: float = 0.005        # LTP 강도
    a_minus: float = 0.006       # LTD 강도
    eta: float = 0.01            # 도파민 조절 학습률
    w_max: float = 1.0
    w_min: float = 0.0

    # Legacy STDP (비교용)
    tau_plus: float = 20.0
    tau_minus: float = 20.0

    # Biological parameters
    innate_boost: float = 3.0       # 선천적 회피 본능 강도
    fear_inhibition: float = 0.8    # 공포가 배고픔 억제하는 강도
    inhibitory_weight: float = -2.0 # 억제 시냅스 가중치

    # === WTA (Winner-Take-All) 측면 억제 ===
    wta_inhibition: float = -3.0    # WTA 억제 강도 (강할수록 승자 독식)
    wta_sparsity: float = 0.02      # WTA 연결 희소성

    # === Adaptive Threshold (Motor neurons only) ===
    # "고인 물은 썩는다" - 승자도 지쳐야 교대가 일어남
    # NOTE: WTA와 함께 사용 시 역효과 발생 - 비활성화
    tau_adaptation: float = 2000.0  # 피로 회복 시간상수 (ms) - 2초
    beta_adaptation: float = 0.0    # 0 = 비활성화 (WTA only mode)

    dt: float = 1.0

    @property
    def total_neurons(self) -> int:
        return (self.n_food_eye + self.n_enemy_eye + self.n_body_eye +
                self.n_hunger_circuit + self.n_fear_circuit + self.n_attack_circuit +
                self.n_integration_1 + self.n_integration_2 +
                self.n_motor_left + self.n_motor_right + self.n_motor_boost)


class BiologicalBrain:
    """
    생물학적 회로 구조의 PyGeNN 뇌

    Architecture:
    ```
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │   Food Eye   │  │  Enemy Eye   │  │   Body Eye   │
    │    (8K)      │  │    (8K)      │  │    (4K)      │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐
    │Hunger Circuit│◄-│ Fear Circuit │  (Fear --| Hunger)
    │    (10K)     │  │    (10K)     │
    └──────┬───────┘  └──────┬───────┘
           │                 │
           └────────┬────────┘
                    ▼
         ┌────────────────────┐
         │   Integration 1    │
         │       (50K)        │
         └─────────┬──────────┘
                   ▼
         ┌────────────────────┐
         │   Integration 2    │
         │       (50K)        │
         └─────────┬──────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    ┌───────┐  ┌───────┐  ┌───────┐
    │ Left  │  │ Right │  │ Boost │
    │ (5K)  │  │ (5K)  │  │ (3K)  │
    └───────┘  └───────┘  └───────┘

    Innate Reflex (Cross-wired, 3x boost):
    Enemy LEFT  ──────────────────► RIGHT Motor
    Enemy RIGHT ──────────────────► LEFT Motor
    ```
    """

    def __init__(self, config: Optional[BiologicalConfig] = None):
        self.config = config or BiologicalConfig()

        print(f"Building Biological PyGeNN Brain ({self.config.total_neurons:,} neurons)...")

        # GeNN 모델 생성
        self.model = GeNNModel("float", "slither_bio")
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

        # === 1. SENSORY POPULATIONS (분리!) ===
        self.food_eye = self.model.add_neuron_population(
            "food_eye", self.config.n_food_eye, "LIF", lif_params, lif_init)
        self.enemy_eye = self.model.add_neuron_population(
            "enemy_eye", self.config.n_enemy_eye, "LIF", lif_params, lif_init)
        self.body_eye = self.model.add_neuron_population(
            "body_eye", self.config.n_body_eye, "LIF", lif_params, lif_init)

        print(f"  Sensory: Food({self.config.n_food_eye:,}) + Enemy({self.config.n_enemy_eye:,}) + Body({self.config.n_body_eye:,})")

        # === 2. SPECIALIZED CIRCUITS ===
        self.hunger = self.model.add_neuron_population(
            "hunger", self.config.n_hunger_circuit, "LIF", lif_params, lif_init)
        self.fear = self.model.add_neuron_population(
            "fear", self.config.n_fear_circuit, "LIF", lif_params, lif_init)
        self.attack = self.model.add_neuron_population(
            "attack", self.config.n_attack_circuit, "LIF", lif_params, lif_init)

        print(f"  Circuits: Hunger({self.config.n_hunger_circuit:,}) + Fear({self.config.n_fear_circuit:,}) + Attack({self.config.n_attack_circuit:,})")

        # === 3. INTEGRATION LAYERS ===
        self.integration_1 = self.model.add_neuron_population(
            "integration_1", self.config.n_integration_1, "LIF", lif_params, lif_init)
        self.integration_2 = self.model.add_neuron_population(
            "integration_2", self.config.n_integration_2, "LIF", lif_params, lif_init)

        print(f"  Integration: {self.config.n_integration_1 + self.config.n_integration_2:,}")

        # === 4. MOTOR POPULATIONS (Standard LIF + WTA) ===
        # Adaptive Threshold 실험 결과: 역효과 확인됨 → Standard LIF 사용
        self.motor_left = self.model.add_neuron_population(
            "motor_left", self.config.n_motor_left, "LIF", lif_params, lif_init)
        self.motor_right = self.model.add_neuron_population(
            "motor_right", self.config.n_motor_right, "LIF", lif_params, lif_init)
        self.motor_boost = self.model.add_neuron_population(
            "motor_boost", self.config.n_motor_boost, "LIF", lif_params, lif_init)

        print(f"  Motor: Left({self.config.n_motor_left:,}) + Right({self.config.n_motor_right:,}) + Boost({self.config.n_motor_boost:,})")

        # === R-STDP 파라미터 (3초 Eligibility Trace) ===
        r_stdp_params = {
            "tauStdp": self.config.tau_stdp,       # 20ms (spike timing)
            "tauElig": self.config.tau_eligibility, # 3000ms (3초 기억)
            "aPlus": self.config.a_plus,
            "aMinus": self.config.a_minus,
            "wMin": self.config.w_min,
            "wMax": self.config.w_max,
            "dopamine": 0.5,
            "eta": self.config.eta,  # 도파민 학습률
        }

        # Legacy STDP 파라미터 (비교용)
        stdp_params = {
            "tauPlus": self.config.tau_plus,
            "tauMinus": self.config.tau_minus,
            "aPlus": self.config.a_plus,
            "aMinus": self.config.a_minus,
            "wMin": self.config.w_min,
            "wMax": self.config.w_max,
            "dopamine": 0.5,
        }

        # 시냅스 생성 헬퍼 (R-STDP 사용)
        def create_synapse(name, pre, post, n_pre, n_post, sparsity=None, w_init=None):
            sp = sparsity or self.config.sparsity
            fan_in = n_pre * sp
            std = w_init if w_init else (1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.1)
            syn = self.model.add_synapse_population(
                name, "SPARSE", pre, post,
                init_weight_update(r_stdp_model, r_stdp_params,
                                   {"g": std, "stdp_trace": 0.0, "eligibility": 0.0}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}),
                init_sparse_connectivity("FixedProbability", {"prob": sp})
            )
            syn.set_wu_param_dynamic("dopamine")
            return syn

        # === SYNAPTIC CONNECTIONS ===
        self.all_synapses = []

        # Sensory → Circuits
        self.syn_food_hunger = create_synapse(
            "food_hunger", self.food_eye, self.hunger,
            self.config.n_food_eye, self.config.n_hunger_circuit)
        # Enemy → Fear/Attack: 가중치 대폭 강화 (신호 전달 보장)
        self.syn_enemy_fear = create_synapse(
            "enemy_fear", self.enemy_eye, self.fear,
            self.config.n_enemy_eye, self.config.n_fear_circuit,
            sparsity=self.config.sparsity * 4,  # 2% 연결
            w_init=2.0)  # 강한 가중치
        self.syn_enemy_attack = create_synapse(
            "enemy_attack", self.enemy_eye, self.attack,
            self.config.n_enemy_eye, self.config.n_attack_circuit,
            sparsity=self.config.sparsity * 4,  # 2% 연결
            w_init=2.5)  # 공격이 공포보다 강하게!
        self.syn_body_fear = create_synapse(
            "body_fear", self.body_eye, self.fear,
            self.config.n_body_eye, self.config.n_fear_circuit,
            sparsity=self.config.sparsity * 0.5)

        self.all_synapses.extend([self.syn_food_hunger, self.syn_enemy_fear, self.syn_enemy_attack, self.syn_body_fear])

        # Circuits → Integration 1
        self.syn_hunger_int1 = create_synapse(
            "hunger_int1", self.hunger, self.integration_1,
            self.config.n_hunger_circuit, self.config.n_integration_1)
        self.syn_fear_int1 = create_synapse(
            "fear_int1", self.fear, self.integration_1,
            self.config.n_fear_circuit, self.config.n_integration_1)
        self.syn_attack_int1 = create_synapse(
            "attack_int1", self.attack, self.integration_1,
            self.config.n_attack_circuit, self.config.n_integration_1)

        self.all_synapses.extend([self.syn_hunger_int1, self.syn_fear_int1, self.syn_attack_int1])

        # Integration 1 → Integration 2
        self.syn_int1_int2 = create_synapse(
            "int1_int2", self.integration_1, self.integration_2,
            self.config.n_integration_1, self.config.n_integration_2)

        self.all_synapses.append(self.syn_int1_int2)

        # Integration 2 → Motor
        self.syn_int2_left = create_synapse(
            "int2_left", self.integration_2, self.motor_left,
            self.config.n_integration_2, self.config.n_motor_left)
        self.syn_int2_right = create_synapse(
            "int2_right", self.integration_2, self.motor_right,
            self.config.n_integration_2, self.config.n_motor_right)
        self.syn_int2_boost = create_synapse(
            "int2_boost", self.integration_2, self.motor_boost,
            self.config.n_integration_2, self.config.n_motor_boost)

        self.all_synapses.extend([self.syn_int2_left, self.syn_int2_right, self.syn_int2_boost])

        # === CROSS-INHIBITION: Fear --| Hunger ===
        # 공포가 배고픔을 억제 (적이 보이면 먹이 추적 중단)
        print(f"  Cross-Inhibition: Fear --| Hunger (weight={self.config.inhibitory_weight})")
        self.syn_fear_hunger_inhib = create_synapse(
            "fear_hunger_inhib", self.fear, self.hunger,
            self.config.n_fear_circuit, self.config.n_hunger_circuit,
            sparsity=self.config.sparsity * 2,
            w_init=abs(self.config.inhibitory_weight))

        self.all_synapses.append(self.syn_fear_hunger_inhib)

        # === FEAR ↔ ATTACK 상호 억제 (Fight-or-Flight Competition) ===
        # 튜닝: 공포가 공격을 너무 억제하면 attack_triggers=0 문제 발생
        # → Fear→Attack 억제를 약하게, Attack→Fear 억제를 강하게 (공격 우선)
        print(f"  Fear ↔ Attack Mutual Inhibition (Attack-biased)")
        self.syn_fear_attack_inhib = create_synapse(
            "fear_attack_inhib", self.fear, self.attack,
            self.config.n_fear_circuit, self.config.n_attack_circuit,
            sparsity=self.config.sparsity * 2,
            w_init=abs(self.config.inhibitory_weight) * 0.3)  # 공포→공격 억제 (약하게!)
        self.syn_attack_fear_inhib = create_synapse(
            "attack_fear_inhib", self.attack, self.fear,
            self.config.n_attack_circuit, self.config.n_fear_circuit,
            sparsity=self.config.sparsity * 2,
            w_init=abs(self.config.inhibitory_weight) * 0.7)  # 공격→공포 억제 (강하게!)

        self.all_synapses.extend([self.syn_fear_attack_inhib, self.syn_attack_fear_inhib])

        # === DIRECT REFLEX: Fear → Boost ===
        self.syn_fear_boost = create_synapse(
            "fear_boost", self.fear, self.motor_boost,
            self.config.n_fear_circuit, self.config.n_motor_boost,
            sparsity=self.config.sparsity * 3)

        self.all_synapses.append(self.syn_fear_boost)

        # === INNATE AVOIDANCE REFLEX (진화된 본능!) ===
        # Cross-wired: Enemy LEFT → Motor RIGHT, Enemy RIGHT → Motor LEFT
        # 3x boosted initial weights
        print(f"  Innate Reflex: Enemy→Motor (cross-wired, {self.config.innate_boost}x boost)")

        # Enemy의 왼쪽 절반 → 오른쪽 모터 (회피를 위해 반대로)
        # Enemy의 오른쪽 절반 → 왼쪽 모터
        # 이건 process()에서 입력을 나눠서 처리
        self.syn_enemy_motor_left = create_synapse(
            "enemy_motor_left", self.enemy_eye, self.motor_left,
            self.config.n_enemy_eye, self.config.n_motor_left,
            sparsity=self.config.sparsity * 2,
            w_init=0.3 * self.config.innate_boost)  # 3x boost!
        self.syn_enemy_motor_right = create_synapse(
            "enemy_motor_right", self.enemy_eye, self.motor_right,
            self.config.n_enemy_eye, self.config.n_motor_right,
            sparsity=self.config.sparsity * 2,
            w_init=0.3 * self.config.innate_boost)  # 3x boost!

        self.all_synapses.extend([self.syn_enemy_motor_left, self.syn_enemy_motor_right])

        # === Direct food reflex ===
        self.syn_food_motor_left = create_synapse(
            "food_motor_left", self.food_eye, self.motor_left,
            self.config.n_food_eye, self.config.n_motor_left,
            sparsity=self.config.sparsity * 2)
        self.syn_food_motor_right = create_synapse(
            "food_motor_right", self.food_eye, self.motor_right,
            self.config.n_food_eye, self.config.n_motor_right,
            sparsity=self.config.sparsity * 2)

        self.all_synapses.extend([self.syn_food_motor_left, self.syn_food_motor_right])

        # === ATTACK CIRCUIT → Motor (적 방향으로 돌진) ===
        # 튜닝: 공격 충동이 모터까지 전달되어야 함 - 가중치 상향
        print(f"  Attack Reflex: Attack→Motor (boosted)")
        self.syn_attack_motor_left = create_synapse(
            "attack_motor_left", self.attack, self.motor_left,
            self.config.n_attack_circuit, self.config.n_motor_left,
            sparsity=self.config.sparsity * 2,
            w_init=0.5)  # 0.2 → 0.5 (2.5x 강화)
        self.syn_attack_motor_right = create_synapse(
            "attack_motor_right", self.attack, self.motor_right,
            self.config.n_attack_circuit, self.config.n_motor_right,
            sparsity=self.config.sparsity * 2,
            w_init=0.5)  # 0.2 → 0.5
        # Attack → Boost (공격 시 가속 - 돌진!)
        self.syn_attack_boost = create_synapse(
            "attack_boost", self.attack, self.motor_boost,
            self.config.n_attack_circuit, self.config.n_motor_boost,
            sparsity=self.config.sparsity * 3,
            w_init=0.4)  # 0.15 → 0.4 (공격 시 부스트 확률 증가)

        self.all_synapses.extend([self.syn_attack_motor_left, self.syn_attack_motor_right, self.syn_attack_boost])

        # === WTA (Winner-Take-All) 측면 억제 회로 ===
        # 가장 강하게 발화한 모터 뉴런이 나머지를 억제 → 깨끗한 STDP 학습
        print(f"  WTA Lateral Inhibition: Motor neurons (weight={self.config.wta_inhibition})")

        # 억제 시냅스용 파라미터 (STDP 비활성화 - 억제 연결은 고정)
        inhib_params = {
            "tauPlus": self.config.tau_plus,
            "tauMinus": self.config.tau_minus,
            "aPlus": 0.0,  # 학습 없음 (고정된 억제)
            "aMinus": 0.0,
            "wMin": self.config.wta_inhibition,  # 음수 가중치
            "wMax": 0.0,
            "dopamine": 0.5,
        }

        # Left ↔ Right 상호 억제 (음수 가중치 = 억제)
        self.syn_left_right_inhib = self.model.add_synapse_population(
            "left_right_inhib", "SPARSE",
            self.motor_left, self.motor_right,
            init_weight_update(da_stdp_model, inhib_params,
                               {"g": self.config.wta_inhibition, "eligibility": 0.0}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": self.config.wta_sparsity})
        )

        self.syn_right_left_inhib = self.model.add_synapse_population(
            "right_left_inhib", "SPARSE",
            self.motor_right, self.motor_left,
            init_weight_update(da_stdp_model, inhib_params,
                               {"g": self.config.wta_inhibition, "eligibility": 0.0}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": self.config.wta_sparsity})
        )

        # Left ↔ Boost 상호 억제
        self.syn_left_boost_inhib = self.model.add_synapse_population(
            "left_boost_inhib", "SPARSE",
            self.motor_left, self.motor_boost,
            init_weight_update(da_stdp_model, inhib_params,
                               {"g": self.config.wta_inhibition * 0.5, "eligibility": 0.0}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": self.config.wta_sparsity})
        )

        self.syn_boost_left_inhib = self.model.add_synapse_population(
            "boost_left_inhib", "SPARSE",
            self.motor_boost, self.motor_left,
            init_weight_update(da_stdp_model, inhib_params,
                               {"g": self.config.wta_inhibition * 0.5, "eligibility": 0.0}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": self.config.wta_sparsity})
        )

        # Right ↔ Boost 상호 억제
        self.syn_right_boost_inhib = self.model.add_synapse_population(
            "right_boost_inhib", "SPARSE",
            self.motor_right, self.motor_boost,
            init_weight_update(da_stdp_model, inhib_params,
                               {"g": self.config.wta_inhibition * 0.5, "eligibility": 0.0}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": self.config.wta_sparsity})
        )

        self.syn_boost_right_inhib = self.model.add_synapse_population(
            "boost_right_inhib", "SPARSE",
            self.motor_boost, self.motor_right,
            init_weight_update(da_stdp_model, inhib_params,
                               {"g": self.config.wta_inhibition * 0.5, "eligibility": 0.0}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": self.config.wta_sparsity})
        )

        # WTA 시냅스는 all_synapses에 추가하지 않음 (도파민 학습 불필요)

        # 빌드 및 로드
        print("  Compiling CUDA code...")
        self.model.build()
        self.model.load()
        print(f"  Model ready! {self.config.total_neurons:,} neurons")

        # State
        self.dopamine = 0.5
        self.fear_level = 0.0
        self.attack_level = 0.0
        self.hunger_level = 0.5
        self.steps = 0
        self.generation = 0  # 윤회 세대
        self.stats = {'food_eaten': 0, 'boosts': 0, 'fear_triggers': 0, 'attack_triggers': 0}

    def process(self, sensor_input: np.ndarray, reward: float = 0.0) -> Tuple[float, float, bool]:
        """센서 입력 처리 및 행동 출력"""
        # Unpack sensor input (3, n_rays)
        food_signal = sensor_input[0]
        enemy_signal = sensor_input[1]
        body_signal = sensor_input[2]

        n_rays = len(food_signal)

        # === ENCODE SENSORY INPUT (분리된 채널) ===
        food_encoded = self._encode_to_population(food_signal, self.config.n_food_eye)
        enemy_encoded = self._encode_to_population(enemy_signal, self.config.n_enemy_eye)
        body_encoded = self._encode_to_population(body_signal, self.config.n_body_eye)

        # Set sensory input (voltage injection)
        self.food_eye.vars["V"].view[:] = self.config.v_rest + food_encoded * 40.0
        self.enemy_eye.vars["V"].view[:] = self.config.v_rest + enemy_encoded * 50.0
        self.body_eye.vars["V"].view[:] = self.config.v_rest + body_encoded * 25.0

        self.food_eye.vars["V"].push_to_device()
        self.enemy_eye.vars["V"].push_to_device()
        self.body_eye.vars["V"].push_to_device()

        # === SIMULATE ===
        self.model.step_time()
        self.steps += 1

        # === READ MOTOR OUTPUT ===
        self.motor_left.vars["V"].pull_from_device()
        self.motor_right.vars["V"].pull_from_device()
        self.motor_boost.vars["V"].pull_from_device()

        left_v = self.motor_left.vars["V"].view.copy()
        right_v = self.motor_right.vars["V"].view.copy()
        boost_v = self.motor_boost.vars["V"].view.copy()

        # Decode motor activity
        left_rate = self._decode_activity(left_v)
        right_rate = self._decode_activity(right_v)
        boost_rate = self._decode_activity(boost_v)

        # Read fear & attack levels (for stats and behavior modulation)
        self.fear.vars["V"].pull_from_device()
        self.attack.vars["V"].pull_from_device()
        fear_v = self.fear.vars["V"].view
        attack_v = self.attack.vars["V"].view
        self.fear_level = self._decode_activity(fear_v)
        self.attack_level = self._decode_activity(attack_v)

        # 공포/공격 활성화 감지 (Fight-or-Flight)
        if self.fear_level > 0.02 or enemy_signal.max() > 0.2:
            self.stats['fear_triggers'] += 1
        if self.attack_level > 0.001:  # 임계값 낮춤 (0.02 → 0.001)
            self.stats['attack_triggers'] += 1

        # === COMPUTE ACTION ===
        # Direction from motor difference
        angle_delta = (right_rate - left_rate) * 0.8

        # Compute target position
        target_x = 0.5 + 0.2 * np.cos(angle_delta)
        target_y = 0.5 + 0.2 * np.sin(angle_delta)

        # Food seeking bias (heuristic to help initial learning)
        if food_signal.max() > 0.1:
            best_ray = np.argmax(food_signal)
            food_angle = (2 * np.pi * best_ray / n_rays) - np.pi
            blend = min(0.4, food_signal.max())
            target_x = target_x * (1 - blend) + (0.5 + 0.15 * np.cos(food_angle)) * blend
            target_y = target_y * (1 - blend) + (0.5 + 0.15 * np.sin(food_angle)) * blend

        # Enemy avoidance bias (적절한 회피)
        if enemy_signal.max() > 0.25:  # 더 가까울 때만
            enemy_ray = np.argmax(enemy_signal)
            enemy_angle = (2 * np.pi * enemy_ray / n_rays) - np.pi
            avoid_angle = enemy_angle + np.pi  # 반대 방향
            blend = min(0.4, enemy_signal.max() * 0.8)  # 적절한 회피
            target_x = target_x * (1 - blend) + (0.5 + 0.2 * np.cos(avoid_angle)) * blend
            target_y = target_y * (1 - blend) + (0.5 + 0.2 * np.sin(avoid_angle)) * blend

        target_x = np.clip(target_x, 0.05, 0.95)
        target_y = np.clip(target_y, 0.05, 0.95)

        # Boost decision - 보수적으로 (부스트는 길이를 소모함)
        enemy_very_close = enemy_signal.max() > 0.6  # 매우 가까운 적만
        boost = boost_rate > 0.3 and enemy_very_close

        if boost:
            self.stats['boosts'] += 1

        # === LEARNING ===
        if reward != 0:
            self._update_dopamine(reward)
            if reward > 0:
                self.stats['food_eaten'] += 1

        return target_x, target_y, boost

    def _encode_to_population(self, signal: np.ndarray, n_neurons: int) -> np.ndarray:
        """신호를 뉴런 population 크기로 확장"""
        n_rays = len(signal)
        repeats = (n_neurons // n_rays) + 1
        expanded = np.tile(signal, repeats)[:n_neurons]

        # Add noise for stochastic activation
        noise = np.random.rand(n_neurons) * 0.2
        encoded = expanded * (1 + noise)

        return encoded.astype(np.float32)

    def _decode_activity(self, v: np.ndarray) -> float:
        """막전위를 활성도로 변환 (0-1)"""
        v_norm = (v - self.config.v_rest) / (self.config.v_thresh - self.config.v_rest)
        return float(np.clip(v_norm, 0, 1).mean())

    def _update_dopamine(self, reward: float):
        """도파민 업데이트 및 GPU 전송"""
        self.dopamine = np.clip(self.dopamine + reward * 0.15, 0.0, 1.0)
        for syn in self.all_synapses:
            syn.set_dynamic_param_value("dopamine", self.dopamine)

    def apply_death_penalty(self):
        """죽음 시 Death Penalty (비활성화됨)

        STDP eligibility trace는 죽기 직전 행동만 기억하지 않고,
        이전 좋은 행동까지 같이 약화시킴 → 비활성화
        """
        # Death Penalty 비활성화: 역효과 확인됨
        # - eligibility trace가 직전 행동뿐 아니라 좋은 행동도 포함
        # - LTD가 모든 최근 시냅스를 약화시켜 학습 저하
        self.generation += 1
        pass

    def reset(self, keep_weights: bool = True):
        """상태 초기화 (윤회 시스템)

        Args:
            keep_weights: True면 시냅스 가중치 유지 (윤회)
                         False면 완전 초기화
        """
        all_pops = [self.food_eye, self.enemy_eye, self.body_eye,
                    self.hunger, self.fear, self.attack,
                    self.integration_1, self.integration_2,
                    self.motor_left, self.motor_right, self.motor_boost]

        # 막전위만 초기화 (가중치는 유지!)
        for pop in all_pops:
            pop.vars["V"].view[:] = self.config.v_rest
            pop.vars["V"].push_to_device()

        self.dopamine = 0.5
        self.fear_level = 0.0
        self.attack_level = 0.0
        self.hunger_level = 0.5
        self.stats = {'food_eaten': 0, 'boosts': 0, 'fear_triggers': 0, 'attack_triggers': 0}

    def save_weights(self, path: Path):
        """시냅스 가중치 저장"""
        weights = {}
        for syn in self.all_synapses:
            syn.vars["g"].pull_from_device()
            # Sparse connectivity uses .values, not .view
            weights[syn.name] = np.array(syn.vars["g"].values)

        np.savez(path, **weights)
        print(f"  Saved weights: {path}")

    def load_weights(self, path: Path) -> bool:
        """시냅스 가중치 로드"""
        if not path.exists():
            print(f"  No checkpoint: {path}")
            return False

        try:
            data = np.load(path)
            for syn in self.all_synapses:
                if syn.name in data:
                    syn.vars["g"].values[:] = data[syn.name]
                    syn.vars["g"].push_to_device()
            print(f"  Loaded weights: {path}")
            return True
        except Exception as e:
            print(f"  Load failed: {e}")
            return False


class BiologicalAgent:
    """생물학적 PyGeNN Slither.io 에이전트"""

    def __init__(self, brain_config: Optional[BiologicalConfig] = None,
                 env_config: Optional[SlitherConfig] = None,
                 render_mode: str = "none"):
        self.brain = BiologicalBrain(brain_config)
        self.env = SlitherGym(env_config, render_mode)

        self.scores = []
        self.best_score = 0

    def save_model(self, name: str):
        """모델 저장"""
        path = CHECKPOINT_DIR / f"{name}.npz"
        self.brain.save_weights(path)

    def load_model(self, name: str) -> bool:
        """모델 로드"""
        path = CHECKPOINT_DIR / f"{name}.npz"
        return self.brain.load_weights(path)

    def run_episode(self, max_steps: int = 1000) -> dict:
        """한 에피소드 실행 (윤회 시스템 적용)"""
        obs = self.env.reset()
        self.brain.reset(keep_weights=True)  # 가중치 유지!

        total_reward = 0
        step = 0

        while step < max_steps:
            sensor = self.env.get_sensor_input(self.brain.config.n_rays)
            target_x, target_y, boost = self.brain.process(sensor)

            obs, reward, done, info = self.env.step((target_x, target_y, boost))
            total_reward += reward

            if reward != 0:
                self.brain.process(sensor, reward)

            self.env.render()
            step += 1

            if done:
                # 죽음! Death Penalty 적용 (윤회)
                self.brain.apply_death_penalty()
                break

        return {
            'length': info['length'],
            'steps': info['steps'],
            'reward': total_reward,
            'food_eaten': info.get('foods_eaten', 0),
            'fear_triggers': self.brain.stats['fear_triggers'],
            'attack_triggers': self.brain.stats['attack_triggers'],
            'boosts': self.brain.stats['boosts'],
            'generation': self.brain.generation
        }

    def train(self, n_episodes: int = 100, resume: bool = False):
        """학습"""
        from gpu_monitor import start_monitoring, stop_monitoring

        # Resume from checkpoint
        if resume:
            if self.load_model("best"):
                print("  ★ Resumed from best checkpoint")
                # Try to get previous best from filename
                import glob
                checkpoints = glob.glob(str(CHECKPOINT_DIR / "best_*.npz"))
                if checkpoints:
                    scores = [int(p.split('_')[-1].replace('.npz', '')) for p in checkpoints]
                    self.best_score = max(scores)
                    print(f"  Previous Best: {self.best_score}")

        print("\n" + "=" * 60)
        print(f"Biological PyGeNN Training ({self.brain.config.total_neurons:,} neurons)")
        print(f"  STDP: τ={self.brain.config.tau_plus}ms, η={self.brain.config.a_plus}")
        print(f"  WTA: inhibition={self.brain.config.wta_inhibition}, sparsity={self.brain.config.wta_sparsity}")
        print("=" * 60)

        monitor = start_monitoring(interval=1.0)
        start_time = time.time()

        for ep in range(n_episodes):
            result = self.run_episode()
            self.scores.append(result['length'])

            if result['length'] > self.best_score:
                self.best_score = result['length']
                print(f"  ★ NEW BEST! Length={result['length']}")
                # Save best model
                self.save_model("best")
                self.save_model(f"best_{result['length']}")

            high = max(self.scores)
            avg = sum(self.scores[-10:]) / min(len(self.scores), 10)

            if ep % 10 == 0:
                monitor.print_status()

            gen = result.get('generation', 0)
            print(f"[Ep {ep+1:3d}] Gen:{gen:3d} | Length: {result['length']:3d} | "
                  f"High: {high} | Avg(10): {avg:.0f} | "
                  f"Food: {result['food_eaten']} | Fear: {result['fear_triggers']} | Attack: {result['attack_triggers']}")

        elapsed = time.time() - start_time

        # Save final model
        self.save_model("final")

        print("\n" + "=" * 60)
        print(f"Training Complete!")
        print(f"  Episodes: {n_episodes}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/n_episodes:.2f}s/ep)")
        print(f"  Best Length: {max(self.scores)}")
        print(f"  Final Avg: {sum(self.scores)/len(self.scores):.1f}")
        print(f"  Saved to: {CHECKPOINT_DIR}")
        print("=" * 60)

        stop_monitoring()

    def close(self):
        """정리"""
        self.env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--render', choices=['none', 'pygame', 'ascii'], default='none')
    parser.add_argument('--enemies', type=int, default=3, help='Number of enemy bots')
    parser.add_argument('--resume', action='store_true', help='Resume from best checkpoint')
    parser.add_argument('--lite', action='store_true', help='Use lite config (50K neurons) - GPU safe')
    parser.add_argument('--dev', action='store_true', help='Use dev config (15K neurons) - debugging')
    args = parser.parse_args()

    print("Biological PyGeNN Slither.io Agent")
    print(f"Render mode: {args.render}")
    print(f"Enemies: {args.enemies}")

    # GPU 안전 모드 선택
    if args.dev:
        brain_config = BiologicalConfig.dev()
        print("Mode: DEV (15K neurons - GPU safe for debugging)")
    elif args.lite:
        brain_config = BiologicalConfig.lite()
        print("Mode: LITE (50K neurons - GPU safe)")
    else:
        brain_config = BiologicalConfig()
        print("Mode: FULL (153K neurons - GPU intensive!)")
    print(f"Total neurons: {brain_config.total_neurons:,}")
    print()

    env_config = SlitherConfig(n_enemies=args.enemies)

    agent = BiologicalAgent(
        brain_config=brain_config,
        env_config=env_config,
        render_mode=args.render
    )

    try:
        agent.train(n_episodes=args.episodes, resume=args.resume)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted.")
        agent.save_model("interrupted")
    finally:
        agent.close()
