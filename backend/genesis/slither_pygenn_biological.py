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
                    init_postsynaptic, create_weight_update_model, create_neuron_model,
                    init_var)
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
# === v24: Soft-Bound R-STDP (Multiplicative) ===
# 기존 Additive STDP의 문제: 가중치가 wMax/wMin으로 빠르게 포화
# 해결: Multiplicative STDP - 가중치가 극단값에 가까울수록 변화량 감소
# 효과: 가중치가 정규분포에 가깝게 퍼지며, 포화 현상 방지
r_stdp_model = create_weight_update_model(
    "R_STDP_SOFT_BOUND",
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

        // 4. 도파민 조절 학습 - SOFT-BOUND (v24)
        scalar da_signal = dopamine - 0.5;
        if (fabs(da_signal) > 0.1) {
            scalar update = eta * da_signal * eligibility;

            if (update > 0) {
                // 강화(LTP): 남은 공간(wMax - g)에 비례해서 증가
                // 가중치가 wMax에 가까울수록 변화량 감소
                g += update * (wMax - g);
            } else {
                // 약화(LTD): 현재 값(g - wMin)에 비례해서 감소
                // 가중치가 wMin에 가까울수록 변화량 감소
                g += update * (g - wMin);
            }
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

# === SensoryLIF Model (v23: 동적 전류 입력) ===
# 핵심: I_input 변수를 통해 외부에서 전류 주입 가능
# 전압 직접 설정 대신 전류 주입 → 정상적인 스파이크 이벤트 발생
sensory_lif_model = create_neuron_model(
    "SensoryLIF",
    params=["C", "TauM", "Vrest", "Vreset", "Vthresh", "TauRefrac"],
    vars=[("V", "scalar"), ("RefracTime", "scalar"), ("I_input", "scalar")],
    sim_code="""
        // Refractory period check
        if (RefracTime > 0.0) {
            RefracTime -= dt;
        } else {
            // LIF dynamics with external current input
            // I_total = I_input (external) + Isyn (synaptic)
            scalar I_total = I_input + Isyn;
            V += (-(V - Vrest) / TauM + I_total / C) * dt;
        }
    """,
    threshold_condition_code="""
        RefracTime <= 0.0 && V >= Vthresh
    """,
    reset_code="""
        V = Vreset;
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

    # === R-STDP parameters (v24: Soft-Bound + 안정적 학습) ===
    tau_stdp: float = 20.0       # STDP 타이밍 윈도우 (빠름)
    tau_eligibility: float = 1000.0  # 1초 eligibility trace
    a_plus: float = 0.005        # LTP 강도
    a_minus: float = 0.006       # LTD 강도
    eta: float = 0.01            # v24: 학습률 낮춤 (Soft-Bound로 안정적)
    w_max: float = 10.0          # 가중치 상한
    w_min: float = -5.0          # 가중치 하한 (억제 허용)

    # Legacy STDP (비교용)
    tau_plus: float = 20.0
    tau_minus: float = 20.0

    # === v36: Long Tau R-STDP for Hunt synapses (Sandbag Training) ===
    # 킬까지 5-10초 걸리므로, 사냥 시냅스는 더 긴 기억이 필요
    tau_eligibility_hunt: float = 10000.0  # 10초 (일반 1초 vs 사냥 10초)

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

        # LIF 파라미터 (일반 뉴런용)
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

        # v23: SensoryLIF 파라미터 (동적 전류 입력)
        sensory_params = {
            "C": 1.0,
            "TauM": self.config.tau_m,
            "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset,
            "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        sensory_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. SENSORY POPULATIONS (v23: SensoryLIF 사용!) ===
        n_food_half = self.config.n_food_eye // 2
        n_enemy_half = self.config.n_enemy_eye // 2

        self.food_eye_left = self.model.add_neuron_population(
            "food_eye_left", n_food_half, sensory_lif_model, sensory_params, sensory_init)
        self.food_eye_right = self.model.add_neuron_population(
            "food_eye_right", n_food_half, sensory_lif_model, sensory_params, sensory_init)
        self.enemy_eye_left = self.model.add_neuron_population(
            "enemy_eye_left", n_enemy_half, sensory_lif_model, sensory_params, sensory_init)
        self.enemy_eye_right = self.model.add_neuron_population(
            "enemy_eye_right", n_enemy_half, sensory_lif_model, sensory_params, sensory_init)
        # v27j: body_eye split into L/R for wall avoidance
        n_body_half = self.config.n_body_eye // 2
        self.body_eye_left = self.model.add_neuron_population(
            "body_eye_left", n_body_half, sensory_lif_model, sensory_params, sensory_init)
        self.body_eye_right = self.model.add_neuron_population(
            "body_eye_right", n_body_half, sensory_lif_model, sensory_params, sensory_init)

        # === v30: ENEMY HEAD SENSOR (적 머리 = 공격 대상!) ===
        # 적 머리 방향으로 회전해서 내 몸으로 막기 (킬!)
        n_enemy_head_half = n_enemy_half // 2  # 적 머리 뉴런은 더 작게
        self.enemy_head_left = self.model.add_neuron_population(
            "enemy_head_left", n_enemy_head_half, sensory_lif_model, sensory_params, sensory_init)
        self.enemy_head_right = self.model.add_neuron_population(
            "enemy_head_right", n_enemy_head_half, sensory_lif_model, sensory_params, sensory_init)

        print(f"  Sensory: Food_L/R({n_food_half:,}x2) + Enemy_L/R({n_enemy_half:,}x2) + Body_L/R({n_body_half:,}x2) + EnemyHead_L/R({n_enemy_head_half:,}x2)")

        # === 2. SPECIALIZED CIRCUITS ===
        self.hunger = self.model.add_neuron_population(
            "hunger", self.config.n_hunger_circuit, "LIF", lif_params, lif_init)
        self.fear = self.model.add_neuron_population(
            "fear", self.config.n_fear_circuit, "LIF", lif_params, lif_init)
        # v28c: Attack uses standard LIF (v29 SensoryLIF 롤백)
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

        # === v36: Long Tau R-STDP for Hunt synapses ===
        # 사냥 시냅스는 10초 기억 필요 (킬까지의 기동 시간)
        r_stdp_hunt_params = {
            "tauStdp": self.config.tau_stdp,           # 20ms (spike timing)
            "tauElig": self.config.tau_eligibility_hunt,  # 10000ms (10초 기억!)
            "aPlus": self.config.a_plus,
            "aMinus": self.config.a_minus,
            "wMin": self.config.w_min,
            "wMax": 40.0,  # Hunt 가중치 상한 확대 (학습 여유)
            "dopamine": 0.5,
            "eta": 0.05,   # 학습률 증가 (보상 빈도가 낮으므로)
        }

        # 시냅스 생성 헬퍼 (R-STDP 사용 - 학습 가능)
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

        # === v24: 고정 시냅스 헬퍼 (StaticPulse - 학습 안 함) ===
        # 생존 본능(공포 회로)은 학습하면 안 됨 - 선천적 본능으로 고정
        def create_static_synapse(name, pre, post, n_pre, n_post, sparsity=None, w_init=1.0):
            sp = sparsity or self.config.sparsity
            syn = self.model.add_synapse_population(
                name, "SPARSE", pre, post,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": w_init})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}),
                init_sparse_connectivity("FixedProbability", {"prob": sp})
            )
            return syn

        # === v36: Hunt 시냅스 헬퍼 (Long Tau R-STDP - 10초 기억) ===
        # 사냥 기술은 학습해야 함 - 긴 eligibility trace로 킬과 행동 연결
        def create_hunt_synapse(name, pre, post, n_pre, n_post, sparsity=None, w_init=1.0):
            sp = sparsity or self.config.sparsity
            syn = self.model.add_synapse_population(
                name, "SPARSE", pre, post,
                init_weight_update(r_stdp_model, r_stdp_hunt_params,
                                   {"g": w_init, "stdp_trace": 0.0, "eligibility": 0.0}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}),
                init_sparse_connectivity("FixedProbability", {"prob": sp})
            )
            syn.set_wu_param_dynamic("dopamine")
            return syn

        # === SYNAPTIC CONNECTIONS ===
        self.all_synapses = []

        # Sensory → Circuits (v19: L/R 분리)
        self.syn_food_left_hunger = create_synapse(
            "food_left_hunger", self.food_eye_left, self.hunger,
            n_food_half, self.config.n_hunger_circuit)
        self.syn_food_right_hunger = create_synapse(
            "food_right_hunger", self.food_eye_right, self.hunger,
            n_food_half, self.config.n_hunger_circuit)

        # === v24: Enemy → Fear (고정 - 공포는 선천적 본능) ===
        # "토끼가 늑대에게 물렸다고 다음엔 덜 무서워하지 않는다"
        self.syn_enemy_left_fear = create_static_synapse(
            "enemy_left_fear", self.enemy_eye_left, self.fear,
            n_enemy_half, self.config.n_fear_circuit,
            sparsity=self.config.sparsity * 4, w_init=5.0)  # 고정: 강한 공포 반응
        self.syn_enemy_right_fear = create_static_synapse(
            "enemy_right_fear", self.enemy_eye_right, self.fear,
            n_enemy_half, self.config.n_fear_circuit,
            sparsity=self.config.sparsity * 4, w_init=5.0)

        # Enemy → Attack (학습 가능 - 사냥 기술은 배움)
        self.syn_enemy_left_attack = create_synapse(
            "enemy_left_attack", self.enemy_eye_left, self.attack,
            n_enemy_half, self.config.n_attack_circuit,
            sparsity=self.config.sparsity * 4, w_init=2.5)
        self.syn_enemy_right_attack = create_synapse(
            "enemy_right_attack", self.enemy_eye_right, self.attack,
            n_enemy_half, self.config.n_attack_circuit,
            sparsity=self.config.sparsity * 4, w_init=2.5)

        # Body → Fear (고정 - 자기 몸/벽 인식도 본능)
        n_body_half = self.config.n_body_eye // 2
        self.syn_body_left_fear = create_static_synapse(
            "body_left_fear", self.body_eye_left, self.fear,
            n_body_half, self.config.n_fear_circuit,
            sparsity=self.config.sparsity * 0.5, w_init=1.0)
        self.syn_body_right_fear = create_static_synapse(
            "body_right_fear", self.body_eye_right, self.fear,
            n_body_half, self.config.n_fear_circuit,
            sparsity=self.config.sparsity * 0.5, w_init=1.0)

        # R-STDP 시냅스만 all_synapses에 추가 (학습 대상)
        self.all_synapses.extend([
            self.syn_food_left_hunger, self.syn_food_right_hunger,
            self.syn_enemy_left_attack, self.syn_enemy_right_attack,
            # enemy_left/right_fear, body_fear는 StaticPulse → 학습 안 함
        ])

        # Circuits → Integration 1
        self.syn_hunger_int1 = create_synapse(
            "hunger_int1", self.hunger, self.integration_1,
            self.config.n_hunger_circuit, self.config.n_integration_1)
        # Fear → Int1 (고정 - 공포 전달 경로도 본능)
        self.syn_fear_int1 = create_static_synapse(
            "fear_int1", self.fear, self.integration_1,
            self.config.n_fear_circuit, self.config.n_integration_1,
            w_init=2.0)  # 고정: 공포가 행동에 영향
        self.syn_attack_int1 = create_synapse(
            "attack_int1", self.attack, self.integration_1,
            self.config.n_attack_circuit, self.config.n_integration_1)

        # fear_int1은 StaticPulse → 학습 대상에서 제외
        self.all_synapses.extend([self.syn_hunger_int1, self.syn_attack_int1])

        # Integration 1 → Integration 2
        self.syn_int1_int2 = create_synapse(
            "int1_int2", self.integration_1, self.integration_2,
            self.config.n_integration_1, self.config.n_integration_2)

        self.all_synapses.append(self.syn_int1_int2)

        # === v25: Integration 2 → Motor (약화 - 기억이 감각을 덮으면 안 됨) ===
        # 기억(Int2)은 감각(Sensory)을 보조해야지, 덮어쓰면 안 됨
        int2_motor_weight = 3.0  # v25: 약한 가중치 (기존 ~9.4 → 3.0)
        self.syn_int2_left = create_synapse(
            "int2_left", self.integration_2, self.motor_left,
            self.config.n_integration_2, self.config.n_motor_left,
            w_init=int2_motor_weight)
        self.syn_int2_right = create_synapse(
            "int2_right", self.integration_2, self.motor_right,
            self.config.n_integration_2, self.config.n_motor_right,
            w_init=int2_motor_weight)
        self.syn_int2_boost = create_synapse(
            "int2_boost", self.integration_2, self.motor_boost,
            self.config.n_integration_2, self.config.n_motor_boost,
            w_init=int2_motor_weight)

        self.all_synapses.extend([self.syn_int2_left, self.syn_int2_right, self.syn_int2_boost])

        # === v24: CROSS-INHIBITION (고정 - 본능적 억제) ===
        # Fear --| Hunger: 공포가 배고픔을 억제 (적이 보이면 먹이 추적 중단)
        print(f"  Cross-Inhibition: Fear --| Hunger (STATIC, weight={self.config.inhibitory_weight})")
        self.syn_fear_hunger_inhib = create_static_synapse(
            "fear_hunger_inhib", self.fear, self.hunger,
            self.config.n_fear_circuit, self.config.n_hunger_circuit,
            sparsity=self.config.sparsity * 2,
            w_init=abs(self.config.inhibitory_weight))
        # 학습 대상 아님 (본능)

        # === v29: FEAR ↔ ATTACK 상호 억제 (공격 우세) ===
        # 공격 모드가 활성화되면 공포를 압도해야 함 (Disinhibition)
        print(f"  Fear ↔ Attack Mutual Inhibition (STATIC, Attack DOMINANT)")
        self.syn_fear_attack_inhib = create_static_synapse(
            "fear_attack_inhib", self.fear, self.attack,
            self.config.n_fear_circuit, self.config.n_attack_circuit,
            sparsity=self.config.sparsity * 2,
            w_init=-2.0)  # v29: 공포→공격 억제 (약함)
        self.syn_attack_fear_inhib = create_static_synapse(
            "attack_fear_inhib", self.attack, self.fear,
            self.config.n_attack_circuit, self.config.n_fear_circuit,
            sparsity=0.4,  # v29: 더 밀집한 연결
            w_init=-8.0)  # v29: 공격→공포 강력 억제! (Disinhibition)
        # 학습 대상 아님 (본능) - 공격이 공포를 이기는 구조

        # === v24: DIRECT REFLEX: Fear → Boost (고정) ===
        # 공포 시 도망 가속은 본능
        self.syn_fear_boost = create_static_synapse(
            "fear_boost", self.fear, self.motor_boost,
            self.config.n_fear_circuit, self.config.n_motor_boost,
            sparsity=self.config.sparsity * 3, w_init=3.0)  # 강한 부스트
        # 학습 대상 아님 (본능)

        # === v37f: PUSH-PULL AVOIDANCE REFLEX (약화!) ===
        # v37b 문제: 5 enemies에서 Fear가 너무 강해서 Hunt 불가
        # v37f: Fear 약화 → Hunt가 이길 수 있음
        push_weight = 80.0   # v37f: 100→80 (약화)
        pull_weight = -60.0  # v37f: -80→-60 (약화)
        push_sparsity = 0.3  # v27d: 0.25→0.3 (더 밀집)
        print(f"  Push-Pull Reflex: Enemy→Motor (PUSH={push_weight}, PULL={pull_weight}, sp={push_sparsity})")

        # PUSH: Enemy_L → Motor_R (오른쪽으로 도망) - 강하고 밀집!
        self.syn_enemy_left_motor_right = create_static_synapse(
            "enemy_left_motor_right", self.enemy_eye_left, self.motor_right,
            n_enemy_half, self.config.n_motor_right,
            sparsity=push_sparsity, w_init=push_weight)
        self.syn_enemy_right_motor_left = create_static_synapse(
            "enemy_right_motor_left", self.enemy_eye_right, self.motor_left,
            n_enemy_half, self.config.n_motor_left,
            sparsity=push_sparsity, w_init=push_weight)

        # PULL (억제): Enemy_L → Motor_L (왼쪽으로 가지 마!)
        self.syn_enemy_left_motor_left_inhib = create_static_synapse(
            "enemy_left_motor_left_inhib", self.enemy_eye_left, self.motor_left,
            n_enemy_half, self.config.n_motor_left,
            sparsity=push_sparsity, w_init=pull_weight)
        self.syn_enemy_right_motor_right_inhib = create_static_synapse(
            "enemy_right_motor_right_inhib", self.enemy_eye_right, self.motor_right,
            n_enemy_half, self.config.n_motor_right,
            sparsity=push_sparsity, w_init=pull_weight)
        # 학습 대상 아님 (본능) - 공포의 거부권

        # === v27j: WALL/BODY AVOIDANCE REFLEX (벽 회피) ===
        # 벽도 적처럼 회피해야 함
        wall_push_weight = 80.0   # 적보다 약간 약하게 (100 vs 80)
        wall_pull_weight = -60.0  # 억제도 약간 약하게
        wall_sparsity = 0.2
        n_body_half = self.config.n_body_eye // 2
        print(f"  Wall Reflex: Body→Motor (PUSH={wall_push_weight}, PULL={wall_pull_weight}, sp={wall_sparsity})")

        # PUSH: Body_L → Motor_R (벽이 왼쪽에 있으면 오른쪽으로)
        self.syn_body_left_motor_right = create_static_synapse(
            "body_left_motor_right", self.body_eye_left, self.motor_right,
            n_body_half, self.config.n_motor_right,
            sparsity=wall_sparsity, w_init=wall_push_weight)
        self.syn_body_right_motor_left = create_static_synapse(
            "body_right_motor_left", self.body_eye_right, self.motor_left,
            n_body_half, self.config.n_motor_left,
            sparsity=wall_sparsity, w_init=wall_push_weight)

        # PULL: Body_L → Motor_L (벽 쪽으로 가지 마!)
        self.syn_body_left_motor_left_inhib = create_static_synapse(
            "body_left_motor_left_inhib", self.body_eye_left, self.motor_left,
            n_body_half, self.config.n_motor_left,
            sparsity=wall_sparsity, w_init=wall_pull_weight)
        self.syn_body_right_motor_right_inhib = create_static_synapse(
            "body_right_motor_right_inhib", self.body_eye_right, self.motor_right,
            n_body_half, self.config.n_motor_right,
            sparsity=wall_sparsity, w_init=wall_pull_weight)
        # 학습 대상 아님 (본능)

        # === v28c: 적 회피 우선, 식욕은 보조 ===
        # 음식 신호가 양쪽 모터를 동시 활성화 → 적 회피 신호 상쇄!
        # 음식은 "방향 유도" 정도만, 생존이 우선
        food_weight = 20.0   # v28c: 30→20 (적 회피 우선)
        food_sparsity = 0.15 # v28c: 0.2→0.15 (더 sparse)
        print(f"  Food Reflex: Food_L→Motor_L, Food_R→Motor_R (STATIC, w={food_weight}, sp={food_sparsity})")

        self.syn_food_left_motor_left = create_static_synapse(
            "food_left_motor_left", self.food_eye_left, self.motor_left,
            n_food_half, self.config.n_motor_left,
            sparsity=food_sparsity, w_init=food_weight)
        self.syn_food_right_motor_right = create_static_synapse(
            "food_right_motor_right", self.food_eye_right, self.motor_right,
            n_food_half, self.config.n_motor_right,
            sparsity=food_sparsity, w_init=food_weight)
        # 학습 대상 아님 (본능)

        # === v37f: SCALED ATTACK - Fear 약화에 맞춘 조정 ===
        # v37f 전략: Fear를 약화시켰으니 Hunt도 비례 조정
        #
        # v37f 설정:
        #   Fear: body × 80 → contralateral (약화됨)
        #   Hunt: head × 180 → ipsilateral (강화!)
        #   Disinhibition: head × (-100) → contralateral
        #
        # 결과 계산 (적 머리 보일 때):
        #   Motor_R: +80 - 100 = -20 (도망 신호 역전!)
        #   Motor_L: -60 + 180 + 80 = +200 (더 강력한 사냥!)

        n_enemy_head_half = self.config.n_enemy_eye // 4
        attack_hunt_weight = 180.0  # ★ v37f: 강화!
        attack_sparsity = 0.5      # ★ v37b: 높은 연결 밀도!

        # === Part 1: 동측 배선 (적 머리 방향으로 회전) - STATIC! ===
        print(f"  Hunt Reflex: EnemyHead→Motor IPSILATERAL (w={attack_hunt_weight}, STATIC - no learning)")
        self.syn_enemy_head_left_motor_left = create_static_synapse(
            "enemy_head_left_motor_left", self.enemy_head_left, self.motor_left,
            n_enemy_head_half, self.config.n_motor_left,
            sparsity=attack_sparsity, w_init=attack_hunt_weight)
        self.syn_enemy_head_right_motor_right = create_static_synapse(
            "enemy_head_right_motor_right", self.enemy_head_right, self.motor_right,
            n_enemy_head_half, self.config.n_motor_right,
            sparsity=attack_sparsity, w_init=attack_hunt_weight)

        # v37: Hunt는 이제 정적 시냅스 (학습 안함)
        self.hunt_synapses = []  # R-STDP 제거됨

        # === Part 2: v37 강화된 탈억제 (Fear 완전 무력화!) ===
        #
        # v36 문제: Fear(100) > Hunt(40) + Disinhibit(70) → 여전히 도망
        # v37 해결: Disinhibition을 Fear보다 강하게!
        #
        # Fear 신호 (Body_L 감지시):
        #   Motor_R: +100 (Push)
        #   Motor_L: -80  (Pull)
        #
        # Hunt 신호 (Head_L 감지시):
        #   Motor_L: +150 (Hunt - v37 강화!)
        #   Motor_R: -120 (Disinhibit Push - Fear 상쇄!)
        #   Motor_L: +100 (Disinhibit Pull - v37 강화!)
        #
        # 결과 (적 머리가 보일 때):
        #   Motor_R: +100 - 120 = -20 (도망 신호 역전!)
        #   Motor_L: -80 + 150 + 100 = +170 (강력한 사냥!)

        disinhibit_push = -100.0  # ★ v37f: Fear Push(80)보다 강한 억제!
        disinhibit_pull = 80.0    # ★ v37f: Fear Pull(-60)을 상쇄!
        print(f"  Disinhibition: Push({disinhibit_push}) + Pull(+{disinhibit_pull}) - v37f scaled!")

        # 적 머리 왼쪽 → 오른쪽 모터 억제 (Push 상쇄)
        self.syn_enemy_head_left_motor_right_inhib = create_static_synapse(
            "enemy_head_left_motor_right_inhib", self.enemy_head_left, self.motor_right,
            n_enemy_head_half, self.config.n_motor_right,
            sparsity=attack_sparsity, w_init=disinhibit_push)
        # 적 머리 오른쪽 → 왼쪽 모터 억제 (Push 상쇄)
        self.syn_enemy_head_right_motor_left_inhib = create_static_synapse(
            "enemy_head_right_motor_left_inhib", self.enemy_head_right, self.motor_left,
            n_enemy_head_half, self.config.n_motor_left,
            sparsity=attack_sparsity, w_init=disinhibit_push)

        # v32b: 적 머리 왼쪽 → 왼쪽 모터 활성화 (Pull 상쇄!)
        # Hunt(35) + DisinhibitPull(60) = +95 vs FearPull(-80) → 넷 +15 (사냥 승리!)
        self.syn_enemy_head_left_motor_left_boost = create_static_synapse(
            "enemy_head_left_motor_left_boost", self.enemy_head_left, self.motor_left,
            n_enemy_head_half, self.config.n_motor_left,
            sparsity=attack_sparsity, w_init=disinhibit_pull)
        self.syn_enemy_head_right_motor_right_boost = create_static_synapse(
            "enemy_head_right_motor_right_boost", self.enemy_head_right, self.motor_right,
            n_enemy_head_half, self.config.n_motor_right,
            sparsity=attack_sparsity, w_init=disinhibit_pull)

        # === v37f: PROXIMITY FEAR - 최소화 ===
        # v37e 결과: -40에서도 킬 0 → 효과 미미
        # v37f: -20으로 더 약화 (거의 무시 수준)
        #
        # 주 방어는 약화된 Fear(80)에 맡기고
        # Hunt(180)가 압도하도록 설계
        proximity_inhibit = -20.0  # v37f: 최소 억제
        print(f"  Proximity Fear: Body→Head INHIBITION (w={proximity_inhibit}) - v37f MINIMAL")

        self.syn_body_inhibits_head_left = create_static_synapse(
            "body_inhibits_head_left", self.enemy_eye_left, self.enemy_head_left,
            n_enemy_half, n_enemy_head_half,
            sparsity=0.3, w_init=proximity_inhibit)
        self.syn_body_inhibits_head_right = create_static_synapse(
            "body_inhibits_head_right", self.enemy_eye_right, self.enemy_head_right,
            n_enemy_half, n_enemy_head_half,
            sparsity=0.3, w_init=proximity_inhibit)

        # === v33: ATTACK WTA (승자 독식) - 사냥 방향 결정! ===
        # 문제: L=0.68, R=0.62 → 양쪽 모터 비슷하게 활성화 → 진동 (부리단의 당나귀)
        # 해결: EnemyHead_L ↔ EnemyHead_R 측면 억제 → 강한 쪽이 약한 쪽을 제압!
        #
        # 결과: L=0.68, R=0.62 → L이 R을 억제 → R≈0 → 왼쪽으로 풀 스로틀!
        attack_wta_weight = -25.0  # 강한 억제로 확실한 결정
        attack_wta_sparsity = 0.3
        print(f"  Attack WTA: EnemyHead L↔R (w={attack_wta_weight}) - 'Choose ONE target!'")

        self.syn_head_left_inhibits_right = create_static_synapse(
            "head_left_inhibits_right", self.enemy_head_left, self.enemy_head_right,
            n_enemy_head_half, n_enemy_head_half,
            sparsity=attack_wta_sparsity, w_init=attack_wta_weight)
        self.syn_head_right_inhibits_left = create_static_synapse(
            "head_right_inhibits_left", self.enemy_head_right, self.enemy_head_left,
            n_enemy_head_half, n_enemy_head_half,
            sparsity=attack_wta_sparsity, w_init=attack_wta_weight)

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

        # v23: DENSE 연결 (전체 연결)
        n_conn = 1000 * 1500  # n_food_half * n_motor_left
        print(f"  Food_L→Motor_L connections: {n_conn} (DENSE)")
        print(f"  Model ready! {self.config.total_neurons:,} neurons")

        # State
        self.dopamine = 0.5
        self.fear_level = 0.0
        self.attack_level = 0.0
        self.hunger_level = 0.5
        self.steps = 0
        self.generation = 0  # 윤회 세대
        self.stats = {'food_eaten': 0, 'boosts': 0, 'fear_triggers': 0, 'attack_triggers': 0}

        # v27g: 모터 출력 스무딩 (EMA) - 빠른 반응
        self.prev_left_rate = 0.5
        self.prev_right_rate = 0.5
        self.motor_smoothing = 0.6  # v27g: 0.3→0.6 (더 빠른 반응)

    def process(self, sensor_input: np.ndarray, reward: float = 0.0) -> Tuple[float, float, bool]:
        """센서 입력 처리 및 행동 출력 (v30: Hunt Mode)"""
        # Unpack sensor input (4 channels: food, enemy_body, body, enemy_head)
        food_signal = sensor_input[0]
        enemy_signal = sensor_input[1]  # enemy body (danger - 회피 대상)
        body_signal = sensor_input[2]
        enemy_head_signal = sensor_input[3] if len(sensor_input) > 3 else np.zeros_like(food_signal)  # v30: 공격 대상!

        n_rays = len(food_signal)
        mid = n_rays // 2

        # === ENCODE SENSORY INPUT (v19: L/R 분리) ===
        # 왼쪽 절반 = 왼쪽 시야, 오른쪽 절반 = 오른쪽 시야
        n_food_half = self.config.n_food_eye // 2
        n_enemy_half = self.config.n_enemy_eye // 2

        food_left_encoded = self._encode_to_population(food_signal[:mid], n_food_half)
        food_right_encoded = self._encode_to_population(food_signal[mid:], n_food_half)
        enemy_left_encoded = self._encode_to_population(enemy_signal[:mid], n_enemy_half)
        enemy_right_encoded = self._encode_to_population(enemy_signal[mid:], n_enemy_half)
        # v27j: body_eye split into L/R for wall avoidance
        n_body_half = self.config.n_body_eye // 2
        body_left_encoded = self._encode_to_population(body_signal[:mid], n_body_half)
        body_right_encoded = self._encode_to_population(body_signal[mid:], n_body_half)
        # v30: enemy_head split into L/R for hunting
        # v33: Apply WTA to head signals BEFORE encoding!
        # Problem: L=0.68, R=0.62 → both sides activate → oscillation
        # Solution: Winner takes all - only stronger side gets signal
        head_left_raw = enemy_head_signal[:mid].max()
        head_right_raw = enemy_head_signal[mid:].max()

        # WTA: If one side is significantly stronger, suppress the other
        wta_threshold = 0.02  # v33: Lowered from 0.05 - be more decisive!
        if head_left_raw > head_right_raw + wta_threshold:
            # Left wins - FULL suppress right (승자 독식!)
            enemy_head_signal_processed = enemy_head_signal.copy()
            enemy_head_signal_processed[mid:] = 0  # 100% suppression - 오직 왼쪽만!
        elif head_right_raw > head_left_raw + wta_threshold:
            # Right wins - FULL suppress left (승자 독식!)
            enemy_head_signal_processed = enemy_head_signal.copy()
            enemy_head_signal_processed[:mid] = 0  # 100% suppression - 오직 오른쪽만!
        else:
            # Too close to call - keep both (defensive posture)
            enemy_head_signal_processed = enemy_head_signal

        n_enemy_head_half = n_enemy_half // 2
        enemy_head_left_encoded = self._encode_to_population(enemy_head_signal_processed[:mid], n_enemy_head_half)
        enemy_head_right_encoded = self._encode_to_population(enemy_head_signal_processed[mid:], n_enemy_head_half)

        # === SIMULATE (v23: I_input 전류 주입) ===
        # 전압 직접 설정 대신 전류 주입 → 정상적인 스파이크 이벤트 발생
        current_scale = 3.0  # 전류 강도 (threshold를 넘을 수 있도록 충분히 강하게)

        # 전류 값 설정 (한 번만 설정, 10 스텝 동안 유지)
        # v28b: 음식 감각 민감도 - 균형
        food_sensitivity = 1.5  # v28b: 1.8→1.5 (균형)
        self.food_eye_left.vars["I_input"].view[:] = food_left_encoded * current_scale * food_sensitivity
        self.food_eye_right.vars["I_input"].view[:] = food_right_encoded * current_scale * food_sensitivity
        self.enemy_eye_left.vars["I_input"].view[:] = enemy_left_encoded * current_scale * 1.2  # 적 신호 강화
        self.enemy_eye_right.vars["I_input"].view[:] = enemy_right_encoded * current_scale * 1.2
        # v27j: body/wall signal split into L/R
        self.body_eye_left.vars["I_input"].view[:] = body_left_encoded * current_scale * 1.0
        self.body_eye_right.vars["I_input"].view[:] = body_right_encoded * current_scale * 1.0

        # === v30: ENEMY HEAD HUNTING (적 머리 방향으로 돌진!) ===
        # 적 머리가 가까울수록 강하게 자극 → 동측 배선으로 그 방향으로 회전
        # 가중치 균형: 회피(100) > 사냥(35) > 음식(20) → 위험하면 회피, 안전하면 사냥
        enemy_head_sensitivity = 1.5  # v30: 적 머리 감지 민감도
        self.enemy_head_left.vars["I_input"].view[:] = enemy_head_left_encoded * current_scale * enemy_head_sensitivity
        self.enemy_head_right.vars["I_input"].view[:] = enemy_head_right_encoded * current_scale * enemy_head_sensitivity

        # GPU로 전송
        self.food_eye_left.push_var_to_device("I_input")
        self.food_eye_right.push_var_to_device("I_input")
        self.enemy_eye_left.push_var_to_device("I_input")
        self.enemy_eye_right.push_var_to_device("I_input")
        self.body_eye_left.push_var_to_device("I_input")
        self.body_eye_right.push_var_to_device("I_input")
        self.enemy_head_left.push_var_to_device("I_input")
        self.enemy_head_right.push_var_to_device("I_input")

        # === v26: 시뮬레이션 + 스파이크 누적 ===
        # RefracTime은 2ms 후 decay하므로, 매 스텝마다 새 스파이크 감지 필요
        n_motor_left = self.config.n_motor_left
        n_motor_right = self.config.n_motor_right
        n_motor_boost = self.config.n_motor_boost

        # 스파이크 누적 카운터
        left_spike_count = 0
        right_spike_count = 0
        boost_spike_count = 0

        # 새 스파이크 감지 임계값 (RefracTime이 TauRefrac 근처면 "방금 스파이크")
        spike_threshold = self.config.tau_refrac - 0.5  # 1.5ms

        for _ in range(10):
            self.model.step_time()

            # 매 스텝마다 새 스파이크 카운트 (RefracTime > threshold)
            self.motor_left.vars["RefracTime"].pull_from_device()
            self.motor_right.vars["RefracTime"].pull_from_device()
            self.motor_boost.vars["RefracTime"].pull_from_device()

            left_spike_count += np.sum(self.motor_left.vars["RefracTime"].view > spike_threshold)
            right_spike_count += np.sum(self.motor_right.vars["RefracTime"].view > spike_threshold)
            boost_spike_count += np.sum(self.motor_boost.vars["RefracTime"].view > spike_threshold)

        self.steps += 10

        # === READ MOTOR OUTPUT (v26: 누적 스파이크로 활성도 계산) ===
        # 최대 가능 스파이크: n_neurons * 10_steps / 2 (refractory 고려)
        max_spikes_left = n_motor_left * 5  # 2ms refractory → 최대 5번 스파이크 가능
        max_spikes_right = n_motor_right * 5
        max_spikes_boost = n_motor_boost * 5

        raw_left_rate = float(min(left_spike_count / max_spikes_left, 1.0))
        raw_right_rate = float(min(right_spike_count / max_spikes_right, 1.0))
        boost_rate = float(min(boost_spike_count / max_spikes_boost, 1.0))

        # v27e: EMA 스무딩 (진동 감소)
        alpha = self.motor_smoothing
        left_rate = alpha * raw_left_rate + (1 - alpha) * self.prev_left_rate
        right_rate = alpha * raw_right_rate + (1 - alpha) * self.prev_right_rate
        self.prev_left_rate = left_rate
        self.prev_right_rate = right_rate

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

        # === COMPUTE ACTION (v27i: RELATIVE angle_delta 출력) ===
        # Direction from motor difference
        # Positive = turn RIGHT (clockwise in screen coords)
        # Negative = turn LEFT (counterclockwise in screen coords)
        angle_delta = (right_rate - left_rate) * 0.3  # Scale to gym's [-0.3, 0.3] range

        # v33 디버그: WTA 효과 확인
        if enemy_signal.max() > 0.3 or enemy_head_signal.max() > 0.2:
            enemy_l = enemy_signal[:len(enemy_signal)//2].max()
            enemy_r = enemy_signal[len(enemy_signal)//2:].max()
            head_l = enemy_head_signal[:len(enemy_head_signal)//2].max()
            head_r = enemy_head_signal[len(enemy_head_signal)//2:].max()
            # v33: WTA 결과 표시
            wta_winner = "=" if abs(head_l - head_r) < 0.02 else ("L⚔" if head_l > head_r else "⚔R")
            turn_dir = "RIGHT" if angle_delta > 0 else "LEFT"
            hunt_active = "🎯" if (head_l > 0.3 or head_r > 0.3) else ""
            print(f"[DBG] Body L={enemy_l:.2f} R={enemy_r:.2f} | Head L={head_l:.2f} R={head_r:.2f} {wta_winner} {hunt_active}| M_L={left_rate:.3f} M_R={right_rate:.3f} | δ={angle_delta:+.3f} → {turn_dir}")

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

        # v27i: Return RELATIVE angle_delta instead of absolute coordinates
        # The gym supports (angle_delta, boost) format for direct control
        return angle_delta, boost

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

    def _decode_spike_rate(self, refrac_time: np.ndarray) -> float:
        """RefracTime으로 스파이크 비율 계산 (v26)

        뉴런이 스파이크하면 RefracTime = tau_refrac (2.0ms)
        RefracTime > 0이면 최근 refractory period 내에 스파이크함
        """
        # 최근 스파이크 (RefracTime > 0)의 비율
        spike_count = np.sum(refrac_time > 0)
        spike_rate = spike_count / len(refrac_time)
        return float(spike_rate)

    def _update_dopamine(self, reward: float):
        """도파민 업데이트 및 GPU 전송 (v20: 보상 신호 강화)"""
        # v20: 0.15 → 0.30 (2배 강화) - 음식/죽음의 영향력 증가
        self.dopamine = np.clip(self.dopamine + reward * 0.30, 0.0, 1.0)
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

    def get_hunt_synapse_stats(self) -> dict:
        """v37f: Hunt는 정적 시냅스 - 고정 가중치 반환"""
        # v37f: R-STDP 제거됨, 정적 가중치 180.0 사용
        return {
            'hunt_L_mean': 180.0,  # 정적 가중치 v37f
            'hunt_R_mean': 180.0,
            'hunt_L_std': 0.0,
            'hunt_R_std': 0.0,
        }

    def reset(self, keep_weights: bool = True):
        """상태 초기화 (윤회 시스템)

        Args:
            keep_weights: True면 시냅스 가중치 유지 (윤회)
                         False면 완전 초기화
        """
        all_pops = [self.food_eye_left, self.food_eye_right,
                    self.enemy_eye_left, self.enemy_eye_right,
                    self.body_eye_left, self.body_eye_right,  # v27j: L/R split
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

        # v27e: EMA 스무딩 상태 초기화
        self.prev_left_rate = 0.5
        self.prev_right_rate = 0.5

    def save_weights(self, path: Path):
        """시냅스 가중치 + 연결 인덱스 저장 (SPARSE connectivity 포함)"""
        checkpoint = {}
        for syn in self.all_synapses:
            # Pull connectivity to get indices
            syn.pull_connectivity_from_device()
            # Pull weights separately (CRITICAL!)
            syn.vars["g"].pull_from_device()
            checkpoint[f"{syn.name}_g"] = np.array(syn.vars["g"].values)
            checkpoint[f"{syn.name}_ind"] = np.array(syn.get_sparse_post_inds())
            checkpoint[f"{syn.name}_row_length"] = np.array(syn._row_lengths.view)

        np.savez(path, **checkpoint)
        print(f"  Saved weights: {path} ({len(self.all_synapses)} synapses)")

    def load_weights(self, path: Path) -> bool:
        """시냅스 가중치 + 연결 인덱스 로드 (SPARSE connectivity 복원)"""
        if not path.exists():
            print(f"  No checkpoint: {path}")
            return False

        try:
            data = np.load(path)
            loaded_count = 0

            for syn in self.all_synapses:
                g_key = f"{syn.name}_g"
                ind_key = f"{syn.name}_ind"
                row_key = f"{syn.name}_row_length"

                if g_key in data and ind_key in data and row_key in data:
                    # New format: restore connectivity + weights
                    saved_g = data[g_key]
                    saved_ind = data[ind_key]
                    saved_row = data[row_key]

                    # Restore connectivity structure
                    syn._row_lengths.view[:] = saved_row
                    syn._ind.view[:len(saved_ind)] = saved_ind
                    syn.push_connectivity_to_device()

                    # Restore weights
                    syn.vars["g"].values[:len(saved_g)] = saved_g
                    syn.vars["g"].push_to_device()
                    loaded_count += 1
                elif syn.name in data:
                    # Old format (weights only) - connectivity may not match
                    syn.pull_connectivity_from_device()
                    saved_weights = data[syn.name]
                    min_size = min(len(saved_weights), len(syn.vars["g"].values))
                    syn.vars["g"].values[:min_size] = saved_weights[:min_size]
                    syn.vars["g"].push_to_device()
                    loaded_count += 1

            print(f"  Loaded weights: {path} ({loaded_count} synapses)")
            return loaded_count > 0
        except Exception as e:
            print(f"  Load failed: {e}")
            import traceback
            traceback.print_exc()
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
            angle_delta, boost = self.brain.process(sensor)  # v27i: relative angle control

            obs, reward, done, info = self.env.step((angle_delta, boost))  # v27i: 2-value format
            total_reward += reward

            if reward != 0:
                self.brain.process(sensor, reward)

            self.env.render()
            step += 1

            if done:
                # 죽음! Death Penalty 적용 (윤회)
                self.brain.apply_death_penalty()
                break

        # v36: Hunt synapse stats for learning tracking
        hunt_stats = self.brain.get_hunt_synapse_stats()

        return {
            'length': info['length'],
            'steps': info['steps'],
            'reward': total_reward,
            'food_eaten': info.get('foods_eaten', 0),
            'fear_triggers': self.brain.stats['fear_triggers'],
            'attack_triggers': self.brain.stats['attack_triggers'],
            'boosts': self.brain.stats['boosts'],
            'generation': self.brain.generation,
            'kills': info.get('total_kills', 0),
            'hunt_L_mean': hunt_stats['hunt_L_mean'],
            'hunt_R_mean': hunt_stats['hunt_R_mean'],
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
        print(f"  Hunt R-STDP: τ={self.brain.config.tau_eligibility_hunt}ms (Long Memory!)")
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
            kills = result.get('kills', 0)
            hunt_L = result.get('hunt_L_mean', 0)
            hunt_R = result.get('hunt_R_mean', 0)

            # v36: Kill 표시 (있으면 강조!)
            kill_str = f"★KILL:{kills}★ " if kills > 0 else ""
            print(f"[Ep {ep+1:3d}] {kill_str}Gen:{gen:3d} | Len:{result['length']:3d} | "
                  f"H:{high} A10:{avg:.0f} | "
                  f"Hunt(L={hunt_L:.1f},R={hunt_R:.1f}) | Atk:{result['attack_triggers']}")

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
    parser.add_argument('--sandbag', action='store_true', help='Sandbag training mode (small map, 1 slow enemy)')
    args = parser.parse_args()

    print("Biological PyGeNN Slither.io Agent")
    print(f"Render mode: {args.render}")

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

    # === v36: SANDBAG MODE (1v1 작은 맵, 느린 적) ===
    if args.sandbag:
        print("=" * 60)
        print("🥊 SANDBAG MODE ACTIVATED: 1v1 Cage Match")
        print("=" * 60)
        print("  Map: 500x500 (16x smaller)")
        print("  Enemy: 1 (slow, speed=0.5)")
        print("  Goal: Learn to hunt with R-STDP (τ=10s)")
        print("=" * 60)
        env_config = SlitherConfig(
            width=500,
            height=500,
            n_food=50,         # 음식 밀도 유지
            n_enemies=1,       # 1:1 결투
            enemy_speed=0.5,   # 매우 느린 적 (에이전트 속도 3.0의 1/6)
        )
    else:
        print(f"Enemies: {args.enemies}")
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
