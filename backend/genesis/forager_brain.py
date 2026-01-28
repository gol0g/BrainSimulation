"""
Forager Brain - Phase 2a+2b: 시상하부 + 편도체

Phase 1 회로 재사용:
- Push-Pull 벽 회피
- 음식 동측 배선
- WTA 모터 경쟁

Phase 2a 신규:
- Energy Sensor (내부 감각)
- Hunger Drive
- Satiety Drive
- Hunger → Food Eye 조절
- Satiety → Motor 억제

Phase 2b 신규:
- Pain Eye (고통 감지)
- Danger Sensor (위험 거리)
- Amygdala (LA, CEA) - 공포 회로
- Fear Response - 회피 반응
- Fear → Motor (회피)
- Hunger ↔ Fear 경쟁

핵심: 내부 상태(Energy) + 공포 학습에 의해 행동이 조절되는 뇌
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import os
import time
import argparse

# GeNN imports (WSL에서 실행)
try:
    from pygenn import (GeNNModel, init_sparse_connectivity, init_weight_update,
                        init_postsynaptic, create_neuron_model, init_var)
    PYGENN_AVAILABLE = True
except ImportError:
    PYGENN_AVAILABLE = False
    print("WARNING: PyGeNN not available. Running in CPU-only mode.")

from forager_gym import ForagerGym, ForagerConfig

# Checkpoint directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "forager_hypothalamus"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# === SensoryLIF Model (Phase 1에서 재사용) ===
if PYGENN_AVAILABLE:
    sensory_lif_model = create_neuron_model(
        "SensoryLIF",
        params=["C", "TauM", "Vrest", "Vreset", "Vthresh", "TauRefrac"],
        vars=[("V", "scalar"), ("RefracTime", "scalar"), ("I_input", "scalar")],
        sim_code="""
            if (RefracTime > 0.0) {
                RefracTime -= dt;
            } else {
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
class ForagerBrainConfig:
    """Phase 2a+2b 뇌 설정"""
    # === SENSORY (Phase 1 재사용) ===
    n_food_eye: int = 800       # Food detection (L: 400, R: 400)
    n_wall_eye: int = 400       # Wall detection (L: 200, R: 200)

    # === HYPOTHALAMUS (Phase 2a 신규) ===
    # 이중 센서 방식: Low와 High 분리
    n_low_energy_sensor: int = 200   # 낮은 에너지 감지 (배고픔 트리거)
    n_high_energy_sensor: int = 200  # 높은 에너지 감지 (포만감 트리거)
    n_hunger_drive: int = 500   # 배고픔 동기
    n_satiety_drive: int = 500  # 포만감 동기

    # === AMYGDALA (Phase 2b 신규) ===
    amygdala_enabled: bool = True           # Amygdala 활성화 여부
    n_pain_eye: int = 400                   # Pain Zone 감지 (L: 200, R: 200)
    n_danger_sensor: int = 200              # 위험 거리 센서
    n_lateral_amygdala: int = 500           # LA: 공포 학습
    n_central_amygdala: int = 300           # CEA: 공포 출력
    n_fear_response: int = 200              # 회피 반응

    # === HIPPOCAMPUS (Phase 3 신규) ===
    hippocampus_enabled: bool = True        # Hippocampus 활성화 여부
    n_place_cells: int = 400                # Place Cells (20x20 격자)
    n_food_memory: int = 200                # 음식 위치 기억
    place_cell_sigma: float = 0.08          # 수용장 크기 (정규화, 맵의 8%)
    place_cell_grid_size: int = 20          # 격자 크기 (20x20)

    # === MOTOR ===
    n_motor_left: int = 500
    n_motor_right: int = 500

    # Network parameters
    sparsity: float = 0.03      # 3% connectivity

    # LIF parameters
    tau_m: float = 20.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    v_thresh: float = -50.0
    tau_refrac: float = 2.0

    # WTA parameters
    wta_inhibition: float = -5.0
    wta_sparsity: float = 0.05

    # === Phase 2a 시냅스 가중치 ===
    # 시상하부 회로 (수정: 이중 센서 방식)
    low_energy_to_hunger_weight: float = 30.0   # 흥분: Low Energy → High Hunger
    high_energy_to_satiety_weight: float = 25.0 # 흥분: High Energy → High Satiety
    hunger_satiety_wta: float = -20.0           # 상호 억제 (강화)

    # 조절 회로
    hunger_to_food_eye_weight: float = 12.0  # Hunger → Food Eye 증폭
    satiety_to_motor_weight: float = -4.0    # Satiety → Motor 억제 (v2b: -8.0 → -4.0, MOTOR DEAD 완화)

    # Phase 1 회로 (벽 회피, 음식 추적)
    wall_push_weight: float = 60.0           # 벽 회피 (Push)
    wall_pull_weight: float = -40.0          # 벽 회피 (Pull)
    food_weight: float = 25.0                # 음식 추적 (동측)

    # === Phase 2b 시냅스 가중치 (신규) ===
    # Pain → LA (무조건 반사, 고정)
    pain_to_la_weight: float = 50.0          # 강한 흥분 (US)

    # Danger → LA (학습 가능 - 일단 고정으로 시작)
    danger_to_la_weight: float = 25.0        # 조건 자극 (CS)

    # LA → CEA → Fear Response (내부 연결)
    la_to_cea_weight: float = 30.0
    cea_to_fear_weight: float = 25.0

    # Pain → Motor (방향성 회피 반사, Push-Pull)
    fear_push_weight: float = 60.0           # Pain_L → Motor_R (반대편 활성화)
    fear_pull_weight: float = -40.0          # Pain_L → Motor_L (같은편 억제)

    # Hunger ↔ Fear 경쟁
    hunger_to_fear_weight: float = -15.0     # Hunger → CEA 억제 (배고프면 공포 감소)
    fear_to_hunger_weight: float = -10.0     # CEA → Hunger 억제 (공포 시 식욕 감소)

    # === Phase 3 시냅스 가중치 (신규) ===
    # Place Cells → Food Memory (Hebbian 학습)
    place_to_food_memory_weight: float = 5.0   # 초기 가중치 (학습으로 강화)
    place_to_food_memory_eta: float = 0.1      # Hebbian 학습률
    place_to_food_memory_w_max: float = 30.0   # 최대 가중치

    # Food Memory → Motor (약한 편향)
    food_memory_to_motor_weight: float = 5.0   # 음식 방향 편향 (15→5, 간섭 최소화)

    # Hunger → Food Memory (배고플 때 기억 활성화)
    hunger_to_food_memory_weight: float = 10.0 # 기억 탐색 활성화 (20→10, 간섭 최소화)

    dt: float = 1.0

    @property
    def total_neurons(self) -> int:
        base = (self.n_food_eye + self.n_wall_eye +
                self.n_low_energy_sensor + self.n_high_energy_sensor +
                self.n_hunger_drive + self.n_satiety_drive +
                self.n_motor_left + self.n_motor_right)
        if self.amygdala_enabled:
            base += (self.n_pain_eye + self.n_danger_sensor +
                     self.n_lateral_amygdala + self.n_central_amygdala +
                     self.n_fear_response)
        if self.hippocampus_enabled:
            base += (self.n_place_cells + self.n_food_memory)
        return base


class ForagerBrain:
    """
    Phase 2a+2b: 시상하부 + 편도체를 포함한 생물학적 뇌

    Phase 2a 핵심 회로:
    1. Energy Sensor: 내부 상태 인코딩
    2. Hunger/Satiety Drive: 동기 경쟁 (WTA)
    3. Hunger → Food Eye: 배고플 때 음식 감도 증가
    4. Satiety → Motor: 배부를 때 활동 감소

    Phase 2b 핵심 회로:
    5. Pain Eye: 고통 자극 감지 (US)
    6. Danger Sensor: 위험 거리 감지 (CS)
    7. Amygdala (LA → CEA): 공포 학습 및 표현
    8. Fear Response → Motor: 회피 반사
    9. Hunger ↔ Fear: 동기 경쟁
    """

    def __init__(self, config: Optional[ForagerBrainConfig] = None):
        self.config = config or ForagerBrainConfig()

        if not PYGENN_AVAILABLE:
            raise RuntimeError("PyGeNN required. Run in WSL!")

        print(f"Building Forager Brain ({self.config.total_neurons:,} neurons)...")
        print(f"  Phase 2a: Hypothalamus Circuit")

        # GeNN 모델 생성
        self.model = GeNNModel("float", "forager_brain")
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

        sensory_params = {
            "C": 1.0,
            "TauM": self.config.tau_m,
            "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset,
            "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        sensory_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. SENSORY POPULATIONS ===
        n_food_half = self.config.n_food_eye // 2
        n_wall_half = self.config.n_wall_eye // 2

        self.food_eye_left = self.model.add_neuron_population(
            "food_eye_left", n_food_half, sensory_lif_model, sensory_params, sensory_init)
        self.food_eye_right = self.model.add_neuron_population(
            "food_eye_right", n_food_half, sensory_lif_model, sensory_params, sensory_init)
        self.wall_eye_left = self.model.add_neuron_population(
            "wall_eye_left", n_wall_half, sensory_lif_model, sensory_params, sensory_init)
        self.wall_eye_right = self.model.add_neuron_population(
            "wall_eye_right", n_wall_half, sensory_lif_model, sensory_params, sensory_init)

        print(f"  Sensory: Food_L/R({n_food_half}x2) + Wall_L/R({n_wall_half}x2)")

        # === 2. HYPOTHALAMUS (Phase 2a 신규!) ===
        # 이중 센서: Low Energy (배고픔 신호), High Energy (포만 신호)
        self.low_energy_sensor = self.model.add_neuron_population(
            "low_energy_sensor", self.config.n_low_energy_sensor,
            sensory_lif_model, sensory_params, sensory_init)
        self.high_energy_sensor = self.model.add_neuron_population(
            "high_energy_sensor", self.config.n_high_energy_sensor,
            sensory_lif_model, sensory_params, sensory_init)
        self.hunger_drive = self.model.add_neuron_population(
            "hunger_drive", self.config.n_hunger_drive, "LIF", lif_params, lif_init)
        self.satiety_drive = self.model.add_neuron_population(
            "satiety_drive", self.config.n_satiety_drive, "LIF", lif_params, lif_init)

        print(f"  Hypothalamus: LowEnergy({self.config.n_low_energy_sensor}) + "
              f"HighEnergy({self.config.n_high_energy_sensor}) + "
              f"Hunger({self.config.n_hunger_drive}) + Satiety({self.config.n_satiety_drive})")

        # === 3. MOTOR POPULATIONS ===
        self.motor_left = self.model.add_neuron_population(
            "motor_left", self.config.n_motor_left, "LIF", lif_params, lif_init)
        self.motor_right = self.model.add_neuron_population(
            "motor_right", self.config.n_motor_right, "LIF", lif_params, lif_init)

        print(f"  Motor: Left({self.config.n_motor_left}) + Right({self.config.n_motor_right})")

        # === Phase 2b: AMYGDALA POPULATIONS ===
        if self.config.amygdala_enabled:
            n_pain_half = self.config.n_pain_eye // 2

            self.pain_eye_left = self.model.add_neuron_population(
                "pain_eye_left", n_pain_half, sensory_lif_model, sensory_params, sensory_init)
            self.pain_eye_right = self.model.add_neuron_population(
                "pain_eye_right", n_pain_half, sensory_lif_model, sensory_params, sensory_init)
            self.danger_sensor = self.model.add_neuron_population(
                "danger_sensor", self.config.n_danger_sensor, sensory_lif_model, sensory_params, sensory_init)
            self.lateral_amygdala = self.model.add_neuron_population(
                "lateral_amygdala", self.config.n_lateral_amygdala, "LIF", lif_params, lif_init)
            self.central_amygdala = self.model.add_neuron_population(
                "central_amygdala", self.config.n_central_amygdala, "LIF", lif_params, lif_init)
            self.fear_response = self.model.add_neuron_population(
                "fear_response", self.config.n_fear_response, "LIF", lif_params, lif_init)

            print(f"  Amygdala: Pain_L/R({n_pain_half}x2) + Danger({self.config.n_danger_sensor}) + "
                  f"LA({self.config.n_lateral_amygdala}) + CEA({self.config.n_central_amygdala}) + "
                  f"Fear({self.config.n_fear_response})")

        # === Phase 3: HIPPOCAMPUS POPULATIONS ===
        if self.config.hippocampus_enabled:
            # Place Cells: 위치에 따라 활성화 (I_input으로 외부 제어)
            self.place_cells = self.model.add_neuron_population(
                "place_cells", self.config.n_place_cells, sensory_lif_model, sensory_params, sensory_init)

            # Food Memory: 음식 위치 기억
            self.food_memory = self.model.add_neuron_population(
                "food_memory", self.config.n_food_memory, "LIF", lif_params, lif_init)

            # Place Cell 중심점 계산 (20x20 격자)
            self.place_cell_centers = []
            grid = self.config.place_cell_grid_size
            for i in range(grid):
                for j in range(grid):
                    cx = (i + 0.5) / grid  # 0~1 정규화
                    cy = (j + 0.5) / grid
                    self.place_cell_centers.append((cx, cy))

            print(f"  Hippocampus: PlaceCells({self.config.n_place_cells}) + "
                  f"FoodMemory({self.config.n_food_memory})")

        # === SYNAPTIC CONNECTIONS ===
        self._build_hypothalamus_circuit()
        self._build_modulation_circuit()
        self._build_reflex_circuit()
        self._build_motor_wta()

        # Phase 2b: Amygdala circuits
        if self.config.amygdala_enabled:
            self._build_amygdala_circuit()
            self._build_fear_motor_circuit()
            self._build_hunger_fear_competition()

        # Phase 3: Hippocampus circuits
        if self.config.hippocampus_enabled:
            self._build_hippocampus_circuit()

        # Build and load
        print("Building model...")
        self.model.build()
        self.model.load()
        print(f"Model ready! Total: {self.config.total_neurons:,} neurons")

        # 스파이크 카운팅용
        self.spike_threshold = self.config.tau_refrac - 0.5

    def _create_static_synapse(self, name: str, pre, post, weight: float,
                               sparsity: Optional[float] = None):
        """고정 가중치 시냅스 생성"""
        sp = sparsity or self.config.sparsity
        return self.model.add_synapse_population(
            name, "SPARSE", pre, post,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": sp})
        )

    def _build_hypothalamus_circuit(self):
        """시상하부 회로: Energy Sensors → Hunger/Satiety (이중 센서 방식)"""
        print("  Building Hypothalamus circuit (Dual Sensor)...")

        # Low Energy Sensor → Hunger (흥분)
        # 에너지가 낮으면 배고픔 활성화
        self._create_static_synapse(
            "low_energy_to_hunger", self.low_energy_sensor, self.hunger_drive,
            self.config.low_energy_to_hunger_weight, sparsity=0.15)

        # High Energy Sensor → Satiety (흥분)
        # 에너지가 높으면 포만감 활성화
        self._create_static_synapse(
            "high_energy_to_satiety", self.high_energy_sensor, self.satiety_drive,
            self.config.high_energy_to_satiety_weight, sparsity=0.15)

        # Hunger ↔ Satiety WTA (상호 억제) - 강화
        self._create_static_synapse(
            "hunger_to_satiety", self.hunger_drive, self.satiety_drive,
            self.config.hunger_satiety_wta, sparsity=0.08)
        self._create_static_synapse(
            "satiety_to_hunger", self.satiety_drive, self.hunger_drive,
            self.config.hunger_satiety_wta, sparsity=0.08)

        print(f"    LowEnergy→Hunger: {self.config.low_energy_to_hunger_weight} (excite)")
        print(f"    HighEnergy→Satiety: {self.config.high_energy_to_satiety_weight} (excite)")
        print(f"    Hunger↔Satiety: {self.config.hunger_satiety_wta} (WTA)")

    def _build_modulation_circuit(self):
        """조절 회로: Hunger/Satiety가 행동 조절"""
        print("  Building Modulation circuit...")

        # Hunger → Food Eye (증폭)
        # 배고프면 음식 신호에 더 민감
        self._create_static_synapse(
            "hunger_to_food_left", self.hunger_drive, self.food_eye_left,
            self.config.hunger_to_food_eye_weight, sparsity=0.08)
        self._create_static_synapse(
            "hunger_to_food_right", self.hunger_drive, self.food_eye_right,
            self.config.hunger_to_food_eye_weight, sparsity=0.08)

        # Satiety → Motor (억제)
        # 배부르면 전반적 활동 감소
        self._create_static_synapse(
            "satiety_to_motor_left", self.satiety_drive, self.motor_left,
            self.config.satiety_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "satiety_to_motor_right", self.satiety_drive, self.motor_right,
            self.config.satiety_to_motor_weight, sparsity=0.1)

        print(f"    Hunger→FoodEye: {self.config.hunger_to_food_eye_weight} (amplify)")
        print(f"    Satiety→Motor: {self.config.satiety_to_motor_weight} (suppress)")

    def _build_reflex_circuit(self):
        """반사 회로: 벽 회피 + 음식 추적 (Phase 1 재사용)"""
        print("  Building Reflex circuit (Phase 1)...")

        n_food_half = self.config.n_food_eye // 2
        n_wall_half = self.config.n_wall_eye // 2

        # Wall avoidance: Push-Pull
        # Wall_L → Motor_R (Push)
        self._create_static_synapse(
            "wall_left_motor_right", self.wall_eye_left, self.motor_right,
            self.config.wall_push_weight, sparsity=0.15)
        self._create_static_synapse(
            "wall_right_motor_left", self.wall_eye_right, self.motor_left,
            self.config.wall_push_weight, sparsity=0.15)

        # Wall_L → Motor_L (Pull - inhibit)
        self._create_static_synapse(
            "wall_left_motor_left_inhib", self.wall_eye_left, self.motor_left,
            self.config.wall_pull_weight, sparsity=0.15)
        self._create_static_synapse(
            "wall_right_motor_right_inhib", self.wall_eye_right, self.motor_right,
            self.config.wall_pull_weight, sparsity=0.15)

        print(f"    Wall Push: {self.config.wall_push_weight}")
        print(f"    Wall Pull: {self.config.wall_pull_weight}")

        # Food tracking: Ipsilateral (동측)
        # Food_L → Motor_L (같은 방향으로)
        self._create_static_synapse(
            "food_left_motor_left", self.food_eye_left, self.motor_left,
            self.config.food_weight, sparsity=0.15)
        self._create_static_synapse(
            "food_right_motor_right", self.food_eye_right, self.motor_right,
            self.config.food_weight, sparsity=0.15)

        print(f"    Food Ipsi: {self.config.food_weight}")

    def _build_motor_wta(self):
        """모터 WTA: 좌우 모터 경쟁"""
        print("  Building Motor WTA...")

        self._create_static_synapse(
            "motor_left_right_wta", self.motor_left, self.motor_right,
            self.config.wta_inhibition, sparsity=self.config.wta_sparsity)
        self._create_static_synapse(
            "motor_right_left_wta", self.motor_right, self.motor_left,
            self.config.wta_inhibition, sparsity=self.config.wta_sparsity)

        print(f"    Motor WTA: {self.config.wta_inhibition}")

    # === Phase 2b: Amygdala Circuits ===

    def _build_amygdala_circuit(self):
        """편도체 회로: Pain/Danger → LA → CEA → Fear"""
        print("  Building Amygdala circuit (Phase 2b)...")

        # 1. Pain → LA (무조건 반사, US)
        # 고통 자극은 LA를 직접 활성화
        self._create_static_synapse(
            "pain_left_to_la", self.pain_eye_left, self.lateral_amygdala,
            self.config.pain_to_la_weight, sparsity=0.15)
        self._create_static_synapse(
            "pain_right_to_la", self.pain_eye_right, self.lateral_amygdala,
            self.config.pain_to_la_weight, sparsity=0.15)

        print(f"    Pain→LA: {self.config.pain_to_la_weight} (US, unconditional)")

        # 2. Danger → LA (조건 자극, CS)
        # 위험 거리 신호가 LA를 활성화 (학습 대상 - 일단 고정)
        self._create_static_synapse(
            "danger_to_la", self.danger_sensor, self.lateral_amygdala,
            self.config.danger_to_la_weight, sparsity=0.15)

        print(f"    Danger→LA: {self.config.danger_to_la_weight} (CS, conditioned)")

        # 3. LA → CEA (내부 연결)
        self._create_static_synapse(
            "la_to_cea", self.lateral_amygdala, self.central_amygdala,
            self.config.la_to_cea_weight, sparsity=0.1)

        print(f"    LA→CEA: {self.config.la_to_cea_weight}")

        # 4. CEA → Fear Response (공포 출력)
        self._create_static_synapse(
            "cea_to_fear", self.central_amygdala, self.fear_response,
            self.config.cea_to_fear_weight, sparsity=0.1)

        print(f"    CEA→Fear: {self.config.cea_to_fear_weight}")

    def _build_fear_motor_circuit(self):
        """Fear → Motor 회피 반사 (방향성 Push-Pull)"""
        print("  Building Fear-Motor circuit...")

        # v2b 수정: Pain Eye L/R → Motor 직접 연결 (방향성 회피)
        # 벽 회피와 동일한 Push-Pull 패턴

        # Pain Left → Motor Right (Push: 왼쪽 고통 → 오른쪽으로 회전)
        self._create_static_synapse(
            "pain_left_to_motor_right", self.pain_eye_left, self.motor_right,
            self.config.fear_push_weight, sparsity=0.15)
        # Pain Left → Motor Left (Pull: 왼쪽 억제)
        self._create_static_synapse(
            "pain_left_to_motor_left", self.pain_eye_left, self.motor_left,
            self.config.fear_pull_weight, sparsity=0.15)

        # Pain Right → Motor Left (Push: 오른쪽 고통 → 왼쪽으로 회전)
        self._create_static_synapse(
            "pain_right_to_motor_left", self.pain_eye_right, self.motor_left,
            self.config.fear_push_weight, sparsity=0.15)
        # Pain Right → Motor Right (Pull: 오른쪽 억제)
        self._create_static_synapse(
            "pain_right_to_motor_right", self.pain_eye_right, self.motor_right,
            self.config.fear_pull_weight, sparsity=0.15)

        print(f"    Pain→Motor (Push-Pull): push={self.config.fear_push_weight}, pull={self.config.fear_pull_weight}")

    def _build_hunger_fear_competition(self):
        """Hunger ↔ Fear 경쟁 회로"""
        print("  Building Hunger-Fear competition...")

        # Hunger → CEA 억제 (배고프면 공포 감소)
        self._create_static_synapse(
            "hunger_to_cea", self.hunger_drive, self.central_amygdala,
            self.config.hunger_to_fear_weight, sparsity=0.08)

        print(f"    Hunger→CEA: {self.config.hunger_to_fear_weight} (suppress fear)")

        # CEA → Hunger 억제 (공포 시 식욕 감소)
        self._create_static_synapse(
            "cea_to_hunger", self.central_amygdala, self.hunger_drive,
            self.config.fear_to_hunger_weight, sparsity=0.08)

        print(f"    CEA→Hunger: {self.config.fear_to_hunger_weight} (suppress appetite)")

    # === Phase 3: Hippocampus Circuits ===

    def _build_hippocampus_circuit(self):
        """해마 회로: Place Cells → Food Memory → Motor"""
        print("  Building Hippocampus circuit (Phase 3)...")

        # 1. Place Cells → Food Memory (고정, 학습은 나중에)
        # 현재 활성화된 Place Cell이 Food Memory를 활성화
        self._create_static_synapse(
            "place_to_food_memory", self.place_cells, self.food_memory,
            self.config.place_to_food_memory_weight, sparsity=0.1)

        print(f"    PlaceCells→FoodMemory: {self.config.place_to_food_memory_weight}")

        # 2. Food Memory → Motor (약한 편향, 동측 배선)
        # Food Memory가 활성화되면 해당 방향으로 약하게 회전
        # 좌/우 분리를 위해 Food Memory를 반으로 나눔
        n_half = self.config.n_food_memory // 2

        # Food Memory 왼쪽 절반 → Motor Left
        self.food_memory_left_to_motor = self.model.add_synapse_population(
            "food_memory_left_to_motor", "SPARSE",
            self.food_memory, self.motor_left,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.food_memory_to_motor_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbabilityNoAutapse", {"prob": 0.1})
        )

        # Food Memory 오른쪽 절반 → Motor Right
        self.food_memory_right_to_motor = self.model.add_synapse_population(
            "food_memory_right_to_motor", "SPARSE",
            self.food_memory, self.motor_right,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.food_memory_to_motor_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbabilityNoAutapse", {"prob": 0.1})
        )

        print(f"    FoodMemory→Motor: {self.config.food_memory_to_motor_weight}")

        # 3. Hunger → Food Memory (배고플 때 기억 활성화 증폭)
        self._create_static_synapse(
            "hunger_to_food_memory", self.hunger_drive, self.food_memory,
            self.config.hunger_to_food_memory_weight, sparsity=0.1)

        print(f"    Hunger→FoodMemory: {self.config.hunger_to_food_memory_weight} (amplify when hungry)")

    def _compute_place_cell_input(self, pos_x: float, pos_y: float) -> np.ndarray:
        """
        위치를 Place Cell 입력 전류로 변환

        Args:
            pos_x, pos_y: 정규화된 위치 (0~1)

        Returns:
            Place Cell 입력 전류 배열
        """
        currents = np.zeros(self.config.n_place_cells)
        sigma = self.config.place_cell_sigma

        for i, (cx, cy) in enumerate(self.place_cell_centers):
            # 가우시안 활성화
            dist_sq = (pos_x - cx)**2 + (pos_y - cy)**2
            activation = np.exp(-dist_sq / (2 * sigma**2))
            currents[i] = activation * 50.0  # 최대 전류 50

        return currents

    def process(self, observation: Dict, debug: bool = False) -> Tuple[float, Dict]:
        """
        관찰을 받아 행동 출력

        Args:
            observation: ForagerGym observation dict
            debug: 상세 로그 출력 여부

        Returns:
            angle_delta, debug_info
        """
        # === 1. 외부 감각 입력 ===
        food_l = np.mean(observation["food_rays_left"])
        food_r = np.mean(observation["food_rays_right"])
        wall_l = np.mean(observation["wall_rays_left"])
        wall_r = np.mean(observation["wall_rays_right"])

        # Food Eye 입력 (감도 스케일링)
        food_sensitivity = 50.0
        self.food_eye_left.vars["I_input"].view[:] = food_l * food_sensitivity
        self.food_eye_right.vars["I_input"].view[:] = food_r * food_sensitivity
        self.food_eye_left.vars["I_input"].push_to_device()
        self.food_eye_right.vars["I_input"].push_to_device()

        # Wall Eye 입력
        wall_sensitivity = 40.0
        self.wall_eye_left.vars["I_input"].view[:] = wall_l * wall_sensitivity
        self.wall_eye_right.vars["I_input"].view[:] = wall_r * wall_sensitivity
        self.wall_eye_left.vars["I_input"].push_to_device()
        self.wall_eye_right.vars["I_input"].push_to_device()

        # === 2. 내부 감각 입력 (Phase 2a 핵심!) ===
        energy = observation["energy"]  # 0~1 정규화됨

        # 임계값 기반 인코딩:
        # - Low Energy Sensor: energy < 0.4 일 때만 강하게 발화
        # - High Energy Sensor: energy > 0.6 일 때만 강하게 발화
        # - 0.4 ~ 0.6 범위: 중립 (둘 다 약함)
        energy_sensitivity = 60.0

        # Low Energy Sensor: 에너지가 40% 이하일 때 강하게 발화
        if energy < 0.4:
            low_signal = (0.4 - energy) / 0.3  # 0.4→0, 0.1→1
            low_signal = min(1.5, low_signal)  # 최대 1.5
        else:
            low_signal = 0.0
        self.low_energy_sensor.vars["I_input"].view[:] = low_signal * energy_sensitivity
        self.low_energy_sensor.vars["I_input"].push_to_device()

        # High Energy Sensor: 에너지가 60% 이상일 때 강하게 발화
        if energy > 0.6:
            high_signal = (energy - 0.6) / 0.2  # 0.6→0, 0.8→1, 1.0→2
            high_signal = min(2.0, high_signal)  # 최대 2.0
        else:
            high_signal = 0.0
        self.high_energy_sensor.vars["I_input"].view[:] = high_signal * energy_sensitivity
        self.high_energy_sensor.vars["I_input"].push_to_device()

        # === Phase 2b: Pain/Danger 감각 입력 ===
        pain_l = 0.0
        pain_r = 0.0
        danger_signal = 0.0

        if self.config.amygdala_enabled:
            pain_l = np.mean(observation.get("pain_rays_left", np.zeros(8)))
            pain_r = np.mean(observation.get("pain_rays_right", np.zeros(8)))
            danger_signal = observation.get("danger_signal", 0.0)

            # Pain Eye 입력
            pain_sensitivity = 60.0
            self.pain_eye_left.vars["I_input"].view[:] = pain_l * pain_sensitivity
            self.pain_eye_right.vars["I_input"].view[:] = pain_r * pain_sensitivity
            self.pain_eye_left.vars["I_input"].push_to_device()
            self.pain_eye_right.vars["I_input"].push_to_device()

            # Danger Sensor 입력
            danger_sensitivity = 50.0
            self.danger_sensor.vars["I_input"].view[:] = danger_signal * danger_sensitivity
            self.danger_sensor.vars["I_input"].push_to_device()

        # === Phase 3: Place Cell 입력 ===
        if self.config.hippocampus_enabled:
            pos_x = observation.get("position_x", 0.5)
            pos_y = observation.get("position_y", 0.5)

            # Place Cell 활성화 계산 (가우시안 수용장)
            place_cell_currents = self._compute_place_cell_input(pos_x, pos_y)
            self.place_cells.vars["I_input"].view[:] = place_cell_currents
            self.place_cells.vars["I_input"].push_to_device()

        # === 3. 시뮬레이션 (10ms) ===
        # 스파이크 카운트 초기화
        motor_left_spikes = 0
        motor_right_spikes = 0
        hunger_spikes = 0
        satiety_spikes = 0
        low_energy_spikes = 0
        high_energy_spikes = 0

        # Phase 2b 스파이크 카운트
        la_spikes = 0
        cea_spikes = 0
        fear_spikes = 0

        # Phase 3 스파이크 카운트
        place_cell_spikes = 0
        food_memory_spikes = 0

        for _ in range(10):
            self.model.step_time()

            # 스파이크 카운팅 (Phase 2a)
            self.motor_left.vars["RefracTime"].pull_from_device()
            self.motor_right.vars["RefracTime"].pull_from_device()
            self.hunger_drive.vars["RefracTime"].pull_from_device()
            self.satiety_drive.vars["RefracTime"].pull_from_device()
            self.low_energy_sensor.vars["RefracTime"].pull_from_device()
            self.high_energy_sensor.vars["RefracTime"].pull_from_device()

            motor_left_spikes += np.sum(self.motor_left.vars["RefracTime"].view > self.spike_threshold)
            motor_right_spikes += np.sum(self.motor_right.vars["RefracTime"].view > self.spike_threshold)
            hunger_spikes += np.sum(self.hunger_drive.vars["RefracTime"].view > self.spike_threshold)
            satiety_spikes += np.sum(self.satiety_drive.vars["RefracTime"].view > self.spike_threshold)
            low_energy_spikes += np.sum(self.low_energy_sensor.vars["RefracTime"].view > self.spike_threshold)
            high_energy_spikes += np.sum(self.high_energy_sensor.vars["RefracTime"].view > self.spike_threshold)

            # Phase 2b 스파이크 카운팅
            if self.config.amygdala_enabled:
                self.lateral_amygdala.vars["RefracTime"].pull_from_device()
                self.central_amygdala.vars["RefracTime"].pull_from_device()
                self.fear_response.vars["RefracTime"].pull_from_device()

                la_spikes += np.sum(self.lateral_amygdala.vars["RefracTime"].view > self.spike_threshold)
                cea_spikes += np.sum(self.central_amygdala.vars["RefracTime"].view > self.spike_threshold)
                fear_spikes += np.sum(self.fear_response.vars["RefracTime"].view > self.spike_threshold)

            # Phase 3 스파이크 카운팅
            if self.config.hippocampus_enabled:
                self.place_cells.vars["RefracTime"].pull_from_device()
                self.food_memory.vars["RefracTime"].pull_from_device()

                place_cell_spikes += np.sum(self.place_cells.vars["RefracTime"].view > self.spike_threshold)
                food_memory_spikes += np.sum(self.food_memory.vars["RefracTime"].view > self.spike_threshold)

        # === 4. 스파이크율 계산 ===
        max_spikes_motor = self.config.n_motor_left * 5  # 10ms / 2ms refrac = 5 max
        max_spikes_drive = self.config.n_hunger_drive * 5
        max_spikes_energy = self.config.n_low_energy_sensor * 5

        motor_left_rate = motor_left_spikes / max_spikes_motor
        motor_right_rate = motor_right_spikes / max_spikes_motor
        hunger_rate = hunger_spikes / max_spikes_drive
        satiety_rate = satiety_spikes / max_spikes_drive
        low_energy_rate = low_energy_spikes / max_spikes_energy
        high_energy_rate = high_energy_spikes / max_spikes_energy

        # Phase 2b 스파이크율
        la_rate = 0.0
        cea_rate = 0.0
        fear_rate = 0.0
        if self.config.amygdala_enabled:
            max_spikes_la = self.config.n_lateral_amygdala * 5
            max_spikes_cea = self.config.n_central_amygdala * 5
            max_spikes_fear = self.config.n_fear_response * 5

            la_rate = la_spikes / max_spikes_la
            cea_rate = cea_spikes / max_spikes_cea
            fear_rate = fear_spikes / max_spikes_fear

        # Phase 3 스파이크율
        place_cell_rate = 0.0
        food_memory_rate = 0.0
        if self.config.hippocampus_enabled:
            max_spikes_place = self.config.n_place_cells * 5
            max_spikes_food_memory = self.config.n_food_memory * 5

            place_cell_rate = place_cell_spikes / max_spikes_place
            food_memory_rate = food_memory_spikes / max_spikes_food_memory

        # === 5. 행동 출력 ===
        angle_delta = (motor_right_rate - motor_left_rate) * 0.5

        # === 6. 디버그 정보 ===
        debug_info = {
            # 입력
            "food_l": food_l,
            "food_r": food_r,
            "wall_l": wall_l,
            "wall_r": wall_r,
            "energy_input": energy,

            # 뉴런 활성화 (Phase 2a)
            "low_energy_rate": low_energy_rate,
            "high_energy_rate": high_energy_rate,
            "hunger_rate": hunger_rate,
            "satiety_rate": satiety_rate,
            "motor_left_rate": motor_left_rate,
            "motor_right_rate": motor_right_rate,

            # Phase 2b 입력
            "pain_l": pain_l,
            "pain_r": pain_r,
            "danger_signal": danger_signal,

            # Phase 2b 뉴런 활성화
            "la_rate": la_rate,
            "cea_rate": cea_rate,
            "fear_rate": fear_rate,

            # Phase 3 뉴런 활성화
            "place_cell_rate": place_cell_rate,
            "food_memory_rate": food_memory_rate,

            # 출력
            "angle_delta": angle_delta,
        }

        # === 7. 이상 감지 ===
        self._check_anomalies(observation, debug_info)

        return angle_delta, debug_info

    def _check_anomalies(self, obs: Dict, info: Dict):
        """이상 상황 자동 감지"""
        energy = obs["energy"]
        hunger = info["hunger_rate"]
        satiety = info["satiety_rate"]
        motor_l = info["motor_left_rate"]
        motor_r = info["motor_right_rate"]

        # === Phase 2a 이상 감지 ===
        # 에너지 낮은데 Hunger 비활성화
        if energy < 0.3 and hunger < 0.1:
            print(f"\n{'!'*50}")
            print(f"  WARNING: LOW ENERGY ({energy:.2f}) but HUNGER LOW ({hunger:.2f})!")
            print(f"{'!'*50}\n")

        # 에너지 낮은데 Satiety 활성화
        if energy < 0.4 and satiety > 0.3:
            print(f"\n{'!'*50}")
            print(f"  WARNING: LOW ENERGY ({energy:.2f}) but SATIETY HIGH ({satiety:.2f})!")
            print(f"{'!'*50}\n")

        # 모터 완전 비활성화
        if motor_l < 0.02 and motor_r < 0.02:
            print(f"\n{'!'*50}")
            print(f"  WARNING: MOTOR DEAD - no movement!")
            print(f"{'!'*50}\n")

        # === Phase 2b 이상 감지 ===
        if self.config.amygdala_enabled:
            danger = info.get("danger_signal", 0)
            fear = info.get("fear_rate", 0)
            pain_l = info.get("pain_l", 0)
            pain_r = info.get("pain_r", 0)

            # Pain 신호 높은데 Fear 비활성화
            if (pain_l > 0.5 or pain_r > 0.5) and fear < 0.2:
                print(f"\n{'!'*50}")
                print(f"  WARNING: HIGH PAIN (L={pain_l:.2f}, R={pain_r:.2f}) but FEAR LOW ({fear:.2f})!")
                print(f"{'!'*50}\n")

            # Danger 신호 높은데 Fear 비활성화
            if danger > 0.7 and fear < 0.3:
                print(f"\n{'!'*50}")
                print(f"  WARNING: HIGH DANGER ({danger:.2f}) but FEAR LOW ({fear:.2f})!")
                print(f"{'!'*50}\n")

            # Hunger와 Fear 모두 높음 (경쟁 테스트!)
            if hunger > 0.5 and fear > 0.5:
                print(f"  [COMPETITION] Hunger={hunger:.2f} vs Fear={fear:.2f}")

    def reset(self):
        """뇌 상태 초기화"""
        # Sensory 뉴런 (I_input 있음)
        sensory_pops = [self.food_eye_left, self.food_eye_right,
                        self.wall_eye_left, self.wall_eye_right,
                        self.low_energy_sensor, self.high_energy_sensor]

        # Phase 2b: Pain/Danger 센서 추가
        if self.config.amygdala_enabled:
            sensory_pops.extend([self.pain_eye_left, self.pain_eye_right,
                                 self.danger_sensor])

        for pop in sensory_pops:
            pop.vars["V"].view[:] = self.config.v_rest
            pop.vars["RefracTime"].view[:] = 0.0
            pop.vars["I_input"].view[:] = 0.0
            pop.vars["V"].push_to_device()
            pop.vars["RefracTime"].push_to_device()
            pop.vars["I_input"].push_to_device()

        # 일반 LIF 뉴런 (I_input 없음)
        lif_pops = [self.hunger_drive, self.satiety_drive,
                    self.motor_left, self.motor_right]

        # Phase 2b: Amygdala 뉴런 추가
        if self.config.amygdala_enabled:
            lif_pops.extend([self.lateral_amygdala, self.central_amygdala,
                             self.fear_response])

        for pop in lif_pops:
            pop.vars["V"].view[:] = self.config.v_rest
            pop.vars["RefracTime"].view[:] = 0.0
            pop.vars["V"].push_to_device()
            pop.vars["RefracTime"].push_to_device()


def run_training(episodes: int = 20, render_mode: str = "none",
                log_level: str = "normal", debug: bool = False,
                no_amygdala: bool = False, no_pain: bool = False):
    """Phase 2a+2b 훈련 실행"""

    print("=" * 70)
    print("Phase 2b: Forager Training with Hypothalamus + Amygdala")
    print("=" * 70)

    # 환경 및 뇌 생성
    env_config = ForagerConfig()
    brain_config = ForagerBrainConfig()

    # 옵션 처리
    if no_pain:
        env_config.pain_zone_enabled = False
        print("  [!] Pain Zone DISABLED (Phase 2a mode)")
    if no_amygdala:
        brain_config.amygdala_enabled = False
        print("  [!] Amygdala DISABLED (Phase 2a mode)")

    env = ForagerGym(env_config, render_mode=render_mode)
    brain = ForagerBrain(brain_config)

    # 통계
    all_steps = []
    all_food = []
    all_homeostasis = []
    all_pain_visits = []
    all_pain_steps = []
    death_causes = {"starve": 0, "timeout": 0, "pain": 0}

    for ep in range(episodes):
        obs = env.reset()
        brain.reset()
        done = False
        total_reward = 0

        # 에피소드 로그
        ep_hunger_rates = []
        ep_satiety_rates = []
        ep_fear_rates = []

        while not done:
            # 뇌 처리
            action_delta, info = brain.process(obs, debug=debug)
            action = (action_delta,)

            # 환경 스텝
            obs, reward, done, env_info = env.step(action)
            total_reward += reward

            # 통계 수집
            ep_hunger_rates.append(info["hunger_rate"])
            ep_satiety_rates.append(info["satiety_rate"])
            if brain_config.amygdala_enabled:
                ep_fear_rates.append(info["fear_rate"])

            # 스텝 로그 (debug 모드 또는 10스텝마다)
            if log_level == "verbose" or (log_level == "debug" and env.steps % 10 == 0):
                pain_str = "PAIN!" if env_info.get('in_pain', False) else "safe"
                fear_str = f"F={info.get('fear_rate', 0):.2f}" if brain_config.amygdala_enabled else ""
                print(f"[Step {env.steps:4d}] "
                      f"E={env_info['energy']:5.1f} | "
                      f"H={info['hunger_rate']:.2f} S={info['satiety_rate']:.2f} {fear_str} | "
                      f"M_L={info['motor_left_rate']:.2f} M_R={info['motor_right_rate']:.2f} | "
                      f"{pain_str}")

            # 음식 섭취 이벤트 (normal 이상)
            if log_level in ["normal", "debug", "verbose"] and env_info["food_eaten"]:
                print(f"  [!] FOOD EATEN at step {env.steps}, Energy: {env_info['energy']:.1f}")

            # Pain Zone 진입 이벤트
            if log_level in ["normal", "debug", "verbose"]:
                if env_info.get('in_pain', False) and env.pain_zone_visits == 1 and env.pain_zone_steps == 1:
                    print(f"  [!] ENTERED Pain Zone at step {env.steps}!")

        # 에피소드 종료
        all_steps.append(env.steps)
        all_food.append(env.total_food_eaten)
        all_homeostasis.append(env_info["homeostasis_ratio"])
        all_pain_visits.append(env_info.get("pain_visits", 0))
        all_pain_steps.append(env_info.get("pain_steps", 0))

        if env_info["death_cause"]:
            death_causes[env_info["death_cause"]] = death_causes.get(env_info["death_cause"], 0) + 1

        # 에피소드 요약
        avg_hunger = np.mean(ep_hunger_rates) if ep_hunger_rates else 0
        avg_satiety = np.mean(ep_satiety_rates) if ep_satiety_rates else 0
        avg_fear = np.mean(ep_fear_rates) if ep_fear_rates else 0

        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{episodes} Summary")
        print(f"{'='*60}")
        print(f"  Steps:        {env.steps}")
        print(f"  Final Energy: {env_info['energy']:.1f}")
        print(f"  Food Eaten:   {env.total_food_eaten}")
        print(f"  Death Cause:  {env_info['death_cause']}")
        print(f"  Reward:       {total_reward:.2f}")
        print(f"  Homeostasis:  {env_info['homeostasis_ratio']*100:.1f}%")
        print(f"  Avg Hunger:   {avg_hunger:.3f}")
        print(f"  Avg Satiety:  {avg_satiety:.3f}")

        if brain_config.amygdala_enabled:
            print(f"  --- Phase 2b ---")
            print(f"  Avg Fear:     {avg_fear:.3f}")
            print(f"  Pain Visits:  {env_info.get('pain_visits', 0)}")
            print(f"  Pain Time:    {env_info.get('pain_steps', 0)} steps")

        print(f"{'='*60}\n")

    # === 최종 요약 ===
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - Final Statistics")
    print("=" * 70)
    print(f"  Episodes:       {episodes}")
    print(f"  Avg Steps:      {np.mean(all_steps):.1f}")
    print(f"  Avg Food:       {np.mean(all_food):.1f}")
    print(f"  Avg Homeostasis:{np.mean(all_homeostasis)*100:.1f}%")
    print(f"  Reward Freq:    {np.sum(all_food) / np.sum(all_steps) * 100:.2f}%")

    if env_config.pain_zone_enabled:
        print(f"\n  === Phase 2b: Pain Zone ===")
        print(f"  Avg Pain Visits: {np.mean(all_pain_visits):.1f}")
        print(f"  Avg Pain Time:   {np.mean(all_pain_steps):.1f} steps")
        pain_pct = np.sum(all_pain_steps) / np.sum(all_steps) * 100
        print(f"  Pain Time Ratio: {pain_pct:.1f}%")

    print(f"\n  Death Causes:")
    for cause, count in death_causes.items():
        if count > 0:
            print(f"    {cause}: {count} ({count/episodes*100:.1f}%)")

    # 성공 기준 체크
    survival_rate = death_causes.get("timeout", 0) / episodes * 100
    reward_freq = np.sum(all_food) / np.sum(all_steps) * 100

    print(f"\n  === Phase 2 Validation ===")
    print(f"  Survival Rate: {survival_rate:.1f}% {'✓' if survival_rate > 40 else '✗'} (target: >40%)")
    print(f"  Reward Freq:   {reward_freq:.2f}% {'✓' if reward_freq > 5 else '✗'} (target: >5%)")

    if env_config.pain_zone_enabled:
        pain_pct = np.sum(all_pain_steps) / np.sum(all_steps) * 100
        print(f"  Pain Avoidance:{100-pain_pct:.1f}% {'✓' if pain_pct < 15 else '✗'} (target: <15% pain time)")

    print("=" * 70)

    env.close()
    return all_steps, all_food, all_homeostasis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forager Brain Training - Phase 2b (Fear Conditioning)")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--render", choices=["none", "pygame"], default="none",
                       help="Render mode")
    parser.add_argument("--log-level", choices=["minimal", "normal", "debug", "verbose"],
                       default="normal", help="Log verbosity")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-amygdala", action="store_true",
                       help="Disable Amygdala (Phase 2a mode)")
    parser.add_argument("--no-pain", action="store_true",
                       help="Disable Pain Zone (Phase 2a mode)")
    args = parser.parse_args()

    run_training(
        episodes=args.episodes,
        render_mode=args.render,
        log_level=args.log_level,
        debug=args.debug,
        no_amygdala=args.no_amygdala,
        no_pain=args.no_pain
    )
