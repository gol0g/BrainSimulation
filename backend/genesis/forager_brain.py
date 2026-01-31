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
    n_food_memory: int = 200                # 음식 위치 기억 (Phase 3c: 좌/우 각 100)
    place_cell_sigma: float = 0.08          # 수용장 크기 (정규화, 맵의 8%)
    place_cell_grid_size: int = 20          # 격자 크기 (20x20)
    directional_food_memory: bool = True    # Phase 3c: 방향성 Food Memory

    # === BASAL GANGLIA (Phase 4 신규) ===
    basal_ganglia_enabled: bool = True      # 기저핵 활성화 여부
    n_striatum: int = 400                   # Striatum (선조체): 행동 선택
    n_direct_pathway: int = 200             # Direct pathway (D1): Go 신호
    n_indirect_pathway: int = 200           # Indirect pathway (D2): NoGo 신호
    n_dopamine: int = 100                   # Dopamine neurons (VTA/SNc)

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
    food_weight: float = 40.0                # 음식 추적 (동측) Phase 7: 25→40

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
    place_to_food_memory_weight: float = 2.0   # 초기 가중치 (Phase 3c: 5→2, 학습 효과 강화)
    place_to_food_memory_eta: float = 0.15     # Hebbian 학습률 (Phase 3c: 0.1→0.15)
    place_to_food_memory_w_max: float = 30.0   # 최대 가중치

    # Food Memory → Motor (약한 편향)
    food_memory_to_motor_weight: float = 12.0  # 음식 방향 편향 (Phase 7: 8→12, 추적 강화)

    # Hunger → Food Memory (배고플 때 기억 활성화)
    hunger_to_food_memory_weight: float = 10.0 # 기억 탐색 활성화 (20→10, 간섭 최소화)

    # === Phase 4 시냅스 가중치 (신규) ===
    # Sensory → Striatum (행동 컨텍스트)
    sensory_to_striatum_weight: float = 15.0   # 감각 입력 통합
    hunger_to_striatum_weight: float = 20.0    # 배고픔 상태 → 행동 촉진

    # Striatum → Direct/Indirect pathways
    striatum_to_direct_weight: float = 20.0    # Go 신호 (25→20)
    striatum_to_indirect_weight: float = 15.0  # NoGo 신호 (20→15)
    direct_indirect_competition: float = -10.0 # D1-D2 상호 억제 (15→10, 완화)

    # Direct/Indirect → Motor
    direct_to_motor_weight: float = 15.0       # Go → Motor 촉진 (20→15, 균형)
    indirect_to_motor_weight: float = -8.0     # NoGo → Motor 억제 (15→8, 과억제 완화)

    # Dopamine modulation (보상 학습)
    dopamine_to_direct_weight: float = 25.0    # DA → D1 강화 (Go 촉진)
    dopamine_to_indirect_weight: float = -20.0 # DA → D2 억제 (NoGo 감소)

    # Dopamine 학습 파라미터
    dopamine_eta: float = 0.1                  # Dopamine 학습률
    dopamine_decay: float = 0.95               # Dopamine 감쇠율

    # === PREFRONTAL CORTEX (Phase 5 신규) ===
    prefrontal_enabled: bool = True            # PFC 활성화 여부
    n_working_memory: int = 200                # 작업 기억 뉴런
    n_goal_food: int = 50                      # 음식 탐색 목표
    n_goal_safety: int = 50                    # 안전 추구 목표
    n_inhibitory_control: int = 100            # 억제 제어

    # === Phase 5 시냅스 가중치 ===
    # 입력 → Working Memory
    place_to_working_memory_weight: float = 10.0   # 위치 정보 유지
    food_memory_to_working_memory_weight: float = 15.0  # 음식 기억 유지
    fear_to_working_memory_weight: float = 12.0    # 위험 기억 유지

    # Working Memory 재귀 연결 (지속 활성화)
    working_memory_recurrent_weight: float = 8.0   # 자기 활성화 유지
    working_memory_decay: float = 0.98             # 지속 활성화 감쇠

    # Working Memory → Goal Unit
    working_memory_to_goal_weight: float = 15.0    # 기억 → 목표 활성화

    # 내부 상태 → Goal Unit
    hunger_to_goal_food_weight: float = 25.0       # 배고픔 → 음식 목표
    fear_to_goal_safety_weight: float = 20.0       # 공포 → 안전 목표
    goal_wta_weight: float = -15.0                 # 목표 간 WTA 경쟁

    # Goal → Inhibitory Control
    goal_safety_to_inhibitory_weight: float = 20.0 # 안전 목표 → 억제 활성화

    # Inhibitory Control 출력
    inhibitory_to_direct_weight: float = -10.0     # 억제 → Direct pathway 억제 (15→10, 완화)
    inhibitory_to_motor_weight: float = -2.0       # 억제 → Motor 억제 (5→2, MOTOR DEAD 완화)

    # Goal → Motor (목표 지향 행동)
    goal_food_to_motor_weight: float = 18.0        # 음식 목표 → Motor 활성화 Phase 7: 10→18

    # === CEREBELLUM (Phase 6a 신규) ===
    cerebellum_enabled: bool = True                # 소뇌 활성화 여부
    n_granule_cells: int = 300                     # 과립세포: 입력 통합
    n_purkinje_cells: int = 100                    # 푸르키네세포: 운동 조절
    n_deep_nuclei: int = 100                       # 심부핵: 최종 출력
    n_error_signal: int = 50                       # 오류 신호 (Climbing Fiber)

    # === THALAMUS (Phase 6b 신규) ===
    thalamus_enabled: bool = True                  # 시상 활성화 여부
    n_food_relay: int = 100                        # 음식 감각 중계
    n_danger_relay: int = 100                      # 위험 감각 중계
    n_trn: int = 100                               # TRN: 억제성 게이팅
    n_arousal: int = 50                            # 각성 수준 조절

    # === Phase 6b 시냅스 가중치 ===
    # 감각 → Relay (감각 중계)
    food_eye_to_food_relay_weight: float = 15.0    # 음식 감각 중계
    pain_to_danger_relay_weight: float = 18.0      # 위험 감각 중계
    wall_to_danger_relay_weight: float = 12.0      # 벽 감각 → 위험 중계

    # 내부 상태 → TRN (게이팅 조절)
    hunger_to_trn_weight: float = -12.0            # 배고픔 → TRN 억제 (Food 게이트 개방)
    fear_to_trn_weight: float = -15.0              # 공포 → TRN 억제 (Danger 게이트 개방)

    # TRN → Relay (억제성 게이팅)
    trn_to_food_relay_weight: float = -10.0        # TRN → Food Relay 억제
    trn_to_danger_relay_weight: float = -10.0      # TRN → Danger Relay 억제

    # Goal → Relay (주의 집중)
    goal_food_to_food_relay_weight: float = 12.0   # 음식 목표 → Food Relay 증폭
    goal_safety_to_danger_relay_weight: float = 12.0  # 안전 목표 → Danger Relay 증폭

    # Relay → 출력 (증폭된 감각)
    food_relay_to_motor_weight: float = 8.0        # Food Relay → Motor (간접)
    danger_relay_to_amygdala_weight: float = 15.0  # Danger Relay → Amygdala 증폭
    danger_relay_to_motor_weight: float = 10.0     # Danger Relay → Motor (회피)

    # 각성 조절
    low_energy_to_arousal_weight: float = 20.0     # 낮은 에너지 → 높은 각성
    high_energy_to_arousal_weight: float = -15.0   # 높은 에너지 → 낮은 각성
    arousal_to_motor_weight: float = 6.0           # 각성 → Motor 활성화
    arousal_to_relay_weight: float = 5.0           # 각성 → 감각 민감도

    # === Phase 6a 시냅스 가중치 ===
    # 입력 → Granule Cells
    motor_to_granule_weight: float = 12.0          # 운동 명령 복사 (efference copy)
    sensory_to_granule_weight: float = 10.0        # 감각 입력

    # Granule → Purkinje (Parallel Fibers, 학습 가능)
    granule_to_purkinje_weight: float = 5.0        # 초기 가중치
    granule_purkinje_eta: float = 0.05             # 학습률 (LTD)

    # Error → Purkinje (Climbing Fibers)
    error_to_purkinje_weight: float = 30.0         # 강한 오류 신호

    # Purkinje → Deep Nuclei (억제)
    purkinje_to_deep_weight: float = -15.0         # 억제성 출력

    # Deep Nuclei → Motor (조절)
    deep_to_motor_weight: float = 8.0              # 운동 조절 출력

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
        if self.basal_ganglia_enabled:
            base += (self.n_striatum + self.n_direct_pathway +
                     self.n_indirect_pathway + self.n_dopamine)
        if self.prefrontal_enabled:
            base += (self.n_working_memory + self.n_goal_food +
                     self.n_goal_safety + self.n_inhibitory_control)
        if self.cerebellum_enabled:
            base += (self.n_granule_cells + self.n_purkinje_cells +
                     self.n_deep_nuclei + self.n_error_signal)
        if self.thalamus_enabled:
            base += (self.n_food_relay + self.n_danger_relay +
                     self.n_trn + self.n_arousal)
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

            # Phase 3c: 방향성 Food Memory (좌/우 분리)
            if self.config.directional_food_memory:
                n_half = self.config.n_food_memory // 2
                self.food_memory_left = self.model.add_neuron_population(
                    "food_memory_left", n_half, "LIF", lif_params, lif_init)
                self.food_memory_right = self.model.add_neuron_population(
                    "food_memory_right", n_half, "LIF", lif_params, lif_init)
                # 호환성을 위한 참조 (단일 food_memory는 None)
                self.food_memory = None
            else:
                # Phase 3b: 단일 Food Memory
                self.food_memory = self.model.add_neuron_population(
                    "food_memory", self.config.n_food_memory, "LIF", lif_params, lif_init)
                self.food_memory_left = None
                self.food_memory_right = None

            # Place Cell 중심점 계산 (20x20 격자)
            self.place_cell_centers = []
            self.place_cell_left_indices = []   # Phase 3c: 좌측 Place Cells
            self.place_cell_right_indices = []  # Phase 3c: 우측 Place Cells
            grid = self.config.place_cell_grid_size
            for i in range(grid):
                for j in range(grid):
                    cx = (i + 0.5) / grid  # 0~1 정규화
                    cy = (j + 0.5) / grid
                    self.place_cell_centers.append((cx, cy))
                    idx = i * grid + j
                    if cx < 0.5:
                        self.place_cell_left_indices.append(idx)
                    else:
                        self.place_cell_right_indices.append(idx)

            print(f"  Hippocampus: PlaceCells({self.config.n_place_cells}) + "
                  f"FoodMemory({self.config.n_food_memory})")

        # === Phase 4: BASAL GANGLIA POPULATIONS ===
        if self.config.basal_ganglia_enabled:
            # Striatum: 감각 입력 통합, 행동 선택
            self.striatum = self.model.add_neuron_population(
                "striatum", self.config.n_striatum, "LIF", lif_params, lif_init)

            # Direct pathway (D1): Go 신호 - 행동 촉진
            self.direct_pathway = self.model.add_neuron_population(
                "direct_pathway", self.config.n_direct_pathway, "LIF", lif_params, lif_init)

            # Indirect pathway (D2): NoGo 신호 - 행동 억제
            self.indirect_pathway = self.model.add_neuron_population(
                "indirect_pathway", self.config.n_indirect_pathway, "LIF", lif_params, lif_init)

            # Dopamine neurons (VTA/SNc): 보상 신호
            self.dopamine_neurons = self.model.add_neuron_population(
                "dopamine_neurons", self.config.n_dopamine, sensory_lif_model, sensory_params, sensory_init)

            # Dopamine 레벨 추적
            self.dopamine_level = 0.0

            print(f"  BasalGanglia: Striatum({self.config.n_striatum}) + "
                  f"Direct({self.config.n_direct_pathway}) + "
                  f"Indirect({self.config.n_indirect_pathway}) + "
                  f"Dopamine({self.config.n_dopamine})")

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

        # Phase 4: Basal Ganglia circuits
        if self.config.basal_ganglia_enabled:
            self._build_basal_ganglia_circuit()

        # Phase 5: Prefrontal Cortex circuits
        if self.config.prefrontal_enabled:
            self._build_prefrontal_cortex_circuit()

        # Phase 6a: Cerebellum circuits
        if self.config.cerebellum_enabled:
            self._build_cerebellum_circuit()

        # Phase 6b: Thalamus circuits
        if self.config.thalamus_enabled:
            self._build_thalamus_circuit()

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
        """해마 회로: Place Cells → Food Memory → Motor (Phase 3c: 방향성 학습)"""

        # 학습 추적을 위한 초기화
        self.food_learning_enabled = True
        self.last_active_place_cells = np.zeros(self.config.n_place_cells)

        if self.config.directional_food_memory:
            # === Phase 3c: 방향성 Food Memory ===
            print("  Building Hippocampus circuit (Phase 3c - Directional)...")

            # 1. Place Cells Left → Food Memory Left (학습 가능)
            self.place_to_food_memory_left = self.model.add_synapse_population(
                "place_to_food_memory_left", "DENSE",
                self.place_cells, self.food_memory_left,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.place_to_food_memory_weight})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0})
            )

            # 2. Place Cells Right → Food Memory Right (학습 가능)
            self.place_to_food_memory_right = self.model.add_synapse_population(
                "place_to_food_memory_right", "DENSE",
                self.place_cells, self.food_memory_right,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.place_to_food_memory_weight})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0})
            )

            # 호환성 참조
            self.place_to_food_memory = None

            print(f"    PlaceCells→FoodMemory L/R: {self.config.place_to_food_memory_weight} (DIRECTIONAL, eta={self.config.place_to_food_memory_eta})")

            # 3. Food Memory Left → Motor Left (동측 배선)
            self.food_memory_left_to_motor = self.model.add_synapse_population(
                "food_memory_left_to_motor", "SPARSE",
                self.food_memory_left, self.motor_left,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.food_memory_to_motor_weight})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}),
                init_sparse_connectivity("FixedProbabilityNoAutapse", {"prob": 0.15})
            )

            # 4. Food Memory Right → Motor Right (동측 배선)
            self.food_memory_right_to_motor = self.model.add_synapse_population(
                "food_memory_right_to_motor", "SPARSE",
                self.food_memory_right, self.motor_right,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.food_memory_to_motor_weight})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}),
                init_sparse_connectivity("FixedProbabilityNoAutapse", {"prob": 0.15})
            )

            print(f"    FoodMemory L→Motor L, R→Motor R: {self.config.food_memory_to_motor_weight}")

            # 5. Hunger → Food Memory (양쪽 모두)
            self._create_static_synapse(
                "hunger_to_food_memory_left", self.hunger_drive, self.food_memory_left,
                self.config.hunger_to_food_memory_weight, sparsity=0.1)
            self._create_static_synapse(
                "hunger_to_food_memory_right", self.hunger_drive, self.food_memory_right,
                self.config.hunger_to_food_memory_weight, sparsity=0.1)

        else:
            # === Phase 3b: 단일 Food Memory (기존) ===
            print("  Building Hippocampus circuit (Phase 3b - Hebbian)...")

            self.place_to_food_memory = self.model.add_synapse_population(
                "place_to_food_memory", "DENSE",
                self.place_cells, self.food_memory,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.place_to_food_memory_weight})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0})
            )

            self.place_to_food_memory_left = None
            self.place_to_food_memory_right = None

            print(f"    PlaceCells→FoodMemory: {self.config.place_to_food_memory_weight} (LEARNABLE, eta={self.config.place_to_food_memory_eta})")

            # Food Memory → Motor (양쪽 동시)
            self.food_memory_left_to_motor = self.model.add_synapse_population(
                "food_memory_left_to_motor", "SPARSE",
                self.food_memory, self.motor_left,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.food_memory_to_motor_weight})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}),
                init_sparse_connectivity("FixedProbabilityNoAutapse", {"prob": 0.1})
            )

            self.food_memory_right_to_motor = self.model.add_synapse_population(
                "food_memory_right_to_motor", "SPARSE",
                self.food_memory, self.motor_right,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.food_memory_to_motor_weight})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}),
                init_sparse_connectivity("FixedProbabilityNoAutapse", {"prob": 0.1})
            )

            print(f"    FoodMemory→Motor: {self.config.food_memory_to_motor_weight}")

            # Hunger → Food Memory
            self._create_static_synapse(
                "hunger_to_food_memory", self.hunger_drive, self.food_memory,
                self.config.hunger_to_food_memory_weight, sparsity=0.1)

        print(f"    Hunger→FoodMemory: {self.config.hunger_to_food_memory_weight} (amplify when hungry)")

    # === Phase 4: Basal Ganglia Circuits ===

    def _build_basal_ganglia_circuit(self):
        """
        기저핵 회로: 행동 선택 및 습관 학습

        구조:
        - Sensory → Striatum: 감각 입력 통합
        - Hunger → Striatum: 동기 상태 전달
        - Striatum → Direct (D1): Go 신호
        - Striatum → Indirect (D2): NoGo 신호
        - D1 ↔ D2: 상호 억제 (경쟁)
        - Direct → Motor: 행동 촉진
        - Indirect → Motor: 행동 억제
        - Dopamine → D1/D2: 보상 시 학습 조절
        """
        print("  Building Basal Ganglia circuit (Phase 4)...")

        # 1. Sensory → Striatum (감각 입력 통합)
        # Food Eye → Striatum (음식 정보)
        self._create_static_synapse(
            "food_eye_left_to_striatum", self.food_eye_left, self.striatum,
            self.config.sensory_to_striatum_weight, sparsity=0.08)
        self._create_static_synapse(
            "food_eye_right_to_striatum", self.food_eye_right, self.striatum,
            self.config.sensory_to_striatum_weight, sparsity=0.08)

        print(f"    FoodEye→Striatum: {self.config.sensory_to_striatum_weight}")

        # 2. Hunger → Striatum (동기 상태)
        self._create_static_synapse(
            "hunger_to_striatum", self.hunger_drive, self.striatum,
            self.config.hunger_to_striatum_weight, sparsity=0.1)

        print(f"    Hunger→Striatum: {self.config.hunger_to_striatum_weight}")

        # 3. Striatum → Direct/Indirect pathways
        # Striatum → Direct (Go)
        self.striatum_to_direct = self.model.add_synapse_population(
            "striatum_to_direct", "DENSE",
            self.striatum, self.direct_pathway,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.striatum_to_direct_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0})
        )

        # Striatum → Indirect (NoGo)
        self.striatum_to_indirect = self.model.add_synapse_population(
            "striatum_to_indirect", "DENSE",
            self.striatum, self.indirect_pathway,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.striatum_to_indirect_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0})
        )

        print(f"    Striatum→Direct: {self.config.striatum_to_direct_weight} (Go)")
        print(f"    Striatum→Indirect: {self.config.striatum_to_indirect_weight} (NoGo)")

        # 4. Direct ↔ Indirect 상호 억제 (경쟁)
        self._create_static_synapse(
            "direct_to_indirect", self.direct_pathway, self.indirect_pathway,
            self.config.direct_indirect_competition, sparsity=0.1)
        self._create_static_synapse(
            "indirect_to_direct", self.indirect_pathway, self.direct_pathway,
            self.config.direct_indirect_competition, sparsity=0.1)

        print(f"    Direct↔Indirect: {self.config.direct_indirect_competition} (competition)")

        # 5. Direct/Indirect → Motor
        # Direct → Motor (양쪽 모두 촉진)
        self._create_static_synapse(
            "direct_to_motor_left", self.direct_pathway, self.motor_left,
            self.config.direct_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "direct_to_motor_right", self.direct_pathway, self.motor_right,
            self.config.direct_to_motor_weight, sparsity=0.1)

        # Indirect → Motor (양쪽 모두 억제)
        self._create_static_synapse(
            "indirect_to_motor_left", self.indirect_pathway, self.motor_left,
            self.config.indirect_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "indirect_to_motor_right", self.indirect_pathway, self.motor_right,
            self.config.indirect_to_motor_weight, sparsity=0.1)

        print(f"    Direct→Motor: {self.config.direct_to_motor_weight} (Go)")
        print(f"    Indirect→Motor: {self.config.indirect_to_motor_weight} (NoGo)")

        # 6. Dopamine → Direct/Indirect (보상 조절)
        # Dopamine → Direct (강화): 보상 시 Go 신호 증가
        self._create_static_synapse(
            "dopamine_to_direct", self.dopamine_neurons, self.direct_pathway,
            self.config.dopamine_to_direct_weight, sparsity=0.15)

        # Dopamine → Indirect (억제): 보상 시 NoGo 신호 감소
        self._create_static_synapse(
            "dopamine_to_indirect", self.dopamine_neurons, self.indirect_pathway,
            self.config.dopamine_to_indirect_weight, sparsity=0.15)

        print(f"    Dopamine→Direct: {self.config.dopamine_to_direct_weight} (reward)")
        print(f"    Dopamine→Indirect: {self.config.dopamine_to_indirect_weight} (reward)")

    def _build_prefrontal_cortex_circuit(self):
        """
        Phase 5: Prefrontal Cortex (전전두엽) 구축

        구성:
        1. Working Memory - 지속 활성화로 정보 유지
        2. Goal Units (Food/Safety) - 목표 표상 및 경쟁
        3. Inhibitory Control - 충동 억제

        연결:
        - 입력: Hippocampus, Amygdala, Hypothalamus → Working Memory
        - 내부: Working Memory → Goal, Goal 간 WTA
        - 출력: Goal → Motor, Inhibitory → Basal Ganglia
        """
        print("  Phase 5: Building PFC (Prefrontal Cortex)...")

        # LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. Working Memory (작업 기억) ===
        self.working_memory = self.model.add_neuron_population(
            "working_memory", self.config.n_working_memory,
            sensory_lif_model, lif_params, lif_init)

        # === 2. Goal Units (목표 단위) ===
        self.goal_food = self.model.add_neuron_population(
            "goal_food", self.config.n_goal_food,
            sensory_lif_model, lif_params, lif_init)

        self.goal_safety = self.model.add_neuron_population(
            "goal_safety", self.config.n_goal_safety,
            sensory_lif_model, lif_params, lif_init)

        # === 3. Inhibitory Control (억제 제어) ===
        self.inhibitory_control = self.model.add_neuron_population(
            "inhibitory_control", self.config.n_inhibitory_control,
            sensory_lif_model, lif_params, lif_init)

        print(f"    Working Memory: {self.config.n_working_memory} neurons")
        print(f"    Goal Food: {self.config.n_goal_food} neurons")
        print(f"    Goal Safety: {self.config.n_goal_safety} neurons")
        print(f"    Inhibitory Control: {self.config.n_inhibitory_control} neurons")

        # === 입력 연결: 다른 영역 → Working Memory ===
        # Hippocampus → Working Memory (공간 정보)
        if self.config.hippocampus_enabled:
            self._create_static_synapse(
                "place_to_working_memory", self.place_cells, self.working_memory,
                self.config.place_to_working_memory_weight, sparsity=0.05)

            # Food Memory → Working Memory (음식 위치 기억)
            if self.config.directional_food_memory:
                self._create_static_synapse(
                    "food_memory_left_to_wm", self.food_memory_left, self.working_memory,
                    self.config.food_memory_to_working_memory_weight, sparsity=0.08)
                self._create_static_synapse(
                    "food_memory_right_to_wm", self.food_memory_right, self.working_memory,
                    self.config.food_memory_to_working_memory_weight, sparsity=0.08)
            elif hasattr(self, 'food_memory') and self.food_memory is not None:
                self._create_static_synapse(
                    "food_memory_to_wm", self.food_memory, self.working_memory,
                    self.config.food_memory_to_working_memory_weight, sparsity=0.08)

        # Amygdala → Working Memory (위험 기억)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_working_memory", self.fear_response, self.working_memory,
                self.config.fear_to_working_memory_weight, sparsity=0.08)

        print(f"    Input→WM: place={self.config.place_to_working_memory_weight}, "
              f"food_mem={self.config.food_memory_to_working_memory_weight}, "
              f"fear={self.config.fear_to_working_memory_weight}")

        # === Working Memory 재귀 연결 (지속 활성화) ===
        self.working_memory_recurrent = self.model.add_synapse_population(
            "wm_recurrent", "SPARSE", self.working_memory, self.working_memory,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.working_memory_recurrent_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 10.0}),  # 느린 감쇠
            init_sparse_connectivity("FixedProbability", {"prob": 0.1})
        )
        print(f"    WM Recurrent: {self.config.working_memory_recurrent_weight} (persistent activity)")

        # === Working Memory → Goal Units ===
        self._create_static_synapse(
            "wm_to_goal_food", self.working_memory, self.goal_food,
            self.config.working_memory_to_goal_weight, sparsity=0.1)
        self._create_static_synapse(
            "wm_to_goal_safety", self.working_memory, self.goal_safety,
            self.config.working_memory_to_goal_weight, sparsity=0.1)

        # === 내부 상태 → Goal Units ===
        # Hunger → Goal_Food (배고프면 음식 목표 활성화)
        self._create_static_synapse(
            "hunger_to_goal_food", self.hunger_drive, self.goal_food,
            self.config.hunger_to_goal_food_weight, sparsity=0.1)

        # Fear → Goal_Safety (공포 시 안전 목표 활성화)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_goal_safety", self.fear_response, self.goal_safety,
                self.config.fear_to_goal_safety_weight, sparsity=0.1)

        print(f"    Hunger→Goal_Food: {self.config.hunger_to_goal_food_weight}")
        print(f"    Fear→Goal_Safety: {self.config.fear_to_goal_safety_weight}")

        # === Goal Unit WTA 경쟁 ===
        self._create_static_synapse(
            "goal_food_to_safety", self.goal_food, self.goal_safety,
            self.config.goal_wta_weight, sparsity=0.15)
        self._create_static_synapse(
            "goal_safety_to_food", self.goal_safety, self.goal_food,
            self.config.goal_wta_weight, sparsity=0.15)

        print(f"    Goal WTA: {self.config.goal_wta_weight} (competition)")

        # === Goal_Safety → Inhibitory Control ===
        self._create_static_synapse(
            "goal_safety_to_inhibitory", self.goal_safety, self.inhibitory_control,
            self.config.goal_safety_to_inhibitory_weight, sparsity=0.15)

        # === Inhibitory Control → Basal Ganglia Direct (억제) ===
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "inhibitory_to_direct", self.inhibitory_control, self.direct_pathway,
                self.config.inhibitory_to_direct_weight, sparsity=0.1)
            print(f"    Inhibitory→Direct: {self.config.inhibitory_to_direct_weight} (suppress impulsive Go)")

        # === Inhibitory Control → Motor (직접 억제) ===
        self._create_static_synapse(
            "inhibitory_to_motor_left", self.inhibitory_control, self.motor_left,
            self.config.inhibitory_to_motor_weight, sparsity=0.08)
        self._create_static_synapse(
            "inhibitory_to_motor_right", self.inhibitory_control, self.motor_right,
            self.config.inhibitory_to_motor_weight, sparsity=0.08)

        print(f"    Inhibitory→Motor: {self.config.inhibitory_to_motor_weight}")

        # === Goal_Food → Motor (목표 지향 행동) ===
        # 음식 목표가 활성화되면 Motor 활성화 (탐색 촉진)
        self._create_static_synapse(
            "goal_food_to_motor_left", self.goal_food, self.motor_left,
            self.config.goal_food_to_motor_weight, sparsity=0.08)
        self._create_static_synapse(
            "goal_food_to_motor_right", self.goal_food, self.motor_right,
            self.config.goal_food_to_motor_weight, sparsity=0.08)

        print(f"    Goal_Food→Motor: {self.config.goal_food_to_motor_weight} (goal-directed)")

        print(f"  PFC circuit complete: {self.config.n_working_memory + self.config.n_goal_food + self.config.n_goal_safety + self.config.n_inhibitory_control} neurons")

    def _build_cerebellum_circuit(self):
        """
        Phase 6a: Cerebellum (소뇌) 구축

        구성:
        1. Granule Cells - 입력 통합, 희소 표현
        2. Purkinje Cells - 운동 조절, 오류 학습
        3. Deep Nuclei - 최종 운동 출력
        4. Error Signal - 오류 감지 (Climbing Fiber)

        연결:
        - 입력: Motor, Sensory → Granule Cells
        - 내부: Granule → Purkinje → Deep Nuclei
        - 출력: Deep Nuclei → Motor (조절)
        - 오류: Error → Purkinje (학습 신호)
        """
        print("  Phase 6a: Building Cerebellum...")

        # LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. Granule Cells (과립세포) ===
        self.granule_cells = self.model.add_neuron_population(
            "granule_cells", self.config.n_granule_cells,
            sensory_lif_model, lif_params, lif_init)

        # === 2. Purkinje Cells (푸르키네세포) ===
        self.purkinje_cells = self.model.add_neuron_population(
            "purkinje_cells", self.config.n_purkinje_cells,
            sensory_lif_model, lif_params, lif_init)

        # === 3. Deep Cerebellar Nuclei (심부핵) ===
        self.deep_nuclei = self.model.add_neuron_population(
            "deep_nuclei", self.config.n_deep_nuclei,
            sensory_lif_model, lif_params, lif_init)

        # === 4. Error Signal (오류 신호 - Climbing Fiber 역할) ===
        self.error_signal = self.model.add_neuron_population(
            "error_signal", self.config.n_error_signal,
            sensory_lif_model, lif_params, lif_init)

        print(f"    Granule Cells: {self.config.n_granule_cells} neurons")
        print(f"    Purkinje Cells: {self.config.n_purkinje_cells} neurons")
        print(f"    Deep Nuclei: {self.config.n_deep_nuclei} neurons")
        print(f"    Error Signal: {self.config.n_error_signal} neurons")

        # === 입력 연결: Motor/Sensory → Granule Cells ===
        # Motor efference copy (운동 명령 복사)
        self._create_static_synapse(
            "motor_left_to_granule", self.motor_left, self.granule_cells,
            self.config.motor_to_granule_weight, sparsity=0.1)
        self._create_static_synapse(
            "motor_right_to_granule", self.motor_right, self.granule_cells,
            self.config.motor_to_granule_weight, sparsity=0.1)

        # Sensory → Granule (현재 감각 상태)
        self._create_static_synapse(
            "food_eye_left_to_granule", self.food_eye_left, self.granule_cells,
            self.config.sensory_to_granule_weight, sparsity=0.08)
        self._create_static_synapse(
            "food_eye_right_to_granule", self.food_eye_right, self.granule_cells,
            self.config.sensory_to_granule_weight, sparsity=0.08)
        self._create_static_synapse(
            "wall_eye_left_to_granule", self.wall_eye_left, self.granule_cells,
            self.config.sensory_to_granule_weight, sparsity=0.08)
        self._create_static_synapse(
            "wall_eye_right_to_granule", self.wall_eye_right, self.granule_cells,
            self.config.sensory_to_granule_weight, sparsity=0.08)

        print(f"    Motor→Granule: {self.config.motor_to_granule_weight} (efference copy)")
        print(f"    Sensory→Granule: {self.config.sensory_to_granule_weight}")

        # === Granule → Purkinje (Parallel Fibers) ===
        # DENSE 연결로 학습 가능하게 설정
        self.granule_to_purkinje = self.model.add_synapse_population(
            "granule_to_purkinje", "DENSE",
            self.granule_cells, self.purkinje_cells,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.granule_to_purkinje_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0})
        )

        print(f"    Granule→Purkinje: {self.config.granule_to_purkinje_weight} (parallel fibers)")

        # === Error Signal → Purkinje (Climbing Fibers) ===
        self._create_static_synapse(
            "error_to_purkinje", self.error_signal, self.purkinje_cells,
            self.config.error_to_purkinje_weight, sparsity=0.2)

        print(f"    Error→Purkinje: {self.config.error_to_purkinje_weight} (climbing fibers)")

        # === Purkinje → Deep Nuclei (억제) ===
        self._create_static_synapse(
            "purkinje_to_deep", self.purkinje_cells, self.deep_nuclei,
            self.config.purkinje_to_deep_weight, sparsity=0.15)

        print(f"    Purkinje→Deep: {self.config.purkinje_to_deep_weight} (inhibitory)")

        # === Deep Nuclei 기저 활성화 ===
        # Purkinje 억제가 없을 때 기본 활성화 (tonic activity)
        # Granule에서 직접 흥분 입력도 받음
        self._create_static_synapse(
            "granule_to_deep", self.granule_cells, self.deep_nuclei,
            8.0, sparsity=0.1)  # 기저 흥분

        # === Deep Nuclei → Motor (운동 조절) ===
        self._create_static_synapse(
            "deep_to_motor_left", self.deep_nuclei, self.motor_left,
            self.config.deep_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "deep_to_motor_right", self.deep_nuclei, self.motor_right,
            self.config.deep_to_motor_weight, sparsity=0.1)

        print(f"    Deep→Motor: {self.config.deep_to_motor_weight} (motor modulation)")

        # === Pain/Wall → Error Signal ===
        # 오류 발생 시 (Pain Zone 진입, 벽 근처) Error Signal 활성화
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "pain_to_error", self.pain_eye_left, self.error_signal,
                25.0, sparsity=0.15)
            self._create_static_synapse(
                "pain_to_error_r", self.pain_eye_right, self.error_signal,
                25.0, sparsity=0.15)
            self._create_static_synapse(
                "fear_to_error", self.fear_response, self.error_signal,
                20.0, sparsity=0.1)

        # Wall → Error (벽에 가까우면 오류)
        self._create_static_synapse(
            "wall_to_error_l", self.wall_eye_left, self.error_signal,
            15.0, sparsity=0.1)
        self._create_static_synapse(
            "wall_to_error_r", self.wall_eye_right, self.error_signal,
            15.0, sparsity=0.1)

        print(f"    Pain/Wall→Error: error signal triggers")

        total_neurons = (self.config.n_granule_cells + self.config.n_purkinje_cells +
                        self.config.n_deep_nuclei + self.config.n_error_signal)
        print(f"  Cerebellum circuit complete: {total_neurons} neurons")

    def _build_thalamus_circuit(self):
        """
        Phase 6b: Thalamus (시상) 구축

        구성:
        1. Food Relay - 음식 감각 중계
        2. Danger Relay - 위험 감각 중계
        3. TRN (Thalamic Reticular Nucleus) - 억제성 게이팅
        4. Arousal - 각성 수준 조절

        연결:
        - 입력: Food Eye → Food Relay, Pain/Wall → Danger Relay
        - 게이팅: Hunger/Fear → TRN → Relay (선택적 통과)
        - 주의: Goal → Relay (목표 관련 증폭)
        - 출력: Relay → Motor/Amygdala (증폭된 신호)
        - 각성: Energy → Arousal → Motor/Relay
        """
        print("  Phase 6b: Building Thalamus...")

        # LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. Food Relay (음식 감각 중계) ===
        self.food_relay = self.model.add_neuron_population(
            "food_relay", self.config.n_food_relay,
            sensory_lif_model, lif_params, lif_init)

        # === 2. Danger Relay (위험 감각 중계) ===
        self.danger_relay = self.model.add_neuron_population(
            "danger_relay", self.config.n_danger_relay,
            sensory_lif_model, lif_params, lif_init)

        # === 3. TRN (Thalamic Reticular Nucleus - 억제성 게이팅) ===
        self.trn = self.model.add_neuron_population(
            "trn", self.config.n_trn,
            sensory_lif_model, lif_params, lif_init)

        # === 4. Arousal (각성 수준 조절) ===
        self.arousal = self.model.add_neuron_population(
            "arousal", self.config.n_arousal,
            sensory_lif_model, lif_params, lif_init)

        print(f"    Food Relay: {self.config.n_food_relay} neurons")
        print(f"    Danger Relay: {self.config.n_danger_relay} neurons")
        print(f"    TRN: {self.config.n_trn} neurons")
        print(f"    Arousal: {self.config.n_arousal} neurons")

        # === 감각 → Relay 연결 ===
        # Food Eye → Food Relay
        self._create_static_synapse(
            "food_eye_left_to_food_relay", self.food_eye_left, self.food_relay,
            self.config.food_eye_to_food_relay_weight, sparsity=0.15)
        self._create_static_synapse(
            "food_eye_right_to_food_relay", self.food_eye_right, self.food_relay,
            self.config.food_eye_to_food_relay_weight, sparsity=0.15)

        print(f"    Food Eye→Food Relay: {self.config.food_eye_to_food_relay_weight}")

        # Pain/Wall → Danger Relay
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "pain_left_to_danger_relay", self.pain_eye_left, self.danger_relay,
                self.config.pain_to_danger_relay_weight, sparsity=0.15)
            self._create_static_synapse(
                "pain_right_to_danger_relay", self.pain_eye_right, self.danger_relay,
                self.config.pain_to_danger_relay_weight, sparsity=0.15)

        self._create_static_synapse(
            "wall_left_to_danger_relay", self.wall_eye_left, self.danger_relay,
            self.config.wall_to_danger_relay_weight, sparsity=0.12)
        self._create_static_synapse(
            "wall_right_to_danger_relay", self.wall_eye_right, self.danger_relay,
            self.config.wall_to_danger_relay_weight, sparsity=0.12)

        print(f"    Pain/Wall→Danger Relay: {self.config.pain_to_danger_relay_weight}/{self.config.wall_to_danger_relay_weight}")

        # === 내부 상태 → TRN (게이팅 조절) ===
        # Hunger → TRN 억제 (배고프면 Food 게이트 개방)
        self._create_static_synapse(
            "hunger_to_trn", self.hunger_drive, self.trn,
            self.config.hunger_to_trn_weight, sparsity=0.1)

        # Fear → TRN 억제 (공포 시 Danger 게이트 개방)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_trn", self.fear_response, self.trn,
                self.config.fear_to_trn_weight, sparsity=0.1)

        print(f"    Hunger/Fear→TRN: {self.config.hunger_to_trn_weight}/{self.config.fear_to_trn_weight} (gate control)")

        # === TRN → Relay (억제성 게이팅) ===
        self._create_static_synapse(
            "trn_to_food_relay", self.trn, self.food_relay,
            self.config.trn_to_food_relay_weight, sparsity=0.15)
        self._create_static_synapse(
            "trn_to_danger_relay", self.trn, self.danger_relay,
            self.config.trn_to_danger_relay_weight, sparsity=0.15)

        print(f"    TRN→Relay: {self.config.trn_to_food_relay_weight} (inhibitory gating)")

        # === Goal → Relay (주의 집중) ===
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "goal_food_to_food_relay", self.goal_food, self.food_relay,
                self.config.goal_food_to_food_relay_weight, sparsity=0.15)
            self._create_static_synapse(
                "goal_safety_to_danger_relay", self.goal_safety, self.danger_relay,
                self.config.goal_safety_to_danger_relay_weight, sparsity=0.15)

            print(f"    Goal→Relay: {self.config.goal_food_to_food_relay_weight} (attention)")

        # === Relay → 출력 (증폭된 감각) ===
        # Food Relay → Motor (음식 방향 편향)
        self._create_static_synapse(
            "food_relay_to_motor_left", self.food_relay, self.motor_left,
            self.config.food_relay_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "food_relay_to_motor_right", self.food_relay, self.motor_right,
            self.config.food_relay_to_motor_weight, sparsity=0.1)

        # Danger Relay → Amygdala (위험 신호 증폭)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "danger_relay_to_la", self.danger_relay, self.lateral_amygdala,
                self.config.danger_relay_to_amygdala_weight, sparsity=0.12)

        # Danger Relay → Motor (직접 회피 촉진)
        self._create_static_synapse(
            "danger_relay_to_motor_left", self.danger_relay, self.motor_left,
            self.config.danger_relay_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "danger_relay_to_motor_right", self.danger_relay, self.motor_right,
            self.config.danger_relay_to_motor_weight, sparsity=0.1)

        print(f"    Relay→Output: Food→Motor {self.config.food_relay_to_motor_weight}, Danger→Amyg {self.config.danger_relay_to_amygdala_weight}")

        # === 각성 조절 ===
        # Low Energy → Arousal (배고프면 각성 상승)
        self._create_static_synapse(
            "low_energy_to_arousal", self.low_energy_sensor, self.arousal,
            self.config.low_energy_to_arousal_weight, sparsity=0.15)

        # High Energy → Arousal (배부르면 각성 감소)
        self._create_static_synapse(
            "high_energy_to_arousal", self.high_energy_sensor, self.arousal,
            self.config.high_energy_to_arousal_weight, sparsity=0.15)

        print(f"    Energy→Arousal: Low {self.config.low_energy_to_arousal_weight}, High {self.config.high_energy_to_arousal_weight}")

        # Arousal → Motor (전체 활동 수준)
        self._create_static_synapse(
            "arousal_to_motor_left", self.arousal, self.motor_left,
            self.config.arousal_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "arousal_to_motor_right", self.arousal, self.motor_right,
            self.config.arousal_to_motor_weight, sparsity=0.1)

        # Arousal → Relay (감각 민감도)
        self._create_static_synapse(
            "arousal_to_food_relay", self.arousal, self.food_relay,
            self.config.arousal_to_relay_weight, sparsity=0.1)
        self._create_static_synapse(
            "arousal_to_danger_relay", self.arousal, self.danger_relay,
            self.config.arousal_to_relay_weight, sparsity=0.1)

        print(f"    Arousal→Motor/Relay: {self.config.arousal_to_motor_weight}/{self.config.arousal_to_relay_weight}")

        total_neurons = (self.config.n_food_relay + self.config.n_danger_relay +
                        self.config.n_trn + self.config.n_arousal)
        print(f"  Thalamus circuit complete: {total_neurons} neurons")

    def trigger_error_signal(self, error_type: str = "general", intensity: float = 1.0):
        """
        Phase 6a: 오류 발생 시 Error Signal 활성화

        Args:
            error_type: 오류 유형 ('wall', 'pain', 'collision')
            intensity: 오류 강도 (0~1)
        """
        if not self.config.cerebellum_enabled:
            return

        # Error Signal 뉴런에 입력 전류 주입
        error_current = intensity * 60.0
        self.error_signal.vars["I_input"].view[:] = error_current
        self.error_signal.vars["I_input"].push_to_device()

        return {"error_type": error_type, "intensity": intensity}

    def release_dopamine(self, reward_magnitude: float = 1.0):
        """
        Phase 4: 보상 시 Dopamine 방출

        음식 섭취 등 보상 이벤트 발생 시 호출하여
        Dopamine 뉴런을 활성화하고 학습을 촉진합니다.

        Args:
            reward_magnitude: 보상 크기 (0~1)
        """
        if not self.config.basal_ganglia_enabled:
            return

        # Dopamine 레벨 업데이트
        self.dopamine_level = min(1.0, self.dopamine_level + reward_magnitude)

        # Dopamine 뉴런에 입력 전류 주입
        dopamine_current = self.dopamine_level * 80.0  # 강한 활성화
        self.dopamine_neurons.vars["I_input"].view[:] = dopamine_current
        self.dopamine_neurons.vars["I_input"].push_to_device()

        return {"dopamine_level": self.dopamine_level}

    def decay_dopamine(self):
        """Dopamine 레벨 감쇠"""
        if not self.config.basal_ganglia_enabled:
            return

        self.dopamine_level *= self.config.dopamine_decay

        # 감쇠된 레벨 반영
        if self.dopamine_level < 0.01:
            self.dopamine_level = 0.0
            self.dopamine_neurons.vars["I_input"].view[:] = 0.0
        else:
            self.dopamine_neurons.vars["I_input"].view[:] = self.dopamine_level * 80.0
        self.dopamine_neurons.vars["I_input"].push_to_device()

    def learn_food_location(self, food_position: tuple = None):
        """
        Phase 3b/3c: 음식 발견 시 Hebbian 학습

        음식을 먹었을 때 호출되어 현재 활성화된 Place Cells와
        Food Memory 사이의 연결을 강화합니다.

        Phase 3c (directional): 음식 위치에 따라 좌/우 Food Memory 선택적 학습

        Args:
            food_position: (x, y) 정규화된 음식 위치 (Phase 3c용)

        Δw = η * pre_activity
        """
        if not self.config.hippocampus_enabled or not self.food_learning_enabled:
            return

        active_cells = self.last_active_place_cells
        eta = self.config.place_to_food_memory_eta
        w_max = self.config.place_to_food_memory_w_max
        n_pre = self.config.n_place_cells

        if self.config.directional_food_memory:
            # === Phase 3c: 방향성 학습 ===
            # 음식 위치에 따라 좌/우 Food Memory 선택적 강화
            n_post = self.config.n_food_memory // 2

            # 음식이 왼쪽에 있으면 Food Memory Left 강화
            # 음식이 오른쪽에 있으면 Food Memory Right 강화
            food_x = food_position[0] if food_position else 0.5

            if food_x < 0.5:
                # 좌측 학습: 좌측 Place Cells → Food Memory Left
                self.place_to_food_memory_left.vars["g"].pull_from_device()
                weights = self.place_to_food_memory_left.vars["g"].view.copy()
                weights = weights.reshape(n_pre, n_post)

                n_strengthened = 0
                for i in self.place_cell_left_indices:
                    if active_cells[i] > 0.1:
                        delta_w = eta * active_cells[i]
                        weights[i, :] += delta_w
                        weights[i, :] = np.clip(weights[i, :], 0.0, w_max)
                        n_strengthened += 1

                self.place_to_food_memory_left.vars["g"].view[:] = weights.flatten()
                self.place_to_food_memory_left.vars["g"].push_to_device()
                side = "LEFT"
            else:
                # 우측 학습: 우측 Place Cells → Food Memory Right
                self.place_to_food_memory_right.vars["g"].pull_from_device()
                weights = self.place_to_food_memory_right.vars["g"].view.copy()
                weights = weights.reshape(n_pre, n_post)

                n_strengthened = 0
                for i in self.place_cell_right_indices:
                    if active_cells[i] > 0.1:
                        delta_w = eta * active_cells[i]
                        weights[i, :] += delta_w
                        weights[i, :] = np.clip(weights[i, :], 0.0, w_max)
                        n_strengthened += 1

                self.place_to_food_memory_right.vars["g"].view[:] = weights.flatten()
                self.place_to_food_memory_right.vars["g"].push_to_device()
                side = "RIGHT"

            return {
                "n_strengthened": n_strengthened,
                "avg_weight": float(np.mean(weights)),
                "max_weight": float(np.max(weights)),
                "side": side
            }

        else:
            # === Phase 3b: 단일 Food Memory (기존) ===
            n_post = self.config.n_food_memory

            self.place_to_food_memory.vars["g"].pull_from_device()
            weights = self.place_to_food_memory.vars["g"].view.copy()
            weights = weights.reshape(n_pre, n_post)

            for i in range(n_pre):
                if active_cells[i] > 0.1:
                    delta_w = eta * active_cells[i]
                    weights[i, :] += delta_w
                    weights[i, :] = np.clip(weights[i, :], 0.0, w_max)

            self.place_to_food_memory.vars["g"].view[:] = weights.flatten()
            self.place_to_food_memory.vars["g"].push_to_device()

            n_strengthened = np.sum(active_cells > 0.1)

            return {
                "n_strengthened": int(n_strengthened),
                "avg_weight": float(np.mean(weights)),
                "max_weight": float(np.max(weights))
            }

    def save_hippocampus_weights(self, filepath: str = None) -> str:
        """
        Phase 3b/3c: Hippocampus 가중치 저장

        학습된 Place Cells → Food Memory 가중치를 파일에 저장합니다.
        에피소드 간 학습 지속을 위해 사용됩니다.

        Args:
            filepath: 저장 경로 (None이면 기본 체크포인트 경로 사용)

        Returns:
            저장된 파일 경로
        """
        if not self.config.hippocampus_enabled or not self.food_learning_enabled:
            return None

        if filepath is None:
            filepath = str(CHECKPOINT_DIR / "hippocampus_weights.npy")

        if self.config.directional_food_memory:
            # Phase 3c: 좌/우 가중치 모두 저장
            self.place_to_food_memory_left.vars["g"].pull_from_device()
            self.place_to_food_memory_right.vars["g"].pull_from_device()
            weights_left = self.place_to_food_memory_left.vars["g"].view.copy()
            weights_right = self.place_to_food_memory_right.vars["g"].view.copy()
            np.savez(filepath.replace('.npy', '.npz'),
                     left=weights_left, right=weights_right)
            return filepath.replace('.npy', '.npz')
        else:
            # Phase 3b: 단일 가중치 저장
            self.place_to_food_memory.vars["g"].pull_from_device()
            weights = self.place_to_food_memory.vars["g"].view.copy()
            np.save(filepath, weights)
            return filepath

    def load_hippocampus_weights(self, filepath: str = None) -> bool:
        """
        Phase 3b/3c: Hippocampus 가중치 복원

        저장된 Place Cells → Food Memory 가중치를 파일에서 복원합니다.

        Args:
            filepath: 로드 경로 (None이면 기본 체크포인트 경로 사용)

        Returns:
            성공 여부
        """
        if not self.config.hippocampus_enabled or not self.food_learning_enabled:
            return False

        if filepath is None:
            filepath = str(CHECKPOINT_DIR / "hippocampus_weights.npy")

        if self.config.directional_food_memory:
            # Phase 3c: 좌/우 가중치 로드
            npz_path = filepath.replace('.npy', '.npz')
            if not os.path.exists(npz_path):
                return False
            data = np.load(npz_path)
            self.place_to_food_memory_left.vars["g"].view[:] = data['left']
            self.place_to_food_memory_left.vars["g"].push_to_device()
            self.place_to_food_memory_right.vars["g"].view[:] = data['right']
            self.place_to_food_memory_right.vars["g"].push_to_device()
            return True
        else:
            # Phase 3b: 단일 가중치 로드
            if not os.path.exists(filepath):
                return False
            weights = np.load(filepath)
            self.place_to_food_memory.vars["g"].view[:] = weights
            self.place_to_food_memory.vars["g"].push_to_device()
            return True

    def get_hippocampus_stats(self) -> dict:
        """
        Phase 3b/3c: Hippocampus 학습 상태 통계

        Returns:
            가중치 통계 (avg, max, min, n_strong)
        """
        if not self.config.hippocampus_enabled or not self.food_learning_enabled:
            return None

        if self.config.directional_food_memory:
            # Phase 3c: 좌/우 통계
            self.place_to_food_memory_left.vars["g"].pull_from_device()
            self.place_to_food_memory_right.vars["g"].pull_from_device()
            weights_left = self.place_to_food_memory_left.vars["g"].view.copy()
            weights_right = self.place_to_food_memory_right.vars["g"].view.copy()
            weights = np.concatenate([weights_left, weights_right])

            n_strong_left = np.sum(weights_left > self.config.place_to_food_memory_weight + 0.5)
            n_strong_right = np.sum(weights_right > self.config.place_to_food_memory_weight + 0.5)

            return {
                "avg_weight": float(np.mean(weights)),
                "max_weight": float(np.max(weights)),
                "min_weight": float(np.min(weights)),
                "n_strong_connections": int(n_strong_left + n_strong_right),
                "n_strong_left": int(n_strong_left),
                "n_strong_right": int(n_strong_right)
            }
        else:
            # Phase 3b: 단일 통계
            self.place_to_food_memory.vars["g"].pull_from_device()
            weights = self.place_to_food_memory.vars["g"].view.copy()

            n_strong = np.sum(weights > self.config.place_to_food_memory_weight + 0.5)

            return {
                "avg_weight": float(np.mean(weights)),
                "max_weight": float(np.max(weights)),
                "min_weight": float(np.min(weights)),
                "n_strong_connections": int(n_strong)
            }

    def _compute_place_cell_input(self, pos_x: float, pos_y: float) -> np.ndarray:
        """
        위치를 Place Cell 입력 전류로 변환

        Args:
            pos_x, pos_y: 정규화된 위치 (0~1)

        Returns:
            Place Cell 입력 전류 배열
        """
        currents = np.zeros(self.config.n_place_cells)
        activations = np.zeros(self.config.n_place_cells)
        sigma = self.config.place_cell_sigma

        for i, (cx, cy) in enumerate(self.place_cell_centers):
            # 가우시안 활성화
            dist_sq = (pos_x - cx)**2 + (pos_y - cy)**2
            activation = np.exp(-dist_sq / (2 * sigma**2))
            activations[i] = activation
            currents[i] = activation * 50.0  # 최대 전류 50

        # Phase 3b: 학습을 위해 활성화 패턴 저장
        self.last_active_place_cells = activations

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

        # Phase 4 스파이크 카운트
        striatum_spikes = 0
        direct_spikes = 0
        indirect_spikes = 0
        dopamine_spikes = 0

        # Phase 5 스파이크 카운트
        working_memory_spikes = 0
        goal_food_spikes = 0
        goal_safety_spikes = 0
        inhibitory_spikes = 0

        # Phase 6a 스파이크 카운트
        granule_spikes = 0
        purkinje_spikes = 0
        deep_nuclei_spikes = 0
        error_spikes = 0

        # Phase 6b 스파이크 카운트
        food_relay_spikes = 0
        danger_relay_spikes = 0
        trn_spikes = 0
        arousal_spikes = 0

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
                place_cell_spikes += np.sum(self.place_cells.vars["RefracTime"].view > self.spike_threshold)

                if self.config.directional_food_memory:
                    self.food_memory_left.vars["RefracTime"].pull_from_device()
                    self.food_memory_right.vars["RefracTime"].pull_from_device()
                    food_memory_spikes += np.sum(self.food_memory_left.vars["RefracTime"].view > self.spike_threshold)
                    food_memory_spikes += np.sum(self.food_memory_right.vars["RefracTime"].view > self.spike_threshold)
                elif self.food_memory is not None:
                    self.food_memory.vars["RefracTime"].pull_from_device()
                    food_memory_spikes += np.sum(self.food_memory.vars["RefracTime"].view > self.spike_threshold)

            # Phase 4 스파이크 카운팅
            if self.config.basal_ganglia_enabled:
                self.striatum.vars["RefracTime"].pull_from_device()
                self.direct_pathway.vars["RefracTime"].pull_from_device()
                self.indirect_pathway.vars["RefracTime"].pull_from_device()
                self.dopamine_neurons.vars["RefracTime"].pull_from_device()

                striatum_spikes += np.sum(self.striatum.vars["RefracTime"].view > self.spike_threshold)
                direct_spikes += np.sum(self.direct_pathway.vars["RefracTime"].view > self.spike_threshold)
                indirect_spikes += np.sum(self.indirect_pathway.vars["RefracTime"].view > self.spike_threshold)
                dopamine_spikes += np.sum(self.dopamine_neurons.vars["RefracTime"].view > self.spike_threshold)

            # Phase 5 스파이크 카운팅
            if self.config.prefrontal_enabled:
                self.working_memory.vars["RefracTime"].pull_from_device()
                self.goal_food.vars["RefracTime"].pull_from_device()
                self.goal_safety.vars["RefracTime"].pull_from_device()
                self.inhibitory_control.vars["RefracTime"].pull_from_device()

                working_memory_spikes += np.sum(self.working_memory.vars["RefracTime"].view > self.spike_threshold)
                goal_food_spikes += np.sum(self.goal_food.vars["RefracTime"].view > self.spike_threshold)
                goal_safety_spikes += np.sum(self.goal_safety.vars["RefracTime"].view > self.spike_threshold)
                inhibitory_spikes += np.sum(self.inhibitory_control.vars["RefracTime"].view > self.spike_threshold)

            # Phase 6a 스파이크 카운팅
            if self.config.cerebellum_enabled:
                self.granule_cells.vars["RefracTime"].pull_from_device()
                self.purkinje_cells.vars["RefracTime"].pull_from_device()
                self.deep_nuclei.vars["RefracTime"].pull_from_device()
                self.error_signal.vars["RefracTime"].pull_from_device()

                granule_spikes += np.sum(self.granule_cells.vars["RefracTime"].view > self.spike_threshold)
                purkinje_spikes += np.sum(self.purkinje_cells.vars["RefracTime"].view > self.spike_threshold)
                deep_nuclei_spikes += np.sum(self.deep_nuclei.vars["RefracTime"].view > self.spike_threshold)
                error_spikes += np.sum(self.error_signal.vars["RefracTime"].view > self.spike_threshold)

            # Phase 6b 스파이크 카운팅
            if self.config.thalamus_enabled:
                self.food_relay.vars["RefracTime"].pull_from_device()
                self.danger_relay.vars["RefracTime"].pull_from_device()
                self.trn.vars["RefracTime"].pull_from_device()
                self.arousal.vars["RefracTime"].pull_from_device()

                food_relay_spikes += np.sum(self.food_relay.vars["RefracTime"].view > self.spike_threshold)
                danger_relay_spikes += np.sum(self.danger_relay.vars["RefracTime"].view > self.spike_threshold)
                trn_spikes += np.sum(self.trn.vars["RefracTime"].view > self.spike_threshold)
                arousal_spikes += np.sum(self.arousal.vars["RefracTime"].view > self.spike_threshold)

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

        # Phase 4 스파이크율
        striatum_rate = 0.0
        direct_rate = 0.0
        indirect_rate = 0.0
        dopamine_rate = 0.0
        if self.config.basal_ganglia_enabled:
            max_spikes_striatum = self.config.n_striatum * 5
            max_spikes_direct = self.config.n_direct_pathway * 5
            max_spikes_indirect = self.config.n_indirect_pathway * 5
            max_spikes_dopamine = self.config.n_dopamine * 5

            striatum_rate = striatum_spikes / max_spikes_striatum
            direct_rate = direct_spikes / max_spikes_direct
            indirect_rate = indirect_spikes / max_spikes_indirect
            dopamine_rate = dopamine_spikes / max_spikes_dopamine

        # Phase 5 스파이크율
        working_memory_rate = 0.0
        goal_food_rate = 0.0
        goal_safety_rate = 0.0
        inhibitory_rate = 0.0
        if self.config.prefrontal_enabled:
            max_spikes_wm = self.config.n_working_memory * 5
            max_spikes_goal_food = self.config.n_goal_food * 5
            max_spikes_goal_safety = self.config.n_goal_safety * 5
            max_spikes_inhibitory = self.config.n_inhibitory_control * 5

            working_memory_rate = working_memory_spikes / max_spikes_wm
            goal_food_rate = goal_food_spikes / max_spikes_goal_food
            goal_safety_rate = goal_safety_spikes / max_spikes_goal_safety
            inhibitory_rate = inhibitory_spikes / max_spikes_inhibitory

        # Phase 6a 스파이크율
        granule_rate = 0.0
        purkinje_rate = 0.0
        deep_nuclei_rate = 0.0
        error_rate = 0.0
        if self.config.cerebellum_enabled:
            max_spikes_granule = self.config.n_granule_cells * 5
            max_spikes_purkinje = self.config.n_purkinje_cells * 5
            max_spikes_deep = self.config.n_deep_nuclei * 5
            max_spikes_error = self.config.n_error_signal * 5

            granule_rate = granule_spikes / max_spikes_granule
            purkinje_rate = purkinje_spikes / max_spikes_purkinje
            deep_nuclei_rate = deep_nuclei_spikes / max_spikes_deep
            error_rate = error_spikes / max_spikes_error

        # Phase 6b 스파이크율
        food_relay_rate = 0.0
        danger_relay_rate = 0.0
        trn_rate = 0.0
        arousal_rate = 0.0
        if self.config.thalamus_enabled:
            max_spikes_food_relay = self.config.n_food_relay * 5
            max_spikes_danger_relay = self.config.n_danger_relay * 5
            max_spikes_trn = self.config.n_trn * 5
            max_spikes_arousal = self.config.n_arousal * 5

            food_relay_rate = food_relay_spikes / max_spikes_food_relay
            danger_relay_rate = danger_relay_spikes / max_spikes_danger_relay
            trn_rate = trn_spikes / max_spikes_trn
            arousal_rate = arousal_spikes / max_spikes_arousal

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

            # Phase 4 뉴런 활성화
            "striatum_rate": striatum_rate,
            "direct_rate": direct_rate,
            "indirect_rate": indirect_rate,
            "dopamine_rate": dopamine_rate,
            "dopamine_level": self.dopamine_level if self.config.basal_ganglia_enabled else 0.0,

            # Phase 5 뉴런 활성화
            "working_memory_rate": working_memory_rate,
            "goal_food_rate": goal_food_rate,
            "goal_safety_rate": goal_safety_rate,
            "inhibitory_rate": inhibitory_rate,

            # Phase 6a 뉴런 활성화
            "granule_rate": granule_rate,
            "purkinje_rate": purkinje_rate,
            "deep_nuclei_rate": deep_nuclei_rate,
            "error_rate": error_rate,

            # Phase 6b 뉴런 활성화
            "food_relay_rate": food_relay_rate,
            "danger_relay_rate": danger_relay_rate,
            "trn_rate": trn_rate,
            "arousal_rate": arousal_rate,

            # 에이전트 위치 (Place Cell 시각화용)
            "agent_grid_x": int(observation.get("position_x", 0.5) * 10),  # 0~10 그리드
            "agent_grid_y": int(observation.get("position_y", 0.5) * 10),

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

        # Phase 3: Place Cells 추가 (I_input 있음)
        if self.config.hippocampus_enabled:
            sensory_pops.append(self.place_cells)

        # Phase 6b: Thalamus 추가 (I_input 있음)
        if self.config.thalamus_enabled:
            sensory_pops.extend([self.food_relay, self.danger_relay,
                                 self.trn, self.arousal])

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

        # Phase 3: Food Memory 추가
        if self.config.hippocampus_enabled:
            if self.config.directional_food_memory:
                lif_pops.extend([self.food_memory_left, self.food_memory_right])
            elif self.food_memory is not None:
                lif_pops.append(self.food_memory)

        # Phase 4: Basal Ganglia 추가
        if self.config.basal_ganglia_enabled:
            lif_pops.extend([self.striatum, self.direct_pathway, self.indirect_pathway])

        # Phase 5: Prefrontal Cortex 추가
        if self.config.prefrontal_enabled:
            lif_pops.extend([self.working_memory, self.goal_food,
                            self.goal_safety, self.inhibitory_control])

        # Phase 6a: Cerebellum 추가
        if self.config.cerebellum_enabled:
            lif_pops.extend([self.granule_cells, self.purkinje_cells,
                            self.deep_nuclei, self.error_signal])

        for pop in lif_pops:
            pop.vars["V"].view[:] = self.config.v_rest
            pop.vars["RefracTime"].view[:] = 0.0
            pop.vars["V"].push_to_device()
            pop.vars["RefracTime"].push_to_device()

        # Phase 4: Dopamine 초기화 (I_input 있는 Sensory 타입)
        if self.config.basal_ganglia_enabled:
            self.dopamine_neurons.vars["V"].view[:] = self.config.v_rest
            self.dopamine_neurons.vars["RefracTime"].view[:] = 0.0
            self.dopamine_neurons.vars["I_input"].view[:] = 0.0
            self.dopamine_neurons.vars["V"].push_to_device()
            self.dopamine_neurons.vars["RefracTime"].push_to_device()
            self.dopamine_neurons.vars["I_input"].push_to_device()
            self.dopamine_level = 0.0


def run_training(episodes: int = 20, render_mode: str = "none",
                log_level: str = "normal", debug: bool = False,
                no_amygdala: bool = False, no_pain: bool = False,
                persist_learning: bool = False, fps: int = 10):
    """Phase 6b 훈련 실행"""

    print("=" * 70)
    print("Phase 6b: Forager Training with Thalamus (Sensory Gating & Attention)")
    print("=" * 70)
    if persist_learning:
        print("  [!] PERSIST LEARNING ENABLED - weights saved/loaded between episodes")

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
    env.render_fps = fps  # FPS 설정 (시각화 속도 조절)
    brain = ForagerBrain(brain_config)

    # 통계
    all_steps = []
    all_food = []
    all_homeostasis = []
    all_pain_visits = []
    all_pain_steps = []
    death_causes = {"starve": 0, "timeout": 0, "pain": 0}

    # Phase 3b: 학습 통계
    all_learn_events = []  # 총 학습 이벤트 수

    for ep in range(episodes):
        obs = env.reset()
        brain.reset()
        done = False
        total_reward = 0

        # Phase 3b: 에피소드 간 학습 지속 - 가중치 로드
        if persist_learning and ep > 0:
            loaded = brain.load_hippocampus_weights()
            if loaded and log_level in ["normal", "debug", "verbose"]:
                stats = brain.get_hippocampus_stats()
                print(f"  [LOAD] Restored weights: avg={stats['avg_weight']:.2f}, "
                      f"max={stats['max_weight']:.2f}, strong={stats['n_strong_connections']}")

        # 에피소드 로그
        ep_hunger_rates = []
        ep_satiety_rates = []
        ep_fear_rates = []
        ep_learn_events = 0  # Phase 3b: 학습 이벤트 카운트

        while not done:
            # 뇌 처리
            action_delta, info = brain.process(obs, debug=debug)
            action = (action_delta,)

            # Phase 4: Dopamine 감쇠 (매 스텝)
            brain.decay_dopamine()

            # 시각화를 위해 뇌 정보 전달 (render 전에 설정)
            env.set_brain_info(info)

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

            # 음식 섭취 이벤트 (normal 이상) + Phase 3b/3c 학습 + Phase 4 Dopamine
            if env_info["food_eaten"]:
                # Phase 3b/3c: Hebbian 학습 실행
                # food_position을 전달하여 방향성 학습 지원
                food_pos = (obs["position_x"], obs["position_y"])
                learn_info = brain.learn_food_location(food_position=food_pos)
                if learn_info:
                    ep_learn_events += 1

                # Phase 4: Dopamine 방출 (보상)
                dopamine_info = brain.release_dopamine(reward_magnitude=0.5)

                if log_level in ["normal", "debug", "verbose"]:
                    da_str = f", DA={dopamine_info['dopamine_level']:.2f}" if dopamine_info else ""
                    if learn_info:
                        side_str = f", side={learn_info.get('side', 'N/A')}" if 'side' in learn_info else ""
                        print(f"  [!] FOOD EATEN at step {env.steps}, Energy: {env_info['energy']:.1f} "
                              f"[LEARN: {learn_info['n_strengthened']} cells, avg_w={learn_info['avg_weight']:.2f}{side_str}{da_str}]")
                    else:
                        print(f"  [!] FOOD EATEN at step {env.steps}, Energy: {env_info['energy']:.1f}{da_str}")

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
        all_learn_events.append(ep_learn_events)  # Phase 3b

        if env_info["death_cause"]:
            death_causes[env_info["death_cause"]] = death_causes.get(env_info["death_cause"], 0) + 1

        # Phase 3b: 에피소드 간 학습 지속 - 가중치 저장
        if persist_learning:
            brain.save_hippocampus_weights()
            if log_level in ["debug", "verbose"]:
                stats = brain.get_hippocampus_stats()
                print(f"  [SAVE] Weights saved: avg={stats['avg_weight']:.2f}, "
                      f"max={stats['max_weight']:.2f}, strong={stats['n_strong_connections']}")

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

    # Phase 3b: 학습 통계
    if brain_config.hippocampus_enabled and sum(all_learn_events) > 0:
        print(f"\n  === Phase 3b: Hippocampus Learning ===")
        print(f"  Total Learn Events: {sum(all_learn_events)}")
        print(f"  Avg Learn/Episode:  {np.mean(all_learn_events):.1f}")
        if persist_learning:
            stats = brain.get_hippocampus_stats()
            if stats:
                print(f"  --- Cumulative Learning ---")
                print(f"  Final Avg Weight:   {stats['avg_weight']:.2f} (initial: 5.0)")
                print(f"  Final Max Weight:   {stats['max_weight']:.2f}")
                print(f"  Strong Connections: {stats['n_strong_connections']}")

    print(f"\n  Death Causes:")
    for cause, count in death_causes.items():
        if count > 0:
            print(f"    {cause}: {count} ({count/episodes*100:.1f}%)")

    # 성공 기준 체크
    survival_rate = death_causes.get("timeout", 0) / episodes * 100
    reward_freq = np.sum(all_food) / np.sum(all_steps) * 100

    print(f"\n  === Phase 2 Validation ===")
    print(f"  Survival Rate: {survival_rate:.1f}% {'✓' if survival_rate > 40 else '✗'} (target: >40%)")
    print(f"  Reward Freq:   {reward_freq:.2f}% {'✓' if reward_freq > 2.5 else '✗'} (target: >2.5%)")

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
    parser.add_argument("--persist-learning", action="store_true",
                       help="Save/load Hippocampus weights between episodes (cumulative learning)")
    parser.add_argument("--fps", type=int, default=10,
                       help="Render FPS (default: 10, slower=easier to observe)")
    args = parser.parse_args()

    run_training(
        episodes=args.episodes,
        render_mode=args.render,
        log_level=args.log_level,
        debug=args.debug,
        no_amygdala=args.no_amygdala,
        no_pain=args.no_pain,
        persist_learning=args.persist_learning,
        fps=args.fps
    )
