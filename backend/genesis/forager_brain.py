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

    # === PRIMARY VISUAL CORTEX (Phase 8 신규) ===
    v1_enabled: bool = True                        # V1 활성화 여부
    n_v1_food_left: int = 100                      # 좌측 음식 시각 처리
    n_v1_food_right: int = 100                     # 우측 음식 시각 처리
    n_v1_danger_left: int = 100                    # 좌측 위험 시각 처리
    n_v1_danger_right: int = 100                   # 우측 위험 시각 처리

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

    # === Phase 8 시냅스 가중치 (V1) ===
    # 입력: Relay → V1
    food_relay_to_v1_weight: float = 20.0          # Food Relay → V1 Food
    danger_relay_to_v1_weight: float = 20.0        # Danger Relay → V1 Danger

    # 내부: Lateral Inhibition (대비 강화)
    v1_lateral_inhibition: float = -8.0            # V1 좌우 상호 억제

    # 출력: V1 → 다른 영역
    v1_to_motor_weight: float = 15.0               # V1 → Motor
    v1_to_hippocampus_weight: float = 10.0         # V1 Food → Place Cells
    v1_to_amygdala_weight: float = 12.0            # V1 Danger → Amygdala LA

    # === V2/V4 고차 시각 피질 (Phase 9 신규) ===
    v2v4_enabled: bool = True                      # V2/V4 활성화 여부
    n_v2_edge_food: int = 150                      # V2 음식 에지/윤곽
    n_v2_edge_danger: int = 150                    # V2 위험 에지/윤곽
    n_v4_food_object: int = 100                    # V4 음식 물체 표상
    n_v4_danger_object: int = 100                  # V4 위험 물체 표상
    n_v4_novel_object: int = 100                   # V4 새로운 물체 표상

    # === Phase 9 시냅스 가중치 ===
    # V1 → V2 (수렴)
    v1_to_v2_weight: float = 15.0                  # V1 → V2 Edge

    # V2 → V4 (분류)
    v2_to_v4_weight: float = 20.0                  # V2 Edge → V4 Object
    v4_wta_inhibition: float = -12.0               # V4 WTA (Food vs Danger vs Novel)

    # V4 → 상위 영역
    v4_food_to_hippocampus_weight: float = 15.0    # V4 Food → Hippocampus
    v4_food_to_hunger_weight: float = 10.0         # V4 Food → Hunger Drive
    v4_danger_to_amygdala_weight: float = 18.0     # V4 Danger → Amygdala
    v4_novel_to_dopamine_weight: float = 20.0      # V4 Novel → Dopamine (호기심)

    # Top-Down 조절
    hunger_to_v4_food_weight: float = 8.0          # Hunger → V4 Food (주의 조절)
    fear_to_v4_danger_weight: float = 10.0         # Fear → V4 Danger
    goal_to_v2_weight: float = 6.0                 # Goal → V2 (선택적 주의)

    # V2→V4 Hebbian 학습
    v2_v4_eta: float = 0.1                         # 학습률
    v2_v4_w_max: float = 40.0                      # 최대 가중치

    # === IT Cortex (Phase 10 신규) ===
    it_enabled: bool = True                        # IT Cortex 활성화 여부
    n_it_food_category: int = 200                  # "음식" 범주 뉴런
    n_it_danger_category: int = 200                # "위험" 범주 뉴런
    n_it_neutral_category: int = 150               # 중립/미분류 물체
    n_it_association: int = 200                    # 범주 간 연합
    n_it_memory_buffer: int = 250                  # 단기 물체 기억

    # === Phase 10 시냅스 가중치 ===
    # V4 → IT (순방향)
    v4_to_it_weight: float = 25.0                  # V4 → IT Category (강한 분류)

    # IT ↔ Hippocampus (양방향)
    it_to_hippocampus_weight: float = 15.0         # IT → Hippocampus (저장)
    hippocampus_to_it_weight: float = 12.0         # Hippocampus → IT (인출)

    # IT ↔ Amygdala (양방향)
    it_to_amygdala_weight: float = 18.0            # IT_Danger → Amygdala
    amygdala_to_it_weight: float = 15.0            # Fear → IT_Danger

    # IT → Motor (행동 출력)
    it_to_motor_weight: float = 12.0               # IT → Motor

    # IT → PFC (목표 설정)
    it_to_pfc_weight: float = 15.0                 # IT → Goal

    # IT 내부 WTA
    it_wta_inhibition: float = -15.0               # IT 범주 간 경쟁

    # Top-Down 조절
    hunger_to_it_food_weight: float = 10.0         # Hunger → IT_Food
    fear_to_it_danger_weight: float = 12.0         # Fear → IT_Danger
    wm_to_it_buffer_weight: float = 8.0            # Working Memory → IT_Buffer

    # === Auditory Cortex (Phase 11 신규) ===
    auditory_enabled: bool = True                   # 청각 피질 활성화 여부
    n_sound_danger_left: int = 100                  # 왼쪽 위험 소리 입력
    n_sound_danger_right: int = 100                 # 오른쪽 위험 소리 입력
    n_sound_food_left: int = 100                    # 왼쪽 음식 소리 입력
    n_sound_food_right: int = 100                   # 오른쪽 음식 소리 입력
    n_a1_danger: int = 150                          # A1 위험 소리 처리
    n_a1_food: int = 150                            # A1 음식 소리 처리
    n_a2_association: int = 200                     # 청각 연합 영역

    # === Phase 11 시냅스 가중치 ===
    # Sound Input → A1
    sound_to_a1_weight: float = 20.0                # Sound → A1 (순방향)

    # A1 → Amygdala (청각-공포 경로)
    a1_danger_to_amygdala_weight: float = 22.0      # A1_Danger → LA (빠른 공포)

    # A1 → IT (청각-시각 통합)
    a1_to_it_weight: float = 15.0                   # A1 → IT Category

    # A1 → Motor (청각 유도 행동)
    a1_to_motor_weight: float = 12.0                # A1 → Motor

    # A2 Association
    a1_to_a2_weight: float = 10.0                   # A1 → A2
    it_to_a2_weight: float = 10.0                   # IT → A2 (다감각 통합)

    # Top-Down
    fear_to_a1_danger_weight: float = 8.0           # Fear → A1_Danger
    hunger_to_a1_food_weight: float = 8.0           # Hunger → A1_Food

    # A1 lateral inhibition
    a1_lateral_inhibition: float = -10.0            # A1 좌우 경쟁

    # === Multimodal Integration (Phase 12 신규) ===
    multimodal_enabled: bool = True                  # 다중 감각 통합 활성화
    n_sts_food: int = 200                            # STS 음식 통합
    n_sts_danger: int = 200                          # STS 위험 통합
    n_sts_congruence: int = 150                      # 시청각 일치 감지
    n_sts_mismatch: int = 100                        # 시청각 불일치 감지
    n_multimodal_buffer: int = 150                   # 다중 감각 작업 기억

    # === Phase 12 시냅스 가중치 ===
    # 시각 → STS
    it_to_sts_weight: float = 20.0                   # IT → STS (시각 입력)

    # 청각 → STS
    a1_to_sts_weight: float = 20.0                   # A1 → STS (청각 입력)
    a2_to_sts_weight: float = 15.0                   # A2 → STS (연합 청각)

    # STS 내부 연결
    sts_congruence_weight: float = 15.0              # 일치 감지 가중치
    sts_mismatch_weight: float = 12.0                # 불일치 감지 가중치
    sts_wta_inhibition: float = -8.0                 # STS WTA 경쟁

    # STS → 출력 (간접 경로도 Pain 반사 간섭 주의 - 2026-02-08 수정)
    sts_to_hippocampus_weight: float = 8.0           # STS → Hippocampus (15→8 약화: Food_Memory→Motor 간접 간섭 방지)
    sts_to_amygdala_weight: float = 8.0              # STS_Danger → Amygdala (18→8 약화: Fear 과다 증폭 방지)
    sts_to_motor_weight: float = 0.0                 # STS → Motor (비활성화 - Pain 반사 간섭 방지)
    sts_to_pfc_weight: float = 10.0                  # STS → PFC

    # Top-Down → STS
    hunger_to_sts_weight: float = 8.0                # Hunger → STS_Food
    fear_to_sts_weight: float = 10.0                 # Fear → STS_Danger
    wm_to_sts_congruence_weight: float = 6.0         # Working Memory → Congruence

    # === Parietal Cortex (Phase 13 신규) ===
    parietal_enabled: bool = True                     # 두정엽 활성화 여부
    n_ppc_space_left: int = 150                       # 왼쪽 공간 표상
    n_ppc_space_right: int = 150                      # 오른쪽 공간 표상
    n_ppc_goal_food: int = 150                        # 음식 목표 벡터
    n_ppc_goal_safety: int = 150                      # 안전 목표 벡터
    n_ppc_attention: int = 200                        # 공간 주의 조절
    n_ppc_path_buffer: int = 200                      # 경로 계획 버퍼

    # === Phase 13 시냅스 가중치 ===
    # 감각 → PPC_Space
    v1_to_ppc_weight: float = 15.0                    # V1 → PPC_Space (시각 위치)
    it_to_ppc_weight: float = 12.0                    # IT → PPC_Space (물체 위치)
    sts_to_ppc_weight: float = 15.0                   # STS → PPC_Space (다감각 위치)
    place_to_ppc_weight: float = 12.0                 # Place Cells → PPC_Space (자기 위치)
    food_memory_to_ppc_weight: float = 10.0           # Food Memory → PPC_Space (기억 위치)

    # PFC → PPC (목표 설정)
    goal_to_ppc_weight: float = 18.0                  # Goal → PPC_Goal_Food/Safety
    wm_to_ppc_path_weight: float = 12.0               # Working Memory → Path Buffer

    # PPC 내부 연결
    ppc_space_goal_integration_weight: float = 15.0   # Space + Goal → Goal Vector
    ppc_path_recurrent_weight: float = 10.0           # Path Buffer 자기 유지
    ppc_wta_inhibition: float = -8.0                  # PPC 좌우/목표 경쟁
    ppc_attention_weight: float = 12.0                # Goal → Attention

    # PPC → 출력
    ppc_to_motor_weight: float = 0.0                  # PPC_Goal → Motor (비활성화 - PMC 경유로 변경)
    ppc_to_v1_attention_weight: float = 8.0           # PPC_Attention → V1 (Top-Down)
    ppc_to_sts_attention_weight: float = 8.0          # PPC_Attention → STS (Top-Down)
    ppc_to_hippocampus_weight: float = 5.0            # PPC_Space → Place Cells (10→5 약화: Food_Memory 노이즈 감소)

    # Top-Down → PPC
    hunger_to_ppc_goal_food_weight: float = 10.0      # Hunger → PPC_Goal_Food
    fear_to_ppc_goal_safety_weight: float = 12.0      # Fear → PPC_Goal_Safety
    dopamine_to_ppc_attention_weight: float = 8.0     # Dopamine → PPC_Attention

    # === Premotor Cortex (Phase 14 신규) ===
    premotor_enabled: bool = True                      # 전운동 피질 활성화 여부
    n_pmd_left: int = 100                              # PMd 왼쪽 방향 운동 계획
    n_pmd_right: int = 100                             # PMd 오른쪽 방향 운동 계획
    n_pmv_approach: int = 100                          # PMv 접근 운동 계획
    n_pmv_avoid: int = 100                             # PMv 회피 운동 계획
    n_sma_sequence: int = 150                          # SMA 시퀀스 생성
    n_pre_sma: int = 100                               # pre-SMA 운동 의도
    n_motor_preparation: int = 150                     # 운동 준비 버퍼

    # === Phase 14 시냅스 가중치 ===
    # PPC → PMd (공간 기반 운동 계획)
    ppc_to_pmd_weight: float = 18.0                    # PPC_Goal/Space → PMd

    # IT/STS → PMv (물체 기반 운동 계획)
    it_to_pmv_weight: float = 15.0                     # IT → PMv
    sts_to_pmv_weight: float = 15.0                    # STS → PMv

    # PFC → SMA (목표 기반 시퀀스)
    pfc_to_sma_weight: float = 15.0                    # Goal/WM → SMA
    inhibitory_to_pre_sma_weight: float = -12.0        # Inhibitory → pre_SMA

    # PMC 내부 연결
    sma_recurrent_weight: float = 8.0                  # SMA 자기 유지
    pre_sma_to_sma_weight: float = 12.0                # pre_SMA → SMA
    pmd_pmv_integration_weight: float = 12.0           # PMd/PMv → Motor_Prep
    sma_to_motor_prep_weight: float = 12.0             # SMA → Motor_Prep
    pmc_wta_inhibition: float = -10.0                  # PMC 내 WTA 경쟁

    # PMC → Motor (운동 출력) - 기존 반사 간섭 방지를 위해 약화
    motor_prep_to_motor_weight: float = 2.0            # Motor_Prep → Motor (15→5→2 약화)
    pmd_to_motor_weight: float = 0.0                   # PMd → Motor (비활성화)
    pmv_to_motor_weight: float = 0.0                   # PMv → Motor (비활성화)

    # PMC → Cerebellum (운동 조정)
    motor_prep_to_cerebellum_weight: float = 10.0      # Motor_Prep → Granule

    # BG → PMC (행동 선택)
    direct_to_motor_prep_weight: float = 12.0          # Direct → Motor_Prep (Go)
    indirect_to_motor_prep_weight: float = -8.0        # Indirect → Motor_Prep (NoGo)
    dopamine_to_sma_weight: float = 10.0               # Dopamine → SMA

    # Top-Down → PMC
    hunger_to_pmv_approach_weight: float = 10.0        # Hunger → PMv_Approach
    fear_to_pmv_avoid_weight: float = 12.0             # Fear → PMv_Avoid
    arousal_to_motor_prep_weight: float = 8.0          # Arousal → Motor_Prep

    # === Phase 15: Social Brain (사회적 뇌) ===
    social_brain_enabled: bool = True

    # 뉴런 수
    n_agent_eye_left: int = 200                        # 에이전트 시각 입력 (좌)
    n_agent_eye_right: int = 200                       # 에이전트 시각 입력 (우)
    n_agent_sound_left: int = 100                      # 에이전트 청각 입력 (좌)
    n_agent_sound_right: int = 100                     # 에이전트 청각 입력 (우)
    n_sts_social: int = 200                            # STS 사회적 처리
    n_tpj_self: int = 100                              # TPJ 자기 표상
    n_tpj_other: int = 100                             # TPJ 타자 표상
    n_tpj_compare: int = 100                           # TPJ 자기-타자 비교
    n_acc_conflict: int = 100                          # ACC 갈등 감지
    n_acc_monitor: int = 100                           # ACC 행동 모니터링
    n_social_approach: int = 100                       # 사회적 접근 동기
    n_social_avoid: int = 100                          # 사회적 회피 동기

    # 시냅스 가중치
    agent_eye_to_sts_social_weight: float = 15.0       # Agent_Eye → STS_Social
    agent_sound_to_sts_social_weight: float = 12.0     # Agent_Sound → STS_Social
    sts_social_recurrent_weight: float = 8.0           # STS_Social 자기 유지
    sts_social_to_tpj_weight: float = 12.0             # STS_Social → TPJ_Other
    internal_to_tpj_self_weight: float = 10.0          # Hunger/Satiety → TPJ_Self
    tpj_compare_weight: float = 10.0                   # TPJ_Self/Other → TPJ_Compare
    tpj_to_acc_weight: float = 12.0                    # TPJ_Compare → ACC_Conflict
    social_proximity_to_acc_weight: float = 8.0        # 근접도 → ACC
    sts_social_to_approach_weight: float = 8.0         # STS_Social → Approach
    acc_to_avoid_weight: float = 10.0                  # ACC → Avoid
    social_wta_inhibition: float = -8.0                # Approach ↔ Avoid WTA

    # 기존 회로 연결 (약한 간접 경로 - Phase 12-14 교훈)
    sts_social_to_pfc_weight: float = 6.0              # STS_Social → WM
    acc_to_amygdala_weight: float = 5.0                # ACC → LA (약하게!)
    social_approach_to_goal_food_weight: float = 5.0   # Approach → Goal_Food
    social_avoid_to_goal_safety_weight: float = 5.0    # Avoid → Goal_Safety
    social_to_motor_weight: float = 0.0                # Motor 직접 연결 없음!

    # Top-Down → Social
    fear_to_sts_social_weight: float = 8.0             # Fear → STS_Social
    hunger_to_social_approach_weight: float = 6.0      # Hunger → Social_Approach

    # === Phase 15b: Mirror Neurons & Social Learning ===
    mirror_enabled: bool = True
    n_social_observation: int = 200                    # NPC 목표지향 움직임 감지
    n_mirror_food: int = 150                           # 거울 뉴런 (자기+타인 먹기)
    n_vicarious_reward: int = 100                      # 관찰 예측 오차 (대리 보상)
    n_social_memory: int = 150                         # 사회적 음식 위치 기억

    # 내부 연결
    agent_eye_to_social_obs_weight: float = 12.0       # Agent_Eye → Social_Obs
    sts_social_to_social_obs_weight: float = 10.0      # STS_Social → Social_Obs
    social_obs_to_mirror_weight: float = 10.0          # Social_Obs → Mirror_Food
    mirror_to_vicarious_weight: float = 12.0           # Mirror_Food → Vicarious_Reward
    vicarious_to_social_memory_weight: float = 15.0    # Vicarious → Social_Memory (Hebbian)

    # 기존 회로 출력 (약한 간접 경로, 모두 ≤6.0!)
    social_memory_to_food_memory_weight: float = 5.0   # Social_Memory → Food_Memory L/R
    social_obs_to_wm_weight: float = 5.0               # Social_Obs → Working_Memory
    social_obs_to_dopamine_weight: float = 6.0         # Social_Obs → Dopamine
    mirror_to_goal_food_weight: float = 5.0            # Mirror_Food → Goal_Food
    mirror_to_hunger_weight: float = 4.0               # Mirror_Food → Hunger
    mirror_to_motor_weight: float = 0.0                # Motor 직접 연결 없음!

    # Top-Down → Mirror
    hunger_to_social_obs_weight: float = 6.0           # Hunger → Social_Obs
    fear_to_social_obs_weight: float = -4.0            # Fear → Social_Obs (억제)
    hunger_to_mirror_weight: float = 8.0               # Hunger → Mirror_Food
    food_eye_to_mirror_weight: float = 6.0             # Food_Eye → Mirror_Food

    # Hebbian 학습 (Social_Memory)
    social_memory_eta: float = 0.1                     # 학습률
    social_memory_w_max: float = 20.0                  # 최대 가중치

    # Recurrent
    social_obs_recurrent_weight: float = 6.0           # Social_Obs 자기 유지
    mirror_food_recurrent_weight: float = 5.0          # Mirror_Food 자기 유지
    social_memory_recurrent_weight: float = 8.0        # Social_Memory 자기 유지

    # === Phase 15c: Theory of Mind & Cooperation/Competition ===
    tom_enabled: bool = True
    n_tom_intention: int = 100                          # NPC 의도 추론 (mPFC)
    n_tom_belief: int = 80                              # NPC 신념 추적 (mPFC)
    n_tom_prediction: int = 80                          # NPC 행동 예측
    n_tom_surprise: int = 60                            # 사회적 예측 오차 (Ant. Insula)
    n_coop_compete_coop: int = 80                       # 협력 가치 (vmPFC)
    n_coop_compete_compete: int = 100                   # 경쟁 감지 (dACC)

    # 내부 연결
    social_obs_to_tom_intention_weight: float = 10.0    # Social_Obs → ToM_Intention
    sts_social_to_tom_intention_weight: float = 8.0     # STS_Social → ToM_Intention
    tom_intention_to_belief_weight: float = 12.0        # ToM_Intention → ToM_Belief
    tpj_other_to_tom_belief_weight: float = 10.0        # TPJ_Other → ToM_Belief
    social_obs_to_tom_belief_weight: float = 8.0        # Social_Obs → ToM_Belief
    tom_intention_to_prediction_weight: float = 15.0    # ToM_Intention → ToM_Prediction
    tom_belief_to_prediction_weight: float = 12.0       # ToM_Belief → ToM_Prediction
    tom_prediction_recurrent_weight: float = 8.0        # ToM_Prediction 자기 유지
    tom_prediction_to_surprise_weight: float = -10.0    # Prediction → Surprise (억제)
    tom_surprise_to_prediction_weight: float = -6.0     # Surprise → Prediction (리셋)

    # Coop/Compete
    tom_intention_to_coop_weight: float = 10.0          # ToM_Intention → Coop (Hebbian)
    social_memory_to_coop_weight: float = 8.0           # Social_Memory → Coop
    coop_recurrent_weight: float = 6.0                  # Coop 자기 유지
    tom_intention_to_compete_weight: float = 8.0        # ToM_Intention → Compete
    acc_conflict_to_compete_weight: float = 8.0         # ACC_Conflict → Compete
    coop_compete_wta_weight: float = -8.0               # Coop ↔ Compete 상호 억제

    # 기존 회로 출력 (모두 ≤6.0, Motor 0.0!)
    coop_to_social_approach_weight: float = 5.0         # Coop → Social_Approach
    coop_to_goal_food_weight: float = 4.0               # Coop → Goal_Food
    compete_to_social_avoid_weight: float = 5.0         # Compete → Social_Avoid
    compete_to_hunger_weight: float = 4.0               # Compete → Hunger (긴급성)
    compete_to_acc_weight: float = 5.0                  # Compete → ACC_Conflict
    tom_surprise_to_acc_weight: float = 4.0             # Surprise → ACC_Monitor
    tom_surprise_to_dopamine_weight: float = 5.0        # Surprise → Dopamine (novelty)
    tom_intention_to_wm_weight: float = 5.0             # Intention → Working_Memory
    tom_to_motor_weight: float = 0.0                    # Motor 직접 연결 없음!

    # Top-Down
    hunger_to_tom_intention_weight: float = 6.0         # Hunger → ToM_Intention
    fear_to_tom_intention_weight: float = -4.0          # Fear → ToM_Intention (억제)
    hunger_to_compete_weight: float = 6.0               # Hunger → Compete

    # Hebbian (Cooperation value learning)
    tom_coop_eta: float = 0.08                          # 학습률
    tom_coop_w_max: float = 20.0                        # 최대 가중치

    # === Phase 16: Association Cortex (연합 피질) ===
    association_cortex_enabled: bool = True
    n_assoc_edible: int = 120                           # "먹을 수 있는 것" 초범주
    n_assoc_threatening: int = 120                      # "위험한 것" 초범주
    n_assoc_animate: int = 100                          # "살아있는 것" 초범주
    n_assoc_context: int = 100                          # "익숙한 장소" 맥락
    n_assoc_valence: int = 80                           # "좋다/나쁘다" 가치
    n_assoc_binding: int = 100                          # 교차 연합 (Hebbian)
    n_assoc_novelty: int = 80                           # 새로운 조합 탐지

    # 입력 연결 (기존 회로 → 연합 피질)
    it_food_to_assoc_edible_weight: float = 12.0
    sts_food_to_assoc_edible_weight: float = 10.0
    a1_food_to_assoc_edible_weight: float = 8.0
    social_memory_to_assoc_edible_weight: float = 6.0
    mirror_food_to_assoc_edible_weight: float = 5.0
    it_danger_to_assoc_threatening_weight: float = 12.0
    sts_danger_to_assoc_threatening_weight: float = 10.0
    a1_danger_to_assoc_threatening_weight: float = 8.0
    fear_to_assoc_threatening_weight: float = 8.0
    tom_intention_to_assoc_animate_weight: float = 10.0
    social_obs_to_assoc_animate_weight: float = 10.0
    sts_social_to_assoc_animate_weight: float = 8.0
    mirror_food_to_assoc_animate_weight: float = 5.0
    place_cells_to_assoc_context_weight: float = 8.0
    ppc_space_to_assoc_context_weight: float = 8.0
    food_memory_to_assoc_context_weight: float = 6.0
    dopamine_to_assoc_valence_weight: float = 10.0
    assoc_edible_to_valence_weight: float = 8.0
    assoc_threatening_to_valence_weight: float = -8.0   # inhibitory
    satiety_to_assoc_valence_weight: float = 5.0

    # 내부 연결
    assoc_edible_threatening_wta: float = -6.0          # WTA 상호 억제
    assoc_edible_recurrent: float = 6.0                 # 개념 지속성
    assoc_threatening_recurrent: float = 6.0
    assoc_context_recurrent: float = 5.0
    assoc_binding_recurrent: float = 6.0
    assoc_edible_to_binding_weight: float = 10.0        # Hebbian
    assoc_context_to_binding_weight: float = 10.0       # Hebbian
    assoc_animate_to_binding_weight: float = 8.0
    assoc_valence_to_binding_weight: float = 8.0
    it_neutral_to_assoc_novelty_weight: float = 10.0
    sts_mismatch_to_assoc_novelty_weight: float = 8.0
    assoc_binding_to_novelty_weight: float = -6.0       # 익숙한 것 억제

    # 출력 연결 (모두 ≤6.0, Motor 0.0!)
    assoc_edible_to_goal_food_weight: float = 5.0
    assoc_edible_to_wm_weight: float = 4.0
    assoc_threatening_to_goal_safety_weight: float = 5.0
    assoc_threatening_to_acc_weight: float = 4.0
    assoc_animate_to_tpj_weight: float = 4.0
    assoc_context_to_wm_weight: float = 4.0
    assoc_context_to_food_memory_weight: float = 3.0
    assoc_valence_to_dopamine_weight: float = 4.0
    assoc_novelty_to_arousal_weight: float = 5.0
    assoc_novelty_to_dopamine_weight: float = 4.0
    assoc_binding_to_it_assoc_weight: float = 5.0
    assoc_to_motor_weight: float = 0.0                  # 절대 비활성!

    # Top-Down
    hunger_to_assoc_edible_weight: float = 6.0
    fear_to_assoc_threatening_topdown_weight: float = 6.0
    wm_to_assoc_binding_weight: float = 5.0

    # Hebbian (Association binding learning)
    assoc_binding_eta: float = 0.06                     # 학습률
    assoc_binding_w_max: float = 18.0                   # 최대 가중치

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
        if self.v1_enabled:
            base += (self.n_v1_food_left + self.n_v1_food_right +
                     self.n_v1_danger_left + self.n_v1_danger_right)
        if self.v2v4_enabled:
            base += (self.n_v2_edge_food + self.n_v2_edge_danger +
                     self.n_v4_food_object + self.n_v4_danger_object +
                     self.n_v4_novel_object)
        if self.it_enabled:
            base += (self.n_it_food_category + self.n_it_danger_category +
                     self.n_it_neutral_category + self.n_it_association +
                     self.n_it_memory_buffer)
        if self.auditory_enabled:
            base += (self.n_sound_danger_left + self.n_sound_danger_right +
                     self.n_sound_food_left + self.n_sound_food_right +
                     self.n_a1_danger + self.n_a1_food + self.n_a2_association)
        if self.multimodal_enabled:
            base += (self.n_sts_food + self.n_sts_danger +
                     self.n_sts_congruence + self.n_sts_mismatch +
                     self.n_multimodal_buffer)
        if self.parietal_enabled:
            base += (self.n_ppc_space_left + self.n_ppc_space_right +
                     self.n_ppc_goal_food + self.n_ppc_goal_safety +
                     self.n_ppc_attention + self.n_ppc_path_buffer)
        if self.premotor_enabled:
            base += (self.n_pmd_left + self.n_pmd_right +
                     self.n_pmv_approach + self.n_pmv_avoid +
                     self.n_sma_sequence + self.n_pre_sma +
                     self.n_motor_preparation)
        if self.social_brain_enabled:
            base += (self.n_agent_eye_left + self.n_agent_eye_right +
                     self.n_agent_sound_left + self.n_agent_sound_right +
                     self.n_sts_social + self.n_tpj_self + self.n_tpj_other +
                     self.n_tpj_compare + self.n_acc_conflict + self.n_acc_monitor +
                     self.n_social_approach + self.n_social_avoid)
            if self.mirror_enabled:
                base += (self.n_social_observation + self.n_mirror_food +
                         self.n_vicarious_reward + self.n_social_memory)
            if self.tom_enabled:
                base += (self.n_tom_intention + self.n_tom_belief +
                         self.n_tom_prediction + self.n_tom_surprise +
                         self.n_coop_compete_coop + self.n_coop_compete_compete)
        if self.association_cortex_enabled:
            base += (self.n_assoc_edible + self.n_assoc_threatening +
                     self.n_assoc_animate + self.n_assoc_context +
                     self.n_assoc_valence + self.n_assoc_binding +
                     self.n_assoc_novelty)
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

        # Phase 15b: Mirror neuron state defaults
        self.mirror_self_eating_timer = 0
        self.last_social_obs_rate = 0.0

        # Phase 15c: Theory of Mind state defaults
        self.last_tom_intention_rate = 0.0

        # Phase 16: Association Cortex state defaults
        self.last_assoc_binding_rate = 0.0

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

        # Phase 8: V1 (Primary Visual Cortex) circuits
        if self.config.v1_enabled:
            self._build_v1_circuit()

        # Phase 9: V2/V4 (Higher Visual Cortex) circuits
        if self.config.v2v4_enabled and self.config.v1_enabled:
            self._build_v2v4_circuit()

        # Phase 10: IT Cortex (Inferior Temporal) circuits
        if self.config.it_enabled and self.config.v2v4_enabled:
            self._build_it_cortex_circuit()

        # Phase 11: Auditory Cortex circuits
        if self.config.auditory_enabled:
            self._build_auditory_cortex_circuit()

        # Phase 12: Multimodal Integration circuits
        if self.config.multimodal_enabled and self.config.it_enabled and self.config.auditory_enabled:
            self._build_multimodal_integration_circuit()

        # Phase 13: Parietal Cortex circuits
        if self.config.parietal_enabled:
            self._build_parietal_cortex_circuit()

        # Phase 14: Premotor Cortex circuits
        if self.config.premotor_enabled:
            self._build_premotor_cortex_circuit()

        # Phase 15: Social Brain circuits
        if self.config.social_brain_enabled:
            self._build_social_brain_circuit()

        # Phase 15b: Mirror Neuron circuits
        if self.config.social_brain_enabled and self.config.mirror_enabled:
            self._build_mirror_neuron_circuit()

        # Phase 15c: Theory of Mind circuits
        if self.config.social_brain_enabled and self.config.tom_enabled:
            self._build_tom_circuit()

        # Phase 16: Association Cortex
        if self.config.association_cortex_enabled:
            self._build_association_cortex_circuit()

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

    def _build_v1_circuit(self):
        """
        Phase 8: Primary Visual Cortex (V1) 구축

        구성:
        1. V1_Food_Left/Right - 좌/우 음식 시각 처리
        2. V1_Danger_Left/Right - 좌/우 위험 시각 처리

        연결:
        - 입력: Thalamus Relay → V1 (방향 정보 보존)
        - 내부: 좌우 Lateral Inhibition (대비 강화)
        - 출력: V1 → Motor (ipsi/contra), Hippocampus, Amygdala
        """
        print("  Phase 8: Building Primary Visual Cortex (V1)...")

        # LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. V1 Food Populations (좌/우 분리) ===
        self.v1_food_left = self.model.add_neuron_population(
            "v1_food_left", self.config.n_v1_food_left,
            sensory_lif_model, lif_params, lif_init)
        self.v1_food_right = self.model.add_neuron_population(
            "v1_food_right", self.config.n_v1_food_right,
            sensory_lif_model, lif_params, lif_init)

        print(f"    V1_Food: L({self.config.n_v1_food_left}) + R({self.config.n_v1_food_right})")

        # === 2. V1 Danger Populations (좌/우 분리) ===
        self.v1_danger_left = self.model.add_neuron_population(
            "v1_danger_left", self.config.n_v1_danger_left,
            sensory_lif_model, lif_params, lif_init)
        self.v1_danger_right = self.model.add_neuron_population(
            "v1_danger_right", self.config.n_v1_danger_right,
            sensory_lif_model, lif_params, lif_init)

        print(f"    V1_Danger: L({self.config.n_v1_danger_left}) + R({self.config.n_v1_danger_right})")

        # === 입력 연결: Thalamus Relay → V1 (방향 정보 보존) ===
        if self.config.thalamus_enabled:
            # Food Relay → V1 Food (방향 정보 보존을 위해 L/R 분리 필요)
            # Thalamus에서 L/R 정보가 합쳐져 있으므로 Food Eye에서 직접 받음
            pass  # Relay에서는 L/R 구분이 없음, 아래에서 Food Eye로부터 직접 연결

        # Food Eye → V1 Food (방향 정보 보존)
        self._create_static_synapse(
            "food_eye_left_to_v1_food_left", self.food_eye_left, self.v1_food_left,
            self.config.food_relay_to_v1_weight, sparsity=0.15)
        self._create_static_synapse(
            "food_eye_right_to_v1_food_right", self.food_eye_right, self.v1_food_right,
            self.config.food_relay_to_v1_weight, sparsity=0.15)

        print(f"    FoodEye→V1_Food: {self.config.food_relay_to_v1_weight}")

        # Pain Eye → V1 Danger (방향 정보 보존)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "pain_eye_left_to_v1_danger_left", self.pain_eye_left, self.v1_danger_left,
                self.config.danger_relay_to_v1_weight, sparsity=0.15)
            self._create_static_synapse(
                "pain_eye_right_to_v1_danger_right", self.pain_eye_right, self.v1_danger_right,
                self.config.danger_relay_to_v1_weight, sparsity=0.15)

            print(f"    PainEye→V1_Danger: {self.config.danger_relay_to_v1_weight}")

        # Wall Eye → V1 Danger (벽도 위험)
        self._create_static_synapse(
            "wall_eye_left_to_v1_danger_left", self.wall_eye_left, self.v1_danger_left,
            self.config.danger_relay_to_v1_weight * 0.6, sparsity=0.12)
        self._create_static_synapse(
            "wall_eye_right_to_v1_danger_right", self.wall_eye_right, self.v1_danger_right,
            self.config.danger_relay_to_v1_weight * 0.6, sparsity=0.12)

        print(f"    WallEye→V1_Danger: {self.config.danger_relay_to_v1_weight * 0.6:.1f}")

        # === 내부 연결: Lateral Inhibition (대비 강화) ===
        # V1 Food 좌우 상호 억제
        self._create_static_synapse(
            "v1_food_left_to_right", self.v1_food_left, self.v1_food_right,
            self.config.v1_lateral_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "v1_food_right_to_left", self.v1_food_right, self.v1_food_left,
            self.config.v1_lateral_inhibition, sparsity=0.1)

        # V1 Danger 좌우 상호 억제
        self._create_static_synapse(
            "v1_danger_left_to_right", self.v1_danger_left, self.v1_danger_right,
            self.config.v1_lateral_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "v1_danger_right_to_left", self.v1_danger_right, self.v1_danger_left,
            self.config.v1_lateral_inhibition, sparsity=0.1)

        print(f"    V1 Lateral Inhibition: {self.config.v1_lateral_inhibition}")

        # === 출력 연결: V1 → Motor (ipsilateral for food, contralateral for danger) ===
        # V1 Food → Motor (동측: 음식 쪽으로 회전)
        self._create_static_synapse(
            "v1_food_left_to_motor_left", self.v1_food_left, self.motor_left,
            self.config.v1_to_motor_weight, sparsity=0.12)
        self._create_static_synapse(
            "v1_food_right_to_motor_right", self.v1_food_right, self.motor_right,
            self.config.v1_to_motor_weight, sparsity=0.12)

        print(f"    V1_Food→Motor (ipsi): {self.config.v1_to_motor_weight}")

        # V1 Danger → Motor (반대측: 위험 반대편으로 회전)
        self._create_static_synapse(
            "v1_danger_left_to_motor_right", self.v1_danger_left, self.motor_right,
            self.config.v1_to_motor_weight, sparsity=0.12)
        self._create_static_synapse(
            "v1_danger_right_to_motor_left", self.v1_danger_right, self.motor_left,
            self.config.v1_to_motor_weight, sparsity=0.12)

        print(f"    V1_Danger→Motor (contra): {self.config.v1_to_motor_weight}")

        # === 출력 연결: V1 → Hippocampus (Place Cells) ===
        if self.config.hippocampus_enabled:
            self._create_static_synapse(
                "v1_food_left_to_place_cells", self.v1_food_left, self.place_cells,
                self.config.v1_to_hippocampus_weight, sparsity=0.08)
            self._create_static_synapse(
                "v1_food_right_to_place_cells", self.v1_food_right, self.place_cells,
                self.config.v1_to_hippocampus_weight, sparsity=0.08)

            print(f"    V1_Food→PlaceCells: {self.config.v1_to_hippocampus_weight}")

        # === 출력 연결: V1 Danger → Amygdala LA ===
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "v1_danger_left_to_la", self.v1_danger_left, self.lateral_amygdala,
                self.config.v1_to_amygdala_weight, sparsity=0.1)
            self._create_static_synapse(
                "v1_danger_right_to_la", self.v1_danger_right, self.lateral_amygdala,
                self.config.v1_to_amygdala_weight, sparsity=0.1)

            print(f"    V1_Danger→Amygdala LA: {self.config.v1_to_amygdala_weight}")

        total_neurons = (self.config.n_v1_food_left + self.config.n_v1_food_right +
                        self.config.n_v1_danger_left + self.config.n_v1_danger_right)
        print(f"  V1 circuit complete: {total_neurons} neurons")

    def _build_v2v4_circuit(self):
        """
        Phase 9: V2/V4 Higher Visual Cortex 구축

        V2 (Secondary Visual Cortex):
        - 에지/윤곽 검출
        - 좌우 V1 정보 수렴 (크기 불변성)

        V4 (Visual Area V4):
        - 물체 분류 (Food, Danger, Novel)
        - WTA 경쟁 (하나의 분류만 활성화)

        연결:
        - 입력: V1_Food L/R → V2_Edge_Food (수렴)
        - 내부: V2 → V4 (분류)
        - 출력: V4 → Hippocampus, Amygdala, Dopamine
        - Top-Down: Hunger/Fear/Goal → V2/V4 (주의 조절)
        """
        print("  Phase 9: Building Higher Visual Cortex (V2/V4)...")

        # LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. V2 Edge Populations (에지/윤곽 검출) ===
        self.v2_edge_food = self.model.add_neuron_population(
            "v2_edge_food", self.config.n_v2_edge_food,
            sensory_lif_model, lif_params, lif_init)
        self.v2_edge_danger = self.model.add_neuron_population(
            "v2_edge_danger", self.config.n_v2_edge_danger,
            sensory_lif_model, lif_params, lif_init)

        print(f"    V2_Edge: Food({self.config.n_v2_edge_food}) + Danger({self.config.n_v2_edge_danger})")

        # === 2. V4 Object Populations (물체 분류) ===
        self.v4_food_object = self.model.add_neuron_population(
            "v4_food_object", self.config.n_v4_food_object,
            sensory_lif_model, lif_params, lif_init)
        self.v4_danger_object = self.model.add_neuron_population(
            "v4_danger_object", self.config.n_v4_danger_object,
            sensory_lif_model, lif_params, lif_init)
        self.v4_novel_object = self.model.add_neuron_population(
            "v4_novel_object", self.config.n_v4_novel_object,
            sensory_lif_model, lif_params, lif_init)

        print(f"    V4_Object: Food({self.config.n_v4_food_object}) + "
              f"Danger({self.config.n_v4_danger_object}) + Novel({self.config.n_v4_novel_object})")

        # === 입력 연결: V1 → V2 (좌우 수렴, 크기 불변성) ===
        # V1_Food_Left + V1_Food_Right → V2_Edge_Food
        self._create_static_synapse(
            "v1_food_left_to_v2_edge", self.v1_food_left, self.v2_edge_food,
            self.config.v1_to_v2_weight, sparsity=0.15)
        self._create_static_synapse(
            "v1_food_right_to_v2_edge", self.v1_food_right, self.v2_edge_food,
            self.config.v1_to_v2_weight, sparsity=0.15)

        # V1_Danger_Left + V1_Danger_Right → V2_Edge_Danger
        self._create_static_synapse(
            "v1_danger_left_to_v2_edge", self.v1_danger_left, self.v2_edge_danger,
            self.config.v1_to_v2_weight, sparsity=0.15)
        self._create_static_synapse(
            "v1_danger_right_to_v2_edge", self.v1_danger_right, self.v2_edge_danger,
            self.config.v1_to_v2_weight, sparsity=0.15)

        print(f"    V1→V2 (수렴): {self.config.v1_to_v2_weight}")

        # === 내부 연결: V2 → V4 (분류) ===
        self._create_static_synapse(
            "v2_edge_food_to_v4_food", self.v2_edge_food, self.v4_food_object,
            self.config.v2_to_v4_weight, sparsity=0.2)
        self._create_static_synapse(
            "v2_edge_danger_to_v4_danger", self.v2_edge_danger, self.v4_danger_object,
            self.config.v2_to_v4_weight, sparsity=0.2)

        print(f"    V2→V4 (분류): {self.config.v2_to_v4_weight}")

        # === V4 WTA (Winner-Take-All) ===
        # Food vs Danger vs Novel 경쟁
        self._create_static_synapse(
            "v4_food_to_danger", self.v4_food_object, self.v4_danger_object,
            self.config.v4_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "v4_danger_to_food", self.v4_danger_object, self.v4_food_object,
            self.config.v4_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "v4_food_to_novel", self.v4_food_object, self.v4_novel_object,
            self.config.v4_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "v4_danger_to_novel", self.v4_danger_object, self.v4_novel_object,
            self.config.v4_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "v4_novel_to_food", self.v4_novel_object, self.v4_food_object,
            self.config.v4_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "v4_novel_to_danger", self.v4_novel_object, self.v4_danger_object,
            self.config.v4_wta_inhibition, sparsity=0.1)

        print(f"    V4 WTA (경쟁): {self.config.v4_wta_inhibition}")

        # === Novelty Detection: V1 활성 + V4 비활성 → Novel ===
        # V1이 활성화되었는데 V4 Food/Danger가 비활성이면 = 새로운 물체
        # 구현: V1 → V4_Novel (약한 흥분) + V4_Food/Danger → V4_Novel (강한 억제)
        # 결과: V4_Food/Danger가 활성화되면 Novel은 억제됨
        self._create_static_synapse(
            "v1_food_left_to_v4_novel", self.v1_food_left, self.v4_novel_object,
            self.config.v1_to_v2_weight * 0.3, sparsity=0.1)
        self._create_static_synapse(
            "v1_food_right_to_v4_novel", self.v1_food_right, self.v4_novel_object,
            self.config.v1_to_v2_weight * 0.3, sparsity=0.1)
        self._create_static_synapse(
            "v1_danger_left_to_v4_novel", self.v1_danger_left, self.v4_novel_object,
            self.config.v1_to_v2_weight * 0.3, sparsity=0.1)
        self._create_static_synapse(
            "v1_danger_right_to_v4_novel", self.v1_danger_right, self.v4_novel_object,
            self.config.v1_to_v2_weight * 0.3, sparsity=0.1)

        print(f"    V1→V4_Novel (novelty): {self.config.v1_to_v2_weight * 0.3:.1f}")

        # === 출력 연결: V4 → 상위 영역 ===
        # V4_Food → Hippocampus (음식 기억 강화)
        if self.config.hippocampus_enabled:
            self._create_static_synapse(
                "v4_food_to_hippocampus", self.v4_food_object, self.place_cells,
                self.config.v4_food_to_hippocampus_weight, sparsity=0.1)
            print(f"    V4_Food→Hippocampus: {self.config.v4_food_to_hippocampus_weight}")

        # V4_Food → Hunger Drive (음식 인지 → 배고픔 활성화)
        self._create_static_synapse(
            "v4_food_to_hunger", self.v4_food_object, self.hunger_drive,
            self.config.v4_food_to_hunger_weight, sparsity=0.1)
        print(f"    V4_Food→Hunger: {self.config.v4_food_to_hunger_weight}")

        # V4_Danger → Amygdala (위험 인지 → 공포 활성화)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "v4_danger_to_amygdala", self.v4_danger_object, self.lateral_amygdala,
                self.config.v4_danger_to_amygdala_weight, sparsity=0.12)
            print(f"    V4_Danger→Amygdala: {self.config.v4_danger_to_amygdala_weight}")

        # V4_Novel → Dopamine (새로운 물체 → 호기심/탐색)
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "v4_novel_to_dopamine", self.v4_novel_object, self.dopamine_neurons,
                self.config.v4_novel_to_dopamine_weight, sparsity=0.15)
            print(f"    V4_Novel→Dopamine: {self.config.v4_novel_to_dopamine_weight}")

        # === Top-Down 조절: Hunger/Fear/Goal → V2/V4 ===
        # Hunger → V4_Food (배고플 때 음식 탐지 증가)
        self._create_static_synapse(
            "hunger_to_v4_food", self.hunger_drive, self.v4_food_object,
            self.config.hunger_to_v4_food_weight, sparsity=0.1)
        print(f"    Hunger→V4_Food (top-down): {self.config.hunger_to_v4_food_weight}")

        # Fear → V4_Danger (공포 시 위험 탐지 증가)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_v4_danger", self.fear_response, self.v4_danger_object,
                self.config.fear_to_v4_danger_weight, sparsity=0.1)
            print(f"    Fear→V4_Danger (top-down): {self.config.fear_to_v4_danger_weight}")

        # Goal → V2_Edge (목표에 따른 선택적 주의)
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "goal_food_to_v2_food", self.goal_food, self.v2_edge_food,
                self.config.goal_to_v2_weight, sparsity=0.1)
            self._create_static_synapse(
                "goal_safety_to_v2_danger", self.goal_safety, self.v2_edge_danger,
                self.config.goal_to_v2_weight, sparsity=0.1)
            print(f"    Goal→V2 (attention): {self.config.goal_to_v2_weight}")

        total_v2v4 = (self.config.n_v2_edge_food + self.config.n_v2_edge_danger +
                      self.config.n_v4_food_object + self.config.n_v4_danger_object +
                      self.config.n_v4_novel_object)
        print(f"  V2/V4 circuit complete: {total_v2v4} neurons")

    def _build_it_cortex_circuit(self):
        """
        Phase 10: Inferior Temporal Cortex (IT) 구축

        IT Cortex는 시각 처리의 최상위 단계로:
        - V4에서 입력을 받아 물체의 정체성(identity) 표상
        - 학습을 통해 범주별 뉴런 군집 형성 ("음식", "위험")
        - 해마와 양방향 연결 (기억 저장/인출)

        구성:
        - IT_Food_Category: "음식" 범주 뉴런
        - IT_Danger_Category: "위험" 범주 뉴런
        - IT_Neutral_Category: 중립/미분류 물체
        - IT_Association: 범주 간 연합
        - IT_Memory_Buffer: 단기 물체 기억
        """
        print("  Phase 10: Building Inferior Temporal Cortex (IT)...")

        # LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. IT Category Populations ===
        self.it_food_category = self.model.add_neuron_population(
            "it_food_category", self.config.n_it_food_category,
            sensory_lif_model, lif_params, lif_init)
        self.it_danger_category = self.model.add_neuron_population(
            "it_danger_category", self.config.n_it_danger_category,
            sensory_lif_model, lif_params, lif_init)
        self.it_neutral_category = self.model.add_neuron_population(
            "it_neutral_category", self.config.n_it_neutral_category,
            sensory_lif_model, lif_params, lif_init)

        print(f"    IT_Category: Food({self.config.n_it_food_category}) + "
              f"Danger({self.config.n_it_danger_category}) + "
              f"Neutral({self.config.n_it_neutral_category})")

        # === 2. IT Association & Memory Buffer ===
        self.it_association = self.model.add_neuron_population(
            "it_association", self.config.n_it_association,
            sensory_lif_model, lif_params, lif_init)
        self.it_memory_buffer = self.model.add_neuron_population(
            "it_memory_buffer", self.config.n_it_memory_buffer,
            sensory_lif_model, lif_params, lif_init)

        print(f"    IT_Association: {self.config.n_it_association} neurons")
        print(f"    IT_Memory_Buffer: {self.config.n_it_memory_buffer} neurons")

        # === 입력 연결: V4 → IT (순방향) ===
        self._create_static_synapse(
            "v4_food_to_it_food", self.v4_food_object, self.it_food_category,
            self.config.v4_to_it_weight, sparsity=0.2)
        self._create_static_synapse(
            "v4_danger_to_it_danger", self.v4_danger_object, self.it_danger_category,
            self.config.v4_to_it_weight, sparsity=0.2)
        self._create_static_synapse(
            "v4_novel_to_it_neutral", self.v4_novel_object, self.it_neutral_category,
            self.config.v4_to_it_weight * 0.8, sparsity=0.15)

        print(f"    V4→IT: {self.config.v4_to_it_weight}")

        # === IT ↔ Hippocampus (양방향) ===
        if self.config.hippocampus_enabled:
            # IT → Hippocampus (음식 범주 기억 저장)
            self._create_static_synapse(
                "it_food_to_place_cells", self.it_food_category, self.place_cells,
                self.config.it_to_hippocampus_weight, sparsity=0.1)

            # Hippocampus → IT (기억 기반 범주 활성화)
            self._create_static_synapse(
                "place_cells_to_it_food", self.place_cells, self.it_food_category,
                self.config.hippocampus_to_it_weight, sparsity=0.1)

            # Food Memory → IT_Food (음식 기억 → 음식 범주)
            if self.config.directional_food_memory:
                self._create_static_synapse(
                    "food_mem_left_to_it_food", self.food_memory_left, self.it_food_category,
                    self.config.hippocampus_to_it_weight, sparsity=0.12)
                self._create_static_synapse(
                    "food_mem_right_to_it_food", self.food_memory_right, self.it_food_category,
                    self.config.hippocampus_to_it_weight, sparsity=0.12)

            print(f"    IT↔Hippocampus: {self.config.it_to_hippocampus_weight}/{self.config.hippocampus_to_it_weight}")

        # === IT ↔ Amygdala (양방향) ===
        if self.config.amygdala_enabled:
            # IT_Danger → Amygdala (위험 범주 → 공포)
            self._create_static_synapse(
                "it_danger_to_la", self.it_danger_category, self.lateral_amygdala,
                self.config.it_to_amygdala_weight, sparsity=0.12)

            # Fear → IT_Danger (공포 → 위험 인식 강화)
            self._create_static_synapse(
                "fear_to_it_danger", self.fear_response, self.it_danger_category,
                self.config.amygdala_to_it_weight, sparsity=0.1)

            print(f"    IT↔Amygdala: {self.config.it_to_amygdala_weight}/{self.config.amygdala_to_it_weight}")

        # === IT → Motor (행동 출력) ===
        # IT_Food → Motor (ipsilateral: 음식 쪽으로)
        self._create_static_synapse(
            "it_food_to_motor_left", self.it_food_category, self.motor_left,
            self.config.it_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "it_food_to_motor_right", self.it_food_category, self.motor_right,
            self.config.it_to_motor_weight, sparsity=0.1)

        # IT_Danger → Motor (contralateral: 회피)
        self._create_static_synapse(
            "it_danger_to_motor_left", self.it_danger_category, self.motor_right,
            self.config.it_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "it_danger_to_motor_right", self.it_danger_category, self.motor_left,
            self.config.it_to_motor_weight, sparsity=0.1)

        print(f"    IT→Motor: {self.config.it_to_motor_weight}")

        # === IT → PFC (목표 설정) ===
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "it_food_to_goal_food", self.it_food_category, self.goal_food,
                self.config.it_to_pfc_weight, sparsity=0.12)
            self._create_static_synapse(
                "it_danger_to_goal_safety", self.it_danger_category, self.goal_safety,
                self.config.it_to_pfc_weight, sparsity=0.12)

            print(f"    IT→PFC Goal: {self.config.it_to_pfc_weight}")

        # === IT 내부 WTA (범주 간 경쟁) ===
        self._create_static_synapse(
            "it_food_to_danger", self.it_food_category, self.it_danger_category,
            self.config.it_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "it_danger_to_food", self.it_danger_category, self.it_food_category,
            self.config.it_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "it_food_to_neutral", self.it_food_category, self.it_neutral_category,
            self.config.it_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "it_danger_to_neutral", self.it_danger_category, self.it_neutral_category,
            self.config.it_wta_inhibition, sparsity=0.1)

        print(f"    IT WTA: {self.config.it_wta_inhibition}")

        # === IT Category → Association ===
        self._create_static_synapse(
            "it_food_to_assoc", self.it_food_category, self.it_association,
            15.0, sparsity=0.12)
        self._create_static_synapse(
            "it_danger_to_assoc", self.it_danger_category, self.it_association,
            15.0, sparsity=0.12)

        # === IT Memory Buffer 연결 ===
        # Categories → Buffer (단기 저장)
        self._create_static_synapse(
            "it_food_to_buffer", self.it_food_category, self.it_memory_buffer,
            12.0, sparsity=0.1)
        self._create_static_synapse(
            "it_danger_to_buffer", self.it_danger_category, self.it_memory_buffer,
            12.0, sparsity=0.1)

        # Buffer → Categories (인출)
        self._create_static_synapse(
            "buffer_to_it_food", self.it_memory_buffer, self.it_food_category,
            10.0, sparsity=0.08)
        self._create_static_synapse(
            "buffer_to_it_danger", self.it_memory_buffer, self.it_danger_category,
            10.0, sparsity=0.08)

        print(f"    IT↔Buffer: 12.0/10.0")

        # === Top-Down 조절 ===
        # Hunger → IT_Food (배고플 때 음식 범주 민감도 증가)
        self._create_static_synapse(
            "hunger_to_it_food", self.hunger_drive, self.it_food_category,
            self.config.hunger_to_it_food_weight, sparsity=0.1)

        # Fear → IT_Danger (공포 시 위험 범주 민감도 증가)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_it_danger_topdown", self.fear_response, self.it_danger_category,
                self.config.fear_to_it_danger_weight, sparsity=0.1)

        # Working Memory → IT_Buffer (작업 기억 유지)
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "wm_to_it_buffer", self.working_memory, self.it_memory_buffer,
                self.config.wm_to_it_buffer_weight, sparsity=0.1)

        print(f"    Top-Down: Hunger→IT_Food {self.config.hunger_to_it_food_weight}")

        total_it = (self.config.n_it_food_category + self.config.n_it_danger_category +
                    self.config.n_it_neutral_category + self.config.n_it_association +
                    self.config.n_it_memory_buffer)
        print(f"  IT Cortex complete: {total_it} neurons")
        print(f"  *** M1 Milestone: Total neurons now = {self.config.total_neurons:,} ***")

    def _build_auditory_cortex_circuit(self):
        """
        Phase 11: Auditory Cortex (청각 피질) 구축

        청각 경로:
        - Sound Input (L/R) → A1 (Primary Auditory Cortex) → A2 (Association)
        - A1 → Amygdala (청각-공포 경로)
        - A1 → IT (청각-시각 통합)
        - A1 → Motor (청각 유도 행동)

        구성:
        - Sound_Danger_L/R: 위험 소리 입력
        - Sound_Food_L/R: 음식 소리 입력
        - A1_Danger: 위험 소리 처리
        - A1_Food: 음식 소리 처리
        - A2_Association: 청각 연합 영역
        """
        print("  Phase 11: Building Auditory Cortex...")

        # LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. Sound Input Populations ===
        self.sound_danger_left = self.model.add_neuron_population(
            "sound_danger_left", self.config.n_sound_danger_left,
            sensory_lif_model, lif_params, lif_init)
        self.sound_danger_right = self.model.add_neuron_population(
            "sound_danger_right", self.config.n_sound_danger_right,
            sensory_lif_model, lif_params, lif_init)
        self.sound_food_left = self.model.add_neuron_population(
            "sound_food_left", self.config.n_sound_food_left,
            sensory_lif_model, lif_params, lif_init)
        self.sound_food_right = self.model.add_neuron_population(
            "sound_food_right", self.config.n_sound_food_right,
            sensory_lif_model, lif_params, lif_init)

        print(f"    Sound Input: Danger L/R({self.config.n_sound_danger_left}x2) + "
              f"Food L/R({self.config.n_sound_food_left}x2)")

        # === 2. A1 (Primary Auditory Cortex) ===
        self.a1_danger = self.model.add_neuron_population(
            "a1_danger", self.config.n_a1_danger,
            sensory_lif_model, lif_params, lif_init)
        self.a1_food = self.model.add_neuron_population(
            "a1_food", self.config.n_a1_food,
            sensory_lif_model, lif_params, lif_init)

        print(f"    A1: Danger({self.config.n_a1_danger}) + Food({self.config.n_a1_food})")

        # === 3. A2 Association ===
        self.a2_association = self.model.add_neuron_population(
            "a2_association", self.config.n_a2_association,
            sensory_lif_model, lif_params, lif_init)

        print(f"    A2_Association: {self.config.n_a2_association} neurons")

        # === Sound Input → A1 (순방향) ===
        # Sound_Danger L/R → A1_Danger (좌우 수렴)
        self._create_static_synapse(
            "sound_danger_left_to_a1", self.sound_danger_left, self.a1_danger,
            self.config.sound_to_a1_weight, sparsity=0.15)
        self._create_static_synapse(
            "sound_danger_right_to_a1", self.sound_danger_right, self.a1_danger,
            self.config.sound_to_a1_weight, sparsity=0.15)

        # Sound_Food L/R → A1_Food (좌우 수렴)
        self._create_static_synapse(
            "sound_food_left_to_a1", self.sound_food_left, self.a1_food,
            self.config.sound_to_a1_weight, sparsity=0.15)
        self._create_static_synapse(
            "sound_food_right_to_a1", self.sound_food_right, self.a1_food,
            self.config.sound_to_a1_weight, sparsity=0.15)

        print(f"    Sound→A1: {self.config.sound_to_a1_weight}")

        # === A1 Lateral Inhibition (좌우 경쟁) ===
        # Sound Input 단계에서 좌우 경쟁
        self._create_static_synapse(
            "sound_danger_left_to_right", self.sound_danger_left, self.sound_danger_right,
            self.config.a1_lateral_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "sound_danger_right_to_left", self.sound_danger_right, self.sound_danger_left,
            self.config.a1_lateral_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "sound_food_left_to_right", self.sound_food_left, self.sound_food_right,
            self.config.a1_lateral_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "sound_food_right_to_left", self.sound_food_right, self.sound_food_left,
            self.config.a1_lateral_inhibition, sparsity=0.1)

        print(f"    Sound Lateral Inhibition: {self.config.a1_lateral_inhibition}")

        # === A1 → Amygdala (청각-공포 경로) ===
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "a1_danger_to_la", self.a1_danger, self.lateral_amygdala,
                self.config.a1_danger_to_amygdala_weight, sparsity=0.12)
            print(f"    A1_Danger→Amygdala LA: {self.config.a1_danger_to_amygdala_weight} (fast fear)")

        # === A1 → IT (청각-시각 통합) ===
        if self.config.it_enabled:
            self._create_static_synapse(
                "a1_danger_to_it_danger", self.a1_danger, self.it_danger_category,
                self.config.a1_to_it_weight, sparsity=0.1)
            self._create_static_synapse(
                "a1_food_to_it_food", self.a1_food, self.it_food_category,
                self.config.a1_to_it_weight, sparsity=0.1)
            print(f"    A1→IT: {self.config.a1_to_it_weight} (multimodal)")

        # === A1 → Motor (청각 유도 행동) ===
        # A1_Danger: 반대편 모터 활성화 (회피)
        # Sound_Danger_Left → Motor_Right (왼쪽 위험 소리 → 오른쪽 회피)
        self._create_static_synapse(
            "sound_danger_left_to_motor_right", self.sound_danger_left, self.motor_right,
            self.config.a1_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "sound_danger_right_to_motor_left", self.sound_danger_right, self.motor_left,
            self.config.a1_to_motor_weight, sparsity=0.1)

        # A1_Food: 같은편 모터 활성화 (접근)
        # Sound_Food_Left → Motor_Left (왼쪽 음식 소리 → 왼쪽 접근)
        self._create_static_synapse(
            "sound_food_left_to_motor_left", self.sound_food_left, self.motor_left,
            self.config.a1_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "sound_food_right_to_motor_right", self.sound_food_right, self.motor_right,
            self.config.a1_to_motor_weight, sparsity=0.1)

        print(f"    Sound→Motor: {self.config.a1_to_motor_weight} (directional)")

        # === A1 → A2 Association ===
        self._create_static_synapse(
            "a1_danger_to_a2", self.a1_danger, self.a2_association,
            self.config.a1_to_a2_weight, sparsity=0.12)
        self._create_static_synapse(
            "a1_food_to_a2", self.a1_food, self.a2_association,
            self.config.a1_to_a2_weight, sparsity=0.12)

        # IT → A2 (다감각 통합)
        if self.config.it_enabled:
            self._create_static_synapse(
                "it_food_to_a2", self.it_food_category, self.a2_association,
                self.config.it_to_a2_weight, sparsity=0.1)
            self._create_static_synapse(
                "it_danger_to_a2", self.it_danger_category, self.a2_association,
                self.config.it_to_a2_weight, sparsity=0.1)

        print(f"    A1/IT→A2: {self.config.a1_to_a2_weight}/{self.config.it_to_a2_weight}")

        # === Top-Down 조절 ===
        # Fear → A1_Danger (공포 시 위험 소리 민감도 증가)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_a1_danger", self.fear_response, self.a1_danger,
                self.config.fear_to_a1_danger_weight, sparsity=0.1)

        # Hunger → A1_Food (배고플 때 음식 소리 민감도 증가)
        self._create_static_synapse(
            "hunger_to_a1_food", self.hunger_drive, self.a1_food,
            self.config.hunger_to_a1_food_weight, sparsity=0.1)

        print(f"    Top-Down: Fear→A1_Danger {self.config.fear_to_a1_danger_weight}, "
              f"Hunger→A1_Food {self.config.hunger_to_a1_food_weight}")

        total_auditory = (self.config.n_sound_danger_left + self.config.n_sound_danger_right +
                         self.config.n_sound_food_left + self.config.n_sound_food_right +
                         self.config.n_a1_danger + self.config.n_a1_food +
                         self.config.n_a2_association)
        print(f"  Auditory Cortex complete: {total_auditory} neurons")

    def _build_multimodal_integration_circuit(self):
        """
        Phase 12: Multimodal Integration (다중 감각 통합) 구축

        상측두고랑 (STS) 모델링:
        - 시각 (IT) + 청각 (A1/A2) 통합
        - 시청각 일치/불일치 감지
        - 통합된 감각 정보 → Hippocampus/Amygdala/Motor/PFC

        구성:
        - STS_Food: 음식 관련 시청각 통합
        - STS_Danger: 위험 관련 시청각 통합
        - STS_Congruence: 일치 감지 (신뢰도 증가)
        - STS_Mismatch: 불일치 감지 (주의 증가)
        - Multimodal_Buffer: 다중 감각 작업 기억
        """
        print("  Phase 12: Building Multimodal Integration (STS)...")

        # LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. STS Populations ===
        self.sts_food = self.model.add_neuron_population(
            "sts_food", self.config.n_sts_food,
            sensory_lif_model, lif_params, lif_init)
        self.sts_danger = self.model.add_neuron_population(
            "sts_danger", self.config.n_sts_danger,
            sensory_lif_model, lif_params, lif_init)
        self.sts_congruence = self.model.add_neuron_population(
            "sts_congruence", self.config.n_sts_congruence,
            sensory_lif_model, lif_params, lif_init)
        self.sts_mismatch = self.model.add_neuron_population(
            "sts_mismatch", self.config.n_sts_mismatch,
            sensory_lif_model, lif_params, lif_init)
        self.multimodal_buffer = self.model.add_neuron_population(
            "multimodal_buffer", self.config.n_multimodal_buffer,
            sensory_lif_model, lif_params, lif_init)

        print(f"    STS: Food({self.config.n_sts_food}) + Danger({self.config.n_sts_danger}) + "
              f"Congruence({self.config.n_sts_congruence}) + Mismatch({self.config.n_sts_mismatch})")
        print(f"    Multimodal Buffer: {self.config.n_multimodal_buffer}")

        # === 2. 시각 → STS (IT Cortex에서) ===
        self._create_static_synapse(
            "it_food_to_sts_food", self.it_food_category, self.sts_food,
            self.config.it_to_sts_weight, sparsity=0.12)
        self._create_static_synapse(
            "it_danger_to_sts_danger", self.it_danger_category, self.sts_danger,
            self.config.it_to_sts_weight, sparsity=0.12)

        print(f"    Visual→STS (IT): {self.config.it_to_sts_weight}")

        # === 3. 청각 → STS (A1/A2에서) ===
        self._create_static_synapse(
            "a1_food_to_sts_food", self.a1_food, self.sts_food,
            self.config.a1_to_sts_weight, sparsity=0.12)
        self._create_static_synapse(
            "a1_danger_to_sts_danger", self.a1_danger, self.sts_danger,
            self.config.a1_to_sts_weight, sparsity=0.12)
        self._create_static_synapse(
            "a2_to_sts_food", self.a2_association, self.sts_food,
            self.config.a2_to_sts_weight, sparsity=0.1)
        self._create_static_synapse(
            "a2_to_sts_danger", self.a2_association, self.sts_danger,
            self.config.a2_to_sts_weight, sparsity=0.1)

        print(f"    Auditory→STS (A1/A2): {self.config.a1_to_sts_weight}/{self.config.a2_to_sts_weight}")

        # === 4. STS 내부 연결 ===

        # 4.1 일치 감지 (Congruence Detection)
        # STS_Food → Congruence (음식 시청각 일치)
        self._create_static_synapse(
            "sts_food_to_congruence", self.sts_food, self.sts_congruence,
            self.config.sts_congruence_weight, sparsity=0.15)
        # STS_Danger → Congruence (위험 시청각 일치)
        self._create_static_synapse(
            "sts_danger_to_congruence", self.sts_danger, self.sts_congruence,
            self.config.sts_congruence_weight, sparsity=0.15)

        print(f"    Congruence Detection: {self.config.sts_congruence_weight}")

        # 4.2 불일치 감지 (Mismatch Detection)
        # IT_Food + A1_Danger → Mismatch (시각 음식 + 청각 위험 = 불일치)
        self._create_static_synapse(
            "it_food_to_mismatch", self.it_food_category, self.sts_mismatch,
            self.config.sts_mismatch_weight * 0.5, sparsity=0.08)
        self._create_static_synapse(
            "a1_danger_to_mismatch", self.a1_danger, self.sts_mismatch,
            self.config.sts_mismatch_weight * 0.5, sparsity=0.08)
        # IT_Danger + A1_Food → Mismatch (시각 위험 + 청각 음식 = 불일치)
        self._create_static_synapse(
            "it_danger_to_mismatch", self.it_danger_category, self.sts_mismatch,
            self.config.sts_mismatch_weight * 0.5, sparsity=0.08)
        self._create_static_synapse(
            "a1_food_to_mismatch", self.a1_food, self.sts_mismatch,
            self.config.sts_mismatch_weight * 0.5, sparsity=0.08)

        print(f"    Mismatch Detection: {self.config.sts_mismatch_weight}")

        # 4.3 WTA 경쟁
        # STS_Food ↔ STS_Danger (상호 억제)
        self._create_static_synapse(
            "sts_food_to_danger_wta", self.sts_food, self.sts_danger,
            self.config.sts_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "sts_danger_to_food_wta", self.sts_danger, self.sts_food,
            self.config.sts_wta_inhibition, sparsity=0.1)
        # Congruence ↔ Mismatch (상호 억제)
        self._create_static_synapse(
            "congruence_to_mismatch_wta", self.sts_congruence, self.sts_mismatch,
            self.config.sts_wta_inhibition * 1.5, sparsity=0.15)
        self._create_static_synapse(
            "mismatch_to_congruence_wta", self.sts_mismatch, self.sts_congruence,
            self.config.sts_wta_inhibition * 1.5, sparsity=0.15)

        print(f"    STS WTA: {self.config.sts_wta_inhibition}")

        # === 5. STS → Hippocampus ===
        if self.config.hippocampus_enabled:
            self._create_static_synapse(
                "sts_food_to_food_memory_l", self.sts_food, self.food_memory_left,
                self.config.sts_to_hippocampus_weight, sparsity=0.1)
            self._create_static_synapse(
                "sts_food_to_food_memory_r", self.sts_food, self.food_memory_right,
                self.config.sts_to_hippocampus_weight, sparsity=0.1)
            # Congruence → Place Cells (일치 시 기억 강화)
            self._create_static_synapse(
                "congruence_to_place", self.sts_congruence, self.place_cells,
                self.config.sts_to_hippocampus_weight * 0.5, sparsity=0.05)

            print(f"    STS→Hippocampus: {self.config.sts_to_hippocampus_weight}")

        # === 6. STS → Amygdala ===
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "sts_danger_to_la", self.sts_danger, self.lateral_amygdala,
                self.config.sts_to_amygdala_weight, sparsity=0.1)
            # Mismatch → LA (불일치 = 경계)
            self._create_static_synapse(
                "mismatch_to_la", self.sts_mismatch, self.lateral_amygdala,
                self.config.sts_to_amygdala_weight * 0.6, sparsity=0.08)

            print(f"    STS→Amygdala: {self.config.sts_to_amygdala_weight}")

        # === 7. STS → Motor ===
        # STS_Food → Motor (ipsi, 통합된 음식 방향)
        self._create_static_synapse(
            "sts_food_to_motor_left", self.sts_food, self.motor_left,
            self.config.sts_to_motor_weight, sparsity=0.08)
        self._create_static_synapse(
            "sts_food_to_motor_right", self.sts_food, self.motor_right,
            self.config.sts_to_motor_weight, sparsity=0.08)
        # STS_Danger → Motor (contra, 통합된 위험 회피)
        self._create_static_synapse(
            "sts_danger_to_motor_left", self.sts_danger, self.motor_right,
            self.config.sts_to_motor_weight, sparsity=0.08)
        self._create_static_synapse(
            "sts_danger_to_motor_right", self.sts_danger, self.motor_left,
            self.config.sts_to_motor_weight, sparsity=0.08)

        print(f"    STS→Motor: {self.config.sts_to_motor_weight}")

        # === 8. STS → PFC ===
        if self.config.prefrontal_enabled:
            # Congruence → Working Memory (확실한 정보)
            self._create_static_synapse(
                "congruence_to_wm", self.sts_congruence, self.working_memory,
                self.config.sts_to_pfc_weight, sparsity=0.1)
            # Mismatch → Goal_Safety (불확실 = 안전 우선)
            self._create_static_synapse(
                "mismatch_to_goal_safety", self.sts_mismatch, self.goal_safety,
                self.config.sts_to_pfc_weight * 1.2, sparsity=0.1)

            print(f"    STS→PFC: {self.config.sts_to_pfc_weight}")

        # === 9. STS → Multimodal Buffer ===
        self._create_static_synapse(
            "sts_food_to_buffer", self.sts_food, self.multimodal_buffer,
            12.0, sparsity=0.1)
        self._create_static_synapse(
            "sts_danger_to_buffer", self.sts_danger, self.multimodal_buffer,
            12.0, sparsity=0.1)
        self._create_static_synapse(
            "congruence_to_buffer", self.sts_congruence, self.multimodal_buffer,
            15.0, sparsity=0.12)

        print(f"    STS→Multimodal Buffer: 12-15")

        # === 10. Top-Down 조절 ===
        # Hunger → STS_Food (배고플 때 음식 통합 민감도 증가)
        self._create_static_synapse(
            "hunger_to_sts_food", self.hunger_drive, self.sts_food,
            self.config.hunger_to_sts_weight, sparsity=0.08)

        # Fear → STS_Danger (공포 시 위험 통합 민감도 증가)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_sts_danger", self.fear_response, self.sts_danger,
                self.config.fear_to_sts_weight, sparsity=0.08)

        # Working Memory → Congruence (목표 집중 시 일치 감지 강화)
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "wm_to_congruence", self.working_memory, self.sts_congruence,
                self.config.wm_to_sts_congruence_weight, sparsity=0.08)

        print(f"    Top-Down: Hunger→STS_Food {self.config.hunger_to_sts_weight}, "
              f"Fear→STS_Danger {self.config.fear_to_sts_weight}")

        total_multimodal = (self.config.n_sts_food + self.config.n_sts_danger +
                          self.config.n_sts_congruence + self.config.n_sts_mismatch +
                          self.config.n_multimodal_buffer)
        print(f"  Multimodal Integration complete: {total_multimodal} neurons")
        print(f"  Total neurons now = {self.config.total_neurons:,}")

    def _build_parietal_cortex_circuit(self):
        """
        Phase 13: Parietal Cortex (두정엽) 구축

        후두정 피질 (PPC) 모델링:
        - 공간 표상: 시각/청각/체감각 통합
        - 목표 벡터: 현재 위치 → 목표 위치 방향 계산
        - 공간 주의: 중요한 위치에 선택적 주의 배분
        - 경로 계획: 연속적 행동 시퀀스 생성 기초

        구성:
        - PPC_Space_Left/Right: 좌우 공간 표상
        - PPC_Goal_Food/Safety: 음식/안전 목표 벡터
        - PPC_Attention: 공간 주의 조절
        - PPC_Path_Buffer: 경로 계획 버퍼
        """
        print("  Phase 13: Building Parietal Cortex (PPC)...")

        # LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. PPC Populations ===
        self.ppc_space_left = self.model.add_neuron_population(
            "ppc_space_left", self.config.n_ppc_space_left,
            sensory_lif_model, lif_params, lif_init)
        self.ppc_space_right = self.model.add_neuron_population(
            "ppc_space_right", self.config.n_ppc_space_right,
            sensory_lif_model, lif_params, lif_init)
        self.ppc_goal_food = self.model.add_neuron_population(
            "ppc_goal_food", self.config.n_ppc_goal_food,
            sensory_lif_model, lif_params, lif_init)
        self.ppc_goal_safety = self.model.add_neuron_population(
            "ppc_goal_safety", self.config.n_ppc_goal_safety,
            sensory_lif_model, lif_params, lif_init)
        self.ppc_attention = self.model.add_neuron_population(
            "ppc_attention", self.config.n_ppc_attention,
            sensory_lif_model, lif_params, lif_init)
        self.ppc_path_buffer = self.model.add_neuron_population(
            "ppc_path_buffer", self.config.n_ppc_path_buffer,
            sensory_lif_model, lif_params, lif_init)

        print(f"    PPC_Space: Left({self.config.n_ppc_space_left}) + Right({self.config.n_ppc_space_right})")
        print(f"    PPC_Goal: Food({self.config.n_ppc_goal_food}) + Safety({self.config.n_ppc_goal_safety})")
        print(f"    PPC_Attention: {self.config.n_ppc_attention}, Path_Buffer: {self.config.n_ppc_path_buffer}")

        # === 2. 감각 → PPC_Space (공간 입력) ===

        # 2.1 V1 → PPC_Space (시각 위치)
        if self.config.v1_enabled:
            self._create_static_synapse(
                "v1_food_left_to_ppc_left", self.v1_food_left, self.ppc_space_left,
                self.config.v1_to_ppc_weight, sparsity=0.1)
            self._create_static_synapse(
                "v1_food_right_to_ppc_right", self.v1_food_right, self.ppc_space_right,
                self.config.v1_to_ppc_weight, sparsity=0.1)
            self._create_static_synapse(
                "v1_danger_left_to_ppc_left", self.v1_danger_left, self.ppc_space_left,
                self.config.v1_to_ppc_weight, sparsity=0.1)
            self._create_static_synapse(
                "v1_danger_right_to_ppc_right", self.v1_danger_right, self.ppc_space_right,
                self.config.v1_to_ppc_weight, sparsity=0.1)

            print(f"    V1→PPC_Space: {self.config.v1_to_ppc_weight}")

        # 2.2 IT → PPC_Space (물체 인식 기반 위치)
        if self.config.it_enabled:
            self._create_static_synapse(
                "it_food_to_ppc_left", self.it_food_category, self.ppc_space_left,
                self.config.it_to_ppc_weight, sparsity=0.08)
            self._create_static_synapse(
                "it_food_to_ppc_right", self.it_food_category, self.ppc_space_right,
                self.config.it_to_ppc_weight, sparsity=0.08)
            self._create_static_synapse(
                "it_danger_to_ppc_left", self.it_danger_category, self.ppc_space_left,
                self.config.it_to_ppc_weight, sparsity=0.08)
            self._create_static_synapse(
                "it_danger_to_ppc_right", self.it_danger_category, self.ppc_space_right,
                self.config.it_to_ppc_weight, sparsity=0.08)

            print(f"    IT→PPC_Space: {self.config.it_to_ppc_weight}")

        # 2.3 STS → PPC_Space (다감각 위치)
        if self.config.multimodal_enabled:
            self._create_static_synapse(
                "sts_food_to_ppc_left", self.sts_food, self.ppc_space_left,
                self.config.sts_to_ppc_weight, sparsity=0.1)
            self._create_static_synapse(
                "sts_food_to_ppc_right", self.sts_food, self.ppc_space_right,
                self.config.sts_to_ppc_weight, sparsity=0.1)
            self._create_static_synapse(
                "sts_danger_to_ppc_left", self.sts_danger, self.ppc_space_left,
                self.config.sts_to_ppc_weight, sparsity=0.1)
            self._create_static_synapse(
                "sts_danger_to_ppc_right", self.sts_danger, self.ppc_space_right,
                self.config.sts_to_ppc_weight, sparsity=0.1)

            print(f"    STS→PPC_Space: {self.config.sts_to_ppc_weight}")

        # 2.4 Hippocampus → PPC_Space (자기 위치, 기억된 음식 위치)
        if self.config.hippocampus_enabled:
            # Place Cells → PPC_Space (왼쪽 Place Cells → 왼쪽 공간, 오른쪽도 마찬가지)
            self._create_static_synapse(
                "place_to_ppc_left", self.place_cells, self.ppc_space_left,
                self.config.place_to_ppc_weight, sparsity=0.08)
            self._create_static_synapse(
                "place_to_ppc_right", self.place_cells, self.ppc_space_right,
                self.config.place_to_ppc_weight, sparsity=0.08)

            # Food Memory → PPC_Space (기억된 음식 위치)
            if self.config.directional_food_memory:
                self._create_static_synapse(
                    "food_mem_left_to_ppc_left", self.food_memory_left, self.ppc_space_left,
                    self.config.food_memory_to_ppc_weight, sparsity=0.1)
                self._create_static_synapse(
                    "food_mem_right_to_ppc_right", self.food_memory_right, self.ppc_space_right,
                    self.config.food_memory_to_ppc_weight, sparsity=0.1)

            print(f"    Hippocampus→PPC_Space: {self.config.place_to_ppc_weight}")

        # === 3. PFC → PPC (목표 설정) ===
        if self.config.prefrontal_enabled:
            # Goal_Food → PPC_Goal_Food
            self._create_static_synapse(
                "pfc_goal_food_to_ppc", self.goal_food, self.ppc_goal_food,
                self.config.goal_to_ppc_weight, sparsity=0.15)
            # Goal_Safety → PPC_Goal_Safety
            self._create_static_synapse(
                "pfc_goal_safety_to_ppc", self.goal_safety, self.ppc_goal_safety,
                self.config.goal_to_ppc_weight, sparsity=0.15)
            # Working Memory → Path Buffer
            self._create_static_synapse(
                "wm_to_ppc_path", self.working_memory, self.ppc_path_buffer,
                self.config.wm_to_ppc_path_weight, sparsity=0.1)

            print(f"    PFC→PPC: Goal({self.config.goal_to_ppc_weight}), WM→Path({self.config.wm_to_ppc_path_weight})")

        # === 4. PPC 내부 연결 ===

        # 4.1 공간-목표 통합 (Space + Goal → Goal Vector)
        # PPC_Space_Left + Goal_Food → PPC_Goal_Food (왼쪽에 음식 목표)
        self._create_static_synapse(
            "ppc_left_to_goal_food", self.ppc_space_left, self.ppc_goal_food,
            self.config.ppc_space_goal_integration_weight, sparsity=0.12)
        self._create_static_synapse(
            "ppc_right_to_goal_food", self.ppc_space_right, self.ppc_goal_food,
            self.config.ppc_space_goal_integration_weight, sparsity=0.12)

        # PPC_Space + Goal_Safety → PPC_Goal_Safety (위험 반대 방향)
        self._create_static_synapse(
            "ppc_left_to_goal_safety", self.ppc_space_left, self.ppc_goal_safety,
            self.config.ppc_space_goal_integration_weight * 0.8, sparsity=0.1)
        self._create_static_synapse(
            "ppc_right_to_goal_safety", self.ppc_space_right, self.ppc_goal_safety,
            self.config.ppc_space_goal_integration_weight * 0.8, sparsity=0.1)

        print(f"    Space-Goal Integration: {self.config.ppc_space_goal_integration_weight}")

        # 4.2 경로 계획 (Path Buffer 연결)
        self._create_static_synapse(
            "goal_food_to_path", self.ppc_goal_food, self.ppc_path_buffer,
            self.config.ppc_path_recurrent_weight, sparsity=0.1)
        self._create_static_synapse(
            "goal_safety_to_path", self.ppc_goal_safety, self.ppc_path_buffer,
            self.config.ppc_path_recurrent_weight, sparsity=0.1)
        # Path Buffer 자기 유지 (재귀 연결)
        self._create_static_synapse(
            "path_buffer_recurrent", self.ppc_path_buffer, self.ppc_path_buffer,
            self.config.ppc_path_recurrent_weight * 0.5, sparsity=0.05)

        print(f"    Path Buffer: {self.config.ppc_path_recurrent_weight}")

        # 4.3 WTA 경쟁
        # PPC_Space_Left ↔ PPC_Space_Right (좌우 경쟁)
        self._create_static_synapse(
            "ppc_left_right_wta", self.ppc_space_left, self.ppc_space_right,
            self.config.ppc_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "ppc_right_left_wta", self.ppc_space_right, self.ppc_space_left,
            self.config.ppc_wta_inhibition, sparsity=0.1)

        # PPC_Goal_Food ↔ PPC_Goal_Safety (목표 경쟁)
        self._create_static_synapse(
            "ppc_goal_food_safety_wta", self.ppc_goal_food, self.ppc_goal_safety,
            self.config.ppc_wta_inhibition * 1.2, sparsity=0.12)
        self._create_static_synapse(
            "ppc_goal_safety_food_wta", self.ppc_goal_safety, self.ppc_goal_food,
            self.config.ppc_wta_inhibition * 1.2, sparsity=0.12)

        print(f"    PPC WTA: {self.config.ppc_wta_inhibition}")

        # 4.4 주의 조절 (Attention)
        self._create_static_synapse(
            "goal_food_to_attention", self.ppc_goal_food, self.ppc_attention,
            self.config.ppc_attention_weight, sparsity=0.1)
        self._create_static_synapse(
            "goal_safety_to_attention", self.ppc_goal_safety, self.ppc_attention,
            self.config.ppc_attention_weight * 1.2, sparsity=0.1)  # 안전 목표 시 주의 더 강화

        # Amygdala Fear → Attention (공포 시 주의 강화)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_ppc_attention", self.fear_response, self.ppc_attention,
                self.config.ppc_attention_weight * 1.5, sparsity=0.12)

        print(f"    Attention: {self.config.ppc_attention_weight}")

        # === 5. PPC → Motor (공간 유도 행동) ===

        # 5.1 PPC_Goal_Food → Motor (음식 방향 이동)
        # 왼쪽 공간 + 음식 목표 → 왼쪽 모터
        self._create_static_synapse(
            "ppc_goal_food_to_motor_left", self.ppc_goal_food, self.motor_left,
            self.config.ppc_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "ppc_goal_food_to_motor_right", self.ppc_goal_food, self.motor_right,
            self.config.ppc_to_motor_weight, sparsity=0.1)

        # 5.2 PPC_Goal_Safety → Motor (위험 반대 방향)
        # Safety 목표는 위험의 반대 방향으로 이동
        self._create_static_synapse(
            "ppc_goal_safety_to_motor_left", self.ppc_goal_safety, self.motor_right,
            self.config.ppc_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "ppc_goal_safety_to_motor_right", self.ppc_goal_safety, self.motor_left,
            self.config.ppc_to_motor_weight, sparsity=0.1)

        # 5.3 PPC_Path_Buffer → Motor (경로 실행)
        self._create_static_synapse(
            "ppc_path_to_motor_left", self.ppc_path_buffer, self.motor_left,
            self.config.ppc_to_motor_weight * 0.7, sparsity=0.08)
        self._create_static_synapse(
            "ppc_path_to_motor_right", self.ppc_path_buffer, self.motor_right,
            self.config.ppc_to_motor_weight * 0.7, sparsity=0.08)

        print(f"    PPC→Motor: {self.config.ppc_to_motor_weight}")

        # === 6. PPC → V1/STS (Top-Down 주의) ===
        if self.config.v1_enabled:
            # PPC_Attention → V1 (시각 처리 강화)
            self._create_static_synapse(
                "ppc_attention_to_v1_food_left", self.ppc_attention, self.v1_food_left,
                self.config.ppc_to_v1_attention_weight, sparsity=0.08)
            self._create_static_synapse(
                "ppc_attention_to_v1_food_right", self.ppc_attention, self.v1_food_right,
                self.config.ppc_to_v1_attention_weight, sparsity=0.08)
            self._create_static_synapse(
                "ppc_attention_to_v1_danger_left", self.ppc_attention, self.v1_danger_left,
                self.config.ppc_to_v1_attention_weight, sparsity=0.08)
            self._create_static_synapse(
                "ppc_attention_to_v1_danger_right", self.ppc_attention, self.v1_danger_right,
                self.config.ppc_to_v1_attention_weight, sparsity=0.08)

            print(f"    PPC→V1 (Top-Down): {self.config.ppc_to_v1_attention_weight}")

        if self.config.multimodal_enabled:
            # PPC_Attention → STS (다감각 주의 조절)
            self._create_static_synapse(
                "ppc_attention_to_sts_food", self.ppc_attention, self.sts_food,
                self.config.ppc_to_sts_attention_weight, sparsity=0.08)
            self._create_static_synapse(
                "ppc_attention_to_sts_danger", self.ppc_attention, self.sts_danger,
                self.config.ppc_to_sts_attention_weight, sparsity=0.08)

            print(f"    PPC→STS (Top-Down): {self.config.ppc_to_sts_attention_weight}")

        # === 7. PPC → Hippocampus (공간 기억) ===
        if self.config.hippocampus_enabled:
            # PPC_Space → Place Cells (공간 표상 업데이트)
            self._create_static_synapse(
                "ppc_left_to_place", self.ppc_space_left, self.place_cells,
                self.config.ppc_to_hippocampus_weight, sparsity=0.08)
            self._create_static_synapse(
                "ppc_right_to_place", self.ppc_space_right, self.place_cells,
                self.config.ppc_to_hippocampus_weight, sparsity=0.08)

            # PPC_Goal_Food → Food Memory (목표 위치 기억)
            if self.config.directional_food_memory:
                self._create_static_synapse(
                    "ppc_goal_food_to_food_mem_left", self.ppc_goal_food, self.food_memory_left,
                    self.config.ppc_to_hippocampus_weight * 0.8, sparsity=0.08)
                self._create_static_synapse(
                    "ppc_goal_food_to_food_mem_right", self.ppc_goal_food, self.food_memory_right,
                    self.config.ppc_to_hippocampus_weight * 0.8, sparsity=0.08)

            print(f"    PPC→Hippocampus: {self.config.ppc_to_hippocampus_weight}")

        # === 8. Top-Down 조절 ===
        # Hunger → PPC_Goal_Food (배고플 때 음식 목표 강화)
        self._create_static_synapse(
            "hunger_to_ppc_goal_food", self.hunger_drive, self.ppc_goal_food,
            self.config.hunger_to_ppc_goal_food_weight, sparsity=0.1)

        # Fear → PPC_Goal_Safety (공포 시 안전 목표 강화)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_ppc_goal_safety", self.fear_response, self.ppc_goal_safety,
                self.config.fear_to_ppc_goal_safety_weight, sparsity=0.1)

        # Dopamine → PPC_Attention (보상 예측 시 주의 강화)
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "dopamine_to_ppc_attention", self.dopamine_neurons, self.ppc_attention,
                self.config.dopamine_to_ppc_attention_weight, sparsity=0.1)

        print(f"    Top-Down: Hunger→Goal_Food {self.config.hunger_to_ppc_goal_food_weight}, "
              f"Fear→Goal_Safety {self.config.fear_to_ppc_goal_safety_weight}")

        total_parietal = (self.config.n_ppc_space_left + self.config.n_ppc_space_right +
                        self.config.n_ppc_goal_food + self.config.n_ppc_goal_safety +
                        self.config.n_ppc_attention + self.config.n_ppc_path_buffer)
        print(f"  Parietal Cortex complete: {total_parietal} neurons")
        print(f"  Total neurons now = {self.config.total_neurons:,}")

    def _build_premotor_cortex_circuit(self):
        """
        Phase 14: Premotor Cortex (전운동 피질) 구축

        전운동 피질 모델링:
        - PMd (Dorsal Premotor): 공간 기반 운동 계획
        - PMv (Ventral Premotor): 물체 기반 운동 계획
        - SMA (Supplementary Motor Area): 시퀀스 생성
        - pre-SMA: 운동 의도/선택
        - Motor_Preparation: 운동 준비 버퍼

        구성:
        - PPC → PMd: 공간 기반 운동 계획
        - IT/STS → PMv: 물체 기반 운동 계획
        - PFC → SMA: 목표 기반 시퀀스
        - PMC → Motor: 운동 출력
        """
        print("  Phase 14: Building Premotor Cortex (PMC)...")

        # LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. PMC Populations ===
        self.pmd_left = self.model.add_neuron_population(
            "pmd_left", self.config.n_pmd_left,
            sensory_lif_model, lif_params, lif_init)
        self.pmd_right = self.model.add_neuron_population(
            "pmd_right", self.config.n_pmd_right,
            sensory_lif_model, lif_params, lif_init)
        self.pmv_approach = self.model.add_neuron_population(
            "pmv_approach", self.config.n_pmv_approach,
            sensory_lif_model, lif_params, lif_init)
        self.pmv_avoid = self.model.add_neuron_population(
            "pmv_avoid", self.config.n_pmv_avoid,
            sensory_lif_model, lif_params, lif_init)
        self.sma_sequence = self.model.add_neuron_population(
            "sma_sequence", self.config.n_sma_sequence,
            sensory_lif_model, lif_params, lif_init)
        self.pre_sma = self.model.add_neuron_population(
            "pre_sma", self.config.n_pre_sma,
            sensory_lif_model, lif_params, lif_init)
        self.motor_preparation = self.model.add_neuron_population(
            "motor_preparation", self.config.n_motor_preparation,
            sensory_lif_model, lif_params, lif_init)

        print(f"    PMd: Left({self.config.n_pmd_left}) + Right({self.config.n_pmd_right})")
        print(f"    PMv: Approach({self.config.n_pmv_approach}) + Avoid({self.config.n_pmv_avoid})")
        print(f"    SMA: Sequence({self.config.n_sma_sequence}), pre_SMA({self.config.n_pre_sma})")
        print(f"    Motor_Preparation: {self.config.n_motor_preparation}")

        # === 2. PPC → PMd (공간 기반 운동 계획) ===
        if self.config.parietal_enabled:
            # PPC_Goal_Food + PPC_Space_Left → PMd_Left (왼쪽 음식 방향)
            self._create_static_synapse(
                "ppc_goal_food_to_pmd_left", self.ppc_goal_food, self.pmd_left,
                self.config.ppc_to_pmd_weight, sparsity=0.12)
            self._create_static_synapse(
                "ppc_space_left_to_pmd_left", self.ppc_space_left, self.pmd_left,
                self.config.ppc_to_pmd_weight * 0.8, sparsity=0.1)

            # PPC_Goal_Food + PPC_Space_Right → PMd_Right (오른쪽 음식 방향)
            self._create_static_synapse(
                "ppc_goal_food_to_pmd_right", self.ppc_goal_food, self.pmd_right,
                self.config.ppc_to_pmd_weight, sparsity=0.12)
            self._create_static_synapse(
                "ppc_space_right_to_pmd_right", self.ppc_space_right, self.pmd_right,
                self.config.ppc_to_pmd_weight * 0.8, sparsity=0.1)

            # PPC_Goal_Safety → PMd (반대 방향 회피)
            self._create_static_synapse(
                "ppc_goal_safety_to_pmd_right", self.ppc_goal_safety, self.pmd_right,
                self.config.ppc_to_pmd_weight * 0.7, sparsity=0.1)
            self._create_static_synapse(
                "ppc_goal_safety_to_pmd_left", self.ppc_goal_safety, self.pmd_left,
                self.config.ppc_to_pmd_weight * 0.7, sparsity=0.1)

            print(f"    PPC→PMd: {self.config.ppc_to_pmd_weight}")

        # === 3. IT/STS → PMv (물체 기반 운동 계획) ===
        if self.config.it_enabled:
            # IT_Food → PMv_Approach
            self._create_static_synapse(
                "it_food_to_pmv_approach", self.it_food_category, self.pmv_approach,
                self.config.it_to_pmv_weight, sparsity=0.12)
            # IT_Danger → PMv_Avoid
            self._create_static_synapse(
                "it_danger_to_pmv_avoid", self.it_danger_category, self.pmv_avoid,
                self.config.it_to_pmv_weight, sparsity=0.12)

            print(f"    IT→PMv: {self.config.it_to_pmv_weight}")

        if self.config.multimodal_enabled:
            # STS_Food → PMv_Approach
            self._create_static_synapse(
                "sts_food_to_pmv_approach", self.sts_food, self.pmv_approach,
                self.config.sts_to_pmv_weight, sparsity=0.12)
            # STS_Danger → PMv_Avoid
            self._create_static_synapse(
                "sts_danger_to_pmv_avoid", self.sts_danger, self.pmv_avoid,
                self.config.sts_to_pmv_weight, sparsity=0.12)

            print(f"    STS→PMv: {self.config.sts_to_pmv_weight}")

        # === 4. PFC → SMA (목표 기반 시퀀스) ===
        if self.config.prefrontal_enabled:
            # Goal_Food → SMA_Sequence
            self._create_static_synapse(
                "goal_food_to_sma", self.goal_food, self.sma_sequence,
                self.config.pfc_to_sma_weight, sparsity=0.12)
            # Goal_Safety → SMA_Sequence
            self._create_static_synapse(
                "goal_safety_to_sma", self.goal_safety, self.sma_sequence,
                self.config.pfc_to_sma_weight, sparsity=0.12)
            # Working_Memory → pre_SMA
            self._create_static_synapse(
                "wm_to_pre_sma", self.working_memory, self.pre_sma,
                self.config.pfc_to_sma_weight, sparsity=0.1)
            # Inhibitory_Control → pre_SMA (억제)
            self._create_static_synapse(
                "inhibitory_to_pre_sma", self.inhibitory_control, self.pre_sma,
                self.config.inhibitory_to_pre_sma_weight, sparsity=0.1)

            print(f"    PFC→SMA: {self.config.pfc_to_sma_weight}")

        # === 5. PMC 내부 연결 ===

        # 5.1 SMA 재귀 연결 (시퀀스 유지)
        self._create_static_synapse(
            "sma_recurrent", self.sma_sequence, self.sma_sequence,
            self.config.sma_recurrent_weight, sparsity=0.05)

        # 5.2 pre_SMA → SMA (의도 → 시퀀스 시작)
        self._create_static_synapse(
            "pre_sma_to_sma", self.pre_sma, self.sma_sequence,
            self.config.pre_sma_to_sma_weight, sparsity=0.12)

        print(f"    SMA Recurrent: {self.config.sma_recurrent_weight}, pre_SMA→SMA: {self.config.pre_sma_to_sma_weight}")

        # 5.3 PMd/PMv → Motor_Preparation (통합)
        self._create_static_synapse(
            "pmd_left_to_motor_prep", self.pmd_left, self.motor_preparation,
            self.config.pmd_pmv_integration_weight, sparsity=0.12)
        self._create_static_synapse(
            "pmd_right_to_motor_prep", self.pmd_right, self.motor_preparation,
            self.config.pmd_pmv_integration_weight, sparsity=0.12)
        self._create_static_synapse(
            "pmv_approach_to_motor_prep", self.pmv_approach, self.motor_preparation,
            self.config.pmd_pmv_integration_weight, sparsity=0.12)
        self._create_static_synapse(
            "pmv_avoid_to_motor_prep", self.pmv_avoid, self.motor_preparation,
            self.config.pmd_pmv_integration_weight, sparsity=0.12)

        # 5.4 SMA → Motor_Preparation (시퀀스 실행)
        self._create_static_synapse(
            "sma_to_motor_prep", self.sma_sequence, self.motor_preparation,
            self.config.sma_to_motor_prep_weight, sparsity=0.1)

        print(f"    PMd/PMv/SMA→Motor_Prep: {self.config.pmd_pmv_integration_weight}")

        # 5.5 WTA 경쟁
        # PMd_Left ↔ PMd_Right
        self._create_static_synapse(
            "pmd_left_right_wta", self.pmd_left, self.pmd_right,
            self.config.pmc_wta_inhibition, sparsity=0.1)
        self._create_static_synapse(
            "pmd_right_left_wta", self.pmd_right, self.pmd_left,
            self.config.pmc_wta_inhibition, sparsity=0.1)

        # PMv_Approach ↔ PMv_Avoid
        self._create_static_synapse(
            "pmv_approach_avoid_wta", self.pmv_approach, self.pmv_avoid,
            self.config.pmc_wta_inhibition * 1.2, sparsity=0.12)
        self._create_static_synapse(
            "pmv_avoid_approach_wta", self.pmv_avoid, self.pmv_approach,
            self.config.pmc_wta_inhibition * 1.2, sparsity=0.12)

        print(f"    PMC WTA: {self.config.pmc_wta_inhibition}")

        # === 6. PMC → Motor (운동 출력) ===

        # 6.1 Motor_Preparation → Motor
        self._create_static_synapse(
            "motor_prep_to_motor_left", self.motor_preparation, self.motor_left,
            self.config.motor_prep_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "motor_prep_to_motor_right", self.motor_preparation, self.motor_right,
            self.config.motor_prep_to_motor_weight, sparsity=0.1)

        # 6.2 PMd → Motor (직접 경로)
        self._create_static_synapse(
            "pmd_left_to_motor_left", self.pmd_left, self.motor_left,
            self.config.pmd_to_motor_weight, sparsity=0.1)
        self._create_static_synapse(
            "pmd_right_to_motor_right", self.pmd_right, self.motor_right,
            self.config.pmd_to_motor_weight, sparsity=0.1)

        # 6.3 PMv → Motor
        # PMv_Approach → 양측 Motor (전진)
        self._create_static_synapse(
            "pmv_approach_to_motor_left", self.pmv_approach, self.motor_left,
            self.config.pmv_to_motor_weight, sparsity=0.08)
        self._create_static_synapse(
            "pmv_approach_to_motor_right", self.pmv_approach, self.motor_right,
            self.config.pmv_to_motor_weight, sparsity=0.08)

        # PMv_Avoid → Motor (회피)
        self._create_static_synapse(
            "pmv_avoid_to_motor_left", self.pmv_avoid, self.motor_right,
            self.config.pmv_to_motor_weight, sparsity=0.08)
        self._create_static_synapse(
            "pmv_avoid_to_motor_right", self.pmv_avoid, self.motor_left,
            self.config.pmv_to_motor_weight, sparsity=0.08)

        print(f"    PMC→Motor: {self.config.motor_prep_to_motor_weight}")

        # === 7. PMC → Cerebellum (운동 조정) ===
        if self.config.cerebellum_enabled:
            self._create_static_synapse(
                "motor_prep_to_granule", self.motor_preparation, self.granule_cells,
                self.config.motor_prep_to_cerebellum_weight, sparsity=0.08)

            print(f"    PMC→Cerebellum: {self.config.motor_prep_to_cerebellum_weight}")

        # === 8. Basal Ganglia → PMC (행동 선택) ===
        if self.config.basal_ganglia_enabled:
            # Direct → Motor_Preparation (Go 신호)
            self._create_static_synapse(
                "direct_to_motor_prep", self.direct_pathway, self.motor_preparation,
                self.config.direct_to_motor_prep_weight, sparsity=0.1)
            # Indirect → Motor_Preparation (NoGo 신호)
            self._create_static_synapse(
                "indirect_to_motor_prep", self.indirect_pathway, self.motor_preparation,
                self.config.indirect_to_motor_prep_weight, sparsity=0.1)
            # Dopamine → SMA (보상 → 시퀀스 강화)
            self._create_static_synapse(
                "dopamine_to_sma", self.dopamine_neurons, self.sma_sequence,
                self.config.dopamine_to_sma_weight, sparsity=0.1)

            print(f"    BG→PMC: Direct {self.config.direct_to_motor_prep_weight}, "
                  f"Indirect {self.config.indirect_to_motor_prep_weight}")

        # === 9. Top-Down 조절 ===
        # Hunger → PMv_Approach
        self._create_static_synapse(
            "hunger_to_pmv_approach", self.hunger_drive, self.pmv_approach,
            self.config.hunger_to_pmv_approach_weight, sparsity=0.1)

        # Fear → PMv_Avoid
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_pmv_avoid", self.fear_response, self.pmv_avoid,
                self.config.fear_to_pmv_avoid_weight, sparsity=0.1)

        # Arousal → Motor_Preparation
        if self.config.thalamus_enabled:
            self._create_static_synapse(
                "arousal_to_motor_prep", self.arousal, self.motor_preparation,
                self.config.arousal_to_motor_prep_weight, sparsity=0.1)

        print(f"    Top-Down: Hunger→PMv_Approach {self.config.hunger_to_pmv_approach_weight}, "
              f"Fear→PMv_Avoid {self.config.fear_to_pmv_avoid_weight}")

        total_premotor = (self.config.n_pmd_left + self.config.n_pmd_right +
                        self.config.n_pmv_approach + self.config.n_pmv_avoid +
                        self.config.n_sma_sequence + self.config.n_pre_sma +
                        self.config.n_motor_preparation)
        print(f"  Premotor Cortex complete: {total_premotor} neurons")
        print(f"  Total neurons now = {self.config.total_neurons:,}")

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

    def _build_social_brain_circuit(self):
        """
        Phase 15: Social Brain (사회적 뇌)

        생물학적 근거:
        - STS: 생물학적 움직임 인식 (Allison et al., 2000)
        - TPJ: 관점 전환, Theory of Mind (Saxe & Kanwisher, 2003)
        - ACC: 갈등 모니터링, 사회적 통증 (Botvinick et al., 2004)
        - vmPFC/OFC: 사회적 보상 평가 (Rushworth et al., 2007)
        """
        from pygenn import init_var, init_weight_update, init_postsynaptic, init_sparse_connectivity

        print("  Phase 15: Building Social Brain...")

        # Sensory populations (with I_input for external current injection)
        s_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        s_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # Internal populations (standard LIF, no I_input needed)
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0, "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0}

        # === 1. 감각 입력 인구 (SensoryLIF - I_input 필요) ===
        self.agent_eye_left = self.model.add_neuron_population(
            "agent_eye_left", self.config.n_agent_eye_left,
            sensory_lif_model, s_params, s_init)
        self.agent_eye_right = self.model.add_neuron_population(
            "agent_eye_right", self.config.n_agent_eye_right,
            sensory_lif_model, s_params, s_init)
        self.agent_sound_left = self.model.add_neuron_population(
            "agent_sound_left", self.config.n_agent_sound_left,
            sensory_lif_model, s_params, s_init)
        self.agent_sound_right = self.model.add_neuron_population(
            "agent_sound_right", self.config.n_agent_sound_right,
            sensory_lif_model, s_params, s_init)

        # === 2. STS_Social (다른 에이전트 통합 인식) ===
        self.sts_social = self.model.add_neuron_population(
            "sts_social", self.config.n_sts_social,
            "LIF", lif_params, lif_init)

        # === 3. TPJ (Temporoparietal Junction) ===
        self.tpj_self = self.model.add_neuron_population(
            "tpj_self", self.config.n_tpj_self,
            "LIF", lif_params, lif_init)
        self.tpj_other = self.model.add_neuron_population(
            "tpj_other", self.config.n_tpj_other,
            "LIF", lif_params, lif_init)
        self.tpj_compare = self.model.add_neuron_population(
            "tpj_compare", self.config.n_tpj_compare,
            "LIF", lif_params, lif_init)

        # === 4. ACC (Anterior Cingulate Cortex) ===
        # acc_conflict uses SensoryLIF (needs I_input for social proximity injection)
        self.acc_conflict = self.model.add_neuron_population(
            "acc_conflict", self.config.n_acc_conflict,
            sensory_lif_model, s_params, s_init)
        self.acc_monitor = self.model.add_neuron_population(
            "acc_monitor", self.config.n_acc_monitor,
            "LIF", lif_params, lif_init)

        # === 5. Social Valuation (접근/회피 동기) ===
        self.social_approach = self.model.add_neuron_population(
            "social_approach", self.config.n_social_approach,
            "LIF", lif_params, lif_init)
        self.social_avoid = self.model.add_neuron_population(
            "social_avoid", self.config.n_social_avoid,
            "LIF", lif_params, lif_init)

        print(f"    Populations: Agent_Eye({self.config.n_agent_eye_left}×2) + "
              f"Agent_Sound({self.config.n_agent_sound_left}×2) + "
              f"STS_Social({self.config.n_sts_social}) + "
              f"TPJ({self.config.n_tpj_self}+{self.config.n_tpj_other}+{self.config.n_tpj_compare}) + "
              f"ACC({self.config.n_acc_conflict}+{self.config.n_acc_monitor}) + "
              f"Social_Val({self.config.n_social_approach}+{self.config.n_social_avoid})")

        # ============================================================
        # 시냅스 연결
        # ============================================================

        # === 6. 감각 입력 → STS_Social ===
        self._create_static_synapse(
            "agent_eye_left_to_sts_social", self.agent_eye_left, self.sts_social,
            self.config.agent_eye_to_sts_social_weight, sparsity=0.15)
        self._create_static_synapse(
            "agent_eye_right_to_sts_social", self.agent_eye_right, self.sts_social,
            self.config.agent_eye_to_sts_social_weight, sparsity=0.15)
        self._create_static_synapse(
            "agent_sound_left_to_sts_social", self.agent_sound_left, self.sts_social,
            self.config.agent_sound_to_sts_social_weight, sparsity=0.12)
        self._create_static_synapse(
            "agent_sound_right_to_sts_social", self.agent_sound_right, self.sts_social,
            self.config.agent_sound_to_sts_social_weight, sparsity=0.12)

        # STS_Social 자기 유지 (recurrent)
        self._create_static_synapse(
            "sts_social_recurrent", self.sts_social, self.sts_social,
            self.config.sts_social_recurrent_weight, sparsity=0.05)

        print(f"    Agent_Eye→STS_Social: {self.config.agent_eye_to_sts_social_weight}")
        print(f"    Agent_Sound→STS_Social: {self.config.agent_sound_to_sts_social_weight}")

        # === 7. STS_Social → TPJ ===
        self._create_static_synapse(
            "sts_social_to_tpj_other", self.sts_social, self.tpj_other,
            self.config.sts_social_to_tpj_weight, sparsity=0.10)

        # TPJ_Self ← 내부 상태 (Hunger/Satiety)
        self._create_static_synapse(
            "hunger_to_tpj_self", self.hunger_drive, self.tpj_self,
            self.config.internal_to_tpj_self_weight, sparsity=0.10)
        self._create_static_synapse(
            "satiety_to_tpj_self", self.satiety_drive, self.tpj_self,
            self.config.internal_to_tpj_self_weight * 0.8, sparsity=0.10)

        # TPJ_Self/Other → TPJ_Compare (자기-타자 비교)
        self._create_static_synapse(
            "tpj_self_to_compare", self.tpj_self, self.tpj_compare,
            self.config.tpj_compare_weight, sparsity=0.08)
        self._create_static_synapse(
            "tpj_other_to_compare", self.tpj_other, self.tpj_compare,
            self.config.tpj_compare_weight, sparsity=0.08)

        print(f"    STS_Social→TPJ: {self.config.sts_social_to_tpj_weight}")

        # === 8. TPJ → ACC (갈등 감지) ===
        self._create_static_synapse(
            "tpj_compare_to_acc", self.tpj_compare, self.acc_conflict,
            self.config.tpj_to_acc_weight, sparsity=0.10)

        # ACC_Conflict ↔ ACC_Monitor (상호 연결)
        self._create_static_synapse(
            "acc_conflict_to_monitor", self.acc_conflict, self.acc_monitor,
            8.0, sparsity=0.08)
        self._create_static_synapse(
            "acc_monitor_to_conflict", self.acc_monitor, self.acc_conflict,
            6.0, sparsity=0.08)

        print(f"    TPJ→ACC: {self.config.tpj_to_acc_weight}")

        # === 9. 사회적 가치 평가 ===
        self._create_static_synapse(
            "sts_social_to_approach", self.sts_social, self.social_approach,
            self.config.sts_social_to_approach_weight, sparsity=0.08)
        self._create_static_synapse(
            "acc_to_avoid", self.acc_conflict, self.social_avoid,
            self.config.acc_to_avoid_weight, sparsity=0.10)

        # Approach ↔ Avoid WTA 경쟁
        self._create_static_synapse(
            "approach_to_avoid_inhib", self.social_approach, self.social_avoid,
            self.config.social_wta_inhibition, sparsity=0.08)
        self._create_static_synapse(
            "avoid_to_approach_inhib", self.social_avoid, self.social_approach,
            self.config.social_wta_inhibition, sparsity=0.08)

        print(f"    Social Approach↔Avoid WTA: {self.config.social_wta_inhibition}")

        # === 10. 기존 회로 연결 (약한 간접 경로) ===
        # STS_Social → PFC Working Memory
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "sts_social_to_wm", self.sts_social, self.working_memory,
                self.config.sts_social_to_pfc_weight, sparsity=0.05)

        # ACC → Amygdala LA (약하게! Phase 12-14 교훈)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "acc_to_la", self.acc_conflict, self.lateral_amygdala,
                self.config.acc_to_amygdala_weight, sparsity=0.05)

        # Social Approach → PFC Goal_Food
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "social_approach_to_goal_food", self.social_approach, self.goal_food,
                self.config.social_approach_to_goal_food_weight, sparsity=0.05)
            self._create_static_synapse(
                "social_avoid_to_goal_safety", self.social_avoid, self.goal_safety,
                self.config.social_avoid_to_goal_safety_weight, sparsity=0.05)

        print(f"    STS_Social→PFC: {self.config.sts_social_to_pfc_weight}")
        print(f"    ACC→Amygdala: {self.config.acc_to_amygdala_weight} (약한 간접)")

        # === 11. Top-Down 조절 ===
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_sts_social", self.fear_response, self.sts_social,
                self.config.fear_to_sts_social_weight, sparsity=0.08)
        self._create_static_synapse(
            "hunger_to_social_approach", self.hunger_drive, self.social_approach,
            self.config.hunger_to_social_approach_weight, sparsity=0.05)

        print(f"    Fear→STS_Social: {self.config.fear_to_sts_social_weight}")
        print(f"    Phase 15 Social Brain: {self.config.n_agent_eye_left * 2 + self.config.n_agent_sound_left * 2 + self.config.n_sts_social + self.config.n_tpj_self + self.config.n_tpj_other + self.config.n_tpj_compare + self.config.n_acc_conflict + self.config.n_acc_monitor + self.config.n_social_approach + self.config.n_social_avoid} neurons")

    def _build_mirror_neuron_circuit(self):
        """
        Phase 15b: Mirror Neurons & Social Learning

        거울 뉴런 시스템: NPC가 음식 먹는 것을 관찰하고 학습

        구조:
        - Social_Observation (200, SensoryLIF): NPC 목표지향 움직임 감지
        - Mirror_Food (150, SensoryLIF): 자기+타인 먹기 거울 뉴런
        - Vicarious_Reward (100, SensoryLIF): 관찰 예측 오차
        - Social_Memory (150, LIF): 사회적 음식 위치 기억 (Hebbian)
        """
        print("  Phase 15b: Building Mirror Neuron circuit...")

        # SensoryLIF 파라미터 (I_input 필요한 인구)
        s_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        s_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # LIF 파라미터 (I_input 불필요한 인구)
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0, "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0}

        # === 1. Social_Observation (SensoryLIF) ===
        self.social_observation = self.model.add_neuron_population(
            "social_observation", self.config.n_social_observation,
            sensory_lif_model, s_params, s_init)
        print(f"    Social_Observation: {self.config.n_social_observation} neurons (SensoryLIF)")

        # === 2. Mirror_Food (SensoryLIF) ===
        self.mirror_food = self.model.add_neuron_population(
            "mirror_food", self.config.n_mirror_food,
            sensory_lif_model, s_params, s_init)
        print(f"    Mirror_Food: {self.config.n_mirror_food} neurons (SensoryLIF)")

        # === 3. Vicarious_Reward (SensoryLIF) ===
        self.vicarious_reward = self.model.add_neuron_population(
            "vicarious_reward", self.config.n_vicarious_reward,
            sensory_lif_model, s_params, s_init)
        print(f"    Vicarious_Reward: {self.config.n_vicarious_reward} neurons (SensoryLIF)")

        # === 4. Social_Memory (LIF, Hebbian 학습 대상) ===
        self.social_memory = self.model.add_neuron_population(
            "social_memory", self.config.n_social_memory,
            "LIF", lif_params, lif_init)
        print(f"    Social_Memory: {self.config.n_social_memory} neurons (LIF)")

        # === 5. 시냅스 연결 ===

        # --- 입력 → Social_Observation ---
        # Agent_Eye L/R → Social_Observation
        self._create_static_synapse(
            "agent_eye_l_to_social_obs", self.agent_eye_left, self.social_observation,
            self.config.agent_eye_to_social_obs_weight, sparsity=0.10)
        self._create_static_synapse(
            "agent_eye_r_to_social_obs", self.agent_eye_right, self.social_observation,
            self.config.agent_eye_to_social_obs_weight, sparsity=0.10)

        # STS_Social → Social_Observation
        self._create_static_synapse(
            "sts_social_to_social_obs", self.sts_social, self.social_observation,
            self.config.sts_social_to_social_obs_weight, sparsity=0.08)

        # Social_Observation 재귀 연결
        self._create_static_synapse(
            "social_obs_recurrent", self.social_observation, self.social_observation,
            self.config.social_obs_recurrent_weight, sparsity=0.05)

        print(f"    Agent_Eye→Social_Obs: {self.config.agent_eye_to_social_obs_weight}")
        print(f"    STS_Social→Social_Obs: {self.config.sts_social_to_social_obs_weight}")

        # --- Social_Observation → Mirror_Food ---
        self._create_static_synapse(
            "social_obs_to_mirror", self.social_observation, self.mirror_food,
            self.config.social_obs_to_mirror_weight, sparsity=0.10)

        # Mirror_Food 재귀 연결
        self._create_static_synapse(
            "mirror_food_recurrent", self.mirror_food, self.mirror_food,
            self.config.mirror_food_recurrent_weight, sparsity=0.05)

        # Hunger → Mirror_Food (자기 배고픔→먹기 모사)
        self._create_static_synapse(
            "hunger_to_mirror", self.hunger_drive, self.mirror_food,
            self.config.hunger_to_mirror_weight, sparsity=0.05)

        # Food_Eye → Mirror_Food (자기 음식 시각)
        self._create_static_synapse(
            "food_eye_l_to_mirror", self.food_eye_left, self.mirror_food,
            self.config.food_eye_to_mirror_weight, sparsity=0.05)
        self._create_static_synapse(
            "food_eye_r_to_mirror", self.food_eye_right, self.mirror_food,
            self.config.food_eye_to_mirror_weight, sparsity=0.05)

        print(f"    Social_Obs→Mirror_Food: {self.config.social_obs_to_mirror_weight}")
        print(f"    Hunger→Mirror_Food: {self.config.hunger_to_mirror_weight}")

        # --- Mirror_Food → Vicarious_Reward ---
        self._create_static_synapse(
            "mirror_to_vicarious", self.mirror_food, self.vicarious_reward,
            self.config.mirror_to_vicarious_weight, sparsity=0.10)

        print(f"    Mirror_Food→Vicarious_Reward: {self.config.mirror_to_vicarious_weight}")

        # --- Vicarious_Reward → Social_Memory (DENSE, Hebbian 학습) ---
        self.vicarious_to_social_memory = self.model.add_synapse_population(
            "vicarious_to_social_memory", "DENSE",
            self.vicarious_reward, self.social_memory,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.vicarious_to_social_memory_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0})
        )

        # Social_Memory 재귀 연결
        self._create_static_synapse(
            "social_memory_recurrent", self.social_memory, self.social_memory,
            self.config.social_memory_recurrent_weight, sparsity=0.05)

        print(f"    Vicarious→Social_Memory: {self.config.vicarious_to_social_memory_weight} (HEBBIAN, eta={self.config.social_memory_eta})")

        # --- 기존 회로 출력 (약한 간접 경로 ≤6.0, Motor 0.0!) ---

        # Social_Memory → Food_Memory L/R
        if self.config.hippocampus_enabled and self.config.directional_food_memory:
            self._create_static_synapse(
                "social_mem_to_food_mem_l", self.social_memory, self.food_memory_left,
                self.config.social_memory_to_food_memory_weight, sparsity=0.05)
            self._create_static_synapse(
                "social_mem_to_food_mem_r", self.social_memory, self.food_memory_right,
                self.config.social_memory_to_food_memory_weight, sparsity=0.05)

        # Social_Obs → Working_Memory
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "social_obs_to_wm", self.social_observation, self.working_memory,
                self.config.social_obs_to_wm_weight, sparsity=0.05)

        # Social_Obs → Dopamine
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "social_obs_to_dopamine", self.social_observation, self.dopamine_neurons,
                self.config.social_obs_to_dopamine_weight, sparsity=0.05)

        # Mirror_Food → Goal_Food
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "mirror_to_goal_food", self.mirror_food, self.goal_food,
                self.config.mirror_to_goal_food_weight, sparsity=0.05)

        # Mirror_Food → Hunger (약한 활성화)
        self._create_static_synapse(
            "mirror_to_hunger", self.mirror_food, self.hunger_drive,
            self.config.mirror_to_hunger_weight, sparsity=0.05)

        print(f"    Social_Memory→Food_Memory: {self.config.social_memory_to_food_memory_weight}")
        print(f"    Mirror→Goal_Food: {self.config.mirror_to_goal_food_weight}")
        print(f"    Mirror→Motor: {self.config.mirror_to_motor_weight} (DISABLED!)")

        # --- Top-Down → Mirror ---
        # Hunger → Social_Observation
        self._create_static_synapse(
            "hunger_to_social_obs", self.hunger_drive, self.social_observation,
            self.config.hunger_to_social_obs_weight, sparsity=0.05)

        # Fear → Social_Observation (억제: 위험 시 관찰 억제)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_social_obs", self.fear_response, self.social_observation,
                self.config.fear_to_social_obs_weight, sparsity=0.05)

        print(f"    Hunger→Social_Obs: {self.config.hunger_to_social_obs_weight}")
        print(f"    Fear→Social_Obs: {self.config.fear_to_social_obs_weight}")

        # 상태 변수 초기화
        self.mirror_self_eating_timer = 0
        self.last_social_obs_rate = 0.0

        n_mirror_total = (self.config.n_social_observation + self.config.n_mirror_food +
                          self.config.n_vicarious_reward + self.config.n_social_memory)
        print(f"    Phase 15b Mirror Neurons: {n_mirror_total} neurons")

    def learn_social_food_location(self, npc_food_pos: tuple):
        """
        Phase 15b: NPC가 음식 먹는 것을 관찰했을 때 Hebbian 학습

        Vicarious_Reward → Social_Memory 가중치 강화

        Args:
            npc_food_pos: (x, y) NPC가 먹은 음식의 정규화 위치
        """
        if not (self.config.social_brain_enabled and self.config.mirror_enabled):
            return None

        eta = self.config.social_memory_eta
        w_max = self.config.social_memory_w_max
        n_pre = self.config.n_vicarious_reward
        n_post = self.config.n_social_memory

        # Vicarious_Reward 뉴런의 최근 활성도 기반 학습
        self.vicarious_to_social_memory.vars["g"].pull_from_device()
        weights = self.vicarious_to_social_memory.vars["g"].view.copy()
        weights = weights.reshape(n_pre, n_post)

        # Surprise factor: social_obs_rate가 낮을수록 더 많이 학습 (놀라움)
        surprise = max(0.1, 1.0 - self.last_social_obs_rate)
        delta_w = eta * surprise
        weights += delta_w
        weights = np.clip(weights, 0.0, w_max)

        self.vicarious_to_social_memory.vars["g"].view[:] = weights.flatten()
        self.vicarious_to_social_memory.vars["g"].push_to_device()

        return {
            "avg_weight": float(np.mean(weights)),
            "max_weight": float(np.max(weights)),
            "surprise": surprise,
        }

    def _build_tom_circuit(self):
        """
        Phase 15c: Theory of Mind & Cooperation/Competition

        생물학적 근거:
        - mPFC: 의도/신념 표상 mentalizing (Frith & Frith, 2006)
        - TPJ: 관점 취하기 (이미 15a에 존재, 확장)
        - Anterior Insula: 사회적 예측 오차 (Singer et al., 2004)
        - vmPFC: 협력 가치 학습 (Rilling et al., 2002)
        - dACC: 경쟁 모니터링 (Behrens et al., 2008)

        구조:
        - ToM_Intention (100, SensoryLIF): NPC 의도 추론
        - ToM_Belief (80, LIF): NPC 신념 추적
        - ToM_Prediction (80, LIF): NPC 행동 예측
        - ToM_Surprise (60, SensoryLIF): 예측 오차
        - CoopCompete_Coop (80, LIF, Hebbian): 협력 가치
        - CoopCompete_Compete (100, SensoryLIF): 경쟁 감지
        """
        print("  Phase 15c: Building Theory of Mind circuit...")

        # SensoryLIF 파라미터 (I_input 필요)
        s_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        s_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # LIF 파라미터 (시냅스 입력만)
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0, "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0}

        # === 1. ToM_Intention (SensoryLIF) - NPC 의도 추론 ===
        self.tom_intention = self.model.add_neuron_population(
            "tom_intention", self.config.n_tom_intention,
            sensory_lif_model, s_params, s_init)
        print(f"    ToM_Intention: {self.config.n_tom_intention} neurons (SensoryLIF)")

        # === 2. ToM_Belief (LIF) - NPC 신념 추적 ===
        self.tom_belief = self.model.add_neuron_population(
            "tom_belief", self.config.n_tom_belief,
            "LIF", lif_params, lif_init)
        print(f"    ToM_Belief: {self.config.n_tom_belief} neurons (LIF)")

        # === 3. ToM_Prediction (LIF) - NPC 행동 예측 ===
        self.tom_prediction = self.model.add_neuron_population(
            "tom_prediction", self.config.n_tom_prediction,
            "LIF", lif_params, lif_init)
        print(f"    ToM_Prediction: {self.config.n_tom_prediction} neurons (LIF)")

        # === 4. ToM_Surprise (SensoryLIF) - 예측 오차 ===
        self.tom_surprise = self.model.add_neuron_population(
            "tom_surprise", self.config.n_tom_surprise,
            sensory_lif_model, s_params, s_init)
        print(f"    ToM_Surprise: {self.config.n_tom_surprise} neurons (SensoryLIF)")

        # === 5. CoopCompete_Coop (LIF, Hebbian 학습 대상) ===
        self.coop_compete_coop = self.model.add_neuron_population(
            "coop_compete_coop", self.config.n_coop_compete_coop,
            "LIF", lif_params, lif_init)
        print(f"    CoopCompete_Coop: {self.config.n_coop_compete_coop} neurons (LIF)")

        # === 6. CoopCompete_Compete (SensoryLIF) ===
        self.coop_compete_compete = self.model.add_neuron_population(
            "coop_compete_compete", self.config.n_coop_compete_compete,
            sensory_lif_model, s_params, s_init)
        print(f"    CoopCompete_Compete: {self.config.n_coop_compete_compete} neurons (SensoryLIF)")

        # === 7. 시냅스 연결 ===

        # --- 입력 → ToM_Intention ---
        self._create_static_synapse(
            "social_obs_to_tom_intention", self.social_observation, self.tom_intention,
            self.config.social_obs_to_tom_intention_weight, sparsity=0.10)
        self._create_static_synapse(
            "sts_social_to_tom_intention", self.sts_social, self.tom_intention,
            self.config.sts_social_to_tom_intention_weight, sparsity=0.08)

        print(f"    Social_Obs→ToM_Intention: {self.config.social_obs_to_tom_intention_weight}")
        print(f"    STS_Social→ToM_Intention: {self.config.sts_social_to_tom_intention_weight}")

        # --- 입력 → ToM_Belief ---
        self._create_static_synapse(
            "tom_intention_to_belief", self.tom_intention, self.tom_belief,
            self.config.tom_intention_to_belief_weight, sparsity=0.10)
        self._create_static_synapse(
            "tpj_other_to_tom_belief", self.tpj_other, self.tom_belief,
            self.config.tpj_other_to_tom_belief_weight, sparsity=0.08)
        self._create_static_synapse(
            "social_obs_to_tom_belief", self.social_observation, self.tom_belief,
            self.config.social_obs_to_tom_belief_weight, sparsity=0.08)

        # --- ToM_Prediction ---
        self._create_static_synapse(
            "tom_intention_to_prediction", self.tom_intention, self.tom_prediction,
            self.config.tom_intention_to_prediction_weight, sparsity=0.10)
        self._create_static_synapse(
            "tom_belief_to_prediction", self.tom_belief, self.tom_prediction,
            self.config.tom_belief_to_prediction_weight, sparsity=0.10)
        self._create_static_synapse(
            "tom_prediction_recurrent", self.tom_prediction, self.tom_prediction,
            self.config.tom_prediction_recurrent_weight, sparsity=0.05)

        # --- Prediction-Surprise 회로 ---
        # 예측 성공 → 놀라움 억제
        self._create_static_synapse(
            "tom_prediction_to_surprise", self.tom_prediction, self.tom_surprise,
            self.config.tom_prediction_to_surprise_weight, sparsity=0.10)
        # 놀라움 → 예측 리셋
        self._create_static_synapse(
            "tom_surprise_to_prediction", self.tom_surprise, self.tom_prediction,
            self.config.tom_surprise_to_prediction_weight, sparsity=0.08)

        print(f"    Prediction→Surprise: {self.config.tom_prediction_to_surprise_weight} (inhibitory)")

        # --- CoopCompete_Coop 입력 ---
        # ToM_Intention → Coop (DENSE, Hebbian 학습)
        self.tom_intention_to_coop_hebbian = self.model.add_synapse_population(
            "tom_intention_to_coop_hebbian", "DENSE",
            self.tom_intention, self.coop_compete_coop,
            init_weight_update("StaticPulse", {},
                {"g": init_var("Constant", {"constant": self.config.tom_intention_to_coop_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0})
        )
        # Social_Memory → Coop
        self._create_static_synapse(
            "social_memory_to_coop", self.social_memory, self.coop_compete_coop,
            self.config.social_memory_to_coop_weight, sparsity=0.08)
        # Coop 재귀
        self._create_static_synapse(
            "coop_recurrent", self.coop_compete_coop, self.coop_compete_coop,
            self.config.coop_recurrent_weight, sparsity=0.05)

        print(f"    ToM_Intention→Coop: {self.config.tom_intention_to_coop_weight} (HEBBIAN)")

        # --- CoopCompete_Compete 입력 ---
        self._create_static_synapse(
            "tom_intention_to_compete", self.tom_intention, self.coop_compete_compete,
            self.config.tom_intention_to_compete_weight, sparsity=0.08)
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "acc_conflict_to_compete", self.acc_conflict, self.coop_compete_compete,
                self.config.acc_conflict_to_compete_weight, sparsity=0.08)

        # --- Coop ↔ Compete WTA ---
        self._create_static_synapse(
            "coop_to_compete_inhib", self.coop_compete_coop, self.coop_compete_compete,
            self.config.coop_compete_wta_weight, sparsity=0.08)
        self._create_static_synapse(
            "compete_to_coop_inhib", self.coop_compete_compete, self.coop_compete_coop,
            self.config.coop_compete_wta_weight, sparsity=0.08)

        print(f"    Coop↔Compete WTA: {self.config.coop_compete_wta_weight}")

        # --- 기존 회로 출력 (모두 ≤6.0, Motor 0.0!) ---
        # Coop → Social_Approach
        self._create_static_synapse(
            "coop_to_social_approach", self.coop_compete_coop, self.social_approach,
            self.config.coop_to_social_approach_weight, sparsity=0.05)
        # Coop → Goal_Food
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "coop_to_goal_food", self.coop_compete_coop, self.goal_food,
                self.config.coop_to_goal_food_weight, sparsity=0.05)
        # Compete → Social_Avoid
        self._create_static_synapse(
            "compete_to_social_avoid", self.coop_compete_compete, self.social_avoid,
            self.config.compete_to_social_avoid_weight, sparsity=0.05)
        # Compete → Hunger (긴급성)
        self._create_static_synapse(
            "compete_to_hunger", self.coop_compete_compete, self.hunger_drive,
            self.config.compete_to_hunger_weight, sparsity=0.05)
        # Compete → ACC_Conflict
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "compete_to_acc", self.coop_compete_compete, self.acc_conflict,
                self.config.compete_to_acc_weight, sparsity=0.05)
        # Surprise → ACC_Monitor
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "tom_surprise_to_acc", self.tom_surprise, self.acc_monitor,
                self.config.tom_surprise_to_acc_weight, sparsity=0.05)
        # Surprise → Dopamine
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "tom_surprise_to_dopamine", self.tom_surprise, self.dopamine_neurons,
                self.config.tom_surprise_to_dopamine_weight, sparsity=0.05)
        # Intention → Working_Memory
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "tom_intention_to_wm", self.tom_intention, self.working_memory,
                self.config.tom_intention_to_wm_weight, sparsity=0.05)

        print(f"    Coop→Social_Approach: {self.config.coop_to_social_approach_weight}")
        print(f"    Compete→Social_Avoid: {self.config.compete_to_social_avoid_weight}")
        print(f"    ToM→Motor: {self.config.tom_to_motor_weight} (DISABLED!)")

        # --- Top-Down ---
        # Hunger → ToM_Intention
        self._create_static_synapse(
            "hunger_to_tom_intention", self.hunger_drive, self.tom_intention,
            self.config.hunger_to_tom_intention_weight, sparsity=0.05)
        # Fear → ToM_Intention (억제)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_tom_intention", self.fear_response, self.tom_intention,
                self.config.fear_to_tom_intention_weight, sparsity=0.05)
        # Hunger → Compete
        self._create_static_synapse(
            "hunger_to_compete", self.hunger_drive, self.coop_compete_compete,
            self.config.hunger_to_compete_weight, sparsity=0.05)

        print(f"    Hunger→ToM_Intention: {self.config.hunger_to_tom_intention_weight}")
        print(f"    Fear→ToM_Intention: {self.config.fear_to_tom_intention_weight}")

        n_tom_total = (self.config.n_tom_intention + self.config.n_tom_belief +
                       self.config.n_tom_prediction + self.config.n_tom_surprise +
                       self.config.n_coop_compete_coop + self.config.n_coop_compete_compete)
        print(f"    Phase 15c Theory of Mind: {n_tom_total} neurons")

    def learn_cooperation_value(self, food_near_npc: bool):
        """
        Phase 15c: 협력 가치 Hebbian 학습

        에이전트가 음식을 먹을 때 ToM_Intention이 활성화되어 있었으면
        ToM_Intention → Coop 가중치 강화.
        "NPC를 관찰/따라가면 음식을 찾는다"를 학습.

        Args:
            food_near_npc: 먹은 음식이 NPC target 근처인지 여부
        """
        if not (self.config.social_brain_enabled and self.config.tom_enabled):
            return None

        eta = self.config.tom_coop_eta
        w_max = self.config.tom_coop_w_max
        n_pre = self.config.n_tom_intention
        n_post = self.config.n_coop_compete_coop

        self.tom_intention_to_coop_hebbian.vars["g"].pull_from_device()
        weights = self.tom_intention_to_coop_hebbian.vars["g"].view.copy()
        weights = weights.reshape(n_pre, n_post)

        # 학습 강도: NPC 근처 음식이면 강한 강화
        if food_near_npc:
            learning_factor = 1.0
        else:
            learning_factor = 0.3  # 약한 시간적 상관

        # ToM_Intention 활성도에 비례
        intention_scale = max(0.1, self.last_tom_intention_rate)
        delta_w = eta * learning_factor * intention_scale
        weights += delta_w
        weights = np.clip(weights, 0.0, w_max)

        self.tom_intention_to_coop_hebbian.vars["g"].view[:] = weights.flatten()
        self.tom_intention_to_coop_hebbian.vars["g"].push_to_device()

        return {
            "avg_weight": float(np.mean(weights)),
            "max_weight": float(np.max(weights)),
            "learning_factor": learning_factor,
        }

    def _build_association_cortex_circuit(self):
        """
        Phase 16: Association Cortex (연합 피질) 구축

        기존 범주 표상(IT, STS, A1)을 통합하여 감각 독립적 초범주 형성:
        - Assoc_Edible: "먹을 수 있는 것" (시각+청각+사회적 음식 통합)
        - Assoc_Threatening: "위험한 것" (시각+청각+공포 통합)
        - Assoc_Animate: "살아있는 것" (ToM+사회적 관찰 통합)
        - Assoc_Context: "익숙한 장소" (공간+기억 통합)
        - Assoc_Valence: "좋다/나쁘다" (보상/처벌 가치)
        - Assoc_Binding: 초범주 간 교차 연합 (Hebbian 학습)
        - Assoc_Novelty: 새로운 조합 탐지 (탐색 유발)
        """
        print("  Phase 16: Building Association Cortex...")

        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0, "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0}

        # === 1. 초범주 인구 생성 (모두 LIF - 내부 통합 전용) ===
        self.assoc_edible = self.model.add_neuron_population(
            "assoc_edible", self.config.n_assoc_edible, "LIF", lif_params, lif_init)
        self.assoc_threatening = self.model.add_neuron_population(
            "assoc_threatening", self.config.n_assoc_threatening, "LIF", lif_params, lif_init)
        self.assoc_animate = self.model.add_neuron_population(
            "assoc_animate", self.config.n_assoc_animate, "LIF", lif_params, lif_init)
        self.assoc_context = self.model.add_neuron_population(
            "assoc_context", self.config.n_assoc_context, "LIF", lif_params, lif_init)
        self.assoc_valence = self.model.add_neuron_population(
            "assoc_valence", self.config.n_assoc_valence, "LIF", lif_params, lif_init)
        self.assoc_binding = self.model.add_neuron_population(
            "assoc_binding", self.config.n_assoc_binding, "LIF", lif_params, lif_init)
        self.assoc_novelty = self.model.add_neuron_population(
            "assoc_novelty", self.config.n_assoc_novelty, "LIF", lif_params, lif_init)

        print(f"    Assoc_Edible: {self.config.n_assoc_edible}")
        print(f"    Assoc_Threatening: {self.config.n_assoc_threatening}")
        print(f"    Assoc_Animate: {self.config.n_assoc_animate}")
        print(f"    Assoc_Context: {self.config.n_assoc_context}")
        print(f"    Assoc_Valence: {self.config.n_assoc_valence}")
        print(f"    Assoc_Binding: {self.config.n_assoc_binding}")
        print(f"    Assoc_Novelty: {self.config.n_assoc_novelty}")

        # === 2. Assoc_Edible 입력 ===
        if self.config.it_enabled:
            self._create_static_synapse(
                "it_food_to_assoc_edible", self.it_food_category, self.assoc_edible,
                self.config.it_food_to_assoc_edible_weight, sparsity=0.10)
        if self.config.multimodal_enabled:
            self._create_static_synapse(
                "sts_food_to_assoc_edible", self.sts_food, self.assoc_edible,
                self.config.sts_food_to_assoc_edible_weight, sparsity=0.08)
        if self.config.auditory_enabled:
            self._create_static_synapse(
                "a1_food_to_assoc_edible", self.a1_food, self.assoc_edible,
                self.config.a1_food_to_assoc_edible_weight, sparsity=0.08)
        if self.config.social_brain_enabled and self.config.mirror_enabled:
            self._create_static_synapse(
                "social_mem_to_assoc_edible", self.social_memory, self.assoc_edible,
                self.config.social_memory_to_assoc_edible_weight, sparsity=0.05)
            self._create_static_synapse(
                "mirror_food_to_assoc_edible", self.mirror_food, self.assoc_edible,
                self.config.mirror_food_to_assoc_edible_weight, sparsity=0.05)
        # Recurrent
        self._create_static_synapse(
            "assoc_edible_recurrent", self.assoc_edible, self.assoc_edible,
            self.config.assoc_edible_recurrent, sparsity=0.05)

        # === 3. Assoc_Threatening 입력 ===
        if self.config.it_enabled:
            self._create_static_synapse(
                "it_danger_to_assoc_threatening", self.it_danger_category, self.assoc_threatening,
                self.config.it_danger_to_assoc_threatening_weight, sparsity=0.10)
        if self.config.multimodal_enabled:
            self._create_static_synapse(
                "sts_danger_to_assoc_threatening", self.sts_danger, self.assoc_threatening,
                self.config.sts_danger_to_assoc_threatening_weight, sparsity=0.08)
        if self.config.auditory_enabled:
            self._create_static_synapse(
                "a1_danger_to_assoc_threatening", self.a1_danger, self.assoc_threatening,
                self.config.a1_danger_to_assoc_threatening_weight, sparsity=0.08)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_assoc_threatening", self.fear_response, self.assoc_threatening,
                self.config.fear_to_assoc_threatening_weight, sparsity=0.08)
        # Recurrent
        self._create_static_synapse(
            "assoc_threatening_recurrent", self.assoc_threatening, self.assoc_threatening,
            self.config.assoc_threatening_recurrent, sparsity=0.05)

        # === 4. Edible ↔ Threatening WTA ===
        self._create_static_synapse(
            "assoc_edible_to_threatening", self.assoc_edible, self.assoc_threatening,
            self.config.assoc_edible_threatening_wta, sparsity=0.08)
        self._create_static_synapse(
            "assoc_threatening_to_edible", self.assoc_threatening, self.assoc_edible,
            self.config.assoc_edible_threatening_wta, sparsity=0.08)

        print(f"    Edible↔Threatening WTA: {self.config.assoc_edible_threatening_wta}")

        # === 5. Assoc_Animate 입력 ===
        if self.config.social_brain_enabled and self.config.tom_enabled:
            self._create_static_synapse(
                "tom_intent_to_assoc_animate", self.tom_intention, self.assoc_animate,
                self.config.tom_intention_to_assoc_animate_weight, sparsity=0.08)
        if self.config.social_brain_enabled and self.config.mirror_enabled:
            self._create_static_synapse(
                "social_obs_to_assoc_animate", self.social_observation, self.assoc_animate,
                self.config.social_obs_to_assoc_animate_weight, sparsity=0.08)
            self._create_static_synapse(
                "mirror_food_to_assoc_animate", self.mirror_food, self.assoc_animate,
                self.config.mirror_food_to_assoc_animate_weight, sparsity=0.05)
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "sts_social_to_assoc_animate", self.sts_social, self.assoc_animate,
                self.config.sts_social_to_assoc_animate_weight, sparsity=0.08)

        # === 6. Assoc_Context 입력 ===
        if self.config.hippocampus_enabled:
            self._create_static_synapse(
                "place_cells_to_assoc_context", self.place_cells, self.assoc_context,
                self.config.place_cells_to_assoc_context_weight, sparsity=0.02)
            if self.config.directional_food_memory:
                self._create_static_synapse(
                    "food_mem_l_to_assoc_context", self.food_memory_left, self.assoc_context,
                    self.config.food_memory_to_assoc_context_weight, sparsity=0.05)
                self._create_static_synapse(
                    "food_mem_r_to_assoc_context", self.food_memory_right, self.assoc_context,
                    self.config.food_memory_to_assoc_context_weight, sparsity=0.05)
        if self.config.parietal_enabled:
            self._create_static_synapse(
                "ppc_space_l_to_assoc_context", self.ppc_space_left, self.assoc_context,
                self.config.ppc_space_to_assoc_context_weight, sparsity=0.05)
            self._create_static_synapse(
                "ppc_space_r_to_assoc_context", self.ppc_space_right, self.assoc_context,
                self.config.ppc_space_to_assoc_context_weight, sparsity=0.05)
        # Recurrent
        self._create_static_synapse(
            "assoc_context_recurrent", self.assoc_context, self.assoc_context,
            self.config.assoc_context_recurrent, sparsity=0.05)

        # === 7. Assoc_Valence 입력 ===
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "dopamine_to_assoc_valence", self.dopamine_neurons, self.assoc_valence,
                self.config.dopamine_to_assoc_valence_weight, sparsity=0.08)
        self._create_static_synapse(
            "assoc_edible_to_valence", self.assoc_edible, self.assoc_valence,
            self.config.assoc_edible_to_valence_weight, sparsity=0.08)
        self._create_static_synapse(
            "assoc_threatening_to_valence", self.assoc_threatening, self.assoc_valence,
            self.config.assoc_threatening_to_valence_weight, sparsity=0.08)
        self._create_static_synapse(
            "satiety_to_assoc_valence", self.satiety_drive, self.assoc_valence,
            self.config.satiety_to_assoc_valence_weight, sparsity=0.05)

        # === 8. Assoc_Binding 입력 (2 Hebbian DENSE + 2 sparse + recurrent) ===
        # Hebbian DENSE: Edible → Binding
        from pygenn import init_weight_update, init_postsynaptic
        self.assoc_edible_to_binding_hebbian = self.model.add_synapse_population(
            "assoc_edible_to_binding_hebb", "DENSE",
            self.assoc_edible, self.assoc_binding,
            init_weight_update("StaticPulse", {},
                               {"g": self.config.assoc_edible_to_binding_weight}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))

        # Hebbian DENSE: Context → Binding
        self.assoc_context_to_binding_hebbian = self.model.add_synapse_population(
            "assoc_context_to_binding_hebb", "DENSE",
            self.assoc_context, self.assoc_binding,
            init_weight_update("StaticPulse", {},
                               {"g": self.config.assoc_context_to_binding_weight}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))

        # Sparse: Animate, Valence → Binding
        self._create_static_synapse(
            "assoc_animate_to_binding", self.assoc_animate, self.assoc_binding,
            self.config.assoc_animate_to_binding_weight, sparsity=0.08)
        self._create_static_synapse(
            "assoc_valence_to_binding", self.assoc_valence, self.assoc_binding,
            self.config.assoc_valence_to_binding_weight, sparsity=0.08)
        # Recurrent
        self._create_static_synapse(
            "assoc_binding_recurrent", self.assoc_binding, self.assoc_binding,
            self.config.assoc_binding_recurrent, sparsity=0.05)

        print(f"    Assoc_Binding: Hebbian DENSE (Edible, Context)")

        # === 9. Assoc_Novelty 입력 ===
        if self.config.it_enabled:
            self._create_static_synapse(
                "it_neutral_to_assoc_novelty", self.it_neutral_category, self.assoc_novelty,
                self.config.it_neutral_to_assoc_novelty_weight, sparsity=0.08)
        if self.config.multimodal_enabled:
            self._create_static_synapse(
                "sts_mismatch_to_assoc_novelty", self.sts_mismatch, self.assoc_novelty,
                self.config.sts_mismatch_to_assoc_novelty_weight, sparsity=0.08)
        self._create_static_synapse(
            "assoc_binding_to_novelty", self.assoc_binding, self.assoc_novelty,
            self.config.assoc_binding_to_novelty_weight, sparsity=0.08)

        # === 10. Top-Down 조절 ===
        self._create_static_synapse(
            "hunger_to_assoc_edible", self.hunger_drive, self.assoc_edible,
            self.config.hunger_to_assoc_edible_weight, sparsity=0.05)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_assoc_threatening_td", self.fear_response, self.assoc_threatening,
                self.config.fear_to_assoc_threatening_topdown_weight, sparsity=0.05)
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "wm_to_assoc_binding", self.working_memory, self.assoc_binding,
                self.config.wm_to_assoc_binding_weight, sparsity=0.05)

        # === 11. 출력 연결 (모두 ≤6.0, Motor = 0.0!) ===
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "assoc_edible_to_goal_food", self.assoc_edible, self.goal_food,
                self.config.assoc_edible_to_goal_food_weight, sparsity=0.05)
            self._create_static_synapse(
                "assoc_edible_to_wm", self.assoc_edible, self.working_memory,
                self.config.assoc_edible_to_wm_weight, sparsity=0.05)
            self._create_static_synapse(
                "assoc_threatening_to_goal_safety", self.assoc_threatening, self.goal_safety,
                self.config.assoc_threatening_to_goal_safety_weight, sparsity=0.05)
            self._create_static_synapse(
                "assoc_context_to_wm", self.assoc_context, self.working_memory,
                self.config.assoc_context_to_wm_weight, sparsity=0.05)

        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "assoc_threatening_to_acc", self.assoc_threatening, self.acc_conflict,
                self.config.assoc_threatening_to_acc_weight, sparsity=0.05)
            self._create_static_synapse(
                "assoc_animate_to_tpj", self.assoc_animate, self.tpj_other,
                self.config.assoc_animate_to_tpj_weight, sparsity=0.05)

        if self.config.hippocampus_enabled and self.config.directional_food_memory:
            self._create_static_synapse(
                "assoc_context_to_food_mem_l", self.assoc_context, self.food_memory_left,
                self.config.assoc_context_to_food_memory_weight, sparsity=0.03)
            self._create_static_synapse(
                "assoc_context_to_food_mem_r", self.assoc_context, self.food_memory_right,
                self.config.assoc_context_to_food_memory_weight, sparsity=0.03)

        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "assoc_valence_to_dopamine", self.assoc_valence, self.dopamine_neurons,
                self.config.assoc_valence_to_dopamine_weight, sparsity=0.05)
            self._create_static_synapse(
                "assoc_novelty_to_dopamine", self.assoc_novelty, self.dopamine_neurons,
                self.config.assoc_novelty_to_dopamine_weight, sparsity=0.05)

        if self.config.thalamus_enabled:
            self._create_static_synapse(
                "assoc_novelty_to_arousal", self.assoc_novelty, self.arousal,
                self.config.assoc_novelty_to_arousal_weight, sparsity=0.05)

        if self.config.it_enabled:
            self._create_static_synapse(
                "assoc_binding_to_it_assoc", self.assoc_binding, self.it_association,
                self.config.assoc_binding_to_it_assoc_weight, sparsity=0.05)

        n_assoc_total = (self.config.n_assoc_edible + self.config.n_assoc_threatening +
                         self.config.n_assoc_animate + self.config.n_assoc_context +
                         self.config.n_assoc_valence + self.config.n_assoc_binding +
                         self.config.n_assoc_novelty)
        print(f"    Phase 16 Association Cortex: {n_assoc_total} neurons")
        print(f"    Motor direct: {self.config.assoc_to_motor_weight} (disabled)")

    def learn_association_binding(self, reward_context: bool):
        """
        Phase 16: 연합 바인딩 Hebbian 학습

        Edible→Binding, Context→Binding DENSE 시냅스 가중치를 조정.
        음식을 먹으면 강한 학습, 그 외에는 약한 배경 학습.
        "이 장소에서 먹을 수 있는 것을 찾았다" 연합 형성.

        Args:
            reward_context: True = 음식 먹기 (강한 학습), False = 배경 (약한 학습)
        """
        if not self.config.association_cortex_enabled:
            return None

        eta = self.config.assoc_binding_eta
        w_max = self.config.assoc_binding_w_max
        learning_factor = 1.0 if reward_context else 0.2

        binding_scale = max(0.1, self.last_assoc_binding_rate)

        # Edible → Binding
        n_pre_e = self.config.n_assoc_edible
        n_post = self.config.n_assoc_binding
        self.assoc_edible_to_binding_hebbian.vars["g"].pull_from_device()
        w_e = self.assoc_edible_to_binding_hebbian.vars["g"].view.copy()
        w_e = w_e.reshape(n_pre_e, n_post)
        w_e += eta * learning_factor * binding_scale
        w_e = np.clip(w_e, 0.0, w_max)
        self.assoc_edible_to_binding_hebbian.vars["g"].view[:] = w_e.flatten()
        self.assoc_edible_to_binding_hebbian.vars["g"].push_to_device()

        # Context → Binding
        n_pre_c = self.config.n_assoc_context
        self.assoc_context_to_binding_hebbian.vars["g"].pull_from_device()
        w_c = self.assoc_context_to_binding_hebbian.vars["g"].view.copy()
        w_c = w_c.reshape(n_pre_c, n_post)
        w_c += eta * learning_factor * binding_scale
        w_c = np.clip(w_c, 0.0, w_max)
        self.assoc_context_to_binding_hebbian.vars["g"].view[:] = w_c.flatten()
        self.assoc_context_to_binding_hebbian.vars["g"].push_to_device()

        return {
            "avg_w_edible": float(np.mean(w_e)),
            "avg_w_context": float(np.mean(w_c)),
            "learning_factor": learning_factor,
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

        # === Phase 11: Sound 감각 입력 ===
        sound_danger_l = 0.0
        sound_danger_r = 0.0
        sound_food_l = 0.0
        sound_food_r = 0.0

        if self.config.auditory_enabled:
            sound_danger_l = observation.get("sound_danger_left", 0.0)
            sound_danger_r = observation.get("sound_danger_right", 0.0)
            sound_food_l = observation.get("sound_food_left", 0.0)
            sound_food_r = observation.get("sound_food_right", 0.0)

            # Sound Input 뉴런에 전류 주입
            sound_sensitivity = 50.0
            self.sound_danger_left.vars["I_input"].view[:] = sound_danger_l * sound_sensitivity
            self.sound_danger_right.vars["I_input"].view[:] = sound_danger_r * sound_sensitivity
            self.sound_food_left.vars["I_input"].view[:] = sound_food_l * sound_sensitivity
            self.sound_food_right.vars["I_input"].view[:] = sound_food_r * sound_sensitivity

            self.sound_danger_left.vars["I_input"].push_to_device()
            self.sound_danger_right.vars["I_input"].push_to_device()
            self.sound_food_left.vars["I_input"].push_to_device()
            self.sound_food_right.vars["I_input"].push_to_device()

        # === Phase 15: Agent 감각 입력 ===
        agent_eye_l = 0.0
        agent_eye_r = 0.0
        agent_sound_l = 0.0
        agent_sound_r = 0.0
        social_proximity = 0.0

        if self.config.social_brain_enabled:
            agent_eye_l = np.mean(observation.get("agent_rays_left", np.zeros(8)))
            agent_eye_r = np.mean(observation.get("agent_rays_right", np.zeros(8)))
            agent_sound_l = observation.get("agent_sound_left", 0.0)
            agent_sound_r = observation.get("agent_sound_right", 0.0)
            social_proximity = observation.get("social_proximity", 0.0)

            # Agent Eye 뉴런에 전류 주입
            agent_sensitivity = 50.0
            self.agent_eye_left.vars["I_input"].view[:] = agent_eye_l * agent_sensitivity
            self.agent_eye_right.vars["I_input"].view[:] = agent_eye_r * agent_sensitivity
            self.agent_sound_left.vars["I_input"].view[:] = agent_sound_l * agent_sensitivity
            self.agent_sound_right.vars["I_input"].view[:] = agent_sound_r * agent_sensitivity

            self.agent_eye_left.vars["I_input"].push_to_device()
            self.agent_eye_right.vars["I_input"].push_to_device()
            self.agent_sound_left.vars["I_input"].push_to_device()
            self.agent_sound_right.vars["I_input"].push_to_device()

            # Social proximity → ACC (직접 전류 주입)
            if social_proximity > 0:
                self.acc_conflict.vars["I_input"].view[:] = social_proximity * 30.0
                self.acc_conflict.vars["I_input"].push_to_device()

        # === Phase 15b: Mirror Neuron 감각 입력 ===
        npc_food_dir_l = 0.0
        npc_food_dir_r = 0.0
        npc_eating_l = 0.0
        npc_eating_r = 0.0
        npc_near_food = 0.0

        if self.config.social_brain_enabled and self.config.mirror_enabled:
            npc_food_dir_l = observation.get("npc_food_direction_left", 0.0)
            npc_food_dir_r = observation.get("npc_food_direction_right", 0.0)
            npc_eating_l = observation.get("npc_eating_left", 0.0)
            npc_eating_r = observation.get("npc_eating_right", 0.0)
            npc_near_food = observation.get("npc_near_food", 0.0)

            # Social_Observation: NPC 목표지향 움직임 (I_input)
            npc_food_dir = max(npc_food_dir_l, npc_food_dir_r)
            self.social_observation.vars["I_input"].view[:] = npc_food_dir * 45.0
            self.social_observation.vars["I_input"].push_to_device()

            # Mirror_Food: 자기 먹기 이벤트 (I_input)
            if self.mirror_self_eating_timer > 0:
                self.mirror_food.vars["I_input"].view[:] = 40.0
                self.mirror_self_eating_timer -= 1
            else:
                self.mirror_food.vars["I_input"].view[:] = 0.0
            self.mirror_food.vars["I_input"].push_to_device()

            # Vicarious_Reward: NPC 먹기 예측 오차 (I_input)
            npc_eating = max(npc_eating_l, npc_eating_r)
            surprise_factor = max(0.1, 1.0 - self.last_social_obs_rate)
            self.vicarious_reward.vars["I_input"].view[:] = npc_eating * surprise_factor * 50.0
            self.vicarious_reward.vars["I_input"].push_to_device()

        # === Phase 15c: Theory of Mind 감각 입력 ===
        npc_intention_food = 0.0
        npc_heading_change = 0.0
        npc_competition = 0.0

        if self.config.social_brain_enabled and self.config.tom_enabled:
            npc_intention_food = observation.get("npc_intention_food", 0.0)
            npc_heading_change = observation.get("npc_heading_change", 0.0)
            npc_competition = observation.get("npc_competition", 0.0)

            # ToM_Intention: NPC 음식 추구 확신도 (I_input)
            self.tom_intention.vars["I_input"].view[:] = npc_intention_food * 45.0
            self.tom_intention.vars["I_input"].push_to_device()

            # ToM_Surprise: NPC 방향 불안정성 (I_input)
            self.tom_surprise.vars["I_input"].view[:] = npc_heading_change * 40.0
            self.tom_surprise.vars["I_input"].push_to_device()

            # CoopCompete_Compete: NPC 경쟁 신호 (I_input)
            self.coop_compete_compete.vars["I_input"].view[:] = npc_competition * 45.0
            self.coop_compete_compete.vars["I_input"].push_to_device()

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

        # Phase 8 스파이크 카운트 (V1)
        v1_food_left_spikes = 0
        v1_food_right_spikes = 0
        v1_danger_left_spikes = 0
        v1_danger_right_spikes = 0

        # Phase 9 스파이크 카운트 (V2/V4)
        v2_edge_food_spikes = 0
        v2_edge_danger_spikes = 0
        v4_food_object_spikes = 0
        v4_danger_object_spikes = 0
        v4_novel_object_spikes = 0

        # Phase 10 스파이크 카운트 (IT Cortex)
        it_food_category_spikes = 0
        it_danger_category_spikes = 0
        it_neutral_category_spikes = 0
        it_association_spikes = 0
        it_memory_buffer_spikes = 0

        # Phase 11 스파이크 카운트 (Auditory Cortex)
        a1_danger_spikes = 0
        a1_food_spikes = 0
        a2_association_spikes = 0

        # Phase 12 스파이크 카운트 (Multimodal Integration)
        sts_food_spikes = 0
        sts_danger_spikes = 0
        sts_congruence_spikes = 0
        sts_mismatch_spikes = 0

        # Phase 13 스파이크 카운트 (Parietal Cortex)
        ppc_space_left_spikes = 0
        ppc_space_right_spikes = 0
        ppc_goal_food_spikes = 0
        ppc_goal_safety_spikes = 0
        ppc_attention_spikes = 0
        ppc_path_buffer_spikes = 0

        # Phase 14 스파이크 카운트 (Premotor Cortex)
        pmd_left_spikes = 0
        pmd_right_spikes = 0
        pmv_approach_spikes = 0
        pmv_avoid_spikes = 0
        sma_sequence_spikes = 0
        motor_prep_spikes = 0

        # Phase 15 스파이크 카운트 (Social Brain)
        sts_social_spikes = 0
        tpj_self_spikes = 0
        tpj_other_spikes = 0
        tpj_compare_spikes = 0
        acc_conflict_spikes = 0
        acc_monitor_spikes = 0
        social_approach_spikes = 0
        social_avoid_spikes = 0

        # Phase 15b 스파이크 카운트 (Mirror Neurons)
        social_obs_spikes = 0
        mirror_food_spikes = 0
        vicarious_reward_spikes = 0
        social_memory_spikes = 0

        # Phase 15c 스파이크 카운트 (Theory of Mind)
        tom_intention_spikes = 0
        tom_belief_spikes = 0
        tom_prediction_spikes = 0
        tom_surprise_spikes = 0
        coop_spikes = 0
        compete_spikes = 0

        # Phase 16 스파이크 카운트 (Association Cortex)
        assoc_edible_spikes = 0
        assoc_threatening_spikes = 0
        assoc_animate_spikes = 0
        assoc_context_spikes = 0
        assoc_valence_spikes = 0
        assoc_binding_spikes = 0
        assoc_novelty_spikes = 0

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

            # Phase 8 스파이크 카운팅 (V1)
            if self.config.v1_enabled:
                self.v1_food_left.vars["RefracTime"].pull_from_device()
                self.v1_food_right.vars["RefracTime"].pull_from_device()
                self.v1_danger_left.vars["RefracTime"].pull_from_device()
                self.v1_danger_right.vars["RefracTime"].pull_from_device()

                v1_food_left_spikes += np.sum(self.v1_food_left.vars["RefracTime"].view > self.spike_threshold)
                v1_food_right_spikes += np.sum(self.v1_food_right.vars["RefracTime"].view > self.spike_threshold)
                v1_danger_left_spikes += np.sum(self.v1_danger_left.vars["RefracTime"].view > self.spike_threshold)
                v1_danger_right_spikes += np.sum(self.v1_danger_right.vars["RefracTime"].view > self.spike_threshold)

            # Phase 9 스파이크 카운팅 (V2/V4)
            if self.config.v2v4_enabled and self.config.v1_enabled:
                self.v2_edge_food.vars["RefracTime"].pull_from_device()
                self.v2_edge_danger.vars["RefracTime"].pull_from_device()
                self.v4_food_object.vars["RefracTime"].pull_from_device()
                self.v4_danger_object.vars["RefracTime"].pull_from_device()
                self.v4_novel_object.vars["RefracTime"].pull_from_device()

                v2_edge_food_spikes += np.sum(self.v2_edge_food.vars["RefracTime"].view > self.spike_threshold)
                v2_edge_danger_spikes += np.sum(self.v2_edge_danger.vars["RefracTime"].view > self.spike_threshold)
                v4_food_object_spikes += np.sum(self.v4_food_object.vars["RefracTime"].view > self.spike_threshold)
                v4_danger_object_spikes += np.sum(self.v4_danger_object.vars["RefracTime"].view > self.spike_threshold)
                v4_novel_object_spikes += np.sum(self.v4_novel_object.vars["RefracTime"].view > self.spike_threshold)

            # Phase 10 스파이크 카운팅 (IT Cortex)
            if self.config.it_enabled and self.config.v2v4_enabled:
                self.it_food_category.vars["RefracTime"].pull_from_device()
                self.it_danger_category.vars["RefracTime"].pull_from_device()
                self.it_neutral_category.vars["RefracTime"].pull_from_device()
                self.it_association.vars["RefracTime"].pull_from_device()
                self.it_memory_buffer.vars["RefracTime"].pull_from_device()

                it_food_category_spikes += np.sum(self.it_food_category.vars["RefracTime"].view > self.spike_threshold)
                it_danger_category_spikes += np.sum(self.it_danger_category.vars["RefracTime"].view > self.spike_threshold)
                it_neutral_category_spikes += np.sum(self.it_neutral_category.vars["RefracTime"].view > self.spike_threshold)
                it_association_spikes += np.sum(self.it_association.vars["RefracTime"].view > self.spike_threshold)
                it_memory_buffer_spikes += np.sum(self.it_memory_buffer.vars["RefracTime"].view > self.spike_threshold)

            # Phase 11 스파이크 카운팅 (Auditory Cortex)
            if self.config.auditory_enabled:
                self.a1_danger.vars["RefracTime"].pull_from_device()
                self.a1_food.vars["RefracTime"].pull_from_device()
                self.a2_association.vars["RefracTime"].pull_from_device()

                a1_danger_spikes += np.sum(self.a1_danger.vars["RefracTime"].view > self.spike_threshold)
                a1_food_spikes += np.sum(self.a1_food.vars["RefracTime"].view > self.spike_threshold)
                a2_association_spikes += np.sum(self.a2_association.vars["RefracTime"].view > self.spike_threshold)

            # Phase 12 스파이크 카운팅 (Multimodal Integration)
            if self.config.multimodal_enabled:
                self.sts_food.vars["RefracTime"].pull_from_device()
                self.sts_danger.vars["RefracTime"].pull_from_device()
                self.sts_congruence.vars["RefracTime"].pull_from_device()
                self.sts_mismatch.vars["RefracTime"].pull_from_device()

                sts_food_spikes += np.sum(self.sts_food.vars["RefracTime"].view > self.spike_threshold)
                sts_danger_spikes += np.sum(self.sts_danger.vars["RefracTime"].view > self.spike_threshold)
                sts_congruence_spikes += np.sum(self.sts_congruence.vars["RefracTime"].view > self.spike_threshold)
                sts_mismatch_spikes += np.sum(self.sts_mismatch.vars["RefracTime"].view > self.spike_threshold)

            # Phase 13 스파이크 카운팅 (Parietal Cortex)
            if self.config.parietal_enabled:
                self.ppc_space_left.vars["RefracTime"].pull_from_device()
                self.ppc_space_right.vars["RefracTime"].pull_from_device()
                self.ppc_goal_food.vars["RefracTime"].pull_from_device()
                self.ppc_goal_safety.vars["RefracTime"].pull_from_device()
                self.ppc_attention.vars["RefracTime"].pull_from_device()
                self.ppc_path_buffer.vars["RefracTime"].pull_from_device()

                ppc_space_left_spikes += np.sum(self.ppc_space_left.vars["RefracTime"].view > self.spike_threshold)
                ppc_space_right_spikes += np.sum(self.ppc_space_right.vars["RefracTime"].view > self.spike_threshold)
                ppc_goal_food_spikes += np.sum(self.ppc_goal_food.vars["RefracTime"].view > self.spike_threshold)
                ppc_goal_safety_spikes += np.sum(self.ppc_goal_safety.vars["RefracTime"].view > self.spike_threshold)
                ppc_attention_spikes += np.sum(self.ppc_attention.vars["RefracTime"].view > self.spike_threshold)
                ppc_path_buffer_spikes += np.sum(self.ppc_path_buffer.vars["RefracTime"].view > self.spike_threshold)

            # Phase 14 스파이크 카운팅 (Premotor Cortex)
            if self.config.premotor_enabled:
                self.pmd_left.vars["RefracTime"].pull_from_device()
                self.pmd_right.vars["RefracTime"].pull_from_device()
                self.pmv_approach.vars["RefracTime"].pull_from_device()
                self.pmv_avoid.vars["RefracTime"].pull_from_device()
                self.sma_sequence.vars["RefracTime"].pull_from_device()
                self.motor_preparation.vars["RefracTime"].pull_from_device()

                pmd_left_spikes += np.sum(self.pmd_left.vars["RefracTime"].view > self.spike_threshold)
                pmd_right_spikes += np.sum(self.pmd_right.vars["RefracTime"].view > self.spike_threshold)
                pmv_approach_spikes += np.sum(self.pmv_approach.vars["RefracTime"].view > self.spike_threshold)
                pmv_avoid_spikes += np.sum(self.pmv_avoid.vars["RefracTime"].view > self.spike_threshold)
                sma_sequence_spikes += np.sum(self.sma_sequence.vars["RefracTime"].view > self.spike_threshold)
                motor_prep_spikes += np.sum(self.motor_preparation.vars["RefracTime"].view > self.spike_threshold)

            # Phase 15 스파이크 카운팅 (Social Brain)
            if self.config.social_brain_enabled:
                self.sts_social.vars["RefracTime"].pull_from_device()
                self.tpj_self.vars["RefracTime"].pull_from_device()
                self.tpj_other.vars["RefracTime"].pull_from_device()
                self.tpj_compare.vars["RefracTime"].pull_from_device()
                self.acc_conflict.vars["RefracTime"].pull_from_device()
                self.acc_monitor.vars["RefracTime"].pull_from_device()
                self.social_approach.vars["RefracTime"].pull_from_device()
                self.social_avoid.vars["RefracTime"].pull_from_device()

                sts_social_spikes += np.sum(self.sts_social.vars["RefracTime"].view > self.spike_threshold)
                tpj_self_spikes += np.sum(self.tpj_self.vars["RefracTime"].view > self.spike_threshold)
                tpj_other_spikes += np.sum(self.tpj_other.vars["RefracTime"].view > self.spike_threshold)
                tpj_compare_spikes += np.sum(self.tpj_compare.vars["RefracTime"].view > self.spike_threshold)
                acc_conflict_spikes += np.sum(self.acc_conflict.vars["RefracTime"].view > self.spike_threshold)
                acc_monitor_spikes += np.sum(self.acc_monitor.vars["RefracTime"].view > self.spike_threshold)
                social_approach_spikes += np.sum(self.social_approach.vars["RefracTime"].view > self.spike_threshold)
                social_avoid_spikes += np.sum(self.social_avoid.vars["RefracTime"].view > self.spike_threshold)

                # Phase 15b 스파이크 카운팅 (Mirror Neurons)
                if self.config.mirror_enabled:
                    self.social_observation.vars["RefracTime"].pull_from_device()
                    self.mirror_food.vars["RefracTime"].pull_from_device()
                    self.vicarious_reward.vars["RefracTime"].pull_from_device()
                    self.social_memory.vars["RefracTime"].pull_from_device()

                    social_obs_spikes += np.sum(self.social_observation.vars["RefracTime"].view > self.spike_threshold)
                    mirror_food_spikes += np.sum(self.mirror_food.vars["RefracTime"].view > self.spike_threshold)
                    vicarious_reward_spikes += np.sum(self.vicarious_reward.vars["RefracTime"].view > self.spike_threshold)
                    social_memory_spikes += np.sum(self.social_memory.vars["RefracTime"].view > self.spike_threshold)

                # Phase 15c 스파이크 카운팅 (Theory of Mind)
                if self.config.tom_enabled:
                    self.tom_intention.vars["RefracTime"].pull_from_device()
                    self.tom_belief.vars["RefracTime"].pull_from_device()
                    self.tom_prediction.vars["RefracTime"].pull_from_device()
                    self.tom_surprise.vars["RefracTime"].pull_from_device()
                    self.coop_compete_coop.vars["RefracTime"].pull_from_device()
                    self.coop_compete_compete.vars["RefracTime"].pull_from_device()

                    tom_intention_spikes += np.sum(self.tom_intention.vars["RefracTime"].view > self.spike_threshold)
                    tom_belief_spikes += np.sum(self.tom_belief.vars["RefracTime"].view > self.spike_threshold)
                    tom_prediction_spikes += np.sum(self.tom_prediction.vars["RefracTime"].view > self.spike_threshold)
                    tom_surprise_spikes += np.sum(self.tom_surprise.vars["RefracTime"].view > self.spike_threshold)
                    coop_spikes += np.sum(self.coop_compete_coop.vars["RefracTime"].view > self.spike_threshold)
                    compete_spikes += np.sum(self.coop_compete_compete.vars["RefracTime"].view > self.spike_threshold)

            # Phase 16 스파이크 카운팅 (Association Cortex)
            if self.config.association_cortex_enabled:
                self.assoc_edible.vars["RefracTime"].pull_from_device()
                self.assoc_threatening.vars["RefracTime"].pull_from_device()
                self.assoc_animate.vars["RefracTime"].pull_from_device()
                self.assoc_context.vars["RefracTime"].pull_from_device()
                self.assoc_valence.vars["RefracTime"].pull_from_device()
                self.assoc_binding.vars["RefracTime"].pull_from_device()
                self.assoc_novelty.vars["RefracTime"].pull_from_device()

                assoc_edible_spikes += np.sum(self.assoc_edible.vars["RefracTime"].view > self.spike_threshold)
                assoc_threatening_spikes += np.sum(self.assoc_threatening.vars["RefracTime"].view > self.spike_threshold)
                assoc_animate_spikes += np.sum(self.assoc_animate.vars["RefracTime"].view > self.spike_threshold)
                assoc_context_spikes += np.sum(self.assoc_context.vars["RefracTime"].view > self.spike_threshold)
                assoc_valence_spikes += np.sum(self.assoc_valence.vars["RefracTime"].view > self.spike_threshold)
                assoc_binding_spikes += np.sum(self.assoc_binding.vars["RefracTime"].view > self.spike_threshold)
                assoc_novelty_spikes += np.sum(self.assoc_novelty.vars["RefracTime"].view > self.spike_threshold)

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

        # Phase 8 스파이크율 (V1)
        v1_food_left_rate = 0.0
        v1_food_right_rate = 0.0
        v1_danger_left_rate = 0.0
        v1_danger_right_rate = 0.0
        if self.config.v1_enabled:
            max_spikes_v1_food = self.config.n_v1_food_left * 5
            max_spikes_v1_danger = self.config.n_v1_danger_left * 5

            v1_food_left_rate = v1_food_left_spikes / max_spikes_v1_food
            v1_food_right_rate = v1_food_right_spikes / max_spikes_v1_food
            v1_danger_left_rate = v1_danger_left_spikes / max_spikes_v1_danger
            v1_danger_right_rate = v1_danger_right_spikes / max_spikes_v1_danger

        # Phase 9 스파이크율 (V2/V4)
        v2_edge_food_rate = 0.0
        v2_edge_danger_rate = 0.0
        v4_food_object_rate = 0.0
        v4_danger_object_rate = 0.0
        v4_novel_object_rate = 0.0
        if self.config.v2v4_enabled and self.config.v1_enabled:
            max_spikes_v2_food = self.config.n_v2_edge_food * 5
            max_spikes_v2_danger = self.config.n_v2_edge_danger * 5
            max_spikes_v4_food = self.config.n_v4_food_object * 5
            max_spikes_v4_danger = self.config.n_v4_danger_object * 5
            max_spikes_v4_novel = self.config.n_v4_novel_object * 5

            v2_edge_food_rate = v2_edge_food_spikes / max_spikes_v2_food
            v2_edge_danger_rate = v2_edge_danger_spikes / max_spikes_v2_danger
            v4_food_object_rate = v4_food_object_spikes / max_spikes_v4_food
            v4_danger_object_rate = v4_danger_object_spikes / max_spikes_v4_danger
            v4_novel_object_rate = v4_novel_object_spikes / max_spikes_v4_novel

        # Phase 10 스파이크율 (IT Cortex)
        it_food_category_rate = 0.0
        it_danger_category_rate = 0.0
        it_neutral_category_rate = 0.0
        it_association_rate = 0.0
        it_memory_buffer_rate = 0.0
        if self.config.it_enabled and self.config.v2v4_enabled:
            max_spikes_it_food = self.config.n_it_food_category * 5
            max_spikes_it_danger = self.config.n_it_danger_category * 5
            max_spikes_it_neutral = self.config.n_it_neutral_category * 5
            max_spikes_it_assoc = self.config.n_it_association * 5
            max_spikes_it_buffer = self.config.n_it_memory_buffer * 5

            it_food_category_rate = it_food_category_spikes / max_spikes_it_food
            it_danger_category_rate = it_danger_category_spikes / max_spikes_it_danger
            it_neutral_category_rate = it_neutral_category_spikes / max_spikes_it_neutral
            it_association_rate = it_association_spikes / max_spikes_it_assoc
            it_memory_buffer_rate = it_memory_buffer_spikes / max_spikes_it_buffer

        # Phase 11 스파이크율 (Auditory Cortex)
        a1_danger_rate = 0.0
        a1_food_rate = 0.0
        a2_association_rate = 0.0
        if self.config.auditory_enabled:
            max_spikes_a1_danger = self.config.n_a1_danger * 5
            max_spikes_a1_food = self.config.n_a1_food * 5
            max_spikes_a2 = self.config.n_a2_association * 5

            a1_danger_rate = a1_danger_spikes / max_spikes_a1_danger
            a1_food_rate = a1_food_spikes / max_spikes_a1_food
            a2_association_rate = a2_association_spikes / max_spikes_a2

        # Phase 12 스파이크율 (Multimodal Integration)
        sts_food_rate = 0.0
        sts_danger_rate = 0.0
        sts_congruence_rate = 0.0
        sts_mismatch_rate = 0.0
        if self.config.multimodal_enabled:
            max_spikes_sts_food = self.config.n_sts_food * 5
            max_spikes_sts_danger = self.config.n_sts_danger * 5
            max_spikes_congruence = self.config.n_sts_congruence * 5
            max_spikes_mismatch = self.config.n_sts_mismatch * 5

            sts_food_rate = sts_food_spikes / max_spikes_sts_food
            sts_danger_rate = sts_danger_spikes / max_spikes_sts_danger
            sts_congruence_rate = sts_congruence_spikes / max_spikes_congruence
            sts_mismatch_rate = sts_mismatch_spikes / max_spikes_mismatch

        # Phase 13 스파이크율 (Parietal Cortex)
        ppc_space_left_rate = 0.0
        ppc_space_right_rate = 0.0
        ppc_goal_food_rate = 0.0
        ppc_goal_safety_rate = 0.0
        ppc_attention_rate = 0.0
        ppc_path_buffer_rate = 0.0
        if self.config.parietal_enabled:
            max_spikes_ppc_space = self.config.n_ppc_space_left * 5
            max_spikes_ppc_goal = self.config.n_ppc_goal_food * 5
            max_spikes_ppc_attention = self.config.n_ppc_attention * 5
            max_spikes_ppc_path = self.config.n_ppc_path_buffer * 5

            ppc_space_left_rate = ppc_space_left_spikes / max_spikes_ppc_space
            ppc_space_right_rate = ppc_space_right_spikes / max_spikes_ppc_space
            ppc_goal_food_rate = ppc_goal_food_spikes / max_spikes_ppc_goal
            ppc_goal_safety_rate = ppc_goal_safety_spikes / max_spikes_ppc_goal
            ppc_attention_rate = ppc_attention_spikes / max_spikes_ppc_attention
            ppc_path_buffer_rate = ppc_path_buffer_spikes / max_spikes_ppc_path

        # Phase 14 스파이크율 (Premotor Cortex)
        pmd_left_rate = 0.0
        pmd_right_rate = 0.0
        pmv_approach_rate = 0.0
        pmv_avoid_rate = 0.0
        sma_sequence_rate = 0.0
        motor_prep_rate = 0.0
        if self.config.premotor_enabled:
            max_spikes_pmd = self.config.n_pmd_left * 5
            max_spikes_pmv = self.config.n_pmv_approach * 5
            max_spikes_sma = self.config.n_sma_sequence * 5
            max_spikes_motor_prep = self.config.n_motor_preparation * 5

            pmd_left_rate = pmd_left_spikes / max_spikes_pmd
            pmd_right_rate = pmd_right_spikes / max_spikes_pmd
            pmv_approach_rate = pmv_approach_spikes / max_spikes_pmv
            pmv_avoid_rate = pmv_avoid_spikes / max_spikes_pmv
            sma_sequence_rate = sma_sequence_spikes / max_spikes_sma
            motor_prep_rate = motor_prep_spikes / max_spikes_motor_prep

        # Phase 15 스파이크율 (Social Brain)
        sts_social_rate = 0.0
        tpj_self_rate = 0.0
        tpj_other_rate = 0.0
        tpj_compare_rate = 0.0
        acc_conflict_rate = 0.0
        acc_monitor_rate = 0.0
        social_approach_rate = 0.0
        social_avoid_rate = 0.0
        if self.config.social_brain_enabled:
            sts_social_rate = sts_social_spikes / (self.config.n_sts_social * 5)
            tpj_self_rate = tpj_self_spikes / (self.config.n_tpj_self * 5)
            tpj_other_rate = tpj_other_spikes / (self.config.n_tpj_other * 5)
            tpj_compare_rate = tpj_compare_spikes / (self.config.n_tpj_compare * 5)
            acc_conflict_rate = acc_conflict_spikes / (self.config.n_acc_conflict * 5)
            acc_monitor_rate = acc_monitor_spikes / (self.config.n_acc_monitor * 5)
            social_approach_rate = social_approach_spikes / (self.config.n_social_approach * 5)
            social_avoid_rate = social_avoid_spikes / (self.config.n_social_avoid * 5)

        # Phase 15b 스파이크율 (Mirror Neurons)
        social_obs_rate = 0.0
        mirror_food_rate = 0.0
        vicarious_reward_rate = 0.0
        social_memory_rate = 0.0
        if self.config.social_brain_enabled and self.config.mirror_enabled:
            social_obs_rate = social_obs_spikes / (self.config.n_social_observation * 5)
            mirror_food_rate = mirror_food_spikes / (self.config.n_mirror_food * 5)
            vicarious_reward_rate = vicarious_reward_spikes / (self.config.n_vicarious_reward * 5)
            social_memory_rate = social_memory_spikes / (self.config.n_social_memory * 5)
            # 예측 오차 계산을 위해 관찰율 저장
            self.last_social_obs_rate = social_obs_rate

        # Phase 15c 스파이크율 (Theory of Mind)
        tom_intention_rate = 0.0
        tom_belief_rate = 0.0
        tom_prediction_rate = 0.0
        tom_surprise_rate = 0.0
        coop_rate = 0.0
        compete_rate = 0.0
        if self.config.social_brain_enabled and self.config.tom_enabled:
            tom_intention_rate = tom_intention_spikes / (self.config.n_tom_intention * 5)
            tom_belief_rate = tom_belief_spikes / (self.config.n_tom_belief * 5)
            tom_prediction_rate = tom_prediction_spikes / (self.config.n_tom_prediction * 5)
            tom_surprise_rate = tom_surprise_spikes / (self.config.n_tom_surprise * 5)
            coop_rate = coop_spikes / (self.config.n_coop_compete_coop * 5)
            compete_rate = compete_spikes / (self.config.n_coop_compete_compete * 5)
            self.last_tom_intention_rate = tom_intention_rate

        # Phase 16 스파이크율 (Association Cortex)
        assoc_edible_rate = 0.0
        assoc_threatening_rate = 0.0
        assoc_animate_rate = 0.0
        assoc_context_rate = 0.0
        assoc_valence_rate = 0.0
        assoc_binding_rate = 0.0
        assoc_novelty_rate = 0.0
        if self.config.association_cortex_enabled:
            assoc_edible_rate = assoc_edible_spikes / (self.config.n_assoc_edible * 5)
            assoc_threatening_rate = assoc_threatening_spikes / (self.config.n_assoc_threatening * 5)
            assoc_animate_rate = assoc_animate_spikes / (self.config.n_assoc_animate * 5)
            assoc_context_rate = assoc_context_spikes / (self.config.n_assoc_context * 5)
            assoc_valence_rate = assoc_valence_spikes / (self.config.n_assoc_valence * 5)
            assoc_binding_rate = assoc_binding_spikes / (self.config.n_assoc_binding * 5)
            assoc_novelty_rate = assoc_novelty_spikes / (self.config.n_assoc_novelty * 5)
            self.last_assoc_binding_rate = assoc_binding_rate

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

            # Phase 8 뉴런 활성화 (V1)
            "v1_food_left_rate": v1_food_left_rate,
            "v1_food_right_rate": v1_food_right_rate,
            "v1_danger_left_rate": v1_danger_left_rate,
            "v1_danger_right_rate": v1_danger_right_rate,

            # Phase 9 뉴런 활성화 (V2/V4)
            "v2_edge_food_rate": v2_edge_food_rate,
            "v2_edge_danger_rate": v2_edge_danger_rate,
            "v4_food_object_rate": v4_food_object_rate,
            "v4_danger_object_rate": v4_danger_object_rate,
            "v4_novel_object_rate": v4_novel_object_rate,

            # Phase 10 뉴런 활성화 (IT Cortex)
            "it_food_category_rate": it_food_category_rate,
            "it_danger_category_rate": it_danger_category_rate,
            "it_neutral_category_rate": it_neutral_category_rate,
            "it_association_rate": it_association_rate,
            "it_memory_buffer_rate": it_memory_buffer_rate,

            # Phase 11 뉴런 활성화 (Auditory Cortex)
            "a1_danger_rate": a1_danger_rate,
            "a1_food_rate": a1_food_rate,
            "a2_association_rate": a2_association_rate,

            # Phase 11 Sound 입력
            "sound_danger_l": sound_danger_l,
            "sound_danger_r": sound_danger_r,
            "sound_food_l": sound_food_l,
            "sound_food_r": sound_food_r,

            # Phase 12 뉴런 활성화 (Multimodal Integration)
            "sts_food_rate": sts_food_rate,
            "sts_danger_rate": sts_danger_rate,
            "sts_congruence_rate": sts_congruence_rate,
            "sts_mismatch_rate": sts_mismatch_rate,

            # Phase 13 뉴런 활성화 (Parietal Cortex)
            "ppc_space_left_rate": ppc_space_left_rate,
            "ppc_space_right_rate": ppc_space_right_rate,
            "ppc_goal_food_rate": ppc_goal_food_rate,
            "ppc_goal_safety_rate": ppc_goal_safety_rate,
            "ppc_attention_rate": ppc_attention_rate,
            "ppc_path_buffer_rate": ppc_path_buffer_rate,

            # Phase 14 뉴런 활성화 (Premotor Cortex)
            "pmd_left_rate": pmd_left_rate,
            "pmd_right_rate": pmd_right_rate,
            "pmv_approach_rate": pmv_approach_rate,
            "pmv_avoid_rate": pmv_avoid_rate,
            "sma_sequence_rate": sma_sequence_rate,
            "motor_prep_rate": motor_prep_rate,

            # Phase 15 뉴런 활성화 (Social Brain)
            "sts_social_rate": sts_social_rate,
            "tpj_self_rate": tpj_self_rate,
            "tpj_other_rate": tpj_other_rate,
            "tpj_compare_rate": tpj_compare_rate,
            "acc_conflict_rate": acc_conflict_rate,
            "acc_monitor_rate": acc_monitor_rate,
            "social_approach_rate": social_approach_rate,
            "social_avoid_rate": social_avoid_rate,

            # Phase 15 입력
            "agent_eye_l": agent_eye_l,
            "agent_eye_r": agent_eye_r,
            "social_proximity": social_proximity,

            # Phase 15b 뉴런 활성화 (Mirror Neurons)
            "social_obs_rate": social_obs_rate,
            "mirror_food_rate": mirror_food_rate,
            "vicarious_reward_rate": vicarious_reward_rate,
            "social_memory_rate": social_memory_rate,

            # Phase 15b 입력
            "npc_food_dir_l": npc_food_dir_l,
            "npc_food_dir_r": npc_food_dir_r,
            "npc_eating_l": npc_eating_l,
            "npc_eating_r": npc_eating_r,
            "npc_near_food": npc_near_food,

            # Phase 15c 뉴런 활성화 (Theory of Mind)
            "tom_intention_rate": tom_intention_rate,
            "tom_belief_rate": tom_belief_rate,
            "tom_prediction_rate": tom_prediction_rate,
            "tom_surprise_rate": tom_surprise_rate,
            "coop_rate": coop_rate,
            "compete_rate": compete_rate,

            # Phase 15c 입력
            "npc_intention_food": npc_intention_food,
            "npc_heading_change": npc_heading_change,
            "npc_competition": npc_competition,

            # Phase 16 뉴런 활성화 (Association Cortex)
            "assoc_edible_rate": assoc_edible_rate,
            "assoc_threatening_rate": assoc_threatening_rate,
            "assoc_animate_rate": assoc_animate_rate,
            "assoc_context_rate": assoc_context_rate,
            "assoc_valence_rate": assoc_valence_rate,
            "assoc_binding_rate": assoc_binding_rate,
            "assoc_novelty_rate": assoc_novelty_rate,

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
                persist_learning: bool = False, no_learning: bool = False,
                fps: int = 10, food_patch: bool = False,
                no_multimodal: bool = False, no_parietal: bool = False,
                no_premotor: bool = False, no_social: bool = False,
                no_mirror: bool = False, no_tom: bool = False,
                no_association: bool = False):
    """Phase 6b 훈련 실행"""

    print("=" * 70)
    print("Phase 6b: Forager Training with Thalamus (Sensory Gating & Attention)")
    print("=" * 70)
    if persist_learning:
        print("  [!] PERSIST LEARNING ENABLED - weights saved/loaded between episodes")
    if no_learning:
        print("  [!] LEARNING DISABLED - baseline mode (no Hebbian learning)")
    if food_patch:
        print("  [!] FOOD PATCH MODE ENABLED - Hebbian learning validation")

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
    if no_multimodal:
        brain_config.multimodal_enabled = False
        print("  [!] Phase 12 (Multimodal Integration) DISABLED")
    if no_parietal:
        brain_config.parietal_enabled = False
        print("  [!] Phase 13 (Parietal Cortex) DISABLED")
    if no_premotor:
        brain_config.premotor_enabled = False
        print("  [!] Phase 14 (Premotor Cortex) DISABLED")
    if no_social:
        brain_config.social_brain_enabled = False
        env_config.social_enabled = False
        print("  [!] Phase 15 (Social Brain) DISABLED")
    if no_mirror:
        brain_config.mirror_enabled = False
        print("  [!] Phase 15b (Mirror Neurons) DISABLED")
    if no_tom:
        brain_config.tom_enabled = False
        print("  [!] Phase 15c (Theory of Mind) DISABLED")
    if no_association:
        brain_config.association_cortex_enabled = False
        print("  [!] Phase 16 (Association Cortex) DISABLED")
    if food_patch:
        env_config.food_patch_enabled = True
        print(f"      Patches: {env_config.n_patches}, radius={env_config.patch_radius}")
        print(f"      Spawn in patch: {env_config.food_spawn_in_patch_prob*100:.0f}%")

    env = ForagerGym(env_config, render_mode=render_mode)
    env.render_fps = fps  # FPS 설정 (시각화 속도 조절)
    brain = ForagerBrain(brain_config)

    # 학습 비활성화 옵션
    if no_learning:
        brain.food_learning_enabled = False

    # 통계
    all_steps = []
    all_food = []
    all_homeostasis = []
    all_pain_visits = []
    all_pain_steps = []
    death_causes = {"starve": 0, "timeout": 0, "pain": 0}

    # Phase 3b: 학습 통계
    all_learn_events = []  # 총 학습 이벤트 수

    # Food Patch 통계
    all_patch_visits = []   # 에피소드별 [patch0_visits, patch1_visits, ...]
    all_patch_food = []     # 에피소드별 [patch0_food, patch1_food, ...]

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

                # Phase 15b: 자기 먹기 → Mirror 활성화
                if brain_config.social_brain_enabled and brain_config.mirror_enabled:
                    brain.mirror_self_eating_timer = env_config.npc_eating_signal_duration

                # Phase 15c: 협력 가치 학습 (음식 먹기 시)
                if brain_config.social_brain_enabled and brain_config.tom_enabled:
                    food_near_npc = False
                    for npc in env.npc_agents:
                        if npc.target_food is not None:
                            tfx, tfy = npc.target_food
                            dist_to_npc_target = np.sqrt(
                                (obs["position_x"] * env_config.width - tfx)**2 +
                                (obs["position_y"] * env_config.height - tfy)**2)
                            if dist_to_npc_target < 50.0:
                                food_near_npc = True
                                break
                    coop_learn = brain.learn_cooperation_value(food_near_npc)
                    if coop_learn and log_level in ["debug", "verbose"]:
                        print(f"  [TOM] Coop learning: avg_w={coop_learn['avg_weight']:.2f}, "
                              f"factor={coop_learn['learning_factor']:.1f}")

                # Phase 16: 연합 바인딩 학습 (음식 먹기 = 강한 학습)
                if brain_config.association_cortex_enabled:
                    assoc_learn = brain.learn_association_binding(reward_context=True)

                if log_level in ["normal", "debug", "verbose"]:
                    da_str = f", DA={dopamine_info['dopamine_level']:.2f}" if dopamine_info else ""
                    if learn_info:
                        side_str = f", side={learn_info.get('side', 'N/A')}" if 'side' in learn_info else ""
                        print(f"  [!] FOOD EATEN at step {env.steps}, Energy: {env_info['energy']:.1f} "
                              f"[LEARN: {learn_info['n_strengthened']} cells, avg_w={learn_info['avg_weight']:.2f}{side_str}{da_str}]")
                    else:
                        print(f"  [!] FOOD EATEN at step {env.steps}, Energy: {env_info['energy']:.1f}{da_str}")

            # Phase 15b: NPC 먹기 관찰 → 사회적 학습
            if brain_config.social_brain_enabled and brain_config.mirror_enabled:
                npc_events = env_info.get("npc_eating_events", [])
                for npc_x, npc_y, npc_step in npc_events:
                    npc_pos = (npc_x / env_config.width, npc_y / env_config.height)
                    social_learn = brain.learn_social_food_location(npc_pos)
                    if social_learn and log_level in ["debug", "verbose"]:
                        print(f"  [SOCIAL] NPC ate at ({npc_x:.0f},{npc_y:.0f}), "
                              f"avg_w={social_learn['avg_weight']:.2f}, surprise={social_learn['surprise']:.2f}")

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

        # Food Patch 통계
        if env_config.food_patch_enabled:
            all_patch_visits.append(env_info.get("patch_visits", []))
            all_patch_food.append(env_info.get("patch_food_eaten", []))

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

        # Food Patch 통계
        if env_config.food_patch_enabled:
            pv = env_info.get("patch_visits", [])
            pf = env_info.get("patch_food_eaten", [])
            print(f"  --- Food Patch ---")
            print(f"  Total Patch Visits: {sum(pv)}")
            print(f"  Patch Food: {sum(pf)}/{env.total_food_eaten} ({100*sum(pf)/max(1,env.total_food_eaten):.0f}%)")
            for i, (v, f) in enumerate(zip(pv, pf)):
                print(f"    Patch {i}: {v} visits, {f} food")

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

    # Food Patch 학습 효과 검증
    if env_config.food_patch_enabled and len(all_patch_visits) > 0:
        print(f"\n  === Food Patch Learning Validation ===")

        # 초반 vs 후반 Patch 방문 비교
        n_early = min(5, episodes // 2)
        n_late = min(5, episodes // 2)

        if episodes >= 6:  # 최소 6 에피소드 필요
            early_visits = sum(sum(v) for v in all_patch_visits[:n_early])
            late_visits = sum(sum(v) for v in all_patch_visits[-n_late:])
            visit_change = (late_visits - early_visits) / max(1, early_visits) * 100

            early_patch_food = sum(sum(f) for f in all_patch_food[:n_early])
            late_patch_food = sum(sum(f) for f in all_patch_food[-n_late:])
            food_change = (late_patch_food - early_patch_food) / max(1, early_patch_food) * 100

            print(f"  Early (ep 1-{n_early}):")
            print(f"    Patch Visits: {early_visits}")
            print(f"    Patch Food:   {early_patch_food}")
            print(f"  Late (ep {episodes-n_late+1}-{episodes}):")
            print(f"    Patch Visits: {late_visits}")
            print(f"    Patch Food:   {late_patch_food}")
            print(f"  Change:")
            print(f"    Visit Change: {visit_change:+.1f}% {'✓' if visit_change > 30 else '✗'} (target: >30%)")
            print(f"    Food Change:  {food_change:+.1f}%")

        # 학습 후 가중치 변화
        if brain_config.hippocampus_enabled:
            stats = brain.get_hippocampus_stats()
            if stats:
                initial_weight = brain_config.place_to_food_memory_weight
                weight_change = (stats['avg_weight'] - initial_weight) / initial_weight * 100
                print(f"  Weight Change:")
                print(f"    Initial: {initial_weight:.2f} → Final: {stats['avg_weight']:.2f}")
                print(f"    Change:  {weight_change:+.1f}% {'✓' if stats['avg_weight'] > 3.0 else '✗'} (target: avg > 3.0)")

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
    parser.add_argument("--no-learning", action="store_true",
                       help="Disable Hebbian learning (for baseline comparison)")
    parser.add_argument("--fps", type=int, default=10,
                       help="Render FPS (default: 10, slower=easier to observe)")
    parser.add_argument("--food-patch", action="store_true",
                       help="Enable Food Patch mode for Hebbian learning validation")
    # Phase 비활성화 플래그 (검증용)
    parser.add_argument("--no-multimodal", action="store_true",
                       help="Disable Phase 12 (Multimodal Integration)")
    parser.add_argument("--no-parietal", action="store_true",
                       help="Disable Phase 13 (Parietal Cortex)")
    parser.add_argument("--no-premotor", action="store_true",
                       help="Disable Phase 14 (Premotor Cortex)")
    parser.add_argument("--no-social", action="store_true",
                       help="Disable Phase 15 (Social Brain)")
    parser.add_argument("--no-mirror", action="store_true",
                       help="Disable Phase 15b (Mirror Neurons)")
    parser.add_argument("--no-tom", action="store_true",
                       help="Disable Phase 15c (Theory of Mind)")
    parser.add_argument("--no-association", action="store_true",
                       help="Disable Phase 16 (Association Cortex)")
    args = parser.parse_args()

    run_training(
        episodes=args.episodes,
        render_mode=args.render,
        log_level=args.log_level,
        debug=args.debug,
        no_amygdala=args.no_amygdala,
        no_pain=args.no_pain,
        persist_learning=args.persist_learning,
        no_learning=args.no_learning,
        fps=args.fps,
        food_patch=args.food_patch,
        no_multimodal=args.no_multimodal,
        no_parietal=args.no_parietal,
        no_premotor=args.no_premotor,
        no_social=args.no_social,
        no_mirror=args.no_mirror,
        no_tom=args.no_tom,
        no_association=args.no_association
    )
