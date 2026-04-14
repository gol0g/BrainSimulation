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
from data_logger import DataLogger

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
    # Obstacle detection (wall_rays에서 분리, 약한 회피)
    obstacle_eye_enabled: bool = True
    n_obstacle_eye: int = 400   # Obstacle detection (L: 200, R: 200)
    obstacle_push_weight: float = 8.0    # 매우 약한 힌트 수준 (wall 60의 13%)
    obstacle_pull_weight: float = -4.0   # 부드러운 회피 (wall -40의 10%)

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

    # === BASAL GANGLIA (Phase 4 / Phase L1-L2: D1/D2 MSN 분리) ===
    basal_ganglia_enabled: bool = True      # 기저핵 활성화 여부
    n_d1_msn: int = 200                    # D1 MSN 총 (100L + 100R) - Go pathway, R-STDP 학습
    n_d2_msn: int = 200                    # D2 MSN 총 (100L + 100R) - NoGo pathway, Static
    n_direct_pathway: int = 200             # Direct 총 (100L + 100R)
    n_indirect_pathway: int = 200           # Indirect 총 (100L + 100R)
    n_dopamine: int = 100                   # Dopamine neurons (VTA/SNc)
    msn_capacitance: float = 30.0           # D1/D2 MSN C (Phase L1: 입력에 비례한 발화율)

    # === MOTOR ===
    n_motor_left: int = 500
    n_motor_right: int = 500
    motor_capacitance: float = 300.0  # Motor neuron C (BUG-001: 1→100→300, 단일입력 포화 방지)

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
    food_weight: float = 35.0                # Legacy: perceptual_learning 비활성 시에만 사용
    # 학습 기반 음식 접근 (food_eye 35.0 하드코딩 대체)
    food_approach_init_w: float = 25.0      # good_food_eye→Motor 초기 가중치 (강한 접근 본능)
    food_approach_w_max: float = 40.0       # 최대 가중치 (충분히 높게 — 학습으로 도달)
    food_approach_eta: float = 0.001        # 학습률 (좋은 음식 먹으면 강화)

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

    # Food Memory → Motor (약한 편향 — L12: GW 경유 라우팅으로 약화)
    food_memory_to_motor_weight: float = 5.0  # L12: 12→5 (GW가 +4.0 조건부 보상)

    # Hunger → Food Memory (배고플 때 기억 활성화)
    hunger_to_food_memory_weight: float = 10.0 # 기억 탐색 활성화 (20→10, 간섭 최소화)

    # === Phase 4 시냅스 가중치 (Phase L2: D1/D2 MSN 분리 + R-STDP) ===
    # Food_Eye → D1 MSN (R-STDP 학습 대상)
    food_to_d1_init_weight: float = 1.0       # R-STDP 초기 가중치 (D1만 학습)
    # Food_Eye → D2 MSN (Static, 학습 안 함)
    food_to_d2_weight: float = 1.0            # D2 정적 가중치

    # D1 → Direct / D2 → Indirect pathways
    d1_to_direct_weight: float = 20.0          # D1→Go 신호
    d2_to_indirect_weight: float = 15.0        # D2→NoGo 신호
    direct_indirect_competition: float = -10.0 # Direct↔Indirect 상호 억제
    d1_d2_competition: float = -5.0            # D1↔D2 측면 경쟁

    # Direct/Indirect → Motor (Phase L1: 측면화로 재활성화)
    direct_to_motor_weight: float = 25.0       # Phase L1: Go 강화
    direct_to_motor_contra_weight: float = -8.0  # Phase L1: BG Push-Pull (교차 억제)
    indirect_to_motor_weight: float = -10.0    # Phase L1: NoGo

    # Dopamine modulation (Phase L2: MSN 레벨로 이동)
    dopamine_to_d1_weight: float = 15.0        # DA → D1 흥분 (D1 수용체)
    dopamine_to_d2_weight: float = -12.0       # DA → D2 억제 (D2 수용체)

    # R-STDP 학습 파라미터 (Phase L3: Homeostatic, D1 MSN)
    rstdp_eta: float = 0.0005                  # R-STDP 학습률 (L3: 0.001→0.0005, 점진적 학습)
    rstdp_trace_decay: float = 0.95            # 적격 추적 감쇠 (τ≈20 steps)
    rstdp_trace_max: float = 1.0               # L3: 추적 상한 (무한 누적 방지)
    rstdp_w_max: float = 5.0                   # R-STDP 최대 가중치
    rstdp_weight_decay: float = 0.00003        # L3: 항상성 가중치 감쇠 (시냅스 스케일링)
    rstdp_w_rest: float = 1.0                  # L3: 감쇠 평형점 (= 초기 가중치)

    # Phase L4: Anti-Hebbian D2 학습 파라미터
    rstdp_d2_eta: float = 0.0003              # D2 Anti-Hebbian 학습률 (D1보다 약하게)
    rstdp_d2_w_min: float = 0.1               # D2 최소 가중치 (완전 소멸 방지)

    # Phase L5: Perceptual Learning (지각 학습)
    perceptual_learning_enabled: bool = True
    n_good_food_eye: int = 400                 # 200L + 200R
    n_bad_food_eye: int = 400                  # 200L + 200R
    good_food_eye_sensitivity: float = 50.0
    bad_food_eye_sensitivity: float = 50.0
    cortical_rstdp_eta: float = 0.0008
    cortical_rstdp_w_max: float = 8.0
    cortical_rstdp_w_min: float = 0.1
    cortical_rstdp_init_w: float = 2.0
    cortical_rstdp_trace_decay: float = 0.90
    cortical_rstdp_trace_max: float = 1.0
    cortical_rstdp_weight_decay: float = 0.00002
    cortical_rstdp_w_rest: float = 2.0
    cortical_anti_hebbian_ratio: float = 0.6   # Anti-Hebbian 약화 비율 (R-STDP 대비)
    taste_aversion_magnitude: float = 15.0     # 맛 혐오 → Lateral Amygdala I_input

    # Phase L13: Conditioned Taste Aversion (bad_food_eye → LA Hebbian)
    taste_aversion_learning_enabled: bool = True
    taste_aversion_hebbian_eta: float = 0.02      # 학습률 (Garcia Effect: one-trial 학습, 0.003→0.02)
    taste_aversion_hebbian_w_max: float = 5.0     # 최대 가중치 (danger_to_la 25.0보다 낮게)
    taste_aversion_hebbian_init_w: float = 0.1    # 초기 가중치 (LA 초기 간섭 방지)

    # Phase L6: Prediction Error Circuit (예측 오차)
    prediction_error_enabled: bool = True
    n_pe_food: int = 100               # 50L + 50R (음식 예측 오차)
    n_pe_danger: int = 100             # 50L + 50R (위험 예측 오차)
    pe_v1_to_pe_weight: float = 10.0   # V1 → PE (excitatory, bottom-up actual)
    pe_it_to_pe_weight: float = -7.0   # IT → PE (inhibitory, top-down prediction)
    pe_to_it_init_w: float = 1.0       # PE → IT 초기 가중치 (gentle modulator)
    pe_to_it_w_max: float = 3.0        # PE → IT 최대 가중치
    pe_to_it_w_min: float = 0.1        # PE → IT 최소 가중치
    pe_rstdp_eta: float = 0.0005       # PE R-STDP 학습률
    pe_trace_decay: float = 0.92       # PE 적격 추적 감쇠
    pe_trace_max: float = 1.0          # PE 적격 추적 최대값
    pe_weight_decay: float = 0.00002   # PE 가중치 항상성 감쇠
    pe_w_rest: float = 1.0             # PE 가중치 평형점 (init_w와 동일)

    # Phase L7: Discriminative BG Learning (음식 유형별 BG 학습)
    discriminative_bg_enabled: bool = True
    typed_food_d1_init_w: float = 1.0     # good/bad food → D1 초기 가중치
    typed_food_d2_init_w: float = 1.0     # good/bad food → D2 초기 가중치
    typed_food_bg_sparsity: float = 0.08  # BG 연결 희소도

    # Phase L8: Aversive Dopamine Dip (나쁜 음식 → 도파민 감소)
    dopamine_dip_enabled: bool = True
    dopamine_dip_magnitude: float = 0.5   # burst(1.0) 대비 50% (생물학적 비대칭)

    # Phase L9: IT Cortex → BG Learning (피질 하향 연결)
    it_bg_enabled: bool = True
    it_to_d1_init_w: float = 0.5      # IT→D1 초기 가중치 (food_eye의 절반, 모듈레이터)
    it_to_d2_init_w: float = 0.5      # IT→D2 초기 가중치
    it_to_bg_sparsity: float = 0.05   # IT는 비측화, 과잉 연결 방지

    # Phase L10: TD Learning (NAc Critic → RPE Dopamine)
    td_learning_enabled: bool = True
    n_nac_value: int = 80                    # NAc shell value neurons (MSN-like)
    n_nac_inhibitory: int = 30               # NAc local inhibition
    nac_food_eye_init_w: float = 1.0         # food_eye → NAc 초기 가중치
    nac_food_eye_sparsity: float = 0.08      # food_eye → NAc SPARSE
    nac_it_food_weight: float = 0.5          # IT_Food → NAc (static)
    nac_place_weight: float = 0.3            # Place_Cells → NAc (static)
    nac_rstdp_eta: float = 0.0005            # NAc R-STDP 학습률 (D1과 동일)
    nac_w_max: float = 5.0                   # NAc max weight
    rpe_discount: float = 0.5                # RPE discount (0=no RPE, 1=full)
    rpe_prediction_threshold: float = 0.3    # NAc rate 30% = 완전 예측
    rpe_floor: float = 0.1                   # 최소 DA 10% (학습 완전 차단 방지)

    # Phase L11: SWR Replay (Hippocampal Sequence)
    swr_replay_enabled: bool = True
    n_ca3_sequence: int = 100                # CA3 시퀀스 뉴런
    n_swr_gate: int = 50                     # SWR 게이트 (I_input 전용)
    n_replay_inhibitory: int = 50            # 리플레이 중 Motor 억제
    swr_replay_count: int = 5               # 에피소드당 리플레이 횟수
    swr_replay_steps: int = 10              # 리플레이 1회당 시뮬레이션 스텝
    swr_experience_max: int = 50            # 경험 버퍼 최대 크기
    swr_place_current_scale: float = 0.3    # 리플레이 시 Place Cell 전류 스케일 (온라인의 30%)
    swr_motor_inhibit_weight: float = -15.0 # replay_inh → Motor 억제 가중치
    swr_gate_to_inh_weight: float = 8.0     # SWR gate → replay_inh 가중치
    place_to_ca3_weight: float = 3.0        # Place → CA3 (static)
    place_to_ca3_sparsity: float = 0.05     # Place → CA3 연결 확률
    # M3 Revaluation: Place→Place transition + Value backup
    place_transition_enabled: bool = True
    place_transition_eta: float = 0.01      # PC_t→PC_t+1 STDP 학습률
    place_transition_w_max: float = 5.0
    place_transition_init_w: float = 0.01     # 0.1→0.01: recurrent excitation 감소
    n_place_value: int = 20                 # Value population (offline value backup)
    place_to_value_eta: float = 0.005       # DA-gated 3-factor
    place_to_value_w_max: float = 3.0
    value_to_d1_weight: float = 1.0         # Value → BG gentle bias
    reverse_replay_steps: int = 5           # reverse replay chain length
    ca3_to_food_memory_weight: float = 2.0  # CA3 → Food Memory (static, 약함)

    # Phase L12: Global Workspace (Attention — Dehaene & Changeux 2011)
    gw_enabled: bool = True
    n_gw_food: int = 50                        # per side (L/R)
    n_gw_safety: int = 60
    gw_food_memory_weight: float = 6.0         # food_memory → GW_Food
    gw_hunger_weight: float = 5.0              # hunger_drive → GW_Food (허용 게이트)
    gw_good_food_eye_weight: float = 3.0       # 직접 감각 부스트
    gw_fear_weight: float = 12.0               # fear → GW_Safety
    gw_la_weight: float = 8.0                  # lateral_amygdala → GW_Safety
    gw_safety_inhibit_weight: float = -12.0    # GW_Safety → GW_Food 억제
    gw_food_to_motor_weight: float = 4.0       # GW_Food → Motor (약!)
    gw_food_to_motor_sparsity: float = 0.05

    # Dopamine 파라미터
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
    inhibitory_to_motor_weight: float = 0.0         # 억제 → Motor (-2→0, 대칭이라 방향성 없음)

    # Goal → Motor (목표 지향 행동)
    goal_food_to_motor_weight: float = 0.0          # 음식 목표 → Motor (18→0, 대칭이라 방향성 없음)

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
    food_relay_to_motor_weight: float = 0.0        # Food Relay → Motor (8→0, 대칭이라 방향성 없음)
    danger_relay_to_amygdala_weight: float = 15.0  # Danger Relay → Amygdala 증폭
    danger_relay_to_motor_weight: float = 0.0      # Danger Relay → Motor (10→0, 대칭이라 방향성 없음)

    # 각성 조절
    low_energy_to_arousal_weight: float = 20.0     # 낮은 에너지 → 높은 각성
    high_energy_to_arousal_weight: float = -15.0   # 높은 에너지 → 낮은 각성
    arousal_to_motor_weight: float = 0.0           # 각성 → Motor (6→0, 대칭이라 방향성 없음)
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
    deep_to_motor_weight: float = 0.0              # 운동 조절 (8→0, 대칭이라 방향성 없음)

    # === Phase 8 시냅스 가중치 (V1) ===
    # 입력: Relay → V1
    food_relay_to_v1_weight: float = 20.0          # Food Relay → V1 Food
    danger_relay_to_v1_weight: float = 20.0        # Danger Relay → V1 Danger

    # 내부: Lateral Inhibition (대비 강화)
    v1_lateral_inhibition: float = -8.0            # V1 좌우 상호 억제

    # 출력: V1 → 다른 영역
    v1_to_motor_weight: float = 0.0                # V1 → Motor (15→0, Food Ipsi 40.0/Pain Push 60.0과 중복)
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
    it_to_motor_weight: float = 0.0                # IT → Motor (12→0, 양쪽 동일이라 방향성 없음)

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

    # A1 → Motor (청각 유도 행동) — C1: Push-Pull 활성화
    a1_to_motor_weight: float = 0.0                 # danger sound (유지: 0.0)
    sound_food_push_weight: float = 8.0             # sound_food ipsi (접근)
    sound_food_pull_weight: float = -4.0            # sound_food contra (억제)

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
    motor_prep_to_motor_weight: float = 0.0            # Motor_Prep → Motor (2→0, 대칭이라 방향성 없음)
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

    # === Phase 17: Language Circuit (언어 회로 - Broca/Wernicke) ===
    language_enabled: bool = True

    # 감각 입력 (SensoryLIF)
    n_call_food_input_left: int = 50
    n_call_food_input_right: int = 50
    n_call_danger_input_left: int = 50
    n_call_danger_input_right: int = 50

    # Wernicke's Area (이해)
    n_wernicke_food: int = 80
    n_wernicke_danger: int = 80
    n_wernicke_social: int = 60
    n_wernicke_context: int = 60

    # Broca's Area (생산)
    n_broca_food: int = 80
    n_broca_danger: int = 80
    n_broca_social: int = 60
    n_broca_sequence: int = 60

    # Vocal Gate / PAG (SensoryLIF)
    n_vocal_gate: int = 80

    # Call Mirror + Call Binding
    n_call_mirror: int = 80
    n_call_binding: int = 80

    # Call Input → Wernicke
    call_to_wernicke_weight: float = 20.0

    # Wernicke 내부
    wernicke_food_danger_wta: float = -10.0
    wernicke_to_social_weight: float = 8.0
    wernicke_to_context_weight: float = 10.0
    wernicke_context_recurrent: float = 6.0

    # Broca 입력 (기존 회로 → Broca)
    assoc_edible_to_broca_food_weight: float = 10.0
    hunger_to_broca_food_weight: float = 8.0
    assoc_threatening_to_broca_danger_weight: float = 10.0
    fear_to_broca_danger_weight: float = 8.0
    sts_social_to_broca_social_weight: float = 8.0
    arousal_to_broca_social_weight: float = 6.0

    # Broca 내부
    broca_food_danger_wta: float = -10.0
    broca_to_sequence_weight: float = 12.0
    broca_sequence_to_broca_inh: float = -6.0
    broca_sequence_recurrent: float = -8.0

    # Arcuate Fasciculus (양방향)
    wernicke_to_broca_weight: float = 8.0       # mirror resonance
    broca_to_wernicke_weight: float = 6.0       # self-monitoring

    # Vocal Gate / PAG
    arousal_to_vocal_gate_weight: float = 10.0
    broca_to_vocal_gate_weight: float = 12.0

    # Call Mirror
    wernicke_to_call_mirror_weight: float = 10.0
    broca_to_call_mirror_weight: float = 10.0

    # Call Binding (Hebbian)
    wernicke_to_call_binding_weight: float = 10.0
    assoc_to_call_binding_weight: float = 8.0
    call_binding_recurrent: float = 6.0
    call_binding_eta: float = 0.06
    call_binding_w_max: float = 18.0

    # 출력 (모두 ≤6.0, Motor=0.0!)
    wernicke_food_to_goal_food_weight: float = 4.0
    wernicke_danger_to_fear_weight: float = 5.0
    wernicke_social_to_sts_social_weight: float = 4.0
    call_mirror_to_assoc_binding_weight: float = 3.0
    call_binding_to_assoc_edible_weight: float = 3.0
    call_binding_to_assoc_threatening_weight: float = 3.0
    call_binding_to_wm_weight: float = 3.0
    language_to_motor_weight: float = 0.0       # 절대 비활성!

    # Top-Down
    hunger_to_wernicke_food_weight: float = 5.0
    fear_to_wernicke_danger_weight: float = 6.0
    wm_to_wernicke_context_weight: float = 4.0

    # Context 입력
    place_to_wernicke_context_weight: float = 4.0
    sts_social_to_wernicke_context_weight: float = 5.0

    # === Phase 18: Working Memory Expansion (작업 기억 확장) ===
    wm_expansion_enabled: bool = True

    # 인구 크기 (8개)
    n_wm_thalamic: int = 100           # MD thalamus analog
    n_wm_update_gate: int = 50         # Dopamine-gated update control
    n_temporal_recent: int = 80        # Current event buffer (~1s)
    n_temporal_prior: int = 40         # Previous event buffer (~3s)
    n_goal_pending: int = 80           # Next goal in queue
    n_goal_switch: int = 70            # Context switch detector
    n_wm_context_binding: int = 100    # Temporal pattern association (Hebbian)
    n_wm_inhibitory: int = 100         # WM-local inhibitory interneurons

    # 시상-피질 루프
    wm_to_wm_thalamic_weight: float = 5.0
    wm_thalamic_to_wm_weight: float = 4.0
    wm_gate_to_thalamic_weight: float = -10.0
    trn_to_wm_thalamic_weight: float = -3.0
    arousal_to_wm_thalamic_weight: float = 3.0

    # Gate inputs (synaptic)
    dopamine_to_wm_gate_weight: float = 6.0
    acc_conflict_to_wm_gate_weight: float = 5.0
    novelty_to_wm_gate_weight: float = 5.0

    # Gate I_input scaling
    wm_gate_dopamine_scale: float = 12.0
    wm_gate_novelty_scale: float = 15.0
    wm_gate_conflict_scale: float = 15.0

    # 시간 버퍼
    temporal_recent_recurrent_weight: float = 7.0
    temporal_prior_recurrent_weight: float = 4.0
    temporal_recent_to_prior_weight: float = 3.0
    temporal_recent_to_wm_weight: float = 4.0

    # 목표 순서화
    wm_to_goal_pending_weight: float = 5.0
    goal_to_pending_inhibit_weight: float = -8.0
    goal_switch_self_inhibit_weight: float = -8.0
    goal_switch_to_goal_inhibit_weight: float = -6.0

    # WM 문맥 학습 (Hebbian)
    wm_context_binding_eta: float = 0.05
    wm_context_binding_w_max: float = 16.0
    wm_context_binding_init_weight: float = 2.0
    wm_context_to_wm_weight: float = 4.0
    wm_context_to_pending_weight: float = 3.0

    # 억제 균형
    wm_to_inhibitory_weight: float = 6.0
    wm_thalamic_to_inhibitory_weight: float = 4.0
    inhibitory_to_wm_weight: float = -5.0
    inhibitory_to_thalamic_weight: float = -4.0
    inhibitory_to_temporal_weight: float = -3.0
    inhibitory_to_pending_weight: float = -3.0

    # WM expansion Motor 직접 연결 = 0.0 (절대 비활성)
    wm_expansion_to_motor_weight: float = 0.0

    # === Phase 19: Metacognition (메타인지) ===
    metacognition_enabled: bool = True

    # Population sizes (5 populations, 380 total)
    n_meta_confidence: int = 80       # Anterior Insula analog
    n_meta_uncertainty: int = 80      # dACC error-likelihood
    n_meta_evaluate: int = 80         # mPFC self-evaluation (SensoryLIF)
    n_meta_arousal_mod: int = 70      # NE uncertainty-arousal coupling
    n_meta_inhibitory: int = 70       # Local inhibitory balance

    # 19a: Confidence inputs
    assoc_valence_to_confidence_weight: float = 5.0
    sts_congruence_to_confidence_weight: float = 4.0
    goal_food_to_confidence_weight: float = 4.0
    goal_safety_to_confidence_weight: float = 4.0
    wm_context_to_confidence_weight: float = 3.0
    meta_confidence_recurrent_weight: float = 5.0

    # 19b: Uncertainty inputs
    acc_conflict_to_uncertainty_weight: float = 4.0
    error_signal_to_uncertainty_weight: float = 4.0
    assoc_novelty_to_uncertainty_weight: float = 4.0
    tom_surprise_to_uncertainty_weight: float = 3.0
    sts_mismatch_to_uncertainty_weight: float = 3.0
    meta_uncertainty_recurrent_weight: float = 4.0

    # 19c: WTA
    meta_confidence_uncertainty_wta_weight: float = -5.0

    # 19d: Meta_Evaluate gate (I_input scaling)
    meta_eval_uncertainty_scale: float = 6.0
    meta_eval_confidence_scale: float = -5.0
    meta_eval_dopamine_scale: float = 4.0

    # 19e: Outputs (ALL <=2.0, NO Motor direct) - very gentle modulator
    meta_confidence_to_goal_food_weight: float = 1.5
    meta_confidence_to_goal_safety_weight: float = 1.5
    meta_confidence_to_goal_switch_weight: float = -2.0
    meta_confidence_to_wm_thalamic_weight: float = 1.0
    meta_evaluate_to_goal_switch_weight: float = 1.5
    meta_evaluate_to_arousal_mod_weight: float = 2.0
    meta_evaluate_to_inhibitory_ctrl_weight: float = 1.5
    meta_arousal_mod_to_arousal_weight: float = 2.0
    meta_arousal_mod_to_dopamine_weight: float = 1.5

    # 19f: Inhibitory balance
    meta_conf_to_inhibitory_weight: float = 4.0
    meta_uncert_to_inhibitory_weight: float = 4.0
    meta_inhibitory_to_conf_weight: float = -3.0
    meta_inhibitory_to_uncert_weight: float = -3.0
    meta_inhibitory_to_eval_weight: float = -2.5

    # 19g: Hebbian learning (Valence → Confidence)
    meta_confidence_binding_eta: float = 0.04
    meta_confidence_binding_w_max: float = 14.0
    meta_confidence_binding_init_weight: float = 2.0

    # Motor direct = 0.0
    metacognition_to_motor_weight: float = 0.0

    # ─── Phase 20: Self-Model ───
    self_model_enabled: bool = True
    n_self_body: int = 80
    n_self_efference: int = 80
    n_self_predict: int = 70
    n_self_agency: int = 70
    n_self_narrative: int = 80
    n_self_inhibitory: int = 60

    # 20a: Body inputs (interoception)
    hunger_to_self_body_weight: float = 4.0
    fear_to_self_body_weight: float = 4.0
    meta_conf_to_self_body_weight: float = 3.0
    meta_uncert_to_self_body_weight: float = 3.0
    dopamine_to_self_body_weight: float = 3.0

    # 20a-I: Body I_input scales
    self_body_energy_scale: float = 8.0
    self_body_hunger_scale: float = -6.0
    self_body_satiety_scale: float = 5.0

    # 20b: Efference inputs (motor copy)
    motor_to_efference_weight: float = 4.0

    # 20c: Predict I_input scales
    self_predict_efference_scale: float = 6.0
    self_predict_food_eye_scale: float = 5.0

    # 20d: Agency inputs
    efference_to_agency_weight: float = 4.0
    predict_to_agency_weight: float = 3.0
    food_memory_to_agency_weight: float = -3.0

    # 20e: Narrative inputs
    body_to_narrative_weight: float = 3.0
    agency_to_narrative_weight: float = 3.0
    wm_context_to_narrative_weight: float = 2.0
    narrative_recurrent_weight: float = 4.0

    # 20f: Outputs (ALL ≤1.5, NO Motor direct)
    self_body_to_meta_conf_weight: float = 1.0
    self_body_to_meta_uncert_weight: float = -1.0
    self_agency_to_goal_food_weight: float = 1.0
    self_agency_to_goal_switch_weight: float = -1.5
    self_narrative_to_wm_weight: float = 1.0
    self_predict_to_error_weight: float = 1.5

    # 20g: Inhibitory balance
    self_to_inhibitory_weight: float = 3.0
    self_inhibitory_to_body_weight: float = -2.5
    self_inhibitory_to_agency_weight: float = -2.5
    self_inhibitory_to_narrative_weight: float = -2.0

    # 20h: Hebbian learning (Body → Narrative)
    self_narrative_binding_eta: float = 0.04
    self_narrative_binding_w_max: float = 14.0
    self_narrative_binding_init_weight: float = 2.0

    # Motor direct = 0.0
    self_model_to_motor_weight: float = 0.0

    # ─── Phase L14: Agency Detection (Forward Model Learning) ───
    agency_detection_enabled: bool = True
    n_agency_pe: int = 50                          # Agency Prediction Error neurons

    # Forward model learning (self_efference → self_predict)
    agency_forward_model_eta: float = 0.005        # Hebbian learning rate (0.04→0.005: 포화 방지)
    agency_forward_model_w_max: float = 10.0       # Max weight
    agency_forward_model_init_w: float = 1.0       # Initial weight

    # Agency PE synaptic weights
    v1_food_to_agency_pe_weight: float = 8.0       # Actual sensory → PE (excitatory)
    predict_to_agency_pe_weight: float = -6.0      # Predicted → PE (inhibitory, cancels)
    agency_pe_to_agency_weight: float = -2.0       # High PE suppresses agency
    agency_pe_to_uncertainty_weight: float = 1.5   # High PE → uncertain
    agency_to_confidence_weight: float = 1.0       # High agency → confident

    # Agency PE inhibitory balance
    agency_pe_to_inhibitory_weight: float = 2.0
    agency_inhibitory_to_pe_weight: float = -1.5

    # ─── Phase L15: Narrative Self (Agency-Gated Autobiographical Learning) ───
    narrative_self_enabled: bool = True
    # Agency→Narrative DENSE Hebbian
    agency_to_narrative_eta: float = 0.01          # Gentle learning rate
    agency_to_narrative_w_max: float = 8.0         # Max weight (gentle modulator)
    agency_to_narrative_init_w: float = 1.0        # Initial weight
    # Agency gating for body→narrative learning
    narrative_agency_gate_baseline: float = 0.15   # Agency rate normalization baseline
    # Body state change detection
    narrative_body_change_scale: float = 10.0      # Amplify Δbody for salience

    # ─── Phase L16: Sparse Expansion Layer (Mushroom Body / DG) ───
    # 단일 KC(3000×2) + 단일 inhibitory(400×2) — 모든 입력이 같은 KC
    sparse_expansion_enabled: bool = True
    n_kc_per_side: int = 3000
    n_kc_inhibitory_per_side: int = 400
    # Legacy compartment sizes (unused, kept for checkpoint compat)
    n_kc_visual_per_side: int = 2000
    n_kc_auditory_per_side: int = 1000
    n_kc_spatial_per_side: int = 500
    n_kc_visual_inh_per_side: int = 200
    n_kc_auditory_inh_per_side: int = 200
    n_kc_spatial_inh_per_side: int = 100
    kc_food_eye_weight: float = 3.0
    kc_food_eye_sparsity: float = 0.10
    kc_good_bad_food_weight: float = 4.0
    kc_good_bad_food_sparsity: float = 0.10
    kc_it_food_weight: float = 2.0
    kc_it_food_sparsity: float = 0.05
    kc_to_inh_weight: float = 5.0
    kc_to_inh_sparsity: float = 0.05
    kc_inh_to_kc_weight: float = -15.0
    kc_inh_to_kc_sparsity: float = 0.08
    kc_to_d1_init_w: float = 0.5
    kc_to_d1_sparsity: float = 0.05
    kc_rstdp_eta: float = 0.0003
    kc_rstdp_w_max: float = 5.0
    kc_rstdp_w_rest: float = 0.5
    kc_d2_eta: float = 0.0002
    kc_d2_w_min: float = 0.05
    # Legacy auditory params (unused after rollback)
    kc_auditory_to_d1_sparsity: float = 0.20
    kc_rstdp_eta_auditory: float = 0.0005
    kc_d2_eta_auditory: float = 0.0003

    # === Phase C4: Contextual Prediction (경험 기반 예측) ===
    contextual_prediction_enabled: bool = True
    n_pred_food_soon: int = 30               # 예측 뉴런 (food 예측 readout)
    n_pred_food_inh: int = 15                # 예측 억제 뉴런 (WTA)
    # Context → Pred (static)
    food_mem_to_pred_weight: float = 3.0     # Food Memory → Pred
    food_mem_to_pred_sparsity: float = 0.05
    temporal_to_pred_weight: float = 2.0     # Temporal_Recent → Pred
    temporal_to_pred_sparsity: float = 0.05
    sound_food_to_pred_weight: float = 2.0   # Sound_Food → Pred
    sound_food_to_pred_sparsity: float = 0.05
    hunger_to_pred_weight: float = 3.0       # Hunger → Pred (need-gating)
    hunger_to_pred_sparsity: float = 0.10
    # Learnable Predictive STDP (DENSE — per-post budget normalization 필요)
    place_to_pred_init_w: float = 0.3
    place_to_pred_w_max: float = 1.5       # 낮은 개별 상한 (budget이 주 제약)
    place_to_pred_eta_ltp: float = 0.0002  # teacher-driven LTP
    place_to_pred_eta_ltd: float = 0.0001  # weight-dependent LTD
    place_to_pred_w_budget: float = 12.0   # per-post neuron incoming weight budget
    wmcb_to_pred_init_w: float = 0.2
    wmcb_to_pred_w_max: float = 1.5
    wmcb_to_pred_eta_ltp: float = 0.0002
    wmcb_to_pred_eta_ltd: float = 0.0001
    # Pred lateral inhibition (symmetry breaking)
    pred_lateral_inh_weight: float = -5.0
    pred_lateral_inh_sparsity: float = 0.15
    # Food teacher → Pred (fixed, strong, drives post spikes)
    food_teacher_to_pred_weight: float = 8.0
    food_teacher_to_pred_sparsity: float = 0.10
    # Pred → outputs (gentle modulator)
    pred_to_goal_food_weight: float = 1.5
    pred_to_goal_food_sparsity: float = 0.05
    pred_to_d1_weight: float = 1.0           # BG approach bias (symmetric L/R)
    pred_to_d1_sparsity: float = 0.03
    # Competition (WTA)
    pred_to_inh_weight: float = 8.0
    pred_to_inh_sparsity: float = 0.10
    pred_inh_to_pred_weight: float = -6.0
    pred_inh_to_pred_sparsity: float = 0.10

    # === Phase C5: Curiosity-Driven Exploration (호기심 기반 탐색) ===
    curiosity_enabled: bool = True
    n_curiosity_gate: int = 20              # 호기심 게이트 (novelty × uncertainty)
    n_curiosity_inh: int = 10               # 안전 억제 (위험 시 호기심 차단)
    # Inputs → Curiosity_Gate (static)
    v4_novel_to_curiosity_weight: float = 1.0
    v4_novel_to_curiosity_sparsity: float = 0.05
    assoc_novelty_to_curiosity_weight: float = 1.2
    assoc_novelty_to_curiosity_sparsity: float = 0.05
    meta_uncertainty_to_curiosity_weight: float = 1.5
    meta_uncertainty_to_curiosity_sparsity: float = 0.08
    acc_conflict_to_curiosity_weight: float = 0.8
    acc_conflict_to_curiosity_sparsity: float = 0.05
    curiosity_recurrent_weight: float = 0.3  # 짧은 지속 (perseveration 방지)
    curiosity_recurrent_sparsity: float = 0.05
    # Safety suppression
    fear_to_curiosity_inh_weight: float = 8.0
    fear_to_curiosity_inh_sparsity: float = 0.10
    gw_safety_to_curiosity_inh_weight: float = 6.0
    gw_safety_to_curiosity_inh_sparsity: float = 0.10
    curiosity_inh_to_gate_weight: float = -2.0
    curiosity_inh_to_gate_sparsity: float = 0.10
    # Output (gentle modulator — Motor 직접 연결 없음)
    curiosity_to_goal_food_weight: float = 1.5
    curiosity_to_goal_food_sparsity: float = 0.05
    curiosity_to_d1_weight: float = 0.8     # 대칭적 approach bias
    curiosity_to_d1_sparsity: float = 0.03

    # === M4: Context-Gated Value Learning (zone-dependent food rules) ===
    context_gate_enabled: bool = True
    n_ctx_a: int = 4                        # Zone A context neurons
    n_ctx_b: int = 4                        # Zone B context neurons
    ctx_place_weight: float = 1.0            # place → ctx (약하게 — WTA 경쟁 여지)
    ctx_wta_weight: float = -15.0           # CtxA ↔ CtxB strong WTA
    ctx_recurrent_weight: float = 3.0       # self-excitation (winner stability)

    # === M3: ACh Uncertainty Gate (환경 변화 감지 → 학습률 조절) ===
    uncertainty_gate_enabled: bool = True
    n_surprise_accum: int = 20            # 놀라움 축적 뉴런 (slow integration)
    n_stability_detector: int = 10        # 안정성 감지 뉴런
    # Inputs → Surprise Accumulator
    agency_pe_to_surprise_weight: float = 2.0   # 예측 오차 → 놀라움
    meta_uncertainty_to_surprise_weight: float = 1.5
    acc_conflict_to_surprise_weight: float = 1.0
    surprise_recurrent_weight: float = 0.5       # slow accumulation
    surprise_recurrent_sparsity: float = 0.08
    # Inputs → Stability Detector
    meta_confidence_to_stability_weight: float = 2.0
    pred_food_to_stability_weight: float = 1.5
    # Competition
    stability_to_surprise_weight: float = -3.0   # 안정하면 놀라움 억제
    stability_to_surprise_sparsity: float = 0.10
    # Output: surprise_rate → process()에서 학습률 조절
    # (시냅스 출력 없음 — rate를 scalar로 읽어서 eta modulation)

    dt: float = 1.0

    @property
    def total_neurons(self) -> int:
        base = (self.n_food_eye + self.n_wall_eye +
                self.n_low_energy_sensor + self.n_high_energy_sensor +
                self.n_hunger_drive + self.n_satiety_drive +
                self.n_motor_left + self.n_motor_right)
        if self.obstacle_eye_enabled:
            base += self.n_obstacle_eye
        if self.amygdala_enabled:
            base += (self.n_pain_eye + self.n_danger_sensor +
                     self.n_lateral_amygdala + self.n_central_amygdala +
                     self.n_fear_response)
        if self.hippocampus_enabled:
            base += (self.n_place_cells + self.n_food_memory)
        if self.basal_ganglia_enabled:
            base += (self.n_d1_msn + self.n_d2_msn + self.n_direct_pathway +
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
        if self.language_enabled:
            base += (self.n_call_food_input_left + self.n_call_food_input_right +
                     self.n_call_danger_input_left + self.n_call_danger_input_right +
                     self.n_wernicke_food + self.n_wernicke_danger +
                     self.n_wernicke_social + self.n_wernicke_context +
                     self.n_broca_food + self.n_broca_danger +
                     self.n_broca_social + self.n_broca_sequence +
                     self.n_vocal_gate + self.n_call_mirror + self.n_call_binding)
        if self.wm_expansion_enabled:
            base += (self.n_wm_thalamic + self.n_wm_update_gate +
                     self.n_temporal_recent + self.n_temporal_prior +
                     self.n_goal_pending + self.n_goal_switch +
                     self.n_wm_context_binding + self.n_wm_inhibitory)
        if self.metacognition_enabled:
            base += (self.n_meta_confidence + self.n_meta_uncertainty +
                     self.n_meta_evaluate + self.n_meta_arousal_mod +
                     self.n_meta_inhibitory)
        if self.self_model_enabled:
            base += (self.n_self_body + self.n_self_efference +
                     self.n_self_predict + self.n_self_agency +
                     self.n_self_narrative + self.n_self_inhibitory)
        if self.perceptual_learning_enabled:
            base += self.n_good_food_eye + self.n_bad_food_eye
        if self.prediction_error_enabled:
            base += self.n_pe_food + self.n_pe_danger
        if self.td_learning_enabled and self.basal_ganglia_enabled:
            base += self.n_nac_value + self.n_nac_inhibitory
        if self.swr_replay_enabled and self.hippocampus_enabled:
            base += self.n_ca3_sequence + self.n_swr_gate + self.n_replay_inhibitory
        if self.gw_enabled:
            base += self.n_gw_food * 2 + self.n_gw_safety  # 50+50+60 = 160
        if self.sparse_expansion_enabled and self.basal_ganglia_enabled and self.perceptual_learning_enabled:
            base += (self.n_kc_per_side * 2 + self.n_kc_inhibitory_per_side * 2)  # 3000×2 + 400×2 = 6800
        if self.contextual_prediction_enabled and self.hippocampus_enabled:
            base += self.n_pred_food_soon + self.n_pred_food_inh  # 30 + 15 = 45
        if self.curiosity_enabled:
            base += self.n_curiosity_gate + self.n_curiosity_inh  # 20 + 10 = 30
        if self.uncertainty_gate_enabled:
            base += self.n_surprise_accum + self.n_stability_detector  # 20 + 10 = 30
        if self.place_transition_enabled and self.hippocampus_enabled:
            base += self.n_place_value  # 20
        if self.context_gate_enabled:
            base += self.n_ctx_a + self.n_ctx_b + 16  # 4+4 ctx + 4×4 ctxval = 24
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

        # Base rate caching
        self.last_hunger_rate = 0.0
        self.last_satiety_rate = 0.0

        # Phase 15b: Mirror neuron state defaults
        self.mirror_self_eating_timer = 0
        self.last_social_obs_rate = 0.0

        # Phase 15c: Theory of Mind state defaults
        self.last_tom_intention_rate = 0.0

        # Phase 16: Association Cortex state defaults
        self.last_assoc_binding_rate = 0.0

        # Phase 16b: cached rates for cross-phase use
        self.last_fear_rate = 0.0

        # Phase 17: Language Circuit state defaults
        self.last_wernicke_food_rate = 0.0
        self.last_wernicke_danger_rate = 0.0
        self.last_broca_food_rate = 0.0
        self.last_broca_danger_rate = 0.0
        self.last_vocal_gate_rate = 0.0
        self.last_call_binding_rate = 0.0
        self.vocalize_type = 0  # 0=none, 1=food_call, 2=danger_call

        # Phase 18: WM Expansion state defaults
        self.last_dopamine_rate = 0.0
        self.last_acc_conflict_rate = 0.0
        self.last_novelty_rate = 0.0
        self.last_wm_thalamic_rate = 0.0
        self.last_wm_update_gate_rate = 0.0
        self.last_temporal_recent_rate = 0.0
        self.last_temporal_prior_rate = 0.0
        self.last_goal_pending_rate = 0.0
        self.last_goal_switch_rate = 0.0
        self.last_wm_context_binding_rate = 0.0
        self.last_wm_inhibitory_rate = 0.0

        # Phase 19: Metacognition state defaults
        self.last_meta_confidence_rate = 0.0
        self.last_meta_uncertainty_rate = 0.0
        self.last_meta_evaluate_rate = 0.0
        self.last_meta_arousal_mod_rate = 0.0
        self.last_meta_inhibitory_rate = 0.0

        # Phase 20: Self-Model state defaults
        self.last_self_body_rate = 0.0
        self.last_self_efference_rate = 0.0
        self.last_self_predict_rate = 0.0
        self.last_self_agency_rate = 0.0
        self.last_self_narrative_rate = 0.0
        self.last_self_inhibitory_rate = 0.0

        # Phase L15: Narrative Self state defaults
        self.prev_self_body_rate = 0.0  # For Δbody change detection

        # Learning weight cache (for real-time graph in render)
        self._last_rstdp_results = {}
        self._last_hippo_avg_w = 0.0
        self._last_garcia_avg_w = 0.0

        # Phase L16: Sparse Expansion (KC) state defaults — single KC
        if self.config.sparse_expansion_enabled:
            self.kc_d1_trace_l = 0.0
            self.kc_d1_trace_r = 0.0
            self.kc_d2_trace_l = 0.0
            self.kc_d2_trace_r = 0.0
            self.last_kc_l_rate = 0.0
            self.last_kc_r_rate = 0.0

        # Phase C4: Contextual Prediction state defaults (always init for debug_info)
        self.last_pred_food_rate = 0.0
        self._pred_food_teacher_active = False

        # Phase C5: Curiosity state defaults
        self.last_curiosity_rate = 0.0  # curiosity gate firing rate (always init)

        # M3: ACh Uncertainty Gate state defaults
        self.last_surprise_rate = 0.0   # surprise accumulator rate
        self.last_stability_rate = 0.0  # stability detector rate
        self.uncertainty_eta_mod = 1.0  # learning rate multiplier (1.0=normal, >1=fast learn)

        # M3: Place transition + revaluation state
        self.prev_place_activation = None
        self.transition_buffer = []

        # M4: Context gate state
        self.last_ctx_a_rate = 0.0
        self.last_ctx_b_rate = 0.0
        self._current_ctx = "a"
        # Context-specific food→ctxval weights (CPU managed, DA-gated)
        # Shape: n_food_eye × 4 (ctxval neurons per side)
        n_food = self.config.n_food_eye // 2  # 400 per side
        self._ctxval_w = {
            "a_l": np.ones(n_food, dtype=np.float32) * 3.0,
            "a_r": np.ones(n_food, dtype=np.float32) * 3.0,
            "b_l": np.ones(n_food, dtype=np.float32) * 3.0,
            "b_r": np.ones(n_food, dtype=np.float32) * 3.0,
        }  # 초기 3.0 = food_to_ctxval static weight와 동일

        # Phase L2: D1/D2 MSN rate defaults
        self.last_d1_l_rate = 0.0
        self.last_d1_r_rate = 0.0

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

        # === Obstacle Eye (wall_rays에서 분리된 장애물 감지) ===
        if self.config.obstacle_eye_enabled:
            n_obs_half = self.config.n_obstacle_eye // 2
            self.obstacle_eye_left = self.model.add_neuron_population(
                "obstacle_eye_left", n_obs_half, sensory_lif_model, sensory_params, sensory_init)
            self.obstacle_eye_right = self.model.add_neuron_population(
                "obstacle_eye_right", n_obs_half, sensory_lif_model, sensory_params, sensory_init)
            print(f"  Obstacle Eye: L/R({n_obs_half}x2) [Push={self.config.obstacle_push_weight}, Pull={self.config.obstacle_pull_weight}]")

        # === Phase L5: Good/Bad Food Eye (지각 학습용) ===
        if self.config.perceptual_learning_enabled:
            n_good_half = self.config.n_good_food_eye // 2
            n_bad_half = self.config.n_bad_food_eye // 2
            self.good_food_eye_left = self.model.add_neuron_population(
                "good_food_eye_left", n_good_half, sensory_lif_model, sensory_params, sensory_init)
            self.good_food_eye_right = self.model.add_neuron_population(
                "good_food_eye_right", n_good_half, sensory_lif_model, sensory_params, sensory_init)
            self.bad_food_eye_left = self.model.add_neuron_population(
                "bad_food_eye_left", n_bad_half, sensory_lif_model, sensory_params, sensory_init)
            self.bad_food_eye_right = self.model.add_neuron_population(
                "bad_food_eye_right", n_bad_half, sensory_lif_model, sensory_params, sensory_init)
            print(f"  Phase L5: Good_Food_L/R({n_good_half}x2) + Bad_Food_L/R({n_bad_half}x2)")

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

        # === 3. MOTOR POPULATIONS (높은 C로 포화 방지) ===
        motor_lif_params = {
            "C": self.config.motor_capacitance,
            "TauM": self.config.tau_m,
            "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset,
            "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0,
            "TauRefrac": self.config.tau_refrac
        }
        self.motor_left = self.model.add_neuron_population(
            "motor_left", self.config.n_motor_left, "LIF", motor_lif_params, lif_init)
        self.motor_right = self.model.add_neuron_population(
            "motor_right", self.config.n_motor_right, "LIF", motor_lif_params, lif_init)
        print(f"  Motor: C={self.config.motor_capacitance} (anti-saturation)")

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

        # === Phase 4: BASAL GANGLIA POPULATIONS (Phase L2: D1/D2 MSN 분리) ===
        if self.config.basal_ganglia_enabled:
            n_d1_half = self.config.n_d1_msn // 2          # 100
            n_d2_half = self.config.n_d2_msn // 2          # 100
            n_dir_half = self.config.n_direct_pathway // 2  # 100
            n_ind_half = self.config.n_indirect_pathway // 2  # 100

            # MSN LIF params with higher C for graded response
            msn_lif_params = {
                "C": self.config.msn_capacitance,  # C=30 (graded, not binary)
                "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
                "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
                "TauRefrac": self.config.tau_refrac, "Ioffset": 0.0
            }

            # D1 MSN L/R: Go pathway (R-STDP 학습)
            self.d1_left = self.model.add_neuron_population(
                "d1_left", n_d1_half, "LIF", msn_lif_params, lif_init)
            self.d1_right = self.model.add_neuron_population(
                "d1_right", n_d1_half, "LIF", msn_lif_params, lif_init)

            # D2 MSN L/R: NoGo pathway (Static)
            self.d2_left = self.model.add_neuron_population(
                "d2_left", n_d2_half, "LIF", msn_lif_params, lif_init)
            self.d2_right = self.model.add_neuron_population(
                "d2_right", n_d2_half, "LIF", msn_lif_params, lif_init)

            # Direct pathway L/R: Go 출력
            self.direct_left = self.model.add_neuron_population(
                "direct_left", n_dir_half, "LIF", lif_params, lif_init)
            self.direct_right = self.model.add_neuron_population(
                "direct_right", n_dir_half, "LIF", lif_params, lif_init)

            # Indirect pathway L/R: NoGo 출력
            self.indirect_left = self.model.add_neuron_population(
                "indirect_left", n_ind_half, "LIF", lif_params, lif_init)
            self.indirect_right = self.model.add_neuron_population(
                "indirect_right", n_ind_half, "LIF", lif_params, lif_init)

            # Dopamine neurons (VTA/SNc): 보상 신호 (비측면화)
            self.dopamine_neurons = self.model.add_neuron_population(
                "dopamine_neurons", self.config.n_dopamine, sensory_lif_model, sensory_params, sensory_init)

            # Dopamine 레벨 추적
            self.dopamine_level = 0.0

            # R-STDP 적격 추적 (Phase L1: D1, Phase L4: D2)
            self.rstdp_trace_l = 0.0
            self.rstdp_trace_r = 0.0
            self.rstdp_d2_trace_l = 0.0  # Phase L4: D2 Anti-Hebbian 추적
            self.rstdp_d2_trace_r = 0.0
            self._rstdp_step = 0  # Phase L3: 항상성 감쇠 스텝 카운터

        # Phase L5: 피질 R-STDP 적격 추적 (좋은/나쁜 음식 × L/R)
        if self.config.perceptual_learning_enabled:
            self.cortical_trace_good_l = 0.0
            self.cortical_trace_good_r = 0.0
            self.cortical_trace_bad_l = 0.0
            self.cortical_trace_bad_r = 0.0
            self._cortical_step = 0
            self._taste_aversion_active = False
            self.last_bad_food_activity_left = 0.0
            self.last_bad_food_activity_right = 0.0
            self.prev_bad_food_activity_left = 0.0
            self.prev_bad_food_activity_right = 0.0

        # Phase L7: Discriminative BG 적격 추적
        if self.config.discriminative_bg_enabled and self.config.perceptual_learning_enabled:
            self.typed_d1_trace_good_l = 0.0
            self.typed_d1_trace_good_r = 0.0
            self.typed_d1_trace_bad_l = 0.0
            self.typed_d1_trace_bad_r = 0.0
            self.typed_d2_trace_good_l = 0.0
            self.typed_d2_trace_good_r = 0.0
            self.typed_d2_trace_bad_l = 0.0
            self.typed_d2_trace_bad_r = 0.0

        # Phase L9: IT → BG 적격 추적
        if self.config.it_bg_enabled and self.config.it_enabled:
            self.it_food_d1_trace_l = 0.0
            self.it_food_d1_trace_r = 0.0
            self.it_food_d2_trace_l = 0.0  # pre-synaptic only (D2 패턴)
            self.it_food_d2_trace_r = 0.0

        # Phase L10: NAc Critic (TD Learning)
        if self.config.td_learning_enabled and self.config.basal_ganglia_enabled:
            nac_msn_params = {
                "C": 30.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
                "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
                "TauRefrac": self.config.tau_refrac, "Ioffset": 0.0
            }
            self.nac_value = self.model.add_neuron_population(
                "nac_value", self.config.n_nac_value, "LIF", nac_msn_params, lif_init)
            self.nac_inhibitory = self.model.add_neuron_population(
                "nac_inhibitory", self.config.n_nac_inhibitory, "LIF", lif_params, lif_init)

            # NAc R-STDP traces
            self.nac_trace_l = 0.0
            self.nac_trace_r = 0.0
            self._nac_value_rate = 0.0

        # Phase L11: SWR Replay (Hippocampal Sequence)
        if self.config.swr_replay_enabled and self.config.hippocampus_enabled:
            ca3_params = {
                "C": 30.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
                "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
                "TauRefrac": self.config.tau_refrac, "Ioffset": 0.0
            }
            self.ca3_sequence = self.model.add_neuron_population(
                "ca3_sequence", self.config.n_ca3_sequence, "LIF", ca3_params, lif_init)

            self.swr_gate = self.model.add_neuron_population(
                "swr_gate", self.config.n_swr_gate, sensory_lif_model, sensory_params, sensory_init)

            self.replay_inhibitory = self.model.add_neuron_population(
                "replay_inhibitory", self.config.n_replay_inhibitory, "LIF", lif_params, lif_init)

            self.experience_buffer = []

        # Phase L12: Global Workspace (Attention)
        if self.config.gw_enabled:
            gw_params = {
                "C": 30.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
                "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
                "TauRefrac": self.config.tau_refrac, "Ioffset": 0.0
            }
            self.gw_food_left = self.model.add_neuron_population(
                "gw_food_left", self.config.n_gw_food, "LIF", gw_params, lif_init)
            self.gw_food_right = self.model.add_neuron_population(
                "gw_food_right", self.config.n_gw_food, "LIF", gw_params, lif_init)
            self.gw_safety = self.model.add_neuron_population(
                "gw_safety", self.config.n_gw_safety, "LIF", gw_params, lif_init)
            # Rate caching
            self.last_gw_food_rate = 0.0
            self.last_gw_safety_rate = 0.0
            self.last_gw_broadcast = "neutral"

        # Phase L6: Prediction Error 적격 추적
        if self.config.prediction_error_enabled:
            self.pe_trace_food_l = 0.0
            self.pe_trace_food_r = 0.0
            self.pe_trace_danger_l = 0.0
            self.pe_trace_danger_r = 0.0
            self._pe_step = 0

        if self.config.basal_ganglia_enabled:
            print(f"  BasalGanglia (L2 D1/D2): "
                  f"D1({n_d1_half}L+{n_d1_half}R) + "
                  f"D2({n_d2_half}L+{n_d2_half}R) + "
                  f"Direct({n_dir_half}L+{n_dir_half}R) + "
                  f"Indirect({n_ind_half}L+{n_ind_half}R) + "
                  f"Dopamine({self.config.n_dopamine})")
            print(f"  Motor: C={self.config.motor_capacitance} (anti-saturation)")
            print(f"  MSN: C={self.config.msn_capacitance} (graded response)")

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
            # C1: Sound_Food → D1 직접 연결 (BG에서는 빌드 순서 때문에 못 만듦)
            if self.config.basal_ganglia_enabled and hasattr(self, 'sound_food_left'):
                sf_init_w = 0.5
                sf_sp = 0.10
                self.sound_food_to_d1_l = self.model.add_synapse_population(
                    "sound_food_l_to_d1_l", "SPARSE", self.sound_food_left, self.d1_left,
                    init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": sf_init_w})}),
                    init_postsynaptic("ExpCurr", {"tau": 5.0}),
                    init_sparse_connectivity("FixedProbability", {"prob": sf_sp}))
                self.sound_food_to_d1_r = self.model.add_synapse_population(
                    "sound_food_r_to_d1_r", "SPARSE", self.sound_food_right, self.d1_right,
                    init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": sf_init_w})}),
                    init_postsynaptic("ExpCurr", {"tau": 5.0}),
                    init_sparse_connectivity("FixedProbability", {"prob": sf_sp}))
                print(f"    C1: Sound_Food→D1 direct (init={sf_init_w}, sp={sf_sp})")

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

        # Phase 17: Language Circuit
        if self.config.language_enabled:
            self._build_language_circuit()

        # Phase 18: Working Memory Expansion
        if self.config.wm_expansion_enabled:
            self._build_wm_expansion_circuit()

        # Phase 19: Metacognition
        if self.config.metacognition_enabled:
            self._build_metacognition_circuit()

        # Phase 20: Self-Model
        if self.config.self_model_enabled:
            self._build_self_model_circuit()

        # Phase L5: Perceptual Learning (좋은/나쁜 음식 → IT 피질)
        if self.config.perceptual_learning_enabled and self.config.it_enabled:
            self._build_perceptual_learning_circuit()

        # Phase L6: Prediction Error Circuit (예측 오차)
        if self.config.prediction_error_enabled and self.config.v1_enabled and self.config.it_enabled:
            self._build_prediction_error_circuit()

        # Phase L9: IT Cortex → BG (피질 하향 연결)
        if (self.config.it_bg_enabled and self.config.it_enabled
                and self.config.basal_ganglia_enabled):
            self._build_it_bg_circuit()

        # Phase L16: Sparse Expansion Layer (Mushroom Body / DG)
        if (self.config.sparse_expansion_enabled and self.config.basal_ganglia_enabled
                and self.config.perceptual_learning_enabled):
            self._build_sparse_expansion_circuit()

        # Phase L10: NAc Critic (TD Learning)
        if self.config.td_learning_enabled and self.config.basal_ganglia_enabled:
            self._build_nac_circuit()

        # Phase L11: SWR Replay
        if self.config.swr_replay_enabled and self.config.hippocampus_enabled:
            self._build_swr_circuit()

        # Phase L12: Global Workspace (Attention)
        if self.config.gw_enabled:
            self._build_gw_circuit()

        # Phase C4: Contextual Prediction (경험 기반 예측)
        if (self.config.contextual_prediction_enabled
                and self.config.hippocampus_enabled
                and self.config.prefrontal_enabled):
            self._build_contextual_prediction_circuit()

        # Phase C5: Curiosity-Driven Exploration (호기심 기반 탐색)
        if (self.config.curiosity_enabled
                and self.config.metacognition_enabled
                and self.config.v2v4_enabled):
            self._build_curiosity_circuit()

        # M3: ACh Uncertainty Gate (환경 변화 감지)
        if (self.config.uncertainty_gate_enabled
                and self.config.metacognition_enabled):
            self._build_uncertainty_gate()

        # M4: Context Gate (zone-dependent value learning)
        if (self.config.context_gate_enabled
                and self.config.hippocampus_enabled):
            self._build_context_gate_circuit()

        # M3: Place transition + Value population (revaluation SWR)
        if (self.config.place_transition_enabled
                and self.config.hippocampus_enabled):
            self._build_place_transition_circuit()

        # Enable spike recording for all populations (batched GPU pull)
        self._enable_spike_recording()

        # Build and load
        print("Building model...")
        self.model.build()
        self.model.load(num_recording_timesteps=10)

        # SPARSE 시냅스는 connectivity를 먼저 pull해야 .values가 동작함 (CRITICAL)
        # connectivity 패턴은 고정이므로 최초 1회만 pull
        if self.config.basal_ganglia_enabled:
            self.food_to_d1_l.pull_connectivity_from_device()
            self.food_to_d1_r.pull_connectivity_from_device()
            self.food_to_d2_l.pull_connectivity_from_device()  # Phase L4: Anti-Hebbian D2
            self.food_to_d2_r.pull_connectivity_from_device()

        # C1: Sound_Food→D1 SPARSE connectivity pull
        if self.config.auditory_enabled and hasattr(self, 'sound_food_to_d1_l'):
            self.sound_food_to_d1_l.pull_connectivity_from_device()
            self.sound_food_to_d1_r.pull_connectivity_from_device()

        # Phase L7: Discriminative BG SPARSE connectivity pull
        if self.config.discriminative_bg_enabled and self.config.perceptual_learning_enabled:
            for syn in [self.good_food_to_d1_l, self.good_food_to_d1_r,
                        self.bad_food_to_d1_l, self.bad_food_to_d1_r,
                        self.good_food_to_d2_l, self.good_food_to_d2_r,
                        self.bad_food_to_d2_l, self.bad_food_to_d2_r]:
                syn.pull_connectivity_from_device()

        # Phase L9: IT→BG SPARSE connectivity pull
        if self.config.it_bg_enabled and self.config.it_enabled:
            for syn in [self.it_food_to_d1_l, self.it_food_to_d1_r,
                        self.it_food_to_d2_l, self.it_food_to_d2_r]:
                syn.pull_connectivity_from_device()

        # Food Approach SPARSE connectivity pull
        if self.config.perceptual_learning_enabled and hasattr(self, 'good_food_to_motor_l'):
            self.good_food_to_motor_l.pull_connectivity_from_device()
            self.good_food_to_motor_r.pull_connectivity_from_device()

        # Phase L16: KC→D1/D2 SPARSE connectivity pull (single KC)
        if self.config.sparse_expansion_enabled and hasattr(self, 'kc_to_d1_l'):
            for syn in [self.kc_to_d1_l, self.kc_to_d1_r,
                        self.kc_to_d2_l, self.kc_to_d2_r]:
                syn.pull_connectivity_from_device()

        # Phase L10: NAc R-STDP SPARSE connectivity pull
        if self.config.td_learning_enabled and self.config.basal_ganglia_enabled:
            self.food_to_nac_l.pull_connectivity_from_device()
            self.food_to_nac_r.pull_connectivity_from_device()

        # Phase L5: 피질 R-STDP 시냅스 connectivity pull
        if self.config.perceptual_learning_enabled and self.config.it_enabled:
            for syn in [self.good_food_to_it_food_l, self.good_food_to_it_food_r,
                        self.good_food_to_it_danger_l, self.good_food_to_it_danger_r,
                        self.bad_food_to_it_danger_l, self.bad_food_to_it_danger_r,
                        self.bad_food_to_it_food_l, self.bad_food_to_it_food_r]:
                syn.pull_connectivity_from_device()

        # Phase L6: PE→IT 시냅스 connectivity pull
        if self.config.prediction_error_enabled and self.config.v1_enabled and self.config.it_enabled:
            for syn in [self.pe_food_to_it_food_l, self.pe_food_to_it_food_r,
                        self.pe_danger_to_it_danger_l, self.pe_danger_to_it_danger_r]:
                syn.pull_connectivity_from_device()

        # Phase C4: Contextual Prediction SPARSE connectivity pull (D1 output only — place/wmcb are DENSE now)
        if self.config.contextual_prediction_enabled and hasattr(self, 'pred_to_d1_l'):
            self.pred_to_d1_l.pull_connectivity_from_device()
            self.pred_to_d1_r.pull_connectivity_from_device()

        # M4: KC→D1/D2 context weights initialization (양쪽 동일 시작)
        if (self.config.context_gate_enabled
                and self.config.sparse_expansion_enabled
                and hasattr(self, 'kc_to_d1_l')):
            for syn_name in ['kc_to_d1_l', 'kc_to_d1_r', 'kc_to_d2_l', 'kc_to_d2_r']:
                syn = getattr(self, syn_name)
                syn.vars["g"].pull_from_device()
                w = syn.vars["g"].values.copy()
                setattr(self, f'_ctx_a_{syn_name}', w.copy())
                setattr(self, f'_ctx_b_{syn_name}', w.copy())
            self._last_rstdp_ctx = "a"  # 마지막 R-STDP 업데이트 시의 context

        n_total = self.config.total_neurons
        print(f"Model ready! Total: {n_total:,} neurons")

        # 스파이크 카운팅용
        self.spike_threshold = self.config.tau_refrac - 0.5

    def _enable_spike_recording(self):
        """모든 스파이크 카운팅 대상 population에 spike_recording_enabled 설정.

        model.build() 호출 전에 실행해야 함.
        process() 루프에서 RefracTime.pull_from_device()를 하던 모든 population 대상.
        이후 pull_recording_buffers_from_device() 한 번으로 전체 스파이크 데이터 수집 가능.
        """
        # Phase 2a: 기본 회로
        always_on = [
            self.motor_left, self.motor_right,
            self.hunger_drive, self.satiety_drive,
            self.low_energy_sensor, self.high_energy_sensor,
        ]
        for pop in always_on:
            pop.spike_recording_enabled = True

        # Obstacle Eye
        if self.config.obstacle_eye_enabled:
            self.obstacle_eye_left.spike_recording_enabled = True
            self.obstacle_eye_right.spike_recording_enabled = True

        # Phase 2b: Amygdala
        if self.config.amygdala_enabled:
            for pop in [self.lateral_amygdala, self.central_amygdala, self.fear_response]:
                pop.spike_recording_enabled = True

        # Phase 3: Hippocampus
        if self.config.hippocampus_enabled:
            self.place_cells.spike_recording_enabled = True
            if self.config.directional_food_memory:
                self.food_memory_left.spike_recording_enabled = True
                self.food_memory_right.spike_recording_enabled = True
            elif self.food_memory is not None:
                self.food_memory.spike_recording_enabled = True

        # Phase 4 / L2: Basal Ganglia (D1/D2 MSN)
        if self.config.basal_ganglia_enabled:
            for pop in [self.d1_left, self.d1_right, self.d2_left, self.d2_right,
                        self.direct_left, self.direct_right,
                        self.indirect_left, self.indirect_right,
                        self.dopamine_neurons]:
                pop.spike_recording_enabled = True

            # Phase L10: NAc
            if self.config.td_learning_enabled:
                self.nac_value.spike_recording_enabled = True

            # Phase L12: Global Workspace
            if self.config.gw_enabled:
                for pop in [self.gw_food_left, self.gw_food_right, self.gw_safety]:
                    pop.spike_recording_enabled = True

        # Phase 5: Prefrontal Cortex
        if self.config.prefrontal_enabled:
            for pop in [self.working_memory, self.goal_food,
                        self.goal_safety, self.inhibitory_control]:
                pop.spike_recording_enabled = True

        # Phase 6a: Cerebellum
        if self.config.cerebellum_enabled:
            for pop in [self.granule_cells, self.purkinje_cells,
                        self.deep_nuclei, self.error_signal]:
                pop.spike_recording_enabled = True

        # Phase 6b: Thalamus
        if self.config.thalamus_enabled:
            for pop in [self.food_relay, self.danger_relay, self.trn, self.arousal]:
                pop.spike_recording_enabled = True

        # Phase 8: V1
        if self.config.v1_enabled:
            for pop in [self.v1_food_left, self.v1_food_right,
                        self.v1_danger_left, self.v1_danger_right]:
                pop.spike_recording_enabled = True

        # Phase 9: V2/V4
        if self.config.v2v4_enabled and self.config.v1_enabled:
            for pop in [self.v2_edge_food, self.v2_edge_danger,
                        self.v4_food_object, self.v4_danger_object,
                        self.v4_novel_object]:
                pop.spike_recording_enabled = True

        # Phase 10: IT Cortex
        if self.config.it_enabled and self.config.v2v4_enabled:
            for pop in [self.it_food_category, self.it_danger_category,
                        self.it_neutral_category, self.it_association,
                        self.it_memory_buffer]:
                pop.spike_recording_enabled = True

        # Phase 11: Auditory Cortex
        if self.config.auditory_enabled:
            for pop in [self.a1_danger, self.a1_food, self.a2_association]:
                pop.spike_recording_enabled = True

        # Phase 12: Multimodal Integration
        if self.config.multimodal_enabled:
            for pop in [self.sts_food, self.sts_danger,
                        self.sts_congruence, self.sts_mismatch]:
                pop.spike_recording_enabled = True

        # Phase 13: Parietal Cortex
        if self.config.parietal_enabled:
            for pop in [self.ppc_space_left, self.ppc_space_right,
                        self.ppc_goal_food, self.ppc_goal_safety,
                        self.ppc_attention, self.ppc_path_buffer]:
                pop.spike_recording_enabled = True

        # Phase 14: Premotor Cortex
        if self.config.premotor_enabled:
            for pop in [self.pmd_left, self.pmd_right,
                        self.pmv_approach, self.pmv_avoid,
                        self.sma_sequence, self.motor_preparation]:
                pop.spike_recording_enabled = True

        # Phase 15: Social Brain
        if self.config.social_brain_enabled:
            for pop in [self.sts_social, self.tpj_self, self.tpj_other,
                        self.tpj_compare, self.acc_conflict, self.acc_monitor,
                        self.social_approach, self.social_avoid]:
                pop.spike_recording_enabled = True

            # Phase 15b: Mirror Neurons
            if self.config.mirror_enabled:
                for pop in [self.social_observation, self.mirror_food,
                            self.vicarious_reward, self.social_memory]:
                    pop.spike_recording_enabled = True

            # Phase 15c: Theory of Mind
            if self.config.tom_enabled:
                for pop in [self.tom_intention, self.tom_belief,
                            self.tom_prediction, self.tom_surprise,
                            self.coop_compete_coop, self.coop_compete_compete]:
                    pop.spike_recording_enabled = True

        # Phase 16: Association Cortex
        if self.config.association_cortex_enabled:
            for pop in [self.assoc_edible, self.assoc_threatening,
                        self.assoc_animate, self.assoc_context,
                        self.assoc_valence, self.assoc_binding,
                        self.assoc_novelty]:
                pop.spike_recording_enabled = True

        # Phase 17: Language Circuit
        if self.config.language_enabled:
            for pop in [self.wernicke_food, self.wernicke_danger,
                        self.wernicke_social, self.wernicke_context,
                        self.broca_food, self.broca_danger,
                        self.broca_social, self.broca_sequence,
                        self.vocal_gate, self.call_mirror, self.call_binding]:
                pop.spike_recording_enabled = True

        # Phase 18: WM Expansion
        if self.config.wm_expansion_enabled:
            for pop in [self.wm_thalamic, self.wm_update_gate,
                        self.temporal_recent, self.temporal_prior,
                        self.goal_pending, self.goal_switch,
                        self.wm_context_binding, self.wm_inhibitory]:
                pop.spike_recording_enabled = True

        # Phase 19: Metacognition
        if self.config.metacognition_enabled:
            for pop in [self.meta_confidence, self.meta_uncertainty,
                        self.meta_evaluate, self.meta_arousal_mod,
                        self.meta_inhibitory_pop]:
                pop.spike_recording_enabled = True

        # Phase 20: Self-Model
        if self.config.self_model_enabled:
            for pop in [self.self_body, self.self_efference, self.self_predict,
                        self.self_agency, self.self_narrative,
                        self.self_inhibitory_sm]:
                pop.spike_recording_enabled = True

        # Phase L14: Agency PE
        if self.config.agency_detection_enabled and hasattr(self, 'agency_pe'):
            self.agency_pe.spike_recording_enabled = True

        # Phase L6: Prediction Error
        if self.config.prediction_error_enabled and self.config.v1_enabled and self.config.it_enabled:
            for pop in [self.pe_food_left, self.pe_food_right,
                        self.pe_danger_left, self.pe_danger_right]:
                pop.spike_recording_enabled = True

        # Phase C4: Contextual Prediction
        if self.config.contextual_prediction_enabled and hasattr(self, 'pred_food_soon'):
            self.pred_food_soon.spike_recording_enabled = True

        # Phase C5: Curiosity
        if self.config.curiosity_enabled and hasattr(self, 'curiosity_gate'):
            self.curiosity_gate.spike_recording_enabled = True

        # M4: Context Gate
        if self.config.context_gate_enabled and hasattr(self, 'ctx_a'):
            self.ctx_a.spike_recording_enabled = True
            self.ctx_b.spike_recording_enabled = True

        # M3: Uncertainty Gate
        if self.config.uncertainty_gate_enabled and hasattr(self, 'surprise_accum'):
            self.surprise_accum.spike_recording_enabled = True

        print("  Spike recording enabled for all monitored populations")

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

        # Food tracking: 학습 기반 접근
        # food_eye(무차별) → 약한 탐색 보조 (5.0, static)
        # good_food_eye(선별) → 강한 접근 (R-STDP, 학습으로 성장)
        if self.config.perceptual_learning_enabled:
            # food_eye: 약한 무차별 탐색 (모든 음식 방향으로 약간 끌림)
            food_explore_w = 10.0
            self._create_static_synapse(
                "food_left_motor_left", self.food_eye_left, self.motor_left,
                food_explore_w, sparsity=0.15)
            self._create_static_synapse(
                "food_right_motor_right", self.food_eye_right, self.motor_right,
                food_explore_w, sparsity=0.15)
            # good_food_eye: 학습 기반 접근 (도파민으로 강화)
            fa_w = self.config.food_approach_init_w
            self.good_food_to_motor_l = self.model.add_synapse_population(
                "good_food_motor_left", "SPARSE", self.good_food_eye_left, self.motor_left,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": fa_w})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}),
                init_sparse_connectivity("FixedProbability", {"prob": 0.15}))
            self.good_food_to_motor_r = self.model.add_synapse_population(
                "good_food_motor_right", "SPARSE", self.good_food_eye_right, self.motor_right,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": fa_w})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}),
                init_sparse_connectivity("FixedProbability", {"prob": 0.15}))
            print(f"    Food Explore: food_eye→Motor {food_explore_w} (static, weak)")
            print(f"    Food Approach: good_food_eye→Motor R-STDP (init={fa_w}, learnable)")
        else:
            # perceptual_learning 비활성 시 기존 방식 유지 (호환)
            self._create_static_synapse(
                "food_left_motor_left", self.food_eye_left, self.motor_left,
                self.config.food_weight, sparsity=0.15)
            self._create_static_synapse(
                "food_right_motor_right", self.food_eye_right, self.motor_right,
                self.config.food_weight, sparsity=0.15)
            print(f"    Food Ipsi: {self.config.food_weight} (static, legacy)")

        # === Obstacle avoidance: Push-Pull (약한 가중치) ===
        if self.config.obstacle_eye_enabled:
            print("  Building Obstacle avoidance circuit...")
            # Obstacle_L → Motor_R (Push)
            self._create_static_synapse(
                "obstacle_left_motor_right", self.obstacle_eye_left, self.motor_right,
                self.config.obstacle_push_weight, sparsity=0.15)
            self._create_static_synapse(
                "obstacle_right_motor_left", self.obstacle_eye_right, self.motor_left,
                self.config.obstacle_push_weight, sparsity=0.15)
            # Obstacle_L → Motor_L (Pull - inhibit)
            self._create_static_synapse(
                "obstacle_left_motor_left_inhib", self.obstacle_eye_left, self.motor_left,
                self.config.obstacle_pull_weight, sparsity=0.15)
            self._create_static_synapse(
                "obstacle_right_motor_right_inhib", self.obstacle_eye_right, self.motor_right,
                self.config.obstacle_pull_weight, sparsity=0.15)
            print(f"    Obstacle Push: {self.config.obstacle_push_weight} (weak, wall={self.config.wall_push_weight})")
            print(f"    Obstacle Pull: {self.config.obstacle_pull_weight} (weak, wall={self.config.wall_pull_weight})")

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

        # 2b. Phase L13: Bad Food Eye → LA (조건화된 맛 혐오, Hebbian 학습)
        # Garcia Effect: 나쁜 음식의 시각 정보가 LA를 활성화하도록 학습
        # DENSE 시냅스 (수동 Hebbian 업데이트용)
        if self.config.taste_aversion_learning_enabled and self.config.perceptual_learning_enabled:
            ta_init_w = self.config.taste_aversion_hebbian_init_w
            self.bad_food_to_la_left = self.model.add_synapse_population(
                "bad_food_to_la_left", "DENSE",
                self.bad_food_eye_left, self.lateral_amygdala,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": ta_init_w})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0})
            )
            self.bad_food_to_la_right = self.model.add_synapse_population(
                "bad_food_to_la_right", "DENSE",
                self.bad_food_eye_right, self.lateral_amygdala,
                init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": ta_init_w})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0})
            )
            print(f"    Phase L13: BadFoodEye→LA: init_w={ta_init_w}, w_max={self.config.taste_aversion_hebbian_w_max} (DENSE, Hebbian)")

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
        Phase L2: D1/D2 MSN 분리 + R-STDP 학습

        구조:
        - Food_Eye_L → D1_L (R-STDP 학습) → Direct_L → Motor_L (Go)
        - Food_Eye_L → D2_L (Static)      → Indirect_L → Motor_L (NoGo)
        - Food_Eye_R → D1_R (R-STDP 학습) → Direct_R → Motor_R (Go)
        - Food_Eye_R → D2_R (Static)      → Indirect_R → Motor_R (NoGo)
        - Dopamine → D1 (흥분) / D2 (억제)
        """
        print("  Building Basal Ganglia circuit (Phase L2: D1/D2 MSN + R-STDP)...")

        d1_init_w = self.config.food_to_d1_init_weight

        # 1. Food_Eye → D1 MSN (R-STDP 학습 대상, SPARSE)
        self.food_to_d1_l = self._create_static_synapse(
            "food_eye_left_to_d1_l", self.food_eye_left, self.d1_left,
            d1_init_w, sparsity=0.08)
        self.food_to_d1_r = self._create_static_synapse(
            "food_eye_right_to_d1_r", self.food_eye_right, self.d1_right,
            d1_init_w, sparsity=0.08)

        print(f"    FoodEye→D1 (R-STDP): init_w={d1_init_w}, w_max={self.config.rstdp_w_max}")

        # 2. Food_Eye → D2 MSN (Phase L4: Anti-Hebbian 학습 대상)
        self.food_to_d2_l = self._create_static_synapse(
            "food_eye_left_to_d2_l", self.food_eye_left, self.d2_left,
            self.config.food_to_d2_weight, sparsity=0.08)
        self.food_to_d2_r = self._create_static_synapse(
            "food_eye_right_to_d2_r", self.food_eye_right, self.d2_right,
            self.config.food_to_d2_weight, sparsity=0.08)

        print(f"    FoodEye→D2 (Anti-Hebbian): init_w={self.config.food_to_d2_weight}, w_min={self.config.rstdp_d2_w_min}")

        # 3. D1 → Direct (Go) - DENSE, lateralized
        self.model.add_synapse_population(
            "d1_l_to_direct_l", "DENSE",
            self.d1_left, self.direct_left,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.d1_to_direct_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))
        self.model.add_synapse_population(
            "d1_r_to_direct_r", "DENSE",
            self.d1_right, self.direct_right,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.d1_to_direct_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))

        # 4. D2 → Indirect (NoGo) - DENSE, lateralized
        self.model.add_synapse_population(
            "d2_l_to_indirect_l", "DENSE",
            self.d2_left, self.indirect_left,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.d2_to_indirect_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))
        self.model.add_synapse_population(
            "d2_r_to_indirect_r", "DENSE",
            self.d2_right, self.indirect_right,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": self.config.d2_to_indirect_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))

        print(f"    D1→Direct: {self.config.d1_to_direct_weight} (Go)")
        print(f"    D2→Indirect: {self.config.d2_to_indirect_weight} (NoGo)")

        # 5. D1 ↔ D2 측면 경쟁 (lateralized)
        for side, d1, d2 in [
            ("l", self.d1_left, self.d2_left),
            ("r", self.d1_right, self.d2_right)
        ]:
            self._create_static_synapse(
                f"d1_to_d2_{side}", d1, d2,
                self.config.d1_d2_competition, sparsity=0.1)
            self._create_static_synapse(
                f"d2_to_d1_{side}", d2, d1,
                self.config.d1_d2_competition, sparsity=0.1)

        # 6. Direct ↔ Indirect 상호 억제 (측면 내부만)
        self._create_static_synapse(
            "direct_l_to_indirect_l", self.direct_left, self.indirect_left,
            self.config.direct_indirect_competition, sparsity=0.1)
        self._create_static_synapse(
            "indirect_l_to_direct_l", self.indirect_left, self.direct_left,
            self.config.direct_indirect_competition, sparsity=0.1)
        self._create_static_synapse(
            "direct_r_to_indirect_r", self.direct_right, self.indirect_right,
            self.config.direct_indirect_competition, sparsity=0.1)
        self._create_static_synapse(
            "indirect_r_to_direct_r", self.indirect_right, self.direct_right,
            self.config.direct_indirect_competition, sparsity=0.1)

        # 7. Direct/Indirect → Motor (측면화: L→L, R→R)
        self._create_static_synapse(
            "direct_l_to_motor_l", self.direct_left, self.motor_left,
            self.config.direct_to_motor_weight, sparsity=0.15)
        self._create_static_synapse(
            "direct_r_to_motor_r", self.direct_right, self.motor_right,
            self.config.direct_to_motor_weight, sparsity=0.15)
        self._create_static_synapse(
            "indirect_l_to_motor_l", self.indirect_left, self.motor_left,
            self.config.indirect_to_motor_weight, sparsity=0.15)
        self._create_static_synapse(
            "indirect_r_to_motor_r", self.indirect_right, self.motor_right,
            self.config.indirect_to_motor_weight, sparsity=0.15)

        # 7b. Direct → Motor 교차 억제 (BG Push-Pull: 방향 차등 신호)
        self._create_static_synapse(
            "direct_l_to_motor_r", self.direct_left, self.motor_right,
            self.config.direct_to_motor_contra_weight, sparsity=0.15)
        self._create_static_synapse(
            "direct_r_to_motor_l", self.direct_right, self.motor_left,
            self.config.direct_to_motor_contra_weight, sparsity=0.15)

        print(f"    Direct→Motor: {self.config.direct_to_motor_weight} (Go, ipsi)")
        print(f"    Direct→Motor: {self.config.direct_to_motor_contra_weight} (Push-Pull, contra)")
        print(f"    Indirect→Motor: {self.config.indirect_to_motor_weight} (NoGo, lateralized)")

        # 8. Dopamine → D1/D2 MSN (보상 조절, MSN 레벨)
        for side, d1, d2 in [
            ("l", self.d1_left, self.d2_left),
            ("r", self.d1_right, self.d2_right)
        ]:
            self._create_static_synapse(
                f"dopamine_to_d1_{side}", self.dopamine_neurons, d1,
                self.config.dopamine_to_d1_weight, sparsity=0.15)
            self._create_static_synapse(
                f"dopamine_to_d2_{side}", self.dopamine_neurons, d2,
                self.config.dopamine_to_d2_weight, sparsity=0.15)

        print(f"    Dopamine→D1: {self.config.dopamine_to_d1_weight} (D1 receptor, excite)")
        print(f"    Dopamine→D2: {self.config.dopamine_to_d2_weight} (D2 receptor, inhibit)")

        # === Phase L7: Discriminative BG (good/bad food → D1/D2) ===
        if self.config.discriminative_bg_enabled and self.config.perceptual_learning_enabled:
            td1_w = self.config.typed_food_d1_init_w
            td2_w = self.config.typed_food_d2_init_w
            t_sp = self.config.typed_food_bg_sparsity

            # good_food_eye → D1 (R-STDP: 좋은 음식 + 도파민 → Go 강화)
            self.good_food_to_d1_l = self._create_static_synapse(
                "good_food_eye_l_to_d1_l", self.good_food_eye_left, self.d1_left,
                td1_w, sparsity=t_sp)
            self.good_food_to_d1_r = self._create_static_synapse(
                "good_food_eye_r_to_d1_r", self.good_food_eye_right, self.d1_right,
                td1_w, sparsity=t_sp)

            # bad_food_eye → D1 (R-STDP: 나쁜 음식에는 도파민 없음 → 학습 안됨)
            self.bad_food_to_d1_l = self._create_static_synapse(
                "bad_food_eye_l_to_d1_l", self.bad_food_eye_left, self.d1_left,
                td1_w, sparsity=t_sp)
            self.bad_food_to_d1_r = self._create_static_synapse(
                "bad_food_eye_r_to_d1_r", self.bad_food_eye_right, self.d1_right,
                td1_w, sparsity=t_sp)

            # good_food_eye → D2 (Anti-Hebbian: 좋은 음식 + 도파민 → NoGo 약화)
            self.good_food_to_d2_l = self._create_static_synapse(
                "good_food_eye_l_to_d2_l", self.good_food_eye_left, self.d2_left,
                td2_w, sparsity=t_sp)
            self.good_food_to_d2_r = self._create_static_synapse(
                "good_food_eye_r_to_d2_r", self.good_food_eye_right, self.d2_right,
                td2_w, sparsity=t_sp)

            # bad_food_eye → D2 (Anti-Hebbian: 나쁜 음식에는 도파민 없음 → NoGo 유지)
            self.bad_food_to_d2_l = self._create_static_synapse(
                "bad_food_eye_l_to_d2_l", self.bad_food_eye_left, self.d2_left,
                td2_w, sparsity=t_sp)
            self.bad_food_to_d2_r = self._create_static_synapse(
                "bad_food_eye_r_to_d2_r", self.bad_food_eye_right, self.d2_right,
                td2_w, sparsity=t_sp)

            print(f"    Phase L7: GoodFood→D1 (R-STDP): init_w={td1_w}")
            print(f"    Phase L7: BadFood→D1 (R-STDP, no DA): init_w={td1_w}")
            print(f"    Phase L7: GoodFood→D2 (Anti-Hebbian): init_w={td2_w}")
            print(f"    Phase L7: BadFood→D2 (Anti-Hebbian, no DA): init_w={td2_w}")
            print(f"    Phase L7: 8 discriminative BG synapses, sparsity={t_sp}")

        # C1: Sound_Food→D1 — 이제 __init__에서 auditory 빌드 직후 생성 (빌드 순서 해결)

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

        # === Inhibitory Control → Basal Ganglia Direct (억제, 양측) ===
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "inhibitory_to_direct_l", self.inhibitory_control, self.direct_left,
                self.config.inhibitory_to_direct_weight, sparsity=0.1)
            self._create_static_synapse(
                "inhibitory_to_direct_r", self.inhibitory_control, self.direct_right,
                self.config.inhibitory_to_direct_weight, sparsity=0.1)
            print(f"    Inhibitory→Direct(L/R): {self.config.inhibitory_to_direct_weight} (suppress impulsive Go)")

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
        # sound_food: C=5 (C=1은 포화 → L/R 발화율 차이 소멸)
        sound_food_params = dict(lif_params)
        sound_food_params["C"] = 5.0  # 1→5: I=40→3.6spk, I=10→0.8spk (4.5x 차이)
        self.sound_food_left = self.model.add_neuron_population(
            "sound_food_left", self.config.n_sound_food_left,
            sensory_lif_model, sound_food_params, lif_init)
        self.sound_food_right = self.model.add_neuron_population(
            "sound_food_right", self.config.n_sound_food_right,
            sensory_lif_model, sound_food_params, lif_init)

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

        # A1_Food: Push-Pull (C1: 소리 방향 → 접근)
        # Sound_Food_Left → Motor_Left (Push: 같은 방향 접근)
        self._create_static_synapse(
            "sound_food_left_to_motor_left", self.sound_food_left, self.motor_left,
            self.config.sound_food_push_weight, sparsity=0.1)
        self._create_static_synapse(
            "sound_food_right_to_motor_right", self.sound_food_right, self.motor_right,
            self.config.sound_food_push_weight, sparsity=0.1)
        # Sound_Food_Left → Motor_Right (Pull: 반대 방향 억제)
        self._create_static_synapse(
            "sound_food_left_to_motor_right_pull", self.sound_food_left, self.motor_right,
            self.config.sound_food_pull_weight, sparsity=0.1)
        self._create_static_synapse(
            "sound_food_right_to_motor_left_pull", self.sound_food_right, self.motor_left,
            self.config.sound_food_pull_weight, sparsity=0.1)

        print(f"    Sound_Food Push-Pull: push={self.config.sound_food_push_weight}, pull={self.config.sound_food_pull_weight}")

        # C1: Sound→Food_Eye 교차 억제 (Webb cricket phonotaxis 원리)
        # 소리가 강한 쪽이 반대쪽 시각을 억제 → 시각 전류 비대칭 → 방향 신호
        # 100뉴런 × 0.15 = 15연결 × -15 = -225 → food_eye 발화 ~17% 감소
        cross_inh_w = -15.0
        self._create_static_synapse(
            "sound_food_l_inhibit_food_eye_r", self.sound_food_left, self.food_eye_right,
            cross_inh_w, sparsity=0.15)
        self._create_static_synapse(
            "sound_food_r_inhibit_food_eye_l", self.sound_food_right, self.food_eye_left,
            cross_inh_w, sparsity=0.15)
        print(f"    C1: Sound→Food_Eye cross-inhibition: {cross_inh_w} (Webb phonotaxis)")

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

        # === 8. Basal Ganglia → PMC (행동 선택, 양측) ===
        if self.config.basal_ganglia_enabled:
            # Direct L/R → Motor_Preparation (Go 신호)
            self._create_static_synapse(
                "direct_l_to_motor_prep", self.direct_left, self.motor_preparation,
                self.config.direct_to_motor_prep_weight, sparsity=0.1)
            self._create_static_synapse(
                "direct_r_to_motor_prep", self.direct_right, self.motor_preparation,
                self.config.direct_to_motor_prep_weight, sparsity=0.1)
            # Indirect L/R → Motor_Preparation (NoGo 신호)
            self._create_static_synapse(
                "indirect_l_to_motor_prep", self.indirect_left, self.motor_preparation,
                self.config.indirect_to_motor_prep_weight, sparsity=0.1)
            self._create_static_synapse(
                "indirect_r_to_motor_prep", self.indirect_right, self.motor_preparation,
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

    def release_dopamine(self, reward_magnitude: float = 1.0, primary_reward: bool = False):
        """
        Phase 4: 보상 시 Dopamine 방출
        Phase L10: primary_reward=True + 양수 보상일 때 NAc 기반 RPE 적용

        Args:
            reward_magnitude: 보상 크기 (-1~1, 음수 = dip)
            primary_reward: True이면 RPE 모듈레이션 적용 (L10)
        """
        if not self.config.basal_ganglia_enabled:
            return

        # Phase L10: RPE modulation (양수 primary rewards만)
        effective_magnitude = reward_magnitude
        rpe_prediction = 0.0
        if (primary_reward and self.config.td_learning_enabled
                and reward_magnitude > 0):
            nac_rate = getattr(self, '_nac_value_rate', 0.0)
            rpe_prediction = min(nac_rate / self.config.rpe_prediction_threshold, 1.0)
            effective_magnitude = reward_magnitude * (
                1.0 - self.config.rpe_discount * rpe_prediction)
            effective_magnitude = max(effective_magnitude, self.config.rpe_floor)

        # Dopamine 레벨 업데이트 (L8: 음수 허용, -1.0 ~ 1.0)
        self.dopamine_level = float(np.clip(
            self.dopamine_level + effective_magnitude, -1.0, 1.0))

        # Dopamine 뉴런에 입력 전류 주입 (L8: dip 시 뉴런 정지, 음수 전류 방지)
        dopamine_current = max(0.0, self.dopamine_level) * 80.0
        self.dopamine_neurons.vars["I_input"].view[:] = dopamine_current
        self.dopamine_neurons.vars["I_input"].push_to_device()

        return {
            "dopamine_level": self.dopamine_level,
            "effective_magnitude": effective_magnitude,
            "rpe_prediction": rpe_prediction,
        }

    def decay_dopamine(self):
        """Dopamine 레벨 감쇠 + R-STDP 가중치 업데이트"""
        if not self.config.basal_ganglia_enabled:
            return

        # R-STDP 가중치 업데이트 (감쇠 전, dopamine_level이 높을 때)
        rstdp_res = self._update_rstdp_weights()
        if rstdp_res:
            self._last_rstdp_results = rstdp_res

        self.dopamine_level *= self.config.dopamine_decay

        # 감쇠된 레벨 반영 (L8: 음수도 0 방향으로 감쇠)
        if abs(self.dopamine_level) < 0.01:
            self.dopamine_level = 0.0
            self.dopamine_neurons.vars["I_input"].view[:] = 0.0
        else:
            self.dopamine_neurons.vars["I_input"].view[:] = max(0.0, self.dopamine_level) * 80.0
        self.dopamine_neurons.vars["I_input"].push_to_device()

    def _update_rstdp_weights(self):
        """Phase L4: R-STDP D1 강화 + D2 Anti-Hebbian 약화 + 항상성 감쇠"""
        has_dopamine = abs(self.dopamine_level) > 0.01  # L8: 음수 dip도 학습 트리거
        decay = self.config.rstdp_weight_decay
        # 항상성 감쇠: 50 스텝마다 배치 적용 (GPU 전송 최소화)
        apply_decay = decay > 0 and self._rstdp_step % 50 == 0

        if not has_dopamine and not apply_decay:
            return None

        eta = self.config.rstdp_eta * self.uncertainty_eta_mod  # M3: ACh modulation
        w_max = self.config.rstdp_w_max
        w_rest = self.config.rstdp_w_rest
        results = {}
        results["eta_mod"] = self.uncertainty_eta_mod

        # === D1: R-STDP 강화 (보상 시 가중치 증가) ===
        for side, trace, syn in [
            ("left", self.rstdp_trace_l, self.food_to_d1_l),
            ("right", self.rstdp_trace_r, self.food_to_d1_r)
        ]:
            need_update = False
            syn.vars["g"].pull_from_device()
            w = syn.vars["g"].values  # SPARSE → .values (not .view)

            # 항상성 감쇠: w → w_rest 방향으로 서서히 감쇠 (50 스텝분 배치)
            if apply_decay:
                w[:] -= (decay * 50) * (w - w_rest)
                need_update = True

            # R-STDP 강화: 도파민 + 적격 추적 기반 (3-factor rule)
            if has_dopamine and trace > 0.01:
                delta_w = eta * trace * self.dopamine_level
                w[:] += delta_w
                need_update = True

            if need_update:
                w[:] = np.clip(w, 0.0, w_max)
                syn.vars["g"].values = w  # write back
                syn.vars["g"].push_to_device()
            results[f"rstdp_avg_w_{side}"] = float(np.nanmean(w))

        # === D2: Anti-Hebbian 약화 (보상 시 가중치 감소) ===
        eta_d2 = self.config.rstdp_d2_eta
        w_min_d2 = self.config.rstdp_d2_w_min
        for side, trace, syn in [
            ("left", self.rstdp_d2_trace_l, self.food_to_d2_l),
            ("right", self.rstdp_d2_trace_r, self.food_to_d2_r)
        ]:
            need_update = False
            syn.vars["g"].pull_from_device()
            w = syn.vars["g"].values  # SPARSE → .values

            # 항상성 감쇠: D2도 w_rest 방향으로 감쇠 (D1과 동일)
            if apply_decay:
                w[:] -= (decay * 50) * (w - w_rest)
                need_update = True

            # Anti-Hebbian: 도파민 + 적격 추적 → 가중치 감소 (부호 반전)
            if has_dopamine and trace > 0.01:
                delta_w = eta_d2 * trace * self.dopamine_level
                w[:] -= delta_w  # 감소 (Anti-Hebbian)
                need_update = True

            if need_update:
                w[:] = np.clip(w, w_min_d2, w_max)
                syn.vars["g"].values = w  # write back
                syn.vars["g"].push_to_device()
            results[f"rstdp_d2_avg_w_{side}"] = float(np.nanmean(w))

        # === Phase L7: Discriminative BG (good/bad food → D1/D2) ===
        if self.config.discriminative_bg_enabled and self.config.perceptual_learning_enabled:
            # D1: good/bad food → D1 (R-STDP 강화, 기존 D1과 동일 규칙)
            for label, trace, syn in [
                ("good_l", self.typed_d1_trace_good_l, self.good_food_to_d1_l),
                ("good_r", self.typed_d1_trace_good_r, self.good_food_to_d1_r),
                ("bad_l", self.typed_d1_trace_bad_l, self.bad_food_to_d1_l),
                ("bad_r", self.typed_d1_trace_bad_r, self.bad_food_to_d1_r),
            ]:
                need_update = False
                syn.vars["g"].pull_from_device()
                w = syn.vars["g"].values
                if apply_decay:
                    w[:] -= (decay * 50) * (w - w_rest)
                    need_update = True
                if has_dopamine and trace > 0.01:
                    w[:] += eta * trace * self.dopamine_level
                    need_update = True
                if need_update:
                    w[:] = np.clip(w, 0.0, w_max)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()
                results[f"typed_d1_{label}"] = float(np.nanmean(w))

            # D2: good/bad food → D2 (Anti-Hebbian 약화, 기존 D2와 동일 규칙)
            for label, trace, syn in [
                ("good_l", self.typed_d2_trace_good_l, self.good_food_to_d2_l),
                ("good_r", self.typed_d2_trace_good_r, self.good_food_to_d2_r),
                ("bad_l", self.typed_d2_trace_bad_l, self.bad_food_to_d2_l),
                ("bad_r", self.typed_d2_trace_bad_r, self.bad_food_to_d2_r),
            ]:
                need_update = False
                syn.vars["g"].pull_from_device()
                w = syn.vars["g"].values
                if apply_decay:
                    w[:] -= (decay * 50) * (w - w_rest)
                    need_update = True
                if has_dopamine and trace > 0.01:
                    w[:] -= eta_d2 * trace * self.dopamine_level
                    need_update = True
                if need_update:
                    w[:] = np.clip(w, w_min_d2, w_max)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()
                results[f"typed_d2_{label}"] = float(np.nanmean(w))

        # === Phase L9: IT_Food → D1/D2 (피질 하향 연결) ===
        if self.config.it_bg_enabled and self.config.it_enabled:
            it_w_max = 3.0  # 피질은 모듈레이터 → food_eye(5.0)보다 낮게

            # IT_Food → D1 (R-STDP)
            for label, trace, syn in [
                ("l", self.it_food_d1_trace_l, self.it_food_to_d1_l),
                ("r", self.it_food_d1_trace_r, self.it_food_to_d1_r),
            ]:
                need_update = False
                syn.vars["g"].pull_from_device()
                w = syn.vars["g"].values
                if apply_decay:
                    w[:] -= (decay * 50) * (w - w_rest)
                    need_update = True
                if has_dopamine and trace > 0.01:
                    w[:] += eta * trace * self.dopamine_level
                    need_update = True
                if need_update:
                    w[:] = np.clip(w, 0.0, it_w_max)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()
                results[f"it_food_d1_{label}"] = float(np.nanmean(w))

            # IT_Food → D2 (Anti-Hebbian)
            for label, trace, syn in [
                ("l", self.it_food_d2_trace_l, self.it_food_to_d2_l),
                ("r", self.it_food_d2_trace_r, self.it_food_to_d2_r),
            ]:
                need_update = False
                syn.vars["g"].pull_from_device()
                w = syn.vars["g"].values
                if apply_decay:
                    w[:] -= (decay * 50) * (w - w_rest)
                    need_update = True
                if has_dopamine and trace > 0.01:
                    w[:] -= eta_d2 * trace * self.dopamine_level  # Anti-Hebbian
                    need_update = True
                if need_update:
                    w[:] = np.clip(w, w_min_d2, it_w_max)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()
                results[f"it_food_d2_{label}"] = float(np.nanmean(w))

        # === Phase L16: KC → D1/D2 (single KC, M4 context-separated) ===
        if self.config.sparse_expansion_enabled and hasattr(self, 'kc_to_d1_l'):
            kc_w_max = self.config.kc_rstdp_w_max
            kc_w_rest = self.config.kc_rstdp_w_rest
            kc_w_min = self.config.kc_d2_w_min
            eta_d1 = self.config.kc_rstdp_eta
            eta_d2 = self.config.kc_d2_eta

            # M4: context swap (50스텝마다만 = GPU safe)
            ctx = self._current_ctx
            ctx_initialized = hasattr(self, '_ctx_a_kc_to_d1_l')
            last_ctx = getattr(self, '_last_rstdp_ctx', ctx)

            # Context 변경됐으면 weights swap
            if ctx_initialized and self.config.context_gate_enabled and ctx != last_ctx:
                for sn in ['kc_to_d1_l', 'kc_to_d1_r', 'kc_to_d2_l', 'kc_to_d2_r']:
                    syn = getattr(self, sn)
                    # 이전 ctx weights 저장
                    syn.vars["g"].pull_from_device()
                    setattr(self, f'_ctx_{last_ctx}_{sn}', syn.vars["g"].values.copy())
                    # 새 ctx weights 로드
                    syn.vars["g"].values = getattr(self, f'_ctx_{ctx}_{sn}')
                    syn.vars["g"].push_to_device()
                self._last_rstdp_ctx = ctx

            for side, d1_syn, d2_syn in [
                ("l", self.kc_to_d1_l, self.kc_to_d2_l),
                ("r", self.kc_to_d1_r, self.kc_to_d2_r),
            ]:
                d1_trace = getattr(self, f'kc_d1_trace_{side}')
                d2_trace = getattr(self, f'kc_d2_trace_{side}')
                d1_name = f'kc_to_d1_{side}'
                d2_name = f'kc_to_d2_{side}'

                # Step 1: GPU에서 현재 weights pull
                d1_syn.vars["g"].pull_from_device()
                w_d1 = d1_syn.vars["g"].values.copy()
                d2_syn.vars["g"].pull_from_device()
                w_d2 = d2_syn.vars["g"].values.copy()

                # M4: 현재 GPU weights를 현재 ctx에 저장
                if ctx_initialized and self.config.context_gate_enabled:
                    setattr(self, f'_ctx_{ctx}_{d1_name}', w_d1.copy())
                    setattr(self, f'_ctx_{ctx}_{d2_name}', w_d2.copy())

                # Step 2: R-STDP update (현재 context의 weights에만 적용)
                need_d1 = False
                if apply_decay:
                    w_d1 -= (decay * 50) * (w_d1 - kc_w_rest)
                    need_d1 = True
                if has_dopamine and d1_trace > 0.01:
                    w_d1 += eta_d1 * d1_trace * self.dopamine_level
                    need_d1 = True
                if need_d1:
                    np.clip(w_d1, 0.0, kc_w_max, out=w_d1)

                need_d2 = False
                if apply_decay:
                    w_d2 -= (decay * 50) * (w_d2 - kc_w_rest)
                    need_d2 = True
                if has_dopamine and d2_trace > 0.01:
                    w_d2 -= eta_d2 * d2_trace * self.dopamine_level
                    need_d2 = True
                if need_d2:
                    np.clip(w_d2, kc_w_min, kc_w_max, out=w_d2)

                # M4: 업데이트된 weights를 현재 ctx CPU에 저장
                if ctx_initialized and self.config.context_gate_enabled:
                    setattr(self, f'_ctx_{ctx}_{d1_name}', w_d1.copy())
                    setattr(self, f'_ctx_{ctx}_{d2_name}', w_d2.copy())

                # Step 3: GPU push
                if need_d1:
                    d1_syn.vars["g"].values = w_d1
                    d1_syn.vars["g"].push_to_device()
                if need_d2:
                    d2_syn.vars["g"].values = w_d2
                    d2_syn.vars["g"].push_to_device()

                results[f"kc_d1_{side}"] = float(np.nanmean(w_d1))
                results[f"kc_d2_{side}"] = float(np.nanmean(w_d2))

        # === Food Approach R-STDP (good_food_eye → Motor, 학습 기반 접근) ===
        if self.config.perceptual_learning_enabled and hasattr(self, 'good_food_to_motor_l'):
            fa_eta = self.config.food_approach_eta
            fa_w_max = self.config.food_approach_w_max
            fa_w_rest = self.config.food_approach_init_w

            for side, syn in [
                ("l", self.good_food_to_motor_l),
                ("r", self.good_food_to_motor_r),
            ]:
                need_update = False
                syn.vars["g"].pull_from_device()
                w = syn.vars["g"].values.copy()
                if apply_decay:
                    w -= (decay * 10) * (w - fa_w_rest)
                    need_update = True
                if has_dopamine and self.rstdp_trace_l + self.rstdp_trace_r > 0.01:
                    # 도파민 양수: 좋은 음식 → 접근 강화
                    # 도파민 음수: 나쁜 음식 → 접근 약화 (dip이 자동 반전)
                    trace = self.rstdp_trace_l if side == "l" else self.rstdp_trace_r
                    w += fa_eta * trace * self.dopamine_level
                    need_update = True
                if need_update:
                    np.clip(w, 0.0, fa_w_max, out=w)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()
                results[f"food_approach_{side}"] = float(np.nanmean(w))

        # === C1: Sound_Food → D1 R-STDP (eligibility bridge trace 사용) ===
        if self.config.auditory_enabled and hasattr(self, 'sound_food_to_d1_l'):
            sf_eta = 0.001
            sf_w_max = 5.0
            sf_w_rest = 0.5
            sound_tag_l = getattr(self, '_sound_elig_tag_l', 0.0)
            sound_tag_r = getattr(self, '_sound_elig_tag_r', 0.0)

            for side, syn, tag in [
                ("l", self.sound_food_to_d1_l, sound_tag_l),
                ("r", self.sound_food_to_d1_r, sound_tag_r),
            ]:
                need_update = False
                syn.vars["g"].pull_from_device()
                w = syn.vars["g"].values.copy()
                if apply_decay:
                    w -= (decay * 10) * (w - sf_w_rest)
                    need_update = True
                if has_dopamine and tag > 0.01:
                    w += sf_eta * tag * self.dopamine_level
                    need_update = True
                if need_update:
                    np.clip(w, 0.0, sf_w_max, out=w)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()
                results[f"sound_d1_{side}"] = float(np.nanmean(w))

        # === Phase L10: NAc R-STDP (food_eye → nac_value) ===
        if self.config.td_learning_enabled:
            nac_eta = self.config.nac_rstdp_eta
            nac_w_max = self.config.nac_w_max
            for side, trace, syn in [
                ("l", self.nac_trace_l, self.food_to_nac_l),
                ("r", self.nac_trace_r, self.food_to_nac_r),
            ]:
                need_update = False
                syn.vars["g"].pull_from_device()
                w = syn.vars["g"].values
                if apply_decay:
                    w[:] -= (decay * 50) * (w - w_rest)
                    need_update = True
                if has_dopamine and trace > 0.01:
                    w[:] += nac_eta * trace * self.dopamine_level
                    need_update = True
                if need_update:
                    w[:] = np.clip(w, 0.0, nac_w_max)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()
                results[f"nac_avg_w_{side}"] = float(np.nanmean(w))

        # === Phase C4: Predictive Plasticity (place→pred, wmcb→pred) ===
        # Teacher-driven LTP + weight-dependent LTD + per-post budget normalization
        # NO dopamine here — representation learning is self-supervised
        if self.config.contextual_prediction_enabled and hasattr(self, 'place_to_pred'):
            eta_ltp = self.config.place_to_pred_eta_ltp
            eta_ltd = self.config.place_to_pred_eta_ltd
            w_max = self.config.place_to_pred_w_max
            w_budget = self.config.place_to_pred_w_budget
            teacher_active = self._pred_food_teacher_active

            # Place → Pred: Predictive STDP (DENSE)
            self.place_to_pred.vars["g"].pull_from_device()
            w = self.place_to_pred.vars["g"].view.copy()  # DENSE: use .view
            n_pre = self.config.n_place_cells
            n_post = self.config.n_pred_food_soon

            # LTD: weight-dependent depression on all active synapses (every update)
            w -= eta_ltd * w  # proportional decay: strong synapses decay more

            # LTP: teacher-driven (food visible → strengthen active context)
            if teacher_active:
                # Use place_cell activation array as pre-trace (continuous values)
                place_activity = self.last_active_place_cells  # shape: (n_place_cells,)
                if len(place_activity) == n_pre:
                    w_2d = w.reshape(n_pre, n_post)
                    # Active cells (above threshold) get LTP proportional to activation
                    active_mask = place_activity > 0.1  # threshold for "active"
                    w_2d[active_mask, :] += eta_ltp * place_activity[active_mask, np.newaxis]
                    w = w_2d.reshape(-1)

            # Per-post budget normalization (heterosynaptic)
            w_2d = w.reshape(n_pre, n_post)
            for j in range(n_post):
                col_sum = np.sum(np.maximum(w_2d[:, j], 0.0))
                if col_sum > w_budget:
                    scale = w_budget / col_sum
                    w_2d[:, j] *= scale
            w = w_2d.reshape(-1)

            # Clip
            np.clip(w, 0.0, w_max, out=w)
            self.place_to_pred.vars["g"].view[:] = w
            self.place_to_pred.vars["g"].push_to_device()
            results["pred_place_w"] = float(np.mean(w))

            # WMCB → Pred: Same predictive rule (DENSE)
            if hasattr(self, 'wmcb_to_pred'):
                eta_ltp_wm = self.config.wmcb_to_pred_eta_ltp
                eta_ltd_wm = self.config.wmcb_to_pred_eta_ltd
                w_max_wm = self.config.wmcb_to_pred_w_max
                self.wmcb_to_pred.vars["g"].pull_from_device()
                w_wm = self.wmcb_to_pred.vars["g"].view.copy()
                n_wm_pre = 100  # wm_context_binding size

                # LTD
                w_wm -= eta_ltd_wm * w_wm

                # LTP on teacher
                if teacher_active:
                    wmcb_active = 1.0 if self.last_wm_context_binding_rate > 0.02 else 0.0
                    if wmcb_active > 0:
                        w_wm += eta_ltp_wm * wmcb_active

                # Budget normalization (shared with place budget)
                w_wm_2d = w_wm.reshape(n_wm_pre, n_post)
                for j in range(n_post):
                    col_sum = np.sum(np.maximum(w_wm_2d[:, j], 0.0))
                    if col_sum > w_budget * 0.3:  # WM gets 30% of budget
                        scale = (w_budget * 0.3) / col_sum
                        w_wm_2d[:, j] *= scale
                w_wm = w_wm_2d.reshape(-1)

                np.clip(w_wm, 0.0, w_max_wm, out=w_wm)
                self.wmcb_to_pred.vars["g"].view[:] = w_wm
                self.wmcb_to_pred.vars["g"].push_to_device()
                results["pred_wmcb_w"] = float(np.mean(w_wm))

        return results if results else None

    def learn_food_location(self, food_position: tuple = None, anti_learn: bool = False):
        """
        Phase 3b/3c: 음식 발견 시 Hebbian 학습
        C0.5: anti_learn=True면 가중치 약화 (나쁜 음식 위치 기억 약화)

        Args:
            food_position: (x, y) 정규화된 음식 위치
            anti_learn: True면 Δw = -η * pre_activity (가중치 감소)

        Δw = ±η * pre_activity
        """
        if not self.config.hippocampus_enabled or not self.food_learning_enabled:
            return

        active_cells = self.last_active_place_cells
        eta = self.config.place_to_food_memory_eta
        if anti_learn:
            eta = -eta * 0.5  # 약화는 강화의 절반 속도
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

            avg_w = float(np.mean(weights))
            self._last_hippo_avg_w = avg_w
            return {
                "n_strengthened": n_strengthened,
                "avg_weight": avg_w,
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
            avg_w = float(np.mean(weights))
            self._last_hippo_avg_w = avg_w

            return {
                "n_strengthened": int(n_strengthened),
                "avg_weight": avg_w,
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

    def save_all_weights(self, filepath: str) -> str:
        """모든 Hebbian 시냅스 가중치를 한 파일에 저장"""
        weights = {}

        # Hippocampus (Phase 3c)
        if self.config.hippocampus_enabled and self.food_learning_enabled:
            if self.config.directional_food_memory:
                self.place_to_food_memory_left.vars["g"].pull_from_device()
                self.place_to_food_memory_right.vars["g"].pull_from_device()
                weights["hippo_left"] = self.place_to_food_memory_left.vars["g"].view.copy()
                weights["hippo_right"] = self.place_to_food_memory_right.vars["g"].view.copy()

        # 나머지 Hebbian 시냅스들
        hebbian_synapses = {
            "vicarious_social": "vicarious_to_social_memory",
            "tom_coop": "tom_intention_to_coop_hebbian",
            "assoc_edible": "assoc_edible_to_binding_hebbian",
            "assoc_context": "assoc_context_to_binding_hebbian",
            "wernicke_food": "wernicke_food_to_binding_hebbian",
            "wernicke_danger": "wernicke_danger_to_binding_hebbian",
            "wm_temporal": "temporal_to_context_hebbian",
            "meta_valence": "valence_to_confidence_hebbian",
            "self_narrative": "body_to_narrative_hebbian",
            "agency_narrative": "agency_to_narrative_hebbian",
        }
        for key, attr in hebbian_synapses.items():
            if hasattr(self, attr):
                syn = getattr(self, attr)
                syn.vars["g"].pull_from_device()
                weights[key] = syn.vars["g"].view.copy()

        # R-STDP (Phase L2: Food_Eye→D1, SPARSE)
        if self.config.basal_ganglia_enabled:
            self.food_to_d1_l.vars["g"].pull_from_device()
            self.food_to_d1_r.vars["g"].pull_from_device()
            weights["rstdp_left"] = self.food_to_d1_l.vars["g"].values.copy()
            weights["rstdp_right"] = self.food_to_d1_r.vars["g"].values.copy()
            # Phase L4: Anti-Hebbian D2
            self.food_to_d2_l.vars["g"].pull_from_device()
            self.food_to_d2_r.vars["g"].pull_from_device()
            weights["rstdp_d2_left"] = self.food_to_d2_l.vars["g"].values.copy()
            weights["rstdp_d2_right"] = self.food_to_d2_r.vars["g"].values.copy()

        # Phase L7: Discriminative BG (8 SPARSE synapses)
        if self.config.discriminative_bg_enabled and self.config.perceptual_learning_enabled:
            typed_bg_synapses = {
                "good_food_d1_left": self.good_food_to_d1_l,
                "good_food_d1_right": self.good_food_to_d1_r,
                "bad_food_d1_left": self.bad_food_to_d1_l,
                "bad_food_d1_right": self.bad_food_to_d1_r,
                "good_food_d2_left": self.good_food_to_d2_l,
                "good_food_d2_right": self.good_food_to_d2_r,
                "bad_food_d2_left": self.bad_food_to_d2_l,
                "bad_food_d2_right": self.bad_food_to_d2_r,
            }
            for key, syn in typed_bg_synapses.items():
                syn.vars["g"].pull_from_device()
                weights[key] = syn.vars["g"].values.copy()

        # Phase L9: IT→BG (4 SPARSE synapses)
        if self.config.it_bg_enabled and self.config.it_enabled:
            it_bg_synapses = {
                "it_food_d1_left": self.it_food_to_d1_l,
                "it_food_d1_right": self.it_food_to_d1_r,
                "it_food_d2_left": self.it_food_to_d2_l,
                "it_food_d2_right": self.it_food_to_d2_r,
            }
            for key, syn in it_bg_synapses.items():
                syn.vars["g"].pull_from_device()
                weights[key] = syn.vars["g"].values.copy()

        # Phase L10: NAc R-STDP (2 SPARSE synapses)
        if self.config.td_learning_enabled and self.config.basal_ganglia_enabled:
            nac_synapses = {
                "nac_food_left": self.food_to_nac_l,
                "nac_food_right": self.food_to_nac_r,
            }
            for key, syn in nac_synapses.items():
                syn.vars["g"].pull_from_device()
                weights[key] = syn.vars["g"].values.copy()

        # Phase L11: Experience buffer
        if self.config.swr_replay_enabled and self.config.hippocampus_enabled:
            weights["swr_experience_buffer"] = np.array(
                self.experience_buffer, dtype=np.float32) if self.experience_buffer else np.array([])

        # Phase L13: Taste Aversion Hebbian (2 DENSE synapses)
        if self.config.taste_aversion_learning_enabled and hasattr(self, 'bad_food_to_la_left'):
            self.bad_food_to_la_left.vars["g"].pull_from_device()
            self.bad_food_to_la_right.vars["g"].pull_from_device()
            weights["bad_food_la_left"] = self.bad_food_to_la_left.vars["g"].view.copy()
            weights["bad_food_la_right"] = self.bad_food_to_la_right.vars["g"].view.copy()

        # Phase L14: Forward Model Hebbian (DENSE)
        if self.config.agency_detection_enabled and hasattr(self, 'efference_to_predict_hebbian'):
            self.efference_to_predict_hebbian.vars["g"].pull_from_device()
            weights["forward_model"] = self.efference_to_predict_hebbian.vars["g"].view.copy()

        # Phase L15: Agency→Narrative Hebbian (DENSE)
        if self.config.narrative_self_enabled and hasattr(self, 'agency_to_narrative_hebbian'):
            self.agency_to_narrative_hebbian.vars["g"].pull_from_device()
            weights["agency_narrative"] = self.agency_to_narrative_hebbian.vars["g"].view.copy()

        # Phase L16: KC→D1/D2 (4 SPARSE synapses, single KC)
        if self.config.sparse_expansion_enabled and hasattr(self, 'kc_to_d1_l'):
            for key, syn in [
                ("kc_d1_left", self.kc_to_d1_l),
                ("kc_d1_right", self.kc_to_d1_r),
                ("kc_d2_left", self.kc_to_d2_l),
                ("kc_d2_right", self.kc_to_d2_r),
            ]:
                syn.vars["g"].pull_from_device()
                weights[key] = syn.vars["g"].values.copy()

        # Food Approach (2 SPARSE synapses)
        if self.config.perceptual_learning_enabled and hasattr(self, 'good_food_to_motor_l'):
            self.good_food_to_motor_l.vars["g"].pull_from_device()
            weights["food_approach_left"] = self.good_food_to_motor_l.vars["g"].values.copy()
            self.good_food_to_motor_r.vars["g"].pull_from_device()
            weights["food_approach_right"] = self.good_food_to_motor_r.vars["g"].values.copy()

        # Phase L5: 피질 R-STDP (8 SPARSE synapses)
        if self.config.perceptual_learning_enabled and self.config.it_enabled:
            cortical_synapses = {
                "cortical_good_food_l": self.good_food_to_it_food_l,
                "cortical_good_food_r": self.good_food_to_it_food_r,
                "cortical_good_danger_l": self.good_food_to_it_danger_l,
                "cortical_good_danger_r": self.good_food_to_it_danger_r,
                "cortical_bad_danger_l": self.bad_food_to_it_danger_l,
                "cortical_bad_danger_r": self.bad_food_to_it_danger_r,
                "cortical_bad_food_l": self.bad_food_to_it_food_l,
                "cortical_bad_food_r": self.bad_food_to_it_food_r,
            }
            for key, syn in cortical_synapses.items():
                syn.vars["g"].pull_from_device()
                weights[key] = syn.vars["g"].values.copy()

        # Phase L6: PE→IT 시냅스 (SPARSE)
        if self.config.prediction_error_enabled and self.config.v1_enabled and self.config.it_enabled:
            pe_synapses = {
                "pe_food_l": self.pe_food_to_it_food_l,
                "pe_food_r": self.pe_food_to_it_food_r,
                "pe_danger_l": self.pe_danger_to_it_danger_l,
                "pe_danger_r": self.pe_danger_to_it_danger_r,
            }
            for key, syn in pe_synapses.items():
                syn.vars["g"].pull_from_device()
                weights[key] = syn.vars["g"].values.copy()

        # Phase C4: Contextual Prediction (2 DENSE synapses — predictive plasticity)
        if self.config.contextual_prediction_enabled and hasattr(self, 'place_to_pred'):
            self.place_to_pred.vars["g"].pull_from_device()
            weights["pred_place"] = self.place_to_pred.vars["g"].view.copy()
            if hasattr(self, 'wmcb_to_pred'):
                self.wmcb_to_pred.vars["g"].pull_from_device()
                weights["pred_wmcb"] = self.wmcb_to_pred.vars["g"].view.copy()

        np.savez(filepath, **weights)
        print(f"  [SAVE] All weights saved to {filepath} ({len(weights)} synapses)")
        return filepath

    def _load_sparse_weights(self, syn, saved_weights):
        """SPARSE 시냅스에 저장된 가중치 로드 (shape 불일치 시 평균값 브로드캐스트)"""
        syn.pull_connectivity_from_device()
        current = syn.vars["g"].values
        if current.shape == saved_weights.shape:
            syn.vars["g"].values = saved_weights
        else:
            # shape 불일치: 랜덤 SPARSE 연결이 달라짐 → 평균값으로 브로드캐스트
            mean_w = float(np.mean(saved_weights))
            new_weights = np.full_like(current, mean_w)
            syn.vars["g"].values = new_weights
            print(f"    [WARN] Shape mismatch ({saved_weights.shape}→{current.shape}), broadcast mean={mean_w:.3f}")
        syn.vars["g"].push_to_device()

    def load_all_weights(self, filepath: str) -> bool:
        """저장된 모든 Hebbian 가중치를 로드"""
        if not os.path.exists(filepath):
            print(f"  [LOAD] File not found: {filepath}")
            return False

        data = np.load(filepath)
        loaded = 0

        # Hippocampus
        if "hippo_left" in data and hasattr(self, "place_to_food_memory_left"):
            self.place_to_food_memory_left.vars["g"].view[:] = data["hippo_left"]
            self.place_to_food_memory_left.vars["g"].push_to_device()
            self.place_to_food_memory_right.vars["g"].view[:] = data["hippo_right"]
            self.place_to_food_memory_right.vars["g"].push_to_device()
            loaded += 2

        hebbian_synapses = {
            "vicarious_social": "vicarious_to_social_memory",
            "tom_coop": "tom_intention_to_coop_hebbian",
            "assoc_edible": "assoc_edible_to_binding_hebbian",
            "assoc_context": "assoc_context_to_binding_hebbian",
            "wernicke_food": "wernicke_food_to_binding_hebbian",
            "wernicke_danger": "wernicke_danger_to_binding_hebbian",
            "wm_temporal": "temporal_to_context_hebbian",
            "meta_valence": "valence_to_confidence_hebbian",
            "self_narrative": "body_to_narrative_hebbian",
            "agency_narrative": "agency_to_narrative_hebbian",
        }
        for key, attr in hebbian_synapses.items():
            if key in data and hasattr(self, attr):
                syn = getattr(self, attr)
                syn.vars["g"].view[:] = data[key]
                syn.vars["g"].push_to_device()
                loaded += 1

        # R-STDP (Phase L2: Food_Eye→D1, SPARSE)
        if "rstdp_left" in data and self.config.basal_ganglia_enabled:
            self._load_sparse_weights(self.food_to_d1_l, data["rstdp_left"])
            self._load_sparse_weights(self.food_to_d1_r, data["rstdp_right"])
            loaded += 2

        # Phase L4: Anti-Hebbian D2 (SPARSE)
        if "rstdp_d2_left" in data and self.config.basal_ganglia_enabled:
            self._load_sparse_weights(self.food_to_d2_l, data["rstdp_d2_left"])
            self._load_sparse_weights(self.food_to_d2_r, data["rstdp_d2_right"])
            loaded += 2

        # Phase L7: Discriminative BG (SPARSE)
        if self.config.discriminative_bg_enabled and self.config.perceptual_learning_enabled:
            typed_bg_synapses = {
                "good_food_d1_left": self.good_food_to_d1_l,
                "good_food_d1_right": self.good_food_to_d1_r,
                "bad_food_d1_left": self.bad_food_to_d1_l,
                "bad_food_d1_right": self.bad_food_to_d1_r,
                "good_food_d2_left": self.good_food_to_d2_l,
                "good_food_d2_right": self.good_food_to_d2_r,
                "bad_food_d2_left": self.bad_food_to_d2_l,
                "bad_food_d2_right": self.bad_food_to_d2_r,
            }
            for key, syn in typed_bg_synapses.items():
                if key in data:
                    self._load_sparse_weights(syn, data[key])
                    loaded += 1

        # Phase L9: IT→BG (SPARSE)
        if self.config.it_bg_enabled and self.config.it_enabled:
            it_bg_synapses = {
                "it_food_d1_left": self.it_food_to_d1_l,
                "it_food_d1_right": self.it_food_to_d1_r,
                "it_food_d2_left": self.it_food_to_d2_l,
                "it_food_d2_right": self.it_food_to_d2_r,
            }
            for key, syn in it_bg_synapses.items():
                if key in data:
                    self._load_sparse_weights(syn, data[key])
                    loaded += 1

        # Phase L10: NAc R-STDP (SPARSE)
        if self.config.td_learning_enabled and self.config.basal_ganglia_enabled:
            nac_synapses = {
                "nac_food_left": self.food_to_nac_l,
                "nac_food_right": self.food_to_nac_r,
            }
            for key, syn in nac_synapses.items():
                if key in data:
                    self._load_sparse_weights(syn, data[key])
                    loaded += 1

        # Phase L11: Experience buffer
        if self.config.swr_replay_enabled and self.config.hippocampus_enabled:
            if "swr_experience_buffer" in data:
                buf = data["swr_experience_buffer"]
                if buf.size > 0:
                    cols = buf.shape[-1] if buf.ndim > 1 else (6 if buf.size % 6 == 0 else 5)
                    self.experience_buffer = [tuple(row) for row in buf.reshape(-1, cols)]
                    loaded += 1

        # Phase L13: Taste Aversion Hebbian (DENSE)
        if self.config.taste_aversion_learning_enabled and hasattr(self, 'bad_food_to_la_left'):
            if "bad_food_la_left" in data:
                self.bad_food_to_la_left.vars["g"].view[:] = data["bad_food_la_left"]
                self.bad_food_to_la_left.vars["g"].push_to_device()
                loaded += 1
            if "bad_food_la_right" in data:
                self.bad_food_to_la_right.vars["g"].view[:] = data["bad_food_la_right"]
                self.bad_food_to_la_right.vars["g"].push_to_device()
                loaded += 1

        # Phase L14: Forward Model Hebbian (DENSE)
        if self.config.agency_detection_enabled and hasattr(self, 'efference_to_predict_hebbian'):
            if "forward_model" in data:
                self.efference_to_predict_hebbian.vars["g"].view[:] = data["forward_model"]
                self.efference_to_predict_hebbian.vars["g"].push_to_device()
                loaded += 1

        # Phase L15: Agency→Narrative Hebbian (DENSE)
        if self.config.narrative_self_enabled and hasattr(self, 'agency_to_narrative_hebbian'):
            if "agency_narrative" in data:
                self.agency_to_narrative_hebbian.vars["g"].view[:] = data["agency_narrative"]
                self.agency_to_narrative_hebbian.vars["g"].push_to_device()
                loaded += 1

        # Phase L16: KC→D1/D2 (SPARSE, single KC)
        if self.config.sparse_expansion_enabled and hasattr(self, 'kc_to_d1_l'):
            kc_syns = [
                ("kc_d1_left", self.kc_to_d1_l),
                ("kc_d1_right", self.kc_to_d1_r),
                ("kc_d2_left", self.kc_to_d2_l),
                ("kc_d2_right", self.kc_to_d2_r),
            ]
            for key, syn in kc_syns:
                if key in data:
                    self._load_sparse_weights(syn, data[key])
                    loaded += 1

        # Food Approach (SPARSE)
        if self.config.perceptual_learning_enabled and hasattr(self, 'good_food_to_motor_l'):
            for key, syn in [
                ("food_approach_left", self.good_food_to_motor_l),
                ("food_approach_right", self.good_food_to_motor_r),
            ]:
                if key in data:
                    self._load_sparse_weights(syn, data[key])
                    loaded += 1

        # Phase L5: 피질 R-STDP (SPARSE)
        if self.config.perceptual_learning_enabled and self.config.it_enabled:
            cortical_synapses = {
                "cortical_good_food_l": self.good_food_to_it_food_l,
                "cortical_good_food_r": self.good_food_to_it_food_r,
                "cortical_good_danger_l": self.good_food_to_it_danger_l,
                "cortical_good_danger_r": self.good_food_to_it_danger_r,
                "cortical_bad_danger_l": self.bad_food_to_it_danger_l,
                "cortical_bad_danger_r": self.bad_food_to_it_danger_r,
                "cortical_bad_food_l": self.bad_food_to_it_food_l,
                "cortical_bad_food_r": self.bad_food_to_it_food_r,
            }
            for key, syn in cortical_synapses.items():
                if key in data:
                    self._load_sparse_weights(syn, data[key])
                    loaded += 1

        # Phase L6: PE→IT (SPARSE)
        if self.config.prediction_error_enabled and self.config.v1_enabled and self.config.it_enabled:
            pe_synapses = {
                "pe_food_l": self.pe_food_to_it_food_l,
                "pe_food_r": self.pe_food_to_it_food_r,
                "pe_danger_l": self.pe_danger_to_it_danger_l,
                "pe_danger_r": self.pe_danger_to_it_danger_r,
            }
            for key, syn in pe_synapses.items():
                if key in data:
                    self._load_sparse_weights(syn, data[key])
                    loaded += 1

        # Phase C4: Contextual Prediction (DENSE — predictive plasticity)
        if self.config.contextual_prediction_enabled and hasattr(self, 'place_to_pred'):
            if "pred_place" in data:
                saved = data["pred_place"]
                current = self.place_to_pred.vars["g"].view
                if current.shape == saved.shape:
                    self.place_to_pred.vars["g"].view[:] = saved
                else:
                    self.place_to_pred.vars["g"].view[:] = np.mean(saved)
                self.place_to_pred.vars["g"].push_to_device()
                loaded += 1
            if "pred_wmcb" in data and hasattr(self, 'wmcb_to_pred'):
                saved = data["pred_wmcb"]
                current = self.wmcb_to_pred.vars["g"].view
                if current.shape == saved.shape:
                    self.wmcb_to_pred.vars["g"].view[:] = saved
                else:
                    self.wmcb_to_pred.vars["g"].view[:] = np.mean(saved)
                self.wmcb_to_pred.vars["g"].push_to_device()
                loaded += 1

        print(f"  [LOAD] Weights loaded from {filepath} ({loaded} synapses)")
        return True

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

    def _build_language_circuit(self):
        """
        Phase 17: Language Circuit (언어 회로 - Broca/Wernicke)

        Vervet monkey alarm call 모델 기반 proto-language:
        - Wernicke's Area: 발성 청취 → 범주 이해 (comprehension)
        - Broca's Area: 내부 상태 → 발성 생산 프로그램 (production)
        - Arcuate Fasciculus: Broca↔Wernicke 양방향 연결
        - Vocal Gate (PAG): 발성 게이팅 (언제 소리낼지)
        - Call Mirror: 듣기+생산 양쪽에서 활성
        - Call Binding: 소리-의미 연합 학습 (Hebbian)
        """
        print("  Phase 17: Building Language Circuit...")

        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0, "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0}

        s_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        s_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 1. 감각 입력 (4 SensoryLIF) ===
        self.call_food_input_left = self.model.add_neuron_population(
            "call_food_input_left", self.config.n_call_food_input_left,
            sensory_lif_model, s_params, s_init)
        self.call_food_input_right = self.model.add_neuron_population(
            "call_food_input_right", self.config.n_call_food_input_right,
            sensory_lif_model, s_params, s_init)
        self.call_danger_input_left = self.model.add_neuron_population(
            "call_danger_input_left", self.config.n_call_danger_input_left,
            sensory_lif_model, s_params, s_init)
        self.call_danger_input_right = self.model.add_neuron_population(
            "call_danger_input_right", self.config.n_call_danger_input_right,
            sensory_lif_model, s_params, s_init)

        # === 2. Wernicke's Area (4 LIF - 이해) ===
        self.wernicke_food = self.model.add_neuron_population(
            "wernicke_food", self.config.n_wernicke_food, "LIF", lif_params, lif_init)
        self.wernicke_danger = self.model.add_neuron_population(
            "wernicke_danger", self.config.n_wernicke_danger, "LIF", lif_params, lif_init)
        self.wernicke_social = self.model.add_neuron_population(
            "wernicke_social", self.config.n_wernicke_social, "LIF", lif_params, lif_init)
        self.wernicke_context = self.model.add_neuron_population(
            "wernicke_context", self.config.n_wernicke_context, "LIF", lif_params, lif_init)

        # === 3. Broca's Area (4 LIF - 생산) ===
        self.broca_food = self.model.add_neuron_population(
            "broca_food", self.config.n_broca_food, "LIF", lif_params, lif_init)
        self.broca_danger = self.model.add_neuron_population(
            "broca_danger", self.config.n_broca_danger, "LIF", lif_params, lif_init)
        self.broca_social = self.model.add_neuron_population(
            "broca_social", self.config.n_broca_social, "LIF", lif_params, lif_init)
        self.broca_sequence = self.model.add_neuron_population(
            "broca_sequence", self.config.n_broca_sequence, "LIF", lif_params, lif_init)

        # === 4. Vocal Gate / PAG (SensoryLIF - Fear 억제용 I_input) ===
        self.vocal_gate = self.model.add_neuron_population(
            "vocal_gate", self.config.n_vocal_gate,
            sensory_lif_model, s_params, s_init)

        # === 5. Call Mirror + Call Binding (LIF) ===
        self.call_mirror = self.model.add_neuron_population(
            "call_mirror", self.config.n_call_mirror, "LIF", lif_params, lif_init)
        self.call_binding = self.model.add_neuron_population(
            "call_binding", self.config.n_call_binding, "LIF", lif_params, lif_init)

        print(f"    Call Input: {self.config.n_call_food_input_left * 2 + self.config.n_call_danger_input_left * 2}")
        print(f"    Wernicke: {self.config.n_wernicke_food + self.config.n_wernicke_danger + self.config.n_wernicke_social + self.config.n_wernicke_context}")
        print(f"    Broca: {self.config.n_broca_food + self.config.n_broca_danger + self.config.n_broca_social + self.config.n_broca_sequence}")
        print(f"    Vocal Gate: {self.config.n_vocal_gate}, Mirror: {self.config.n_call_mirror}, Binding: {self.config.n_call_binding}")

        # === 6. Call Input → Wernicke (4 synapses) ===
        self._create_static_synapse(
            "call_food_l_to_wernicke_food", self.call_food_input_left, self.wernicke_food,
            self.config.call_to_wernicke_weight, sparsity=0.10)
        self._create_static_synapse(
            "call_food_r_to_wernicke_food", self.call_food_input_right, self.wernicke_food,
            self.config.call_to_wernicke_weight, sparsity=0.10)
        self._create_static_synapse(
            "call_danger_l_to_wernicke_danger", self.call_danger_input_left, self.wernicke_danger,
            self.config.call_to_wernicke_weight, sparsity=0.10)
        self._create_static_synapse(
            "call_danger_r_to_wernicke_danger", self.call_danger_input_right, self.wernicke_danger,
            self.config.call_to_wernicke_weight, sparsity=0.10)

        # === 7. Wernicke 내부 (7 synapses) ===
        # Food ↔ Danger WTA
        self._create_static_synapse(
            "wernicke_food_to_danger_wta", self.wernicke_food, self.wernicke_danger,
            self.config.wernicke_food_danger_wta, sparsity=0.08)
        self._create_static_synapse(
            "wernicke_danger_to_food_wta", self.wernicke_danger, self.wernicke_food,
            self.config.wernicke_food_danger_wta, sparsity=0.08)
        # Food/Danger → Social
        self._create_static_synapse(
            "wernicke_food_to_social", self.wernicke_food, self.wernicke_social,
            self.config.wernicke_to_social_weight, sparsity=0.08)
        self._create_static_synapse(
            "wernicke_danger_to_social", self.wernicke_danger, self.wernicke_social,
            self.config.wernicke_to_social_weight, sparsity=0.08)
        # Food/Danger → Context
        self._create_static_synapse(
            "wernicke_food_to_context", self.wernicke_food, self.wernicke_context,
            self.config.wernicke_to_context_weight, sparsity=0.08)
        self._create_static_synapse(
            "wernicke_danger_to_context", self.wernicke_danger, self.wernicke_context,
            self.config.wernicke_to_context_weight, sparsity=0.08)
        # Context recurrent
        self._create_static_synapse(
            "wernicke_context_recurrent", self.wernicke_context, self.wernicke_context,
            self.config.wernicke_context_recurrent, sparsity=0.05)

        # === 8. Broca 입력 (6 synapses from existing circuits) ===
        if self.config.association_cortex_enabled:
            self._create_static_synapse(
                "assoc_edible_to_broca_food", self.assoc_edible, self.broca_food,
                self.config.assoc_edible_to_broca_food_weight, sparsity=0.08)
            self._create_static_synapse(
                "assoc_threatening_to_broca_danger", self.assoc_threatening, self.broca_danger,
                self.config.assoc_threatening_to_broca_danger_weight, sparsity=0.08)
        self._create_static_synapse(
            "hunger_to_broca_food", self.hunger_drive, self.broca_food,
            self.config.hunger_to_broca_food_weight, sparsity=0.05)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_broca_danger", self.fear_response, self.broca_danger,
                self.config.fear_to_broca_danger_weight, sparsity=0.05)
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "sts_social_to_broca_social", self.sts_social, self.broca_social,
                self.config.sts_social_to_broca_social_weight, sparsity=0.08)
        if self.config.thalamus_enabled:
            self._create_static_synapse(
                "arousal_to_broca_social", self.arousal, self.broca_social,
                self.config.arousal_to_broca_social_weight, sparsity=0.05)

        # === 9. Broca 내부 (6 synapses) ===
        # Food ↔ Danger WTA
        self._create_static_synapse(
            "broca_food_to_danger_wta", self.broca_food, self.broca_danger,
            self.config.broca_food_danger_wta, sparsity=0.08)
        self._create_static_synapse(
            "broca_danger_to_food_wta", self.broca_danger, self.broca_food,
            self.config.broca_food_danger_wta, sparsity=0.08)
        # Food/Danger → Sequence
        self._create_static_synapse(
            "broca_food_to_sequence", self.broca_food, self.broca_sequence,
            self.config.broca_to_sequence_weight, sparsity=0.08)
        self._create_static_synapse(
            "broca_danger_to_sequence", self.broca_danger, self.broca_sequence,
            self.config.broca_to_sequence_weight, sparsity=0.08)
        # Sequence → Broca inhibition (prevents continuous vocalization)
        self._create_static_synapse(
            "broca_seq_to_food_inh", self.broca_sequence, self.broca_food,
            self.config.broca_sequence_to_broca_inh, sparsity=0.08)
        self._create_static_synapse(
            "broca_seq_to_danger_inh", self.broca_sequence, self.broca_danger,
            self.config.broca_sequence_to_broca_inh, sparsity=0.08)
        # Sequence self-inhibition timer
        self._create_static_synapse(
            "broca_seq_recurrent", self.broca_sequence, self.broca_sequence,
            self.config.broca_sequence_recurrent, sparsity=0.05)

        # === 10. Arcuate Fasciculus (4 synapses - bidirectional mirror) ===
        self._create_static_synapse(
            "wernicke_food_to_broca_food", self.wernicke_food, self.broca_food,
            self.config.wernicke_to_broca_weight, sparsity=0.08)
        self._create_static_synapse(
            "wernicke_danger_to_broca_danger", self.wernicke_danger, self.broca_danger,
            self.config.wernicke_to_broca_weight, sparsity=0.08)
        self._create_static_synapse(
            "broca_food_to_wernicke_food", self.broca_food, self.wernicke_food,
            self.config.broca_to_wernicke_weight, sparsity=0.05)
        self._create_static_synapse(
            "broca_danger_to_wernicke_danger", self.broca_danger, self.wernicke_danger,
            self.config.broca_to_wernicke_weight, sparsity=0.05)

        # === 11. Vocal Gate / PAG (3 excitatory synapses) ===
        if self.config.thalamus_enabled:
            self._create_static_synapse(
                "arousal_to_vocal_gate", self.arousal, self.vocal_gate,
                self.config.arousal_to_vocal_gate_weight, sparsity=0.08)
        self._create_static_synapse(
            "broca_food_to_vocal_gate", self.broca_food, self.vocal_gate,
            self.config.broca_to_vocal_gate_weight, sparsity=0.10)
        self._create_static_synapse(
            "broca_danger_to_vocal_gate", self.broca_danger, self.vocal_gate,
            self.config.broca_to_vocal_gate_weight, sparsity=0.10)
        # Fear inhibition via I_input in process()

        # === 12. Call Mirror (4 synapses) ===
        self._create_static_synapse(
            "wernicke_food_to_mirror", self.wernicke_food, self.call_mirror,
            self.config.wernicke_to_call_mirror_weight, sparsity=0.08)
        self._create_static_synapse(
            "wernicke_danger_to_mirror", self.wernicke_danger, self.call_mirror,
            self.config.wernicke_to_call_mirror_weight, sparsity=0.08)
        self._create_static_synapse(
            "broca_food_to_mirror", self.broca_food, self.call_mirror,
            self.config.broca_to_call_mirror_weight, sparsity=0.08)
        self._create_static_synapse(
            "broca_danger_to_mirror", self.broca_danger, self.call_mirror,
            self.config.broca_to_call_mirror_weight, sparsity=0.08)

        # === 13. Call Binding (Hebbian DENSE + sparse inputs + recurrent) ===
        from pygenn import init_weight_update, init_postsynaptic
        # Hebbian DENSE: Wernicke_Food → Call_Binding
        self.wernicke_food_to_binding_hebbian = self.model.add_synapse_population(
            "wernicke_food_to_call_binding_hebb", "DENSE",
            self.wernicke_food, self.call_binding,
            init_weight_update("StaticPulse", {},
                               {"g": self.config.wernicke_to_call_binding_weight}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))
        # Hebbian DENSE: Wernicke_Danger → Call_Binding
        self.wernicke_danger_to_binding_hebbian = self.model.add_synapse_population(
            "wernicke_danger_to_call_binding_hebb", "DENSE",
            self.wernicke_danger, self.call_binding,
            init_weight_update("StaticPulse", {},
                               {"g": self.config.wernicke_to_call_binding_weight}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))
        # Sparse: Assoc → Call_Binding
        if self.config.association_cortex_enabled:
            self._create_static_synapse(
                "assoc_edible_to_call_binding", self.assoc_edible, self.call_binding,
                self.config.assoc_to_call_binding_weight, sparsity=0.05)
            self._create_static_synapse(
                "assoc_threatening_to_call_binding", self.assoc_threatening, self.call_binding,
                self.config.assoc_to_call_binding_weight, sparsity=0.05)
        # Recurrent
        self._create_static_synapse(
            "call_binding_recurrent", self.call_binding, self.call_binding,
            self.config.call_binding_recurrent, sparsity=0.05)

        # === 14. 출력 연결 (모두 ≤6.0, Motor=0.0!) ===
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "wernicke_food_to_goal_food", self.wernicke_food, self.goal_food,
                self.config.wernicke_food_to_goal_food_weight, sparsity=0.05)
            self._create_static_synapse(
                "call_binding_to_wm", self.call_binding, self.working_memory,
                self.config.call_binding_to_wm_weight, sparsity=0.05)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "wernicke_danger_to_la", self.wernicke_danger, self.lateral_amygdala,
                self.config.wernicke_danger_to_fear_weight, sparsity=0.05)
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "wernicke_social_to_sts_social", self.wernicke_social, self.sts_social,
                self.config.wernicke_social_to_sts_social_weight, sparsity=0.05)
        if self.config.association_cortex_enabled:
            self._create_static_synapse(
                "call_mirror_to_assoc_binding", self.call_mirror, self.assoc_binding,
                self.config.call_mirror_to_assoc_binding_weight, sparsity=0.05)
            self._create_static_synapse(
                "call_binding_to_assoc_edible", self.call_binding, self.assoc_edible,
                self.config.call_binding_to_assoc_edible_weight, sparsity=0.05)
            self._create_static_synapse(
                "call_binding_to_assoc_threatening", self.call_binding, self.assoc_threatening,
                self.config.call_binding_to_assoc_threatening_weight, sparsity=0.05)

        # === 15. Top-Down 조절 (3 synapses) ===
        self._create_static_synapse(
            "hunger_to_wernicke_food_td", self.hunger_drive, self.wernicke_food,
            self.config.hunger_to_wernicke_food_weight, sparsity=0.05)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_wernicke_danger_td", self.fear_response, self.wernicke_danger,
                self.config.fear_to_wernicke_danger_weight, sparsity=0.05)
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "wm_to_wernicke_context_td", self.working_memory, self.wernicke_context,
                self.config.wm_to_wernicke_context_weight, sparsity=0.05)

        # === 16. Context 입력 (2 synapses) ===
        if self.config.hippocampus_enabled:
            self._create_static_synapse(
                "place_to_wernicke_context", self.place_cells, self.wernicke_context,
                self.config.place_to_wernicke_context_weight, sparsity=0.02)
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "sts_social_to_wernicke_context", self.sts_social, self.wernicke_context,
                self.config.sts_social_to_wernicke_context_weight, sparsity=0.05)

        n_lang_total = (self.config.n_call_food_input_left + self.config.n_call_food_input_right +
                        self.config.n_call_danger_input_left + self.config.n_call_danger_input_right +
                        self.config.n_wernicke_food + self.config.n_wernicke_danger +
                        self.config.n_wernicke_social + self.config.n_wernicke_context +
                        self.config.n_broca_food + self.config.n_broca_danger +
                        self.config.n_broca_social + self.config.n_broca_sequence +
                        self.config.n_vocal_gate + self.config.n_call_mirror + self.config.n_call_binding)
        # C3: NPC Call Push-Pull → Motor (방향성 있는 call 반응)
        # call_food_input L/R는 방향 정보를 가지므로 Sound Push-Pull과 동일 패턴 적용
        call_push = 10.0  # sound_food(8)보다 살짝 강하게 — NPC call 명확한 방향 신호
        call_pull = -5.0
        self._create_static_synapse(
            "call_food_l_to_motor_l", self.call_food_input_left, self.motor_left,
            call_push, sparsity=0.08)
        self._create_static_synapse(
            "call_food_r_to_motor_r", self.call_food_input_right, self.motor_right,
            call_push, sparsity=0.08)
        self._create_static_synapse(
            "call_food_l_to_motor_r_pull", self.call_food_input_left, self.motor_right,
            call_pull, sparsity=0.08)
        self._create_static_synapse(
            "call_food_r_to_motor_l_pull", self.call_food_input_right, self.motor_left,
            call_pull, sparsity=0.08)
        print(f"    C3: NPC Call Push-Pull: push={call_push}, pull={call_pull}")

        print(f"    Phase 17 Language Circuit: {n_lang_total} neurons")
        print(f"    Motor direct: {self.config.language_to_motor_weight} (disabled, except C3 Call Push-Pull)")
        print(f"    Total neurons now = {self.config.total_neurons:,}")

    def learn_call_binding(self, reward_context: bool):
        """
        Phase 17: Call Binding Hebbian 학습

        Wernicke_Food/Danger → Call_Binding DENSE 시냅스 가중치를 조정.
        NPC call을 듣고 음식을 찾으면 강한 학습, 그 외에는 약한 배경 학습.
        "이 소리 = 음식이 있다" 연합 형성.

        Args:
            reward_context: True = 소리 듣고 음식/위험 확인 (강한 학습), False = 배경
        """
        if not self.config.language_enabled:
            return None

        eta = self.config.call_binding_eta
        w_max = self.config.call_binding_w_max
        learning_factor = 1.0 if reward_context else 0.2

        binding_scale = max(0.1, self.last_call_binding_rate)

        # Wernicke_Food → Call_Binding
        n_pre_f = self.config.n_wernicke_food
        n_post = self.config.n_call_binding
        self.wernicke_food_to_binding_hebbian.vars["g"].pull_from_device()
        w_f = self.wernicke_food_to_binding_hebbian.vars["g"].view.copy()
        w_f = w_f.reshape(n_pre_f, n_post)
        w_f += eta * learning_factor * binding_scale
        w_f = np.clip(w_f, 0.0, w_max)
        self.wernicke_food_to_binding_hebbian.vars["g"].view[:] = w_f.flatten()
        self.wernicke_food_to_binding_hebbian.vars["g"].push_to_device()

        # Wernicke_Danger → Call_Binding
        n_pre_d = self.config.n_wernicke_danger
        self.wernicke_danger_to_binding_hebbian.vars["g"].pull_from_device()
        w_d = self.wernicke_danger_to_binding_hebbian.vars["g"].view.copy()
        w_d = w_d.reshape(n_pre_d, n_post)
        w_d += eta * learning_factor * binding_scale
        w_d = np.clip(w_d, 0.0, w_max)
        self.wernicke_danger_to_binding_hebbian.vars["g"].view[:] = w_d.flatten()
        self.wernicke_danger_to_binding_hebbian.vars["g"].push_to_device()

        return {
            "avg_w_food": float(np.mean(w_f)),
            "avg_w_danger": float(np.mean(w_d)),
            "learning_factor": learning_factor,
        }

    def _build_wm_expansion_circuit(self):
        """
        Phase 18: Working Memory Expansion (작업 기억 확장)

        시상-피질 루프 기반 WM 유지, 도파민 게이팅, 시간 버퍼, 목표 순서화.
        생물학적 메커니즘: MD thalamus ↔ PFC loop, dopamine gating, TRN modulation.
        LSTM-style 3게이트 사용하지 않음.
        """
        print("  Phase 18: Building WM Expansion Circuit...")

        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0, "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0}

        s_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        s_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === 18a: WM Thalamocortical Loop ===
        self.wm_thalamic = self.model.add_neuron_population(
            "wm_thalamic", self.config.n_wm_thalamic, "LIF", lif_params, lif_init)
        self.wm_update_gate = self.model.add_neuron_population(
            "wm_update_gate", self.config.n_wm_update_gate,
            sensory_lif_model, s_params, s_init)

        # === 18b: Temporal Buffer ===
        self.temporal_recent = self.model.add_neuron_population(
            "temporal_recent", self.config.n_temporal_recent, "LIF", lif_params, lif_init)
        self.temporal_prior = self.model.add_neuron_population(
            "temporal_prior", self.config.n_temporal_prior, "LIF", lif_params, lif_init)

        # === 18c: Goal Sequencer ===
        self.goal_pending = self.model.add_neuron_population(
            "goal_pending", self.config.n_goal_pending, "LIF", lif_params, lif_init)
        self.goal_switch = self.model.add_neuron_population(
            "goal_switch", self.config.n_goal_switch, "LIF", lif_params, lif_init)

        # === 18d: WM Context Learning ===
        self.wm_context_binding = self.model.add_neuron_population(
            "wm_context_binding", self.config.n_wm_context_binding, "LIF", lif_params, lif_init)

        # === 18e: WM Inhibitory Balance ===
        self.wm_inhibitory = self.model.add_neuron_population(
            "wm_inhibitory", self.config.n_wm_inhibitory, "LIF", lif_params, lif_init)

        print(f"    Populations: WM_Thalamic({self.config.n_wm_thalamic}) + "
              f"Gate({self.config.n_wm_update_gate}) + "
              f"Temporal({self.config.n_temporal_recent}+{self.config.n_temporal_prior}) + "
              f"GoalSeq({self.config.n_goal_pending}+{self.config.n_goal_switch}) + "
              f"Context({self.config.n_wm_context_binding}) + "
              f"Inhibitory({self.config.n_wm_inhibitory})")

        # ============================================================
        # 18a: WM Thalamocortical Loop (8 synapses)
        # ============================================================

        # WM → WM_Thalamic (cortex → thalamus)
        self._create_static_synapse(
            "wm_to_wm_thalamic", self.working_memory, self.wm_thalamic,
            self.config.wm_to_wm_thalamic_weight, sparsity=0.05)
        # WM_Thalamic → WM (thalamus → cortex, maintenance)
        self._create_static_synapse(
            "wm_thalamic_to_wm", self.wm_thalamic, self.working_memory,
            self.config.wm_thalamic_to_wm_weight, sparsity=0.03)
        # WM_Update_Gate → WM_Thalamic (inhibitory: gate breaks maintenance)
        self._create_static_synapse(
            "wm_gate_to_thalamic", self.wm_update_gate, self.wm_thalamic,
            self.config.wm_gate_to_thalamic_weight, sparsity=0.10)
        # Dopamine → WM_Update_Gate
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "dopamine_to_wm_gate", self.dopamine_neurons, self.wm_update_gate,
                self.config.dopamine_to_wm_gate_weight, sparsity=0.05)
        # ACC_Conflict → WM_Update_Gate
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "acc_conflict_to_wm_gate", self.acc_conflict, self.wm_update_gate,
                self.config.acc_conflict_to_wm_gate_weight, sparsity=0.05)
        # Assoc_Novelty → WM_Update_Gate
        if self.config.association_cortex_enabled:
            self._create_static_synapse(
                "novelty_to_wm_gate", self.assoc_novelty, self.wm_update_gate,
                self.config.novelty_to_wm_gate_weight, sparsity=0.05)
        # TRN → WM_Thalamic (thalamic gating consistency)
        if self.config.thalamus_enabled:
            self._create_static_synapse(
                "trn_to_wm_thalamic", self.trn, self.wm_thalamic,
                self.config.trn_to_wm_thalamic_weight, sparsity=0.05)
            # Arousal → WM_Thalamic (arousal supports maintenance)
            self._create_static_synapse(
                "arousal_to_wm_thalamic", self.arousal, self.wm_thalamic,
                self.config.arousal_to_wm_thalamic_weight, sparsity=0.05)

        print(f"    18a: Thalamocortical loop - WM↔Thalamic + Gate")

        # ============================================================
        # 18b: Temporal Buffer (10 synapses)
        # ============================================================

        # Temporal_Recent recurrent
        self._create_static_synapse(
            "temporal_recent_recurrent", self.temporal_recent, self.temporal_recent,
            self.config.temporal_recent_recurrent_weight, sparsity=0.08)
        # Temporal_Prior recurrent
        self._create_static_synapse(
            "temporal_prior_recurrent", self.temporal_prior, self.temporal_prior,
            self.config.temporal_prior_recurrent_weight, sparsity=0.05)
        # Temporal_Recent → Temporal_Prior (slow transfer)
        self._create_static_synapse(
            "temporal_recent_to_prior", self.temporal_recent, self.temporal_prior,
            self.config.temporal_recent_to_prior_weight, sparsity=0.05)
        # Wernicke → Temporal_Recent (language events)
        if self.config.language_enabled:
            self._create_static_synapse(
                "wernicke_food_to_temporal", self.wernicke_food, self.temporal_recent,
                5.0, sparsity=0.05)
            self._create_static_synapse(
                "wernicke_danger_to_temporal", self.wernicke_danger, self.temporal_recent,
                6.0, sparsity=0.05)
        # Assoc → Temporal_Recent (concept events)
        if self.config.association_cortex_enabled:
            self._create_static_synapse(
                "assoc_edible_to_temporal", self.assoc_edible, self.temporal_recent,
                4.0, sparsity=0.05)
            self._create_static_synapse(
                "assoc_threatening_to_temporal", self.assoc_threatening, self.temporal_recent,
                5.0, sparsity=0.05)
        # Fear → Temporal_Recent (pain events)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_temporal", self.fear_response, self.temporal_recent,
                6.0, sparsity=0.05)
        # STS_Congruence → Temporal_Recent (multimodal events)
        if self.config.multimodal_enabled:
            self._create_static_synapse(
                "sts_congruence_to_temporal", self.sts_congruence, self.temporal_recent,
                4.0, sparsity=0.05)
        # Temporal_Recent → WM
        self._create_static_synapse(
            "temporal_recent_to_wm", self.temporal_recent, self.working_memory,
            self.config.temporal_recent_to_wm_weight, sparsity=0.03)

        print(f"    18b: Temporal buffer - Recent({self.config.n_temporal_recent}) + Prior({self.config.n_temporal_prior})")

        # ============================================================
        # 18c: Goal Sequencer (12 synapses)
        # ============================================================

        # WM → Goal_Pending
        self._create_static_synapse(
            "wm_to_goal_pending", self.working_memory, self.goal_pending,
            self.config.wm_to_goal_pending_weight, sparsity=0.05)
        # Temporal_Recent → Goal_Pending
        self._create_static_synapse(
            "temporal_to_goal_pending", self.temporal_recent, self.goal_pending,
            4.0, sparsity=0.05)
        # Assoc → Goal_Pending
        if self.config.association_cortex_enabled:
            self._create_static_synapse(
                "assoc_edible_to_pending", self.assoc_edible, self.goal_pending,
                4.0, sparsity=0.05)
            self._create_static_synapse(
                "assoc_threatening_to_pending", self.assoc_threatening, self.goal_pending,
                4.0, sparsity=0.05)
        # Goal_Food/Safety → Goal_Pending (inhibitory: active goal suppresses pending)
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "goal_food_to_pending", self.goal_food, self.goal_pending,
                self.config.goal_to_pending_inhibit_weight, sparsity=0.08)
            self._create_static_synapse(
                "goal_safety_to_pending", self.goal_safety, self.goal_pending,
                self.config.goal_to_pending_inhibit_weight, sparsity=0.08)
        # ACC_Conflict → Goal_Switch
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "acc_conflict_to_goal_switch", self.acc_conflict, self.goal_switch,
                6.0, sparsity=0.08)
        # Assoc_Novelty → Goal_Switch
        if self.config.association_cortex_enabled:
            self._create_static_synapse(
                "novelty_to_goal_switch", self.assoc_novelty, self.goal_switch,
                5.0, sparsity=0.08)
        # Goal_Pending → Goal_Switch (pending enables switch)
        self._create_static_synapse(
            "pending_to_goal_switch", self.goal_pending, self.goal_switch,
            4.0, sparsity=0.05)
        # Goal_Switch self-inhibition (burst only)
        self._create_static_synapse(
            "goal_switch_self_inhibit", self.goal_switch, self.goal_switch,
            self.config.goal_switch_self_inhibit_weight, sparsity=0.15)
        # Goal_Switch → Goal_Food/Safety (inhibitory: disrupts current goals)
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "goal_switch_to_food", self.goal_switch, self.goal_food,
                self.config.goal_switch_to_goal_inhibit_weight, sparsity=0.05)
            self._create_static_synapse(
                "goal_switch_to_safety", self.goal_switch, self.goal_safety,
                self.config.goal_switch_to_goal_inhibit_weight, sparsity=0.05)

        print(f"    18c: Goal sequencer - Pending({self.config.n_goal_pending}) + Switch({self.config.n_goal_switch})")

        # ============================================================
        # 18d: WM Context Learning (6 synapses, 1 Hebbian DENSE)
        # ============================================================

        from pygenn import init_weight_update, init_postsynaptic
        # Hebbian DENSE: Temporal_Recent → WM_Context_Binding
        self.temporal_to_context_hebbian = self.model.add_synapse_population(
            "temporal_to_wm_context_hebb", "DENSE",
            self.temporal_recent, self.wm_context_binding,
            init_weight_update("StaticPulse", {},
                               {"g": self.config.wm_context_binding_init_weight}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))
        # WM_Context recurrent
        self._create_static_synapse(
            "wm_context_recurrent", self.wm_context_binding, self.wm_context_binding,
            5.0, sparsity=0.05)
        # WM_Context → WM
        self._create_static_synapse(
            "wm_context_to_wm", self.wm_context_binding, self.working_memory,
            self.config.wm_context_to_wm_weight, sparsity=0.03)
        # WM_Context → Goal_Pending
        self._create_static_synapse(
            "wm_context_to_pending", self.wm_context_binding, self.goal_pending,
            self.config.wm_context_to_pending_weight, sparsity=0.03)
        # Dopamine → WM_Context
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "dopamine_to_wm_context", self.dopamine_neurons, self.wm_context_binding,
                4.0, sparsity=0.05)
        # Hunger → WM_Context (drive context)
        self._create_static_synapse(
            "hunger_to_wm_context", self.hunger_drive, self.wm_context_binding,
            3.0, sparsity=0.03)

        print(f"    18d: WM Context (Hebbian DENSE, eta={self.config.wm_context_binding_eta}, "
              f"w_max={self.config.wm_context_binding_w_max})")

        # ============================================================
        # 18e: WM Inhibitory Balance (6 synapses)
        # ============================================================

        # WM → WM_Inhibitory (excitatory)
        self._create_static_synapse(
            "wm_to_wm_inhibitory", self.working_memory, self.wm_inhibitory,
            self.config.wm_to_inhibitory_weight, sparsity=0.08)
        # WM_Thalamic → WM_Inhibitory
        self._create_static_synapse(
            "wm_thalamic_to_inhibitory", self.wm_thalamic, self.wm_inhibitory,
            self.config.wm_thalamic_to_inhibitory_weight, sparsity=0.05)
        # WM_Inhibitory → WM (negative feedback)
        self._create_static_synapse(
            "inhibitory_to_wm", self.wm_inhibitory, self.working_memory,
            self.config.inhibitory_to_wm_weight, sparsity=0.08)
        # WM_Inhibitory → WM_Thalamic
        self._create_static_synapse(
            "inhibitory_to_thalamic", self.wm_inhibitory, self.wm_thalamic,
            self.config.inhibitory_to_thalamic_weight, sparsity=0.05)
        # WM_Inhibitory → Temporal_Recent
        self._create_static_synapse(
            "inhibitory_to_temporal", self.wm_inhibitory, self.temporal_recent,
            self.config.inhibitory_to_temporal_weight, sparsity=0.03)
        # WM_Inhibitory → Goal_Pending
        self._create_static_synapse(
            "inhibitory_to_pending", self.wm_inhibitory, self.goal_pending,
            self.config.inhibitory_to_pending_weight, sparsity=0.03)

        print(f"    18e: WM Inhibitory balance ({self.config.n_wm_inhibitory} neurons)")
        print(f"    Motor direct: {self.config.wm_expansion_to_motor_weight} (disabled)")

    def learn_wm_context(self, reward_context: bool):
        """
        Phase 18d: WM Context Binding Hebbian 학습

        Temporal_Recent → WM_Context_Binding DENSE 시냅스 가중치를 조정.
        음식 먹기/pain 시 강한 학습, 그 외에는 약한 배경 학습.
        "이 시간 문맥에서 좋은/나쁜 일이 발생" 연합 형성.
        """
        if not self.config.wm_expansion_enabled:
            return None

        eta = self.config.wm_context_binding_eta
        w_max = self.config.wm_context_binding_w_max
        learning_factor = 1.0 if reward_context else 0.2

        binding_scale = max(0.1, self.last_wm_context_binding_rate)

        n_pre = self.config.n_temporal_recent
        n_post = self.config.n_wm_context_binding
        self.temporal_to_context_hebbian.vars["g"].pull_from_device()
        w = self.temporal_to_context_hebbian.vars["g"].view.copy()
        w = w.reshape(n_pre, n_post)
        w += eta * learning_factor * binding_scale
        w = np.clip(w, 0.0, w_max)
        self.temporal_to_context_hebbian.vars["g"].view[:] = w.flatten()
        self.temporal_to_context_hebbian.vars["g"].push_to_device()

        return {
            "avg_w": float(np.mean(w)),
            "max_w": float(np.max(w)),
            "learning_factor": learning_factor,
        }

    def _build_metacognition_circuit(self):
        """
        Phase 19: Metacognition (메타인지)

        확신/불확실성 경쟁 회로. 전방 섬엽(확신), dACC(불확실성), mPFC(평가 게이트),
        청반(불확실성→각성). 행동 조절: 확신→목표 유지, 불확실성→탐색 증가.
        """
        print("  Phase 19: Building Metacognition Circuit...")

        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0, "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0}

        s_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        s_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # === Populations ===
        self.meta_confidence = self.model.add_neuron_population(
            "meta_confidence", self.config.n_meta_confidence, "LIF", lif_params, lif_init)
        self.meta_uncertainty = self.model.add_neuron_population(
            "meta_uncertainty", self.config.n_meta_uncertainty, "LIF", lif_params, lif_init)
        self.meta_evaluate = self.model.add_neuron_population(
            "meta_evaluate", self.config.n_meta_evaluate,
            sensory_lif_model, s_params, s_init)
        self.meta_arousal_mod = self.model.add_neuron_population(
            "meta_arousal_mod", self.config.n_meta_arousal_mod, "LIF", lif_params, lif_init)
        self.meta_inhibitory_pop = self.model.add_neuron_population(
            "meta_inhibitory_pop", self.config.n_meta_inhibitory, "LIF", lif_params, lif_init)

        print(f"    Populations: Confidence({self.config.n_meta_confidence}) + "
              f"Uncertainty({self.config.n_meta_uncertainty}) + "
              f"Evaluate({self.config.n_meta_evaluate}) + "
              f"ArousalMod({self.config.n_meta_arousal_mod}) + "
              f"Inhibitory({self.config.n_meta_inhibitory})")

        # ============================================================
        # 19a: Confidence Inputs (6 synapses)
        # ============================================================
        if self.config.association_cortex_enabled:
            self._create_static_synapse(
                "assoc_valence_to_meta_conf", self.assoc_valence, self.meta_confidence,
                self.config.assoc_valence_to_confidence_weight, sparsity=0.05)
        if self.config.multimodal_enabled:
            self._create_static_synapse(
                "sts_congruence_to_meta_conf", self.sts_congruence, self.meta_confidence,
                self.config.sts_congruence_to_confidence_weight, sparsity=0.05)
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "goal_food_to_meta_conf", self.goal_food, self.meta_confidence,
                self.config.goal_food_to_confidence_weight, sparsity=0.05)
            self._create_static_synapse(
                "goal_safety_to_meta_conf", self.goal_safety, self.meta_confidence,
                self.config.goal_safety_to_confidence_weight, sparsity=0.05)
        if self.config.wm_expansion_enabled:
            self._create_static_synapse(
                "wm_context_to_meta_conf", self.wm_context_binding, self.meta_confidence,
                self.config.wm_context_to_confidence_weight, sparsity=0.05)
        # Confidence recurrent
        self._create_static_synapse(
            "meta_conf_recurrent", self.meta_confidence, self.meta_confidence,
            self.config.meta_confidence_recurrent_weight, sparsity=0.08)

        print(f"    19a: Confidence inputs (valence, congruence, goals, context)")

        # ============================================================
        # 19b: Uncertainty Inputs (6 synapses)
        # ============================================================
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "acc_conflict_to_meta_uncert", self.acc_conflict, self.meta_uncertainty,
                self.config.acc_conflict_to_uncertainty_weight, sparsity=0.05)
        if self.config.cerebellum_enabled:
            self._create_static_synapse(
                "error_signal_to_meta_uncert", self.error_signal, self.meta_uncertainty,
                self.config.error_signal_to_uncertainty_weight, sparsity=0.05)
        if self.config.association_cortex_enabled:
            self._create_static_synapse(
                "assoc_novelty_to_meta_uncert", self.assoc_novelty, self.meta_uncertainty,
                self.config.assoc_novelty_to_uncertainty_weight, sparsity=0.05)
        if self.config.social_brain_enabled:
            self._create_static_synapse(
                "tom_surprise_to_meta_uncert", self.tom_surprise, self.meta_uncertainty,
                self.config.tom_surprise_to_uncertainty_weight, sparsity=0.05)
        if self.config.multimodal_enabled:
            self._create_static_synapse(
                "sts_mismatch_to_meta_uncert", self.sts_mismatch, self.meta_uncertainty,
                self.config.sts_mismatch_to_uncertainty_weight, sparsity=0.05)
        # Uncertainty recurrent
        self._create_static_synapse(
            "meta_uncert_recurrent", self.meta_uncertainty, self.meta_uncertainty,
            self.config.meta_uncertainty_recurrent_weight, sparsity=0.08)

        print(f"    19b: Uncertainty inputs (conflict, error, novelty, surprise, mismatch)")

        # ============================================================
        # 19c: WTA (2 synapses)
        # ============================================================
        self._create_static_synapse(
            "meta_conf_to_uncert_wta", self.meta_confidence, self.meta_uncertainty,
            self.config.meta_confidence_uncertainty_wta_weight, sparsity=0.10)
        self._create_static_synapse(
            "meta_uncert_to_conf_wta", self.meta_uncertainty, self.meta_confidence,
            self.config.meta_confidence_uncertainty_wta_weight, sparsity=0.10)

        print(f"    19c: Confidence↔Uncertainty WTA ({self.config.meta_confidence_uncertainty_wta_weight})")

        # ============================================================
        # 19d: Outputs (8 synapses + 1 conditional)
        # ============================================================
        # Confidence outputs (stabilize)
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "meta_conf_to_goal_food", self.meta_confidence, self.goal_food,
                self.config.meta_confidence_to_goal_food_weight, sparsity=0.05)
            self._create_static_synapse(
                "meta_conf_to_goal_safety", self.meta_confidence, self.goal_safety,
                self.config.meta_confidence_to_goal_safety_weight, sparsity=0.05)
        if self.config.wm_expansion_enabled:
            self._create_static_synapse(
                "meta_conf_to_goal_switch", self.meta_confidence, self.goal_switch,
                self.config.meta_confidence_to_goal_switch_weight, sparsity=0.05)
            self._create_static_synapse(
                "meta_conf_to_wm_thalamic", self.meta_confidence, self.wm_thalamic,
                self.config.meta_confidence_to_wm_thalamic_weight, sparsity=0.05)

        # Evaluate outputs (exploration)
        if self.config.wm_expansion_enabled:
            self._create_static_synapse(
                "meta_eval_to_goal_switch", self.meta_evaluate, self.goal_switch,
                self.config.meta_evaluate_to_goal_switch_weight, sparsity=0.08)
        self._create_static_synapse(
            "meta_eval_to_arousal_mod", self.meta_evaluate, self.meta_arousal_mod,
            self.config.meta_evaluate_to_arousal_mod_weight, sparsity=0.08)
        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "meta_eval_to_inhibitory_ctrl", self.meta_evaluate, self.inhibitory_control,
                self.config.meta_evaluate_to_inhibitory_ctrl_weight, sparsity=0.05)

        # Arousal modulation
        if self.config.thalamus_enabled:
            self._create_static_synapse(
                "meta_arousal_mod_to_arousal", self.meta_arousal_mod, self.arousal,
                self.config.meta_arousal_mod_to_arousal_weight, sparsity=0.05)
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "meta_arousal_mod_to_dopamine", self.meta_arousal_mod, self.dopamine_neurons,
                self.config.meta_arousal_mod_to_dopamine_weight, sparsity=0.05)

        print(f"    19d: Outputs (conf→goals, eval→switch/arousal, arousal_mod→arousal/DA)")

        # ============================================================
        # 19e: Inhibitory Balance (5 synapses)
        # ============================================================
        self._create_static_synapse(
            "meta_conf_to_meta_inhib", self.meta_confidence, self.meta_inhibitory_pop,
            self.config.meta_conf_to_inhibitory_weight, sparsity=0.08)
        self._create_static_synapse(
            "meta_uncert_to_meta_inhib", self.meta_uncertainty, self.meta_inhibitory_pop,
            self.config.meta_uncert_to_inhibitory_weight, sparsity=0.08)
        self._create_static_synapse(
            "meta_inhib_to_conf", self.meta_inhibitory_pop, self.meta_confidence,
            self.config.meta_inhibitory_to_conf_weight, sparsity=0.08)
        self._create_static_synapse(
            "meta_inhib_to_uncert", self.meta_inhibitory_pop, self.meta_uncertainty,
            self.config.meta_inhibitory_to_uncert_weight, sparsity=0.08)
        self._create_static_synapse(
            "meta_inhib_to_eval", self.meta_inhibitory_pop, self.meta_evaluate,
            self.config.meta_inhibitory_to_eval_weight, sparsity=0.05)

        print(f"    19e: Inhibitory balance ({self.config.n_meta_inhibitory} neurons)")

        # ============================================================
        # 19f: Hebbian DENSE (Valence → Confidence)
        # ============================================================
        if self.config.association_cortex_enabled:
            from pygenn import init_weight_update, init_postsynaptic
            self.valence_to_confidence_hebbian = self.model.add_synapse_population(
                "valence_to_confidence_hebb", "DENSE",
                self.assoc_valence, self.meta_confidence,
                init_weight_update("StaticPulse", {},
                                   {"g": self.config.meta_confidence_binding_init_weight}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}))

            print(f"    19f: Hebbian DENSE (Valence→Confidence, eta={self.config.meta_confidence_binding_eta}, "
                  f"w_max={self.config.meta_confidence_binding_w_max})")

        print(f"    Motor direct: {self.config.metacognition_to_motor_weight} (disabled)")
        print(f"    Phase 19 Metacognition: {self.config.n_meta_confidence + self.config.n_meta_uncertainty + self.config.n_meta_evaluate + self.config.n_meta_arousal_mod + self.config.n_meta_inhibitory} neurons")

    def learn_metacognitive_confidence(self, reward_context: bool):
        """
        Phase 19f: Valence → Meta_Confidence Hebbian 학습

        보상 시(food eaten, pain) 강한 학습, 배경 시 약한 감쇠.
        어떤 valence 패턴이 성공적 결과를 예측하는지 학습.
        """
        if not self.config.metacognition_enabled or not self.config.association_cortex_enabled:
            return None

        eta = self.config.meta_confidence_binding_eta
        w_max = self.config.meta_confidence_binding_w_max
        learning_factor = 1.0 if reward_context else 0.15

        confidence_scale = max(0.1, self.last_meta_confidence_rate)

        n_pre = self.config.n_assoc_valence
        n_post = self.config.n_meta_confidence
        self.valence_to_confidence_hebbian.vars["g"].pull_from_device()
        w = self.valence_to_confidence_hebbian.vars["g"].view.copy()
        w = w.reshape(n_pre, n_post)
        w += eta * learning_factor * confidence_scale
        w = np.clip(w, 0.0, w_max)
        self.valence_to_confidence_hebbian.vars["g"].view[:] = w.flatten()
        self.valence_to_confidence_hebbian.vars["g"].push_to_device()

        return {
            "avg_w": float(np.mean(w)),
            "max_w": float(np.max(w)),
            "learning_factor": learning_factor,
        }

    def _build_self_model_circuit(self):
        """
        Phase 20: Self-Model (자기 모델)

        생물학적 근거:
        - 섬엽(Insula): 내수용감각 통합 → self_body
        - 소뇌 순행 모델: Efference Copy → self_efference, self_predict
        - 각회(Angular Gyrus)/TPJ: 행위주체감 → self_agency
        - DMN(Default Mode Network): 자기 서사 → self_narrative

        "나는 누구인가" - 메타인지의 자연스러운 확장
        Phase 19: "내가 뭘 모르는지 안다" → Phase 20: "나는 존재한다"
        """
        print("  Phase 20: Self-Model (자기 모델)")

        # SensoryLIF 파라미터 (I_input 필요한 인구)
        s_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "TauRefrac": self.config.tau_refrac
        }
        s_init = {"V": self.config.v_rest, "RefracTime": 0.0, "I_input": 0.0}

        # Standard LIF 파라미터
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0, "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0}

        # ============================================================
        # 20-1: Populations (6개, 440 neurons)
        # ============================================================

        # Self_Body (Insular Cortex) - SensoryLIF (I_input: energy/hunger/satiety)
        self.self_body = self.model.add_neuron_population(
            "self_body", self.config.n_self_body,
            sensory_lif_model, s_params, s_init)

        # Self_Efference (Cerebellum efference copy) - LIF
        self.self_efference = self.model.add_neuron_population(
            "self_efference", self.config.n_self_efference,
            "LIF", lif_params, lif_init)

        # Self_Predict (Cerebellar forward model) - SensoryLIF (I_input: efference + food_eye)
        self.self_predict = self.model.add_neuron_population(
            "self_predict", self.config.n_self_predict,
            sensory_lif_model, s_params, s_init)

        # Self_Agency (Angular Gyrus / IPL) - LIF
        self.self_agency = self.model.add_neuron_population(
            "self_agency", self.config.n_self_agency,
            "LIF", lif_params, lif_init)

        # Self_Narrative (DMN / mPFC) - LIF
        self.self_narrative = self.model.add_neuron_population(
            "self_narrative", self.config.n_self_narrative,
            "LIF", lif_params, lif_init)

        # Self_Inhibitory (Local interneurons) - LIF
        self.self_inhibitory_sm = self.model.add_neuron_population(
            "self_inhibitory_sm", self.config.n_self_inhibitory,
            "LIF", lif_params, lif_init)

        print(f"    20-1: Populations ({self.config.n_self_body}+{self.config.n_self_efference}+"
              f"{self.config.n_self_predict}+{self.config.n_self_agency}+"
              f"{self.config.n_self_narrative}+{self.config.n_self_inhibitory} = "
              f"{self.config.n_self_body + self.config.n_self_efference + self.config.n_self_predict + self.config.n_self_agency + self.config.n_self_narrative + self.config.n_self_inhibitory} neurons)")

        # ============================================================
        # 20a: Self_Body inputs (5 synapses) - 내수용감각 통합
        # ============================================================
        self._create_static_synapse(
            "hunger_to_self_body", self.hunger_drive, self.self_body,
            self.config.hunger_to_self_body_weight, sparsity=0.05)
        if self.config.amygdala_enabled:
            self._create_static_synapse(
                "fear_to_self_body", self.fear_response, self.self_body,
                self.config.fear_to_self_body_weight, sparsity=0.05)
        if self.config.metacognition_enabled:
            self._create_static_synapse(
                "meta_conf_to_self_body", self.meta_confidence, self.self_body,
                self.config.meta_conf_to_self_body_weight, sparsity=0.05)
            self._create_static_synapse(
                "meta_uncert_to_self_body", self.meta_uncertainty, self.self_body,
                self.config.meta_uncert_to_self_body_weight, sparsity=0.05)
        if self.config.basal_ganglia_enabled:
            self._create_static_synapse(
                "dopamine_to_self_body", self.dopamine_neurons, self.self_body,
                self.config.dopamine_to_self_body_weight, sparsity=0.05)

        print(f"    20a: Self_Body inputs (hunger/fear/meta_conf/meta_uncert/dopamine)")

        # ============================================================
        # 20b: Self_Efference inputs (2 synapses) - 운동 명령 복사
        # ============================================================
        self._create_static_synapse(
            "motor_l_to_efference", self.motor_left, self.self_efference,
            self.config.motor_to_efference_weight, sparsity=0.05)
        self._create_static_synapse(
            "motor_r_to_efference", self.motor_right, self.self_efference,
            self.config.motor_to_efference_weight, sparsity=0.05)

        print(f"    20b: Self_Efference inputs (motor_left/motor_right)")

        # ============================================================
        # 20d: Self_Agency inputs (3 synapses) - 행위주체감
        # ============================================================
        self._create_static_synapse(
            "efference_to_agency", self.self_efference, self.self_agency,
            self.config.efference_to_agency_weight, sparsity=0.05)
        self._create_static_synapse(
            "predict_to_agency", self.self_predict, self.self_agency,
            self.config.predict_to_agency_weight, sparsity=0.05)
        if self.config.hippocampus_enabled:
            self._create_static_synapse(
                "food_mem_to_agency", self.food_memory_left, self.self_agency,
                self.config.food_memory_to_agency_weight, sparsity=0.05)

        print(f"    20d: Self_Agency inputs (efference/predict/food_memory)")

        # ============================================================
        # 20e: Self_Narrative inputs (4 synapses) - 자기 서사
        # ============================================================
        self._create_static_synapse(
            "body_to_narrative", self.self_body, self.self_narrative,
            self.config.body_to_narrative_weight, sparsity=0.05)
        self._create_static_synapse(
            "agency_to_narrative", self.self_agency, self.self_narrative,
            self.config.agency_to_narrative_weight, sparsity=0.05)
        if self.config.wm_expansion_enabled:
            self._create_static_synapse(
                "wm_ctx_to_narrative", self.wm_context_binding, self.self_narrative,
                self.config.wm_context_to_narrative_weight, sparsity=0.05)
        # Recurrent (self-referential maintenance)
        self._create_static_synapse(
            "narrative_recurrent", self.self_narrative, self.self_narrative,
            self.config.narrative_recurrent_weight, sparsity=0.08)

        print(f"    20e: Self_Narrative inputs (body/agency/wm_context/recurrent)")

        # ============================================================
        # 20f: Outputs (6 synapses, ALL ≤1.5)
        # ============================================================
        if self.config.metacognition_enabled:
            self._create_static_synapse(
                "self_body_to_meta_conf", self.self_body, self.meta_confidence,
                self.config.self_body_to_meta_conf_weight, sparsity=0.05)
            self._create_static_synapse(
                "self_body_to_meta_uncert", self.self_body, self.meta_uncertainty,
                self.config.self_body_to_meta_uncert_weight, sparsity=0.05)

        if self.config.prefrontal_enabled:
            self._create_static_synapse(
                "self_agency_to_goal_food", self.self_agency, self.goal_food,
                self.config.self_agency_to_goal_food_weight, sparsity=0.05)
        if self.config.wm_expansion_enabled:
            self._create_static_synapse(
                "self_agency_to_goal_switch", self.self_agency, self.goal_switch,
                self.config.self_agency_to_goal_switch_weight, sparsity=0.05)
            self._create_static_synapse(
                "self_narrative_to_wm", self.self_narrative, self.working_memory,
                self.config.self_narrative_to_wm_weight, sparsity=0.05)
        if self.config.cerebellum_enabled:
            self._create_static_synapse(
                "self_predict_to_error", self.self_predict, self.error_signal,
                self.config.self_predict_to_error_weight, sparsity=0.05)

        print(f"    20f: Outputs (body→meta, agency→goals, narrative→WM, predict→error) ALL ≤1.5")

        # ============================================================
        # 20g: Inhibitory balance (6 synapses)
        # ============================================================
        self._create_static_synapse(
            "self_body_to_sm_inhib", self.self_body, self.self_inhibitory_sm,
            self.config.self_to_inhibitory_weight, sparsity=0.05)
        self._create_static_synapse(
            "self_eff_to_sm_inhib", self.self_efference, self.self_inhibitory_sm,
            self.config.self_to_inhibitory_weight, sparsity=0.05)
        self._create_static_synapse(
            "self_agency_to_sm_inhib", self.self_agency, self.self_inhibitory_sm,
            self.config.self_to_inhibitory_weight, sparsity=0.05)
        self._create_static_synapse(
            "sm_inhib_to_body", self.self_inhibitory_sm, self.self_body,
            self.config.self_inhibitory_to_body_weight, sparsity=0.05)
        self._create_static_synapse(
            "sm_inhib_to_agency", self.self_inhibitory_sm, self.self_agency,
            self.config.self_inhibitory_to_agency_weight, sparsity=0.05)
        self._create_static_synapse(
            "sm_inhib_to_narrative", self.self_inhibitory_sm, self.self_narrative,
            self.config.self_inhibitory_to_narrative_weight, sparsity=0.05)

        print(f"    20g: Inhibitory balance ({self.config.n_self_inhibitory} neurons)")

        # ============================================================
        # 20h: Hebbian DENSE (Body → Narrative)
        # ============================================================
        from pygenn import init_weight_update, init_postsynaptic
        self.body_to_narrative_hebbian = self.model.add_synapse_population(
            "body_to_narrative_hebb", "DENSE",
            self.self_body, self.self_narrative,
            init_weight_update("StaticPulse", {},
                               {"g": self.config.self_narrative_binding_init_weight}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))

        print(f"    20h: Hebbian DENSE (Body→Narrative, eta={self.config.self_narrative_binding_eta}, "
              f"w_max={self.config.self_narrative_binding_w_max})")

        # ============================================================
        # Phase L15: Agency → Narrative DENSE Hebbian
        # ============================================================
        if self.config.narrative_self_enabled:
            self.agency_to_narrative_hebbian = self.model.add_synapse_population(
                "agency_to_narrative_hebb", "DENSE",
                self.self_agency, self.self_narrative,
                init_weight_update("StaticPulse", {},
                                   {"g": self.config.agency_to_narrative_init_w}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}))
            print(f"    L15: Hebbian DENSE (Agency→Narrative, eta={self.config.agency_to_narrative_eta}, "
                  f"w_max={self.config.agency_to_narrative_w_max})")

        print(f"    Motor direct: {self.config.self_model_to_motor_weight} (disabled)")
        total_neurons = (self.config.n_self_body + self.config.n_self_efference +
                        self.config.n_self_predict + self.config.n_self_agency +
                        self.config.n_self_narrative + self.config.n_self_inhibitory)
        print(f"    Phase 20 Self-Model: {total_neurons} neurons")

        # ============================================================
        # Phase L14: Agency Detection (Forward Model + Agency PE)
        # ============================================================
        if self.config.agency_detection_enabled:
            self._build_agency_detection_circuit()

    def _build_agency_detection_circuit(self):
        """Phase L14: Agency Detection — Forward Model Learning + Agency PE"""
        from pygenn import init_weight_update, init_postsynaptic

        lif_params = {"C": 1.0, "TauM": 20.0, "Vrest": -60.0,
                      "Vreset": -60.0, "Vthresh": -50.0, "Ioffset": 0.0,
                      "TauRefrac": 2.0}
        lif_init = {"V": -60.0, "RefracTime": 0.0}

        # --- Agency_PE population (50 LIF) ---
        self.agency_pe = self.model.add_neuron_population(
            "agency_pe", self.config.n_agency_pe,
            "LIF", lif_params, lif_init)

        # --- Forward Model Hebbian: self_efference → self_predict (DENSE) ---
        self.efference_to_predict_hebbian = self.model.add_synapse_population(
            "efference_to_predict_hebb", "DENSE",
            self.self_efference, self.self_predict,
            init_weight_update("StaticPulse", {},
                               {"g": self.config.agency_forward_model_init_w}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))

        # --- Agency PE inputs ---
        # V1_Food → Agency_PE (excitatory: actual sensory)
        if hasattr(self, 'v1_food_left'):
            self._create_static_synapse(
                "v1_food_l_to_ape", self.v1_food_left, self.agency_pe,
                self.config.v1_food_to_agency_pe_weight, sparsity=0.08)
            self._create_static_synapse(
                "v1_food_r_to_ape", self.v1_food_right, self.agency_pe,
                self.config.v1_food_to_agency_pe_weight, sparsity=0.08)

        # Self_Predict → Agency_PE (inhibitory: predicted sensory cancels)
        self._create_static_synapse(
            "predict_to_ape", self.self_predict, self.agency_pe,
            self.config.predict_to_agency_pe_weight, sparsity=0.1)

        # --- Agency PE outputs ---
        # Agency_PE → Self_Agency (inhibitory: high PE = low agency)
        self._create_static_synapse(
            "ape_to_agency", self.agency_pe, self.self_agency,
            self.config.agency_pe_to_agency_weight, sparsity=0.1)

        # Agency_PE → Meta_Uncertainty (excitatory: high PE = uncertain)
        if hasattr(self, 'meta_uncertainty'):
            self._create_static_synapse(
                "ape_to_uncert", self.agency_pe, self.meta_uncertainty,
                self.config.agency_pe_to_uncertainty_weight, sparsity=0.05)

        # Self_Agency → Meta_Confidence (excitatory: high agency = confident)
        if hasattr(self, 'meta_confidence'):
            self._create_static_synapse(
                "agency_to_conf", self.self_agency, self.meta_confidence,
                self.config.agency_to_confidence_weight, sparsity=0.05)

        # --- Inhibitory balance for Agency_PE ---
        self._create_static_synapse(
            "ape_to_sm_inhib", self.agency_pe, self.self_inhibitory_sm,
            self.config.agency_pe_to_inhibitory_weight, sparsity=0.05)
        self._create_static_synapse(
            "sm_inhib_to_ape", self.self_inhibitory_sm, self.agency_pe,
            self.config.agency_inhibitory_to_pe_weight, sparsity=0.05)

        # Init cached rates
        self.last_agency_pe_rate = 0.0

        n_syn_static = 7 if hasattr(self, 'v1_food_left') else 5
        n_syn_static += 2  # meta synapses
        print(f"\n  === Phase L14: Agency Detection ===")
        print(f"    Agency_PE: {self.config.n_agency_pe} neurons (LIF)")
        print(f"    Forward Model: self_efference→self_predict DENSE Hebbian "
              f"(eta={self.config.agency_forward_model_eta}, w_max={self.config.agency_forward_model_w_max})")
        print(f"    V1_Food→Agency_PE: {self.config.v1_food_to_agency_pe_weight} (actual sensory)")
        print(f"    Self_Predict→Agency_PE: {self.config.predict_to_agency_pe_weight} (predicted, cancels)")
        print(f"    Agency_PE→Self_Agency: {self.config.agency_pe_to_agency_weight} (high PE = low agency)")
        print(f"    Motor direct: 0.0 (disabled)")

    def _normalize_weight_budget(self, w_2d, budget_per_post, w_max):
        """Per-post neuron incoming weight budget normalization (heterosynaptic).

        강화된 시냅스가 있으면 같은 post 뉴런의 다른 시냅스가 약해짐.
        Lesson #40: predictive plasticity에서 검증된 패턴.
        """
        n_pre, n_post = w_2d.shape
        for j in range(n_post):
            col_sum = np.sum(np.maximum(w_2d[:, j], 0.0))
            if col_sum > budget_per_post:
                w_2d[:, j] *= budget_per_post / col_sum
        np.clip(w_2d, 0.0, w_max, out=w_2d)
        return w_2d

    def learn_forward_model(self, reward_context: bool):
        """
        Phase L14: Forward Model Hebbian 학습 (self_efference → self_predict)

        운동 명령→감각 예측 매핑 학습. 보상 시 강한 학습, 배경 시 약한 학습.
        + Per-post weight budget normalization (heterosynaptic stabilization)
        """
        if not self.config.agency_detection_enabled or not self.config.self_model_enabled:
            return None

        eta = self.config.agency_forward_model_eta
        w_max = self.config.agency_forward_model_w_max
        learning_factor = 1.0 if reward_context else 0.1

        predict_scale = max(0.1, self.last_self_predict_rate)

        n_pre = self.config.n_self_efference
        n_post = self.config.n_self_predict
        self.efference_to_predict_hebbian.vars["g"].pull_from_device()
        w = self.efference_to_predict_hebbian.vars["g"].view.copy()
        w = w.reshape(n_pre, n_post)
        w += eta * learning_factor * predict_scale
        # Heterosynaptic budget: prevent uniform saturation
        w = self._normalize_weight_budget(w, w_max * n_pre * 0.3, w_max)
        self.efference_to_predict_hebbian.vars["g"].view[:] = w.flatten()
        self.efference_to_predict_hebbian.vars["g"].push_to_device()

        return {
            "avg_w": float(np.mean(w)),
            "max_w": float(np.max(w)),
            "learning_factor": learning_factor,
        }

    def learn_self_narrative(self, reward_context: bool):
        """
        Phase 20h + L15: Body → Self_Narrative Hebbian 학습

        L15 Agency-Gated: 자기 원인(high agency)일수록 강한 학습,
        신체 상태 변화(salience)가 클수록 강한 학습.
        Damasio (2010): 자기 서사는 자기 원인 경험에서 더 강하게 형성.
        """
        if not self.config.self_model_enabled:
            return None

        eta = self.config.self_narrative_binding_eta
        w_max = self.config.self_narrative_binding_w_max
        learning_factor = 1.0 if reward_context else 0.15

        narrative_scale = max(0.1, self.last_self_narrative_rate)

        # Phase L15: Agency gate — high agency = stronger learning
        agency_gate = 1.0
        if self.config.narrative_self_enabled:
            baseline = self.config.narrative_agency_gate_baseline
            agency_gate = max(0.3, min(2.0, self.last_self_agency_rate / max(0.01, baseline)))

        # Phase L15: Salience gate — body state change = stronger learning
        salience_gate = 1.0
        if self.config.narrative_self_enabled:
            delta_body = abs(self.last_self_body_rate - self.prev_self_body_rate)
            salience_gate = 1.0 + min(2.0, delta_body * self.config.narrative_body_change_scale)

        n_pre = self.config.n_self_body
        n_post = self.config.n_self_narrative
        self.body_to_narrative_hebbian.vars["g"].pull_from_device()
        w = self.body_to_narrative_hebbian.vars["g"].view.copy()
        w = w.reshape(n_pre, n_post)
        w += eta * learning_factor * narrative_scale * agency_gate * salience_gate
        # Heterosynaptic budget: prevent uniform saturation at w_max
        w = self._normalize_weight_budget(w, w_max * n_pre * 0.3, w_max)
        self.body_to_narrative_hebbian.vars["g"].view[:] = w.flatten()
        self.body_to_narrative_hebbian.vars["g"].push_to_device()

        return {
            "avg_w": float(np.mean(w)),
            "max_w": float(np.max(w)),
            "learning_factor": learning_factor,
            "agency_gate": agency_gate,
            "salience_gate": salience_gate,
        }

    def learn_agency_narrative(self, reward_context: bool):
        """
        Phase L15: Agency → Self_Narrative DENSE Hebbian 학습

        자기 주체성(agency) 패턴을 서사(narrative)에 연결.
        높은 agency일 때 더 강하게 학습 → 자기 원인 행동의 기억 강화.
        """
        if not self.config.self_model_enabled or not self.config.narrative_self_enabled:
            return None
        if not hasattr(self, 'agency_to_narrative_hebbian'):
            return None

        eta = self.config.agency_to_narrative_eta
        w_max = self.config.agency_to_narrative_w_max
        learning_factor = 1.0 if reward_context else 0.15

        # Agency-modulated: stronger when agency is high
        agency_mod = max(0.1, self.last_self_agency_rate)

        n_pre = self.config.n_self_agency
        n_post = self.config.n_self_narrative
        self.agency_to_narrative_hebbian.vars["g"].pull_from_device()
        w = self.agency_to_narrative_hebbian.vars["g"].view.copy()
        w = w.reshape(n_pre, n_post)
        w += eta * learning_factor * agency_mod
        # Heterosynaptic budget
        w = self._normalize_weight_budget(w, w_max * n_pre * 0.3, w_max)
        self.agency_to_narrative_hebbian.vars["g"].view[:] = w.flatten()
        self.agency_to_narrative_hebbian.vars["g"].push_to_device()

        return {
            "avg_w": float(np.mean(w)),
            "max_w": float(np.max(w)),
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

    def _build_perceptual_learning_circuit(self):
        """Phase L5: 지각 학습 회로 — 좋은/나쁜 음식 → IT_Food/IT_Danger (R-STDP)

        생물학적 근거:
        - 맛 혐오 학습 (Garcia Effect): 나쁜 음식 → Amygdala → 회피
        - 피질 STDP: 보상 조절 시냅스 가소성으로 범주 학습
        - 음식 타입별 시각 경로가 IT 피질에서 범주별로 분화

        학습 시냅스 (8개, SPARSE 0.08):
        - good_food→IT_Food: R-STDP 강화 (좋은 음식 = 먹을 것)
        - good_food→IT_Danger: Anti-Hebbian 약화 (좋은 음식 ≠ 위험)
        - bad_food→IT_Danger: R-STDP 강화 (나쁜 음식 = 위험)
        - bad_food→IT_Food: Anti-Hebbian 약화 (나쁜 음식 ≠ 먹을 것)
        """
        print("\n  === Phase L5: Perceptual Learning Circuit ===")
        init_w = self.config.cortical_rstdp_init_w

        # 좋은 음식 → IT_Food (R-STDP 강화: 좋은 음식이면 "먹을 것" 학습)
        self.good_food_to_it_food_l = self._create_static_synapse(
            "good_food_l_to_it_food", self.good_food_eye_left, self.it_food_category,
            init_w, sparsity=0.08)
        self.good_food_to_it_food_r = self._create_static_synapse(
            "good_food_r_to_it_food", self.good_food_eye_right, self.it_food_category,
            init_w, sparsity=0.08)

        # 좋은 음식 → IT_Danger (Anti-Hebbian 약화: 좋은 음식 ≠ 위험)
        self.good_food_to_it_danger_l = self._create_static_synapse(
            "good_food_l_to_it_danger", self.good_food_eye_left, self.it_danger_category,
            init_w, sparsity=0.08)
        self.good_food_to_it_danger_r = self._create_static_synapse(
            "good_food_r_to_it_danger", self.good_food_eye_right, self.it_danger_category,
            init_w, sparsity=0.08)

        # 나쁜 음식 → IT_Danger (R-STDP 강화: 나쁜 음식 = 위험)
        self.bad_food_to_it_danger_l = self._create_static_synapse(
            "bad_food_l_to_it_danger", self.bad_food_eye_left, self.it_danger_category,
            init_w, sparsity=0.08)
        self.bad_food_to_it_danger_r = self._create_static_synapse(
            "bad_food_r_to_it_danger", self.bad_food_eye_right, self.it_danger_category,
            init_w, sparsity=0.08)

        # 나쁜 음식 → IT_Food (Anti-Hebbian 약화: 나쁜 음식 ≠ 먹을 것)
        self.bad_food_to_it_food_l = self._create_static_synapse(
            "bad_food_l_to_it_food", self.bad_food_eye_left, self.it_food_category,
            init_w, sparsity=0.08)
        self.bad_food_to_it_food_r = self._create_static_synapse(
            "bad_food_r_to_it_food", self.bad_food_eye_right, self.it_food_category,
            init_w, sparsity=0.08)

        print(f"    Good→IT_Food (R-STDP), Good→IT_Danger (Anti-Hebbian)")
        print(f"    Bad→IT_Danger (R-STDP), Bad→IT_Food (Anti-Hebbian)")
        print(f"    Init weight: {init_w}, Sparsity: 0.08")
        print(f"    Total: 8 learning synapses")

    def _build_prediction_error_circuit(self):
        """Phase L6: 예측 오차 회로 — 계층적 예측 코딩

        생물학적 근거:
        - 예측 코딩 (Predictive Coding): 뇌는 감각 입력을 예측하고 오차만 전파
        - IT→V1 하향 예측이 V1 상향 신호를 억제 → 오차만 남음
        - 예측 오차 뉴런이 IT 표상을 정교화 → 내부 모델 형성

        회로:
        - PE_Food_L/R: V1_Food(+10) - IT_Food(-7) = 예측 오차
        - PE_Danger_L/R: V1_Danger(+10) - IT_Danger(-7) = 예측 오차
        - PE → IT (R-STDP): 오차가 IT 범주 표상을 정교화
        """
        print("\n  === Phase L6: Prediction Error Circuit ===")

        # LIF 파라미터 (PE는 standard LIF, I_input 불필요)
        lif_params = {
            "C": 1.0, "TauM": self.config.tau_m, "Vrest": self.config.v_rest,
            "Vreset": self.config.v_reset, "Vthresh": self.config.v_thresh,
            "Ioffset": 0.0, "TauRefrac": self.config.tau_refrac
        }
        lif_init = {"V": self.config.v_rest, "RefracTime": 0.0}

        n_pe_food_half = self.config.n_pe_food // 2
        n_pe_danger_half = self.config.n_pe_danger // 2

        # === 1. Prediction Error Populations ===
        self.pe_food_left = self.model.add_neuron_population(
            "pe_food_left", n_pe_food_half, "LIF", lif_params, lif_init)
        self.pe_food_right = self.model.add_neuron_population(
            "pe_food_right", n_pe_food_half, "LIF", lif_params, lif_init)
        self.pe_danger_left = self.model.add_neuron_population(
            "pe_danger_left", n_pe_danger_half, "LIF", lif_params, lif_init)
        self.pe_danger_right = self.model.add_neuron_population(
            "pe_danger_right", n_pe_danger_half, "LIF", lif_params, lif_init)

        print(f"    PE_Food: L({n_pe_food_half}) + R({n_pe_food_half})")
        print(f"    PE_Danger: L({n_pe_danger_half}) + R({n_pe_danger_half})")

        # === 2. Bottom-up: V1 → PE (excitatory, 실제 감각 신호) ===
        v1_pe_w = self.config.pe_v1_to_pe_weight
        self._create_static_synapse(
            "v1_food_l_to_pe_food_l", self.v1_food_left, self.pe_food_left,
            v1_pe_w, sparsity=0.15)
        self._create_static_synapse(
            "v1_food_r_to_pe_food_r", self.v1_food_right, self.pe_food_right,
            v1_pe_w, sparsity=0.15)
        self._create_static_synapse(
            "v1_danger_l_to_pe_danger_l", self.v1_danger_left, self.pe_danger_left,
            v1_pe_w, sparsity=0.15)
        self._create_static_synapse(
            "v1_danger_r_to_pe_danger_r", self.v1_danger_right, self.pe_danger_right,
            v1_pe_w, sparsity=0.15)

        print(f"    V1→PE (bottom-up excitatory): {v1_pe_w}")

        # === 3. Top-down: IT → PE (inhibitory, 예측이 오차를 억제) ===
        it_pe_w = self.config.pe_it_to_pe_weight  # negative = inhibitory
        self._create_static_synapse(
            "it_food_to_pe_food_l", self.it_food_category, self.pe_food_left,
            it_pe_w, sparsity=0.10)
        self._create_static_synapse(
            "it_food_to_pe_food_r", self.it_food_category, self.pe_food_right,
            it_pe_w, sparsity=0.10)
        self._create_static_synapse(
            "it_danger_to_pe_danger_l", self.it_danger_category, self.pe_danger_left,
            it_pe_w, sparsity=0.10)
        self._create_static_synapse(
            "it_danger_to_pe_danger_r", self.it_danger_category, self.pe_danger_right,
            it_pe_w, sparsity=0.10)

        print(f"    IT→PE (top-down inhibitory): {it_pe_w}")

        # === 4. Error → IT (learning synapses, R-STDP: 오차가 IT 정교화) ===
        init_w = self.config.pe_to_it_init_w
        self.pe_food_to_it_food_l = self._create_static_synapse(
            "pe_food_l_to_it_food", self.pe_food_left, self.it_food_category,
            init_w, sparsity=0.10)
        self.pe_food_to_it_food_r = self._create_static_synapse(
            "pe_food_r_to_it_food", self.pe_food_right, self.it_food_category,
            init_w, sparsity=0.10)
        self.pe_danger_to_it_danger_l = self._create_static_synapse(
            "pe_danger_l_to_it_danger", self.pe_danger_left, self.it_danger_category,
            init_w, sparsity=0.10)
        self.pe_danger_to_it_danger_r = self._create_static_synapse(
            "pe_danger_r_to_it_danger", self.pe_danger_right, self.it_danger_category,
            init_w, sparsity=0.10)

        print(f"    PE→IT (R-STDP learning): init_w={init_w}, eta={self.config.pe_rstdp_eta}")

        total_pe = self.config.n_pe_food + self.config.n_pe_danger
        print(f"  Prediction Error circuit complete: {total_pe} neurons, 4 learning synapses")

    def _build_sparse_expansion_circuit(self):
        """Phase L16: Sparse Expansion Layer (Mushroom Body / DG)
        Single KC(3000×2) + Inh(400×2) — all 9 inputs to same KC

        생물학적 근거:
        - 초파리 Mushroom Body: 희소한 Kenyon Cell 표현 (Aso et al., 2014)
        - 해마 DG: pattern separation via sparse coding (Leutgeb et al., 2007)
        - 적은 입력 → 많은 KC (expansion) → WTA로 희소화 → D1/D2 학습
        """
        from pygenn import init_var, init_weight_update, init_postsynaptic, init_sparse_connectivity

        n_kc = self.config.n_kc_per_side
        n_inh = self.config.n_kc_inhibitory_per_side

        print(f"  Phase L16: Building Sparse Expansion (KC) — single KC({n_kc}×2)...")

        # === KC LIF parameters (high C for sparse firing) ===
        kc_params = {
            "C": 30.0, "TauM": 20.0, "Vrest": -65.0,
            "Vreset": -65.0, "Vthresh": -50.0,
            "Ioffset": 0.0, "TauRefrac": 2.0
        }
        kc_init = {"V": -65.0, "RefracTime": 0.0}

        # KC inhibitory (SensoryLIF for dynamic I_input control)
        kc_inh_params = {
            "C": 1.0, "TauM": 20.0, "Vrest": -65.0,
            "Vreset": -65.0, "Vthresh": -50.0, "TauRefrac": 2.0
        }
        kc_inh_init = {"V": -65.0, "RefracTime": 0.0, "I_input": 0.0}

        # === A) KC Populations — single KC × 2 sides ===
        self.kc_left = self.model.add_neuron_population(
            "kc_left", n_kc, "LIF", kc_params, kc_init)
        self.kc_right = self.model.add_neuron_population(
            "kc_right", n_kc, "LIF", kc_params, kc_init)
        self.kc_inh_left = self.model.add_neuron_population(
            "kc_inh_left", n_inh, sensory_lif_model, kc_inh_params, kc_inh_init)
        self.kc_inh_right = self.model.add_neuron_population(
            "kc_inh_right", n_inh, sensory_lif_model, kc_inh_params, kc_inh_init)

        # Spike recording
        for pop in [self.kc_left, self.kc_right,
                    self.kc_inh_left, self.kc_inh_right]:
            pop.spike_recording_enabled = True

        # === B) Input synapses: all inputs → single KC ===

        # food_eye → KC
        self._create_static_synapse(
            "food_eye_l_to_kc_l", self.food_eye_left, self.kc_left,
            self.config.kc_food_eye_weight, sparsity=self.config.kc_food_eye_sparsity)
        self._create_static_synapse(
            "food_eye_r_to_kc_r", self.food_eye_right, self.kc_right,
            self.config.kc_food_eye_weight, sparsity=self.config.kc_food_eye_sparsity)
        # good_food_eye → KC
        self._create_static_synapse(
            "good_food_eye_l_to_kc_l", self.good_food_eye_left, self.kc_left,
            self.config.kc_good_bad_food_weight, sparsity=self.config.kc_good_bad_food_sparsity)
        self._create_static_synapse(
            "good_food_eye_r_to_kc_r", self.good_food_eye_right, self.kc_right,
            self.config.kc_good_bad_food_weight, sparsity=self.config.kc_good_bad_food_sparsity)
        # bad_food_eye → KC
        self._create_static_synapse(
            "bad_food_eye_l_to_kc_l", self.bad_food_eye_left, self.kc_left,
            self.config.kc_good_bad_food_weight, sparsity=self.config.kc_good_bad_food_sparsity)
        self._create_static_synapse(
            "bad_food_eye_r_to_kc_r", self.bad_food_eye_right, self.kc_right,
            self.config.kc_good_bad_food_weight, sparsity=self.config.kc_good_bad_food_sparsity)
        # it_food_category → KC (bilateral)
        self._create_static_synapse(
            "it_food_to_kc_l", self.it_food_category, self.kc_left,
            self.config.kc_it_food_weight, sparsity=self.config.kc_it_food_sparsity)
        self._create_static_synapse(
            "it_food_to_kc_r", self.it_food_category, self.kc_right,
            self.config.kc_it_food_weight, sparsity=self.config.kc_it_food_sparsity)
        # assoc_edible → KC (bilateral)
        if hasattr(self, 'assoc_edible'):
            self._create_static_synapse(
                "assoc_edible_to_kc_l", self.assoc_edible, self.kc_left,
                2.0, sparsity=0.05)
            self._create_static_synapse(
                "assoc_edible_to_kc_r", self.assoc_edible, self.kc_right,
                2.0, sparsity=0.05)
            print(f"    Assoc_Edible→KC: 2.0, sparsity=0.05")

        # sound_food L/R → KC L/R (lateralized)
        if self.config.auditory_enabled and hasattr(self, 'sound_food_left'):
            self._create_static_synapse(
                "sound_food_l_to_kc_l", self.sound_food_left, self.kc_left,
                4.0, sparsity=0.08)
            self._create_static_synapse(
                "sound_food_r_to_kc_r", self.sound_food_right, self.kc_right,
                4.0, sparsity=0.08)
            print(f"    Sound_Food→KC: 4.0, sparsity=0.08 (lateralized)")
        # wernicke_food → KC (Call Semantics)
        if self.config.language_enabled and hasattr(self, 'wernicke_food'):
            self._create_static_synapse(
                "wernicke_food_to_kc_l", self.wernicke_food, self.kc_left,
                3.0, sparsity=0.05)
            self._create_static_synapse(
                "wernicke_food_to_kc_r", self.wernicke_food, self.kc_right,
                3.0, sparsity=0.05)
            print(f"    Wernicke_Food→KC: 3.0, sparsity=0.05 (call semantics)")

        # ppc_goal_food → KC
        if hasattr(self, 'ppc_goal_food'):
            self._create_static_synapse(
                "ppc_goal_food_to_kc_l", self.ppc_goal_food, self.kc_left,
                2.0, sparsity=0.05)
            self._create_static_synapse(
                "ppc_goal_food_to_kc_r", self.ppc_goal_food, self.kc_right,
                2.0, sparsity=0.05)
            print(f"    PPC_Goal_Food→KC: 2.0, sparsity=0.05")
        # social_memory → KC
        if self.config.social_brain_enabled and self.config.mirror_enabled and hasattr(self, 'social_memory'):
            self._create_static_synapse(
                "social_mem_to_kc_l", self.social_memory, self.kc_left,
                1.5, sparsity=0.03)
            self._create_static_synapse(
                "social_mem_to_kc_r", self.social_memory, self.kc_right,
                1.5, sparsity=0.03)
            print(f"    Social_Memory→KC: 1.5, sparsity=0.03")

        # Assoc_Binding → KC
        if hasattr(self, 'assoc_binding'):
            self._create_static_synapse(
                "assoc_bind_to_kc_l", self.assoc_binding, self.kc_left,
                2.0, sparsity=0.05)
            self._create_static_synapse(
                "assoc_bind_to_kc_r", self.assoc_binding, self.kc_right,
                2.0, sparsity=0.05)
            print(f"    Assoc_Binding→KC: 2.0, sparsity=0.05 (C2: learned category→BG)")

        # === C) WTA synapses: single inhibition loop ===
        self._create_static_synapse(
            "kc_l_to_inh_l", self.kc_left, self.kc_inh_left,
            self.config.kc_to_inh_weight, sparsity=self.config.kc_to_inh_sparsity)
        self._create_static_synapse(
            "kc_inh_l_to_kc_l", self.kc_inh_left, self.kc_left,
            self.config.kc_inh_to_kc_weight, sparsity=self.config.kc_inh_to_kc_sparsity)
        self._create_static_synapse(
            "kc_r_to_inh_r", self.kc_right, self.kc_inh_right,
            self.config.kc_to_inh_weight, sparsity=self.config.kc_to_inh_sparsity)
        self._create_static_synapse(
            "kc_inh_r_to_kc_r", self.kc_inh_right, self.kc_right,
            self.config.kc_inh_to_kc_weight, sparsity=self.config.kc_inh_to_kc_sparsity)

        # === D) Output learning synapses: 4 SPARSE (1 KC × 2 sides × D1+D2) ===
        kc_d1_w = self.config.kc_to_d1_init_w
        kc_sp = self.config.kc_to_d1_sparsity

        self.kc_to_d1_l = self.model.add_synapse_population(
            "kc_l_to_d1_l", "SPARSE", self.kc_left, self.d1_left,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": kc_d1_w})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": kc_sp}))
        self.kc_to_d1_r = self.model.add_synapse_population(
            "kc_r_to_d1_r", "SPARSE", self.kc_right, self.d1_right,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": kc_d1_w})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": kc_sp}))
        self.kc_to_d2_l = self.model.add_synapse_population(
            "kc_l_to_d2_l", "SPARSE", self.kc_left, self.d2_left,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": kc_d1_w})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": kc_sp}))
        self.kc_to_d2_r = self.model.add_synapse_population(
            "kc_r_to_d2_r", "SPARSE", self.kc_right, self.d2_right,
            init_weight_update("StaticPulse", {}, {"g": init_var("Constant", {"constant": kc_d1_w})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": kc_sp}))

        # === E) Log ===
        n_total = n_kc * 2
        n_inh_total = n_inh * 2
        print(f"  Phase L16: Sparse Expansion (KC) — single KC")
        print(f"    KC: {n_kc}×2 = {n_total}, KC_Inh: {n_inh}×2 = {n_inh_total}")
        print(f"    Total KC neurons: {n_total} + {n_inh_total} inh = {n_total + n_inh_total}")
        print(f"    Output: 4 SPARSE learning (1 KC × 2 sides × D1+D2)")

    def _build_it_bg_circuit(self):
        """Phase L9: IT Cortex → BG (피질 하향 연결)

        학습된 피질 표상(IT_Food)을 BG 의사결정에 연결.
        생물학적 근거: IT cortex → caudate tail (Hikosaka 2013)
        """
        it_d1_w = self.config.it_to_d1_init_w
        it_d2_w = self.config.it_to_d2_init_w
        it_sp = self.config.it_to_bg_sparsity

        # IT_Food → D1 (R-STDP: 학습된 음식 카테고리 → Go 강화)
        self.it_food_to_d1_l = self._create_static_synapse(
            "it_food_to_d1_l", self.it_food_category, self.d1_left,
            it_d1_w, sparsity=it_sp)
        self.it_food_to_d1_r = self._create_static_synapse(
            "it_food_to_d1_r", self.it_food_category, self.d1_right,
            it_d1_w, sparsity=it_sp)

        # IT_Food → D2 (Anti-Hebbian: 학습된 음식 → NoGo 약화)
        self.it_food_to_d2_l = self._create_static_synapse(
            "it_food_to_d2_l", self.it_food_category, self.d2_left,
            it_d2_w, sparsity=it_sp)
        self.it_food_to_d2_r = self._create_static_synapse(
            "it_food_to_d2_r", self.it_food_category, self.d2_right,
            it_d2_w, sparsity=it_sp)

        print(f"    Phase L9: 4 IT_Food→D1/D2 SPARSE synapses, "
              f"init_w(D1)={it_d1_w} init_w(D2)={it_d2_w} sparsity={it_sp}")

    def _build_nac_circuit(self):
        """Phase L10: NAc Critic — TD Learning (RPE Dopamine)

        NAc shell이 상태→보상 가치를 학습.
        생물학적 근거: NAc shell → VP → VTA (Schultz 1997)
        """
        nac_sp = self.config.nac_food_eye_sparsity
        nac_w = self.config.nac_food_eye_init_w

        # Learning: food_eye → nac_value (R-STDP, DA 조절)
        self.food_to_nac_l = self._create_static_synapse(
            "food_to_nac_l", self.food_eye_left, self.nac_value,
            nac_w, sparsity=nac_sp)
        self.food_to_nac_r = self._create_static_synapse(
            "food_to_nac_r", self.food_eye_right, self.nac_value,
            nac_w, sparsity=nac_sp)

        # Static context: IT_Food → nac_value
        if self.config.it_enabled:
            self._create_static_synapse(
                "it_food_to_nac", self.it_food_category, self.nac_value,
                self.config.nac_it_food_weight, sparsity=0.05)

        # Static context: Place_Cells → nac_value
        if self.config.hippocampus_enabled:
            self._create_static_synapse(
                "place_to_nac", self.place_cells, self.nac_value,
                self.config.nac_place_weight, sparsity=0.05)

        # Local inhibition
        self._create_static_synapse(
            "nac_value_to_inh", self.nac_value, self.nac_inhibitory,
            3.0, sparsity=0.2)
        self._create_static_synapse(
            "nac_inh_to_value", self.nac_inhibitory, self.nac_value,
            -5.0, sparsity=0.2)

        print(f"    Phase L10: NAc({self.config.n_nac_value}+{self.config.n_nac_inhibitory}), "
              f"2 R-STDP + 4 static, RPE discount={self.config.rpe_discount}")

    def _build_swr_circuit(self):
        """Phase L11: SWR Replay Circuit — 오프라인 기억 재생

        생물학적 근거: Buzsáki 2015 — SWR이 해마 시퀀스를 압축 재생
        Static synapses only (7개): place→ca3, ca3→food_mem L/R,
        swr_gate→replay_inh, replay_inh→motor L/R
        """
        # Place → CA3 (리플레이 시 시퀀스 인코딩)
        self._create_static_synapse(
            "place_to_ca3", self.place_cells, self.ca3_sequence,
            self.config.place_to_ca3_weight, sparsity=self.config.place_to_ca3_sparsity)

        # CA3 → Food Memory (리플레이 시 맥락 전달)
        if self.config.directional_food_memory:
            self._create_static_synapse(
                "ca3_to_food_mem_l", self.ca3_sequence, self.food_memory_left,
                self.config.ca3_to_food_memory_weight, sparsity=0.05)
            self._create_static_synapse(
                "ca3_to_food_mem_r", self.ca3_sequence, self.food_memory_right,
                self.config.ca3_to_food_memory_weight, sparsity=0.05)

        # SWR Gate → Replay Inhibitory (게이트 ON → Motor 억제)
        self._create_static_synapse(
            "swr_to_replay_inh", self.swr_gate, self.replay_inhibitory,
            self.config.swr_gate_to_inh_weight, sparsity=0.3)

        # Replay Inhibitory → Motor (DENSE, 리플레이 중 움직임 차단)
        self._create_static_synapse(
            "replay_inh_to_motor_l", self.replay_inhibitory, self.motor_left,
            self.config.swr_motor_inhibit_weight)
        self._create_static_synapse(
            "replay_inh_to_motor_r", self.replay_inhibitory, self.motor_right,
            self.config.swr_motor_inhibit_weight)

        print(f"    Phase L11: SWR({self.config.n_ca3_sequence}+{self.config.n_swr_gate}+{self.config.n_replay_inhibitory}), "
              f"7 static, replay_count={self.config.swr_replay_count}")

    def _build_gw_circuit(self):
        """Phase L12: Global Workspace — 주의 기반 경쟁적 브로드캐스트

        생물학적 근거: Dehaene & Changeux (2011) — Global Neuronal Workspace
        음식 탐색 vs 안전이 경쟁, 승자가 motor에 브로드캐스트

        Static synapses only (12개):
        - Input to GW_Food: food_memory(2), hunger(2), good_food_eye(2)
        - Input to GW_Safety: fear(1), lateral_amygdala(1)
        - Competition: GW_Safety → GW_Food 억제(2)
        - Motor output: GW_Food → Motor(2)
        """
        # --- GW_Food 입력 (음식 탐색 채널) ---
        # food_memory → GW_Food (기억 기반 방향)
        self._create_static_synapse(
            "food_mem_l_to_gw_food_l", self.food_memory_left, self.gw_food_left,
            self.config.gw_food_memory_weight, sparsity=0.1)
        self._create_static_synapse(
            "food_mem_r_to_gw_food_r", self.food_memory_right, self.gw_food_right,
            self.config.gw_food_memory_weight, sparsity=0.1)

        # hunger → GW_Food (허기 게이팅 — 배고플 때만 활성)
        self._create_static_synapse(
            "hunger_to_gw_food_l", self.hunger_drive, self.gw_food_left,
            self.config.gw_hunger_weight, sparsity=0.05)
        self._create_static_synapse(
            "hunger_to_gw_food_r", self.hunger_drive, self.gw_food_right,
            self.config.gw_hunger_weight, sparsity=0.05)

        # good_food_eye → GW_Food (직접 감각 부스트)
        self._create_static_synapse(
            "good_eye_l_to_gw_food_l", self.good_food_eye_left, self.gw_food_left,
            self.config.gw_good_food_eye_weight, sparsity=0.08)
        self._create_static_synapse(
            "good_eye_r_to_gw_food_r", self.good_food_eye_right, self.gw_food_right,
            self.config.gw_good_food_eye_weight, sparsity=0.08)

        # --- GW_Safety 입력 (안전 채널) ---
        self._create_static_synapse(
            "fear_to_gw_safety", self.fear_response, self.gw_safety,
            self.config.gw_fear_weight, sparsity=0.1)
        self._create_static_synapse(
            "la_to_gw_safety", self.lateral_amygdala, self.gw_safety,
            self.config.gw_la_weight, sparsity=0.05)

        # --- WTA Competition (안전이 음식 억제) ---
        self._create_static_synapse(
            "gw_safety_to_gw_food_l", self.gw_safety, self.gw_food_left,
            self.config.gw_safety_inhibit_weight, sparsity=0.1)
        self._create_static_synapse(
            "gw_safety_to_gw_food_r", self.gw_safety, self.gw_food_right,
            self.config.gw_safety_inhibit_weight, sparsity=0.1)

        # --- Motor Output (약한 방향 편향) ---
        self._create_static_synapse(
            "gw_food_l_to_motor_l", self.gw_food_left, self.motor_left,
            self.config.gw_food_to_motor_weight, sparsity=self.config.gw_food_to_motor_sparsity)
        self._create_static_synapse(
            "gw_food_r_to_motor_r", self.gw_food_right, self.motor_right,
            self.config.gw_food_to_motor_weight, sparsity=self.config.gw_food_to_motor_sparsity)

        print(f"    Phase L12: GW({self.config.n_gw_food}x2+{self.config.n_gw_safety}), "
              f"12 static, food_mem->motor reduced to {self.config.food_memory_to_motor_weight}")

    def _build_contextual_prediction_circuit(self):
        """Phase C4: Contextual Prediction — 경험 기반 음식 예측

        생물학적 근거: Hippocampal predictive map (Bono et al.)
        "이 장소 + 이 소리 + 최근 문맥 → 곧 음식이 나올 확률 높음"

        핵심: 기존 회로(place, food_memory, WM, sound) 출력을 작은 readout에 수렴시켜
        예측→BG approach bias. Motor 직접 연결 없음 (안전).

        Populations: pred_food_soon(30 LIF) + pred_food_inh(15 LIF) = +45 neurons
        Learnable: 2 SPARSE R-STDP (place→pred, wmcb→pred)
        Output: pred→goal_food(1.5) + pred→D1 L/R(1.0) — gentle modulator
        """
        print("  Building C4: Contextual Prediction circuit...")

        cfg = self.config
        lif_params = {
            "C": 30.0,  # 높은 capacitance for integration
            "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
            "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0
        }
        lif_init = {"V": -65.0, "RefracTime": 0.0}

        inh_params = {
            "C": 10.0,
            "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
            "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0
        }

        # === A) Populations ===
        self.pred_food_soon = self.model.add_neuron_population(
            "pred_food_soon", cfg.n_pred_food_soon, "LIF", lif_params, lif_init)
        self.pred_food_inh = self.model.add_neuron_population(
            "pred_food_inh", cfg.n_pred_food_inh, "LIF", inh_params, lif_init)

        # === B) Context inputs (static) ===
        # Food Memory → Pred ("음식이 여기 있었다" 기억)
        self._create_static_synapse(
            "food_mem_l_to_pred", self.food_memory_left, self.pred_food_soon,
            cfg.food_mem_to_pred_weight, sparsity=cfg.food_mem_to_pred_sparsity)
        self._create_static_synapse(
            "food_mem_r_to_pred", self.food_memory_right, self.pred_food_soon,
            cfg.food_mem_to_pred_weight, sparsity=cfg.food_mem_to_pred_sparsity)

        # Temporal_Recent → Pred ("최근에 무슨 일이 있었다")
        if hasattr(self, 'temporal_recent'):
            self._create_static_synapse(
                "temporal_to_pred", self.temporal_recent, self.pred_food_soon,
                cfg.temporal_to_pred_weight, sparsity=cfg.temporal_to_pred_sparsity)

        # Sound_Food → Pred ("음식 소리가 들린다")
        if hasattr(self, 'sound_food_left'):
            self._create_static_synapse(
                "sound_food_l_to_pred", self.sound_food_left, self.pred_food_soon,
                cfg.sound_food_to_pred_weight, sparsity=cfg.sound_food_to_pred_sparsity)
            self._create_static_synapse(
                "sound_food_r_to_pred", self.sound_food_right, self.pred_food_soon,
                cfg.sound_food_to_pred_weight, sparsity=cfg.sound_food_to_pred_sparsity)

        # Hunger → Pred (need-gating: 배고플 때만 예측 의미 있음)
        self._create_static_synapse(
            "hunger_to_pred", self.hunger_drive, self.pred_food_soon,
            cfg.hunger_to_pred_weight, sparsity=cfg.hunger_to_pred_sparsity)

        # === C) Learnable Predictive STDP (DENSE — per-post budget normalization) ===
        # Place_Cells → Pred ("이 장소가 음식을 예측하는가" — teacher-driven, not DA)
        n_place = cfg.n_place_cells
        n_pred = cfg.n_pred_food_soon
        self.place_to_pred = self.model.add_synapse_population(
            "place_to_pred", "DENSE", self.place_cells, self.pred_food_soon,
            init_weight_update("StaticPulse", {},
                {"g": init_var("Constant", {"constant": cfg.place_to_pred_init_w})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))

        # WM_Context_Binding → Pred (같은 predictive rule)
        if hasattr(self, 'wm_context_binding'):
            self.wmcb_to_pred = self.model.add_synapse_population(
                "wmcb_to_pred", "DENSE", self.wm_context_binding, self.pred_food_soon,
                init_weight_update("StaticPulse", {},
                    {"g": init_var("Constant", {"constant": cfg.wmcb_to_pred_init_w})}),
                init_postsynaptic("ExpCurr", {"tau": 5.0}))

        # === C2) Food Teacher → Pred (fixed, strong — drives post spikes on food events)
        # food_eye_left/right → Pred (음식이 보일 때 teacher signal)
        self._create_static_synapse(
            "food_teacher_l_to_pred", self.food_eye_left, self.pred_food_soon,
            cfg.food_teacher_to_pred_weight, sparsity=cfg.food_teacher_to_pred_sparsity)
        self._create_static_synapse(
            "food_teacher_r_to_pred", self.food_eye_right, self.pred_food_soon,
            cfg.food_teacher_to_pred_weight, sparsity=cfg.food_teacher_to_pred_sparsity)

        # === C3) Lateral inhibition inside Pred_FoodSoon (symmetry breaking)
        self._create_static_synapse(
            "pred_lateral_inh", self.pred_food_soon, self.pred_food_soon,
            cfg.pred_lateral_inh_weight, sparsity=cfg.pred_lateral_inh_sparsity)

        # === D) Output to BG (gentle modulator) ===
        # Pred → Goal_Food (예측 활성 → 음식 탐색 목표 강화)
        self._create_static_synapse(
            "pred_to_goal_food", self.pred_food_soon, self.goal_food,
            cfg.pred_to_goal_food_weight, sparsity=cfg.pred_to_goal_food_sparsity)

        # Pred → D1 L/R (대칭적 접근 편향 — 방향 무관, 전반적 approach)
        self.pred_to_d1_l = self.model.add_synapse_population(
            "pred_to_d1_l", "SPARSE", self.pred_food_soon, self.d1_left,
            init_weight_update("StaticPulse", {},
                {"g": init_var("Constant", {"constant": cfg.pred_to_d1_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": cfg.pred_to_d1_sparsity}))
        self.pred_to_d1_r = self.model.add_synapse_population(
            "pred_to_d1_r", "SPARSE", self.pred_food_soon, self.d1_right,
            init_weight_update("StaticPulse", {},
                {"g": init_var("Constant", {"constant": cfg.pred_to_d1_weight})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}),
            init_sparse_connectivity("FixedProbability", {"prob": cfg.pred_to_d1_sparsity}))

        # === E) WTA competition ===
        self._create_static_synapse(
            "pred_to_inh", self.pred_food_soon, self.pred_food_inh,
            cfg.pred_to_inh_weight, sparsity=cfg.pred_to_inh_sparsity)
        self._create_static_synapse(
            "pred_inh_to_pred", self.pred_food_inh, self.pred_food_soon,
            cfg.pred_inh_to_pred_weight, sparsity=cfg.pred_inh_to_pred_sparsity)

        print(f"    Pred_FoodSoon: {cfg.n_pred_food_soon} + Inh: {cfg.n_pred_food_inh} = "
              f"{cfg.n_pred_food_soon + cfg.n_pred_food_inh} neurons")
        print(f"    Predictive STDP: place({n_place})→pred DENSE + wmcb→pred DENSE")
        print(f"    Teacher: food_eye→pred({cfg.food_teacher_to_pred_weight}), "
              f"lateral_inh({cfg.pred_lateral_inh_weight})")
        print(f"    Budget: W_budget={cfg.place_to_pred_w_budget} per post neuron")
        print(f"    Output: pred→goal_food({cfg.pred_to_goal_food_weight}), "
              f"pred→D1({cfg.pred_to_d1_weight})")

    def _build_curiosity_circuit(self):
        """Phase C5: Curiosity-Driven Exploration — 호기심 기반 탐색

        생물학적 근거: Hippocampal-VTA novelty loop, ACC-BG information-seeking
        (Gottlieb & Oudeyer, 2013; Bromberg-Martin & Hikosaka, 2009)

        핵심 원리: novelty-gated, uncertainty-reduction-seeking curiosity
        - Novelty (V4_Novel, Assoc_Novelty): "이것은 탐색할 가치 있음"
        - Uncertainty (Meta_Uncertainty): "아직 모르는 것이 있음"
        - Safety_Gate: 위험 시 호기심 차단 (생존 우선)
        - 출력: Goal_Food + D1 (gentle bias) — Motor 직접 연결 없음

        Populations: curiosity_gate(20 LIF) + curiosity_inh(10 LIF) = +30 neurons
        All static synapses (no new learning — 기존 BG R-STDP가 탐색 결과 학습)
        """
        print("  Building C5: Curiosity-Driven Exploration circuit...")

        cfg = self.config
        gate_params = {
            "C": 30.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
            "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0
        }
        inh_params = {
            "C": 10.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
            "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0
        }
        lif_init = {"V": -65.0, "RefracTime": 0.0}

        # === A) Populations ===
        self.curiosity_gate = self.model.add_neuron_population(
            "curiosity_gate", cfg.n_curiosity_gate, "LIF", gate_params, lif_init)
        self.curiosity_inh = self.model.add_neuron_population(
            "curiosity_inh", cfg.n_curiosity_inh, "LIF", inh_params, lif_init)

        # === B) Novelty + Uncertainty inputs → Curiosity_Gate ===
        # V4_Novel → "새로운 객체 감지됨"
        if hasattr(self, 'v4_novel_object'):
            self._create_static_synapse(
                "v4_novel_to_curiosity", self.v4_novel_object, self.curiosity_gate,
                cfg.v4_novel_to_curiosity_weight, sparsity=cfg.v4_novel_to_curiosity_sparsity)

        # Assoc_Novelty → "새로운 조합 감지됨"
        if hasattr(self, 'assoc_novelty'):
            self._create_static_synapse(
                "assoc_novelty_to_curiosity", self.assoc_novelty, self.curiosity_gate,
                cfg.assoc_novelty_to_curiosity_weight, sparsity=cfg.assoc_novelty_to_curiosity_sparsity)

        # Meta_Uncertainty → "아직 모르는 것이 있다"
        if hasattr(self, 'meta_uncertainty'):
            self._create_static_synapse(
                "meta_uncertainty_to_curiosity", self.meta_uncertainty, self.curiosity_gate,
                cfg.meta_uncertainty_to_curiosity_weight, sparsity=cfg.meta_uncertainty_to_curiosity_sparsity)

        # ACC_Conflict → "갈등/모호성 감지됨"
        if hasattr(self, 'acc_conflict'):
            self._create_static_synapse(
                "acc_conflict_to_curiosity", self.acc_conflict, self.curiosity_gate,
                cfg.acc_conflict_to_curiosity_weight, sparsity=cfg.acc_conflict_to_curiosity_sparsity)

        # Self-recurrent (brief persistence, not perseveration)
        self._create_static_synapse(
            "curiosity_recurrent", self.curiosity_gate, self.curiosity_gate,
            cfg.curiosity_recurrent_weight, sparsity=cfg.curiosity_recurrent_sparsity)

        # === C) Safety suppression ===
        # Fear → Safety_Gate (위험하면 호기심 억제)
        if hasattr(self, 'fear_response'):
            self._create_static_synapse(
                "fear_to_curiosity_inh", self.fear_response, self.curiosity_inh,
                cfg.fear_to_curiosity_inh_weight, sparsity=cfg.fear_to_curiosity_inh_sparsity)

        # GW_Safety → Safety_Gate (안전 모드면 호기심 억제)
        if hasattr(self, 'gw_safety'):
            self._create_static_synapse(
                "gw_safety_to_curiosity_inh", self.gw_safety, self.curiosity_inh,
                cfg.gw_safety_to_curiosity_inh_weight, sparsity=cfg.gw_safety_to_curiosity_inh_sparsity)

        # Safety_Gate → Curiosity_Gate (억제)
        self._create_static_synapse(
            "curiosity_inh_to_gate", self.curiosity_inh, self.curiosity_gate,
            cfg.curiosity_inh_to_gate_weight, sparsity=cfg.curiosity_inh_to_gate_sparsity)

        # === D) Output to BG (gentle modulator — Motor 직접 없음) ===
        # Curiosity → Goal_Food (호기심이 음식 탐색 목표를 부스트)
        self._create_static_synapse(
            "curiosity_to_goal_food", self.curiosity_gate, self.goal_food,
            cfg.curiosity_to_goal_food_weight, sparsity=cfg.curiosity_to_goal_food_sparsity)

        # Curiosity → D1 L/R (대칭적 approach bias)
        self._create_static_synapse(
            "curiosity_to_d1_l", self.curiosity_gate, self.d1_left,
            cfg.curiosity_to_d1_weight, sparsity=cfg.curiosity_to_d1_sparsity)
        self._create_static_synapse(
            "curiosity_to_d1_r", self.curiosity_gate, self.d1_right,
            cfg.curiosity_to_d1_weight, sparsity=cfg.curiosity_to_d1_sparsity)

        n_syn = 10  # 4 input + 1 recurrent + 3 safety + 3 output (approx)
        print(f"    Curiosity_Gate: {cfg.n_curiosity_gate} + Inh: {cfg.n_curiosity_inh} = "
              f"{cfg.n_curiosity_gate + cfg.n_curiosity_inh} neurons")
        print(f"    Synapses: ~{n_syn} static (all static, no learning)")
        print(f"    Input: V4_Novel({cfg.v4_novel_to_curiosity_weight}), "
              f"Assoc_Novelty({cfg.assoc_novelty_to_curiosity_weight}), "
              f"Meta_Uncertainty({cfg.meta_uncertainty_to_curiosity_weight})")
        print(f"    Safety: Fear({cfg.fear_to_curiosity_inh_weight})→Inh→Gate({cfg.curiosity_inh_to_gate_weight})")
        print(f"    Output: →Goal_Food({cfg.curiosity_to_goal_food_weight}), "
              f"→D1({cfg.curiosity_to_d1_weight})")

    def _build_uncertainty_gate(self):
        """M3: ACh Uncertainty Gate — 환경 변화 감지 + 학습률 조절

        생물학적 근거: Tu et al. (eLife 2025) — ACh modulates prefrontal
        outcome coding during threat learning under uncertainty.

        핵심: 지속적 예측 오차 → "세계가 바뀌었다" → 학습률 증가 + 탐색 강화
        일시적 예측 오차 → "단발성 노이즈" → 무시

        Populations: surprise_accum(20) + stability_detector(10) = +30 neurons
        Output: surprise_rate를 process()에서 scalar로 읽어 eta modulation
        Motor 연결 없음 (안전).
        """
        print("  Building M3: ACh Uncertainty Gate...")

        cfg = self.config
        # Surprise Accumulator: slow integration (high C for temporal smoothing)
        surprise_params = {
            "C": 50.0,  # 높은 capacitance → 느린 통합 → 일시적 PE 무시
            "TauM": 30.0, "Vrest": -65.0, "Vreset": -65.0,
            "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 3.0
        }
        stability_params = {
            "C": 20.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
            "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0
        }
        lif_init = {"V": -65.0, "RefracTime": 0.0}

        # === Populations ===
        self.surprise_accum = self.model.add_neuron_population(
            "surprise_accum", cfg.n_surprise_accum, "LIF", surprise_params, lif_init)
        self.stability_detector = self.model.add_neuron_population(
            "stability_detector", cfg.n_stability_detector, "LIF", stability_params, lif_init)

        # === Surprise inputs (PE signals → accumulate) ===
        if hasattr(self, 'agency_pe'):
            self._create_static_synapse(
                "agency_pe_to_surprise", self.agency_pe, self.surprise_accum,
                cfg.agency_pe_to_surprise_weight, sparsity=0.08)
        if hasattr(self, 'meta_uncertainty'):
            self._create_static_synapse(
                "meta_uncert_to_surprise", self.meta_uncertainty, self.surprise_accum,
                cfg.meta_uncertainty_to_surprise_weight, sparsity=0.08)
        if hasattr(self, 'acc_conflict'):
            self._create_static_synapse(
                "acc_conflict_to_surprise", self.acc_conflict, self.surprise_accum,
                cfg.acc_conflict_to_surprise_weight, sparsity=0.05)
        # Self-recurrent (slow accumulation — persistent surprise)
        self._create_static_synapse(
            "surprise_recurrent", self.surprise_accum, self.surprise_accum,
            cfg.surprise_recurrent_weight, sparsity=cfg.surprise_recurrent_sparsity)

        # === Stability inputs (confidence/prediction → stable) ===
        if hasattr(self, 'meta_confidence'):
            self._create_static_synapse(
                "meta_conf_to_stability", self.meta_confidence, self.stability_detector,
                cfg.meta_confidence_to_stability_weight, sparsity=0.08)
        if hasattr(self, 'pred_food_soon'):
            self._create_static_synapse(
                "pred_food_to_stability", self.pred_food_soon, self.stability_detector,
                cfg.pred_food_to_stability_weight, sparsity=0.08)

        # === Competition: stability suppresses surprise ===
        self._create_static_synapse(
            "stability_to_surprise", self.stability_detector, self.surprise_accum,
            cfg.stability_to_surprise_weight, sparsity=cfg.stability_to_surprise_sparsity)

        print(f"    Surprise_Accum: {cfg.n_surprise_accum} (C=50, slow)")
        print(f"    Stability_Detector: {cfg.n_stability_detector}")
        print(f"    Input: Agency_PE({cfg.agency_pe_to_surprise_weight}), "
              f"Meta_Uncertainty({cfg.meta_uncertainty_to_surprise_weight})")
        print(f"    Output: surprise_rate → eta_mod (no Motor connection)")

    def _build_context_gate_circuit(self):
        """M4: Context Gate — zone-dependent value learning

        CtxA(4 LIF): 맵 왼쪽에서 활성 (place cells 왼쪽 절반 편향)
        CtxB(4 LIF): 맵 오른쪽에서 활성
        WTA 경쟁: 한 번에 하나만 활성
        Output: ctx_a_rate / ctx_b_rate → R-STDP eligibility gating
        """
        print("  Building M4: Context Gate circuit...")
        cfg = self.config
        n_pc = cfg.n_place_cells  # 400 (20x20 grid)

        ctx_params = {
            "C": 100.0, "TauM": 25.0, "Vrest": -65.0, "Vreset": -65.0,
            "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0
        }  # 높은 C → 입력 차이에 민감 (포화 방지)
        lif_init = {"V": -65.0, "RefracTime": 0.0}

        # SensoryLIF for direct I_input injection (place cell 분포가 아닌 위치 좌표 기반)
        sensory_params = {
            "C": 5.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
            "Vthresh": -50.0, "TauRefrac": 2.0
        }
        sensory_init = {"V": -65.0, "RefracTime": 0.0, "I_input": 0.0}

        self.ctx_a = self.model.add_neuron_population(
            "ctx_a", cfg.n_ctx_a, sensory_lif_model, sensory_params, sensory_init)
        self.ctx_b = self.model.add_neuron_population(
            "ctx_b", cfg.n_ctx_b, sensory_lif_model, sensory_params, sensory_init)
        # I_input은 process()에서 에이전트 x좌표 기반으로 설정
        # 왼쪽이면 CtxA에 전류, 오른쪽이면 CtxB에 전류
        # 이건 place cells가 위치 기반으로 I_input 받는 것과 동일한 원리

        # WTA: CtxA ↔ CtxB mutual inhibition
        self._create_static_synapse(
            "ctx_a_to_b_inh", self.ctx_a, self.ctx_b,
            cfg.ctx_wta_weight, sparsity=0.5)
        self._create_static_synapse(
            "ctx_b_to_a_inh", self.ctx_b, self.ctx_a,
            cfg.ctx_wta_weight, sparsity=0.5)

        # Self-excitation (winner stability)
        self._create_static_synapse(
            "ctx_a_recurrent", self.ctx_a, self.ctx_a,
            cfg.ctx_recurrent_weight, sparsity=0.5)
        self._create_static_synapse(
            "ctx_b_recurrent", self.ctx_b, self.ctx_b,
            cfg.ctx_recurrent_weight, sparsity=0.5)

        # === Context-specific value populations (별도 D1 없이 Motor 직접) ===
        val_params = {
            "C": 5.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
            "Vthresh": -50.0, "TauRefrac": 2.0
        }
        val_init = {"V": -65.0, "RefracTime": 0.0, "I_input": 0.0}
        # SensoryLIF — I_input으로 food×context weight 직접 주입 (GPU synapse 불필요)
        self.ctx_val_a_l = self.model.add_neuron_population(
            "ctx_val_a_l", 4, sensory_lif_model, val_params, val_init)
        self.ctx_val_a_r = self.model.add_neuron_population(
            "ctx_val_a_r", 4, sensory_lif_model, val_params, val_init)
        self.ctx_val_b_l = self.model.add_neuron_population(
            "ctx_val_b_l", 4, sensory_lif_model, val_params, val_init)
        self.ctx_val_b_r = self.model.add_neuron_population(
            "ctx_val_b_r", 4, sensory_lif_model, val_params, val_init)
        # food_eye → CtxVal 시냅스 없음 — I_input으로 직접 주입 (process()에서)

        # Context gating: CtxA → CtxVal_A 흥분, CtxVal_B 억제 (그리고 반대)
        self._create_static_synapse("ctx_a_excite_val_a_l", self.ctx_a, self.ctx_val_a_l, 8.0, sparsity=0.5)
        self._create_static_synapse("ctx_a_excite_val_a_r", self.ctx_a, self.ctx_val_a_r, 8.0, sparsity=0.5)
        self._create_static_synapse("ctx_a_inhibit_val_b_l", self.ctx_a, self.ctx_val_b_l, -10.0, sparsity=0.5)
        self._create_static_synapse("ctx_a_inhibit_val_b_r", self.ctx_a, self.ctx_val_b_r, -10.0, sparsity=0.5)
        self._create_static_synapse("ctx_b_excite_val_b_l", self.ctx_b, self.ctx_val_b_l, 8.0, sparsity=0.5)
        self._create_static_synapse("ctx_b_excite_val_b_r", self.ctx_b, self.ctx_val_b_r, 8.0, sparsity=0.5)
        self._create_static_synapse("ctx_b_inhibit_val_a_l", self.ctx_b, self.ctx_val_a_l, -10.0, sparsity=0.5)
        self._create_static_synapse("ctx_b_inhibit_val_a_r", self.ctx_b, self.ctx_val_a_r, -10.0, sparsity=0.5)

        # CtxVal → Motor (gentle push-pull — 기존 D1 경로와 병렬)
        for side, val_a, val_b, motor_push, motor_pull in [
            ("l", self.ctx_val_a_l, self.ctx_val_b_l, self.motor_left, self.motor_right),
            ("r", self.ctx_val_a_r, self.ctx_val_b_r, self.motor_right, self.motor_left),
        ]:
            self._create_static_synapse(f"ctxval_a_{side}_push", val_a, motor_push, 15.0, sparsity=0.2)
            self._create_static_synapse(f"ctxval_b_{side}_push", val_b, motor_push, 15.0, sparsity=0.2)

        print(f"    CtxA: {cfg.n_ctx_a} (SensoryLIF), CtxB: {cfg.n_ctx_b}")
        print(f"    CtxVal: 4×4=16 neurons (A_L/R + B_L/R)")
        print(f"    Context gating: excite matched(8.0), inhibit mismatched(-10.0)")
        print(f"    CtxVal→Motor: push 2.0 (gentle, parallel to existing D1)")

    def _build_place_transition_circuit(self):
        """M3: Place→Place transition graph + Value population

        생물학적 근거: Hippocampal place cell sequence replay with
        backward value propagation (Mattar & Daw 2018, Shin et al. 2022).

        1. W_pp: place→place recurrent STDP (spatial topology)
        2. Value population: offline value backup target
        3. Value→D1: gentle BG bias from learned spatial value

        0 new learning synapses at build time — W_pp는 process()에서 수동 업데이트,
        Value는 reverse replay에서 업데이트.
        """
        print("  Building M3: Place Transition + Value circuit...")

        cfg = self.config
        n_place = cfg.n_place_cells
        lif_init = {"V": -65.0, "RefracTime": 0.0}

        # Place→Place transition — CPU numpy only (GPU 메모리 절약)
        # SNN 시뮬레이션에 사용 안 함 (place cells는 I_input으로 구동)
        # reverse replay에서 역방향 탐색용으로만 사용
        n_pc = cfg.n_place_cells
        self.w_pp = np.zeros((n_pc, n_pc), dtype=np.float32)  # CPU only

        # Value population
        value_params = {
            "C": 30.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
            "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0
        }
        self.place_value = self.model.add_neuron_population(
            "place_value", cfg.n_place_value, "LIF", value_params, lif_init)

        # Place → Value (DENSE, reverse replay에서 수동 업데이트)
        self.place_to_value = self.model.add_synapse_population(
            "place_to_value", "DENSE", self.place_cells, self.place_value,
            init_weight_update("StaticPulse", {},
                {"g": init_var("Constant", {"constant": 0.1})}),
            init_postsynaptic("ExpCurr", {"tau": 5.0}))

        # Value → D1 L/R (gentle approach bias from spatial value)
        self._create_static_synapse(
            "value_to_d1_l", self.place_value, self.d1_left,
            cfg.value_to_d1_weight, sparsity=0.05)
        self._create_static_synapse(
            "value_to_d1_r", self.place_value, self.d1_right,
            cfg.value_to_d1_weight, sparsity=0.05)

        print(f"    Place→Place transition: DENSE {n_place}x{n_place} (STDP online)")
        print(f"    Value population: {cfg.n_place_value} neurons")
        print(f"    Value→D1: {cfg.value_to_d1_weight} (gentle)")
        print(f"    Reverse replay: {cfg.reverse_replay_steps} steps backward")

    def update_prediction_error_rstdp(self, reward_type: str):
        """Phase L6: 예측 오차 R-STDP 가중치 업데이트

        Args:
            reward_type: "food" (음식 섭취 시) 또는 "danger" (고통 경험 시)
        """
        if not self.config.prediction_error_enabled:
            return None
        if not (self.config.v1_enabled and self.config.it_enabled):
            return None

        eta = self.config.pe_rstdp_eta
        w_max = self.config.pe_to_it_w_max
        w_min = self.config.pe_to_it_w_min
        results = {}

        if reward_type == "food":
            for side, trace, syn in [
                ("left", self.pe_trace_food_l, self.pe_food_to_it_food_l),
                ("right", self.pe_trace_food_r, self.pe_food_to_it_food_r),
            ]:
                if trace > 0.01:
                    syn.vars["g"].pull_from_device()
                    w = syn.vars["g"].values
                    w[:] += eta * trace
                    w[:] = np.clip(w, w_min, w_max)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()
                    results[f"pe_food_it_{side}"] = float(np.nanmean(w))

        elif reward_type == "danger":
            for side, trace, syn in [
                ("left", self.pe_trace_danger_l, self.pe_danger_to_it_danger_l),
                ("right", self.pe_trace_danger_r, self.pe_danger_to_it_danger_r),
            ]:
                if trace > 0.01:
                    syn.vars["g"].pull_from_device()
                    w = syn.vars["g"].values
                    w[:] += eta * trace
                    w[:] = np.clip(w, w_min, w_max)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()
                    results[f"pe_danger_it_{side}"] = float(np.nanmean(w))

        return results if results else None

    def update_cortical_rstdp(self, reward_type: str):
        """Phase L5: 피질 R-STDP 가중치 업데이트

        Args:
            reward_type: "good_food" (좋은 음식 섭취) 또는 "bad_food" (나쁜 음식 섭취)
        """
        if not self.config.perceptual_learning_enabled or not self.config.it_enabled:
            return None

        eta = self.config.cortical_rstdp_eta
        anti_ratio = self.config.cortical_anti_hebbian_ratio
        w_max = self.config.cortical_rstdp_w_max
        w_min = self.config.cortical_rstdp_w_min
        results = {}

        if reward_type == "good_food":
            # 좋은 음식: good→IT_Food 강화, good→IT_Danger 약화
            for side, trace, syn_strengthen, syn_weaken in [
                ("left", self.cortical_trace_good_l,
                 self.good_food_to_it_food_l, self.good_food_to_it_danger_l),
                ("right", self.cortical_trace_good_r,
                 self.good_food_to_it_food_r, self.good_food_to_it_danger_r),
            ]:
                if trace > 0.01:
                    # R-STDP 강화
                    syn_strengthen.vars["g"].pull_from_device()
                    w = syn_strengthen.vars["g"].values
                    w[:] += eta * trace
                    w[:] = np.clip(w, w_min, w_max)
                    syn_strengthen.vars["g"].values = w
                    syn_strengthen.vars["g"].push_to_device()
                    results[f"good_it_food_{side}"] = float(np.nanmean(w))

                    # Anti-Hebbian 약화
                    syn_weaken.vars["g"].pull_from_device()
                    w2 = syn_weaken.vars["g"].values
                    w2[:] -= eta * anti_ratio * trace
                    w2[:] = np.clip(w2, w_min, w_max)
                    syn_weaken.vars["g"].values = w2
                    syn_weaken.vars["g"].push_to_device()
                    results[f"good_it_danger_{side}"] = float(np.nanmean(w2))

        elif reward_type == "bad_food":
            # 나쁜 음식: bad→IT_Danger 강화, bad→IT_Food 약화
            for side, trace, syn_strengthen, syn_weaken in [
                ("left", self.cortical_trace_bad_l,
                 self.bad_food_to_it_danger_l, self.bad_food_to_it_food_l),
                ("right", self.cortical_trace_bad_r,
                 self.bad_food_to_it_danger_r, self.bad_food_to_it_food_r),
            ]:
                if trace > 0.01:
                    # R-STDP 강화
                    syn_strengthen.vars["g"].pull_from_device()
                    w = syn_strengthen.vars["g"].values
                    w[:] += eta * trace
                    w[:] = np.clip(w, w_min, w_max)
                    syn_strengthen.vars["g"].values = w
                    syn_strengthen.vars["g"].push_to_device()
                    results[f"bad_it_danger_{side}"] = float(np.nanmean(w))

                    # Anti-Hebbian 약화
                    syn_weaken.vars["g"].pull_from_device()
                    w2 = syn_weaken.vars["g"].values
                    w2[:] -= eta * anti_ratio * trace
                    w2[:] = np.clip(w2, w_min, w_max)
                    syn_weaken.vars["g"].values = w2
                    syn_weaken.vars["g"].push_to_device()
                    results[f"bad_it_food_{side}"] = float(np.nanmean(w2))

        return results if results else None

    def trigger_taste_aversion(self, magnitude: float = 0.5):
        """Phase L5: 맛 혐오 — 나쁜 음식 섭취 시 danger_sensor 활성화

        Garcia Effect: 나쁜 음식 → 내장 불쾌 → 편도체 공포 반응
        danger_sensor(SensoryLIF)의 I_input으로 공포 경로 활성화
        기존 Pain→Fear→Motor 회피 경로를 재사용하여 맛 혐오 회피 학습
        """
        if not self.config.amygdala_enabled:
            return
        aversion_current = magnitude * self.config.taste_aversion_magnitude
        self.danger_sensor.vars["I_input"].view[:] = aversion_current
        self.danger_sensor.vars["I_input"].push_to_device()
        self._taste_aversion_active = True

    def _clear_taste_aversion(self):
        """맛 혐오 I_input 초기화 (다음 스텝에서 호출)"""
        if not self.config.amygdala_enabled:
            return
        if not getattr(self, '_taste_aversion_active', False):
            return
        self.danger_sensor.vars["I_input"].view[:] = 0.0
        self.danger_sensor.vars["I_input"].push_to_device()
        self._taste_aversion_active = False

    def learn_taste_aversion(self):
        """Phase L13: 조건화된 맛 혐오 Hebbian 학습 — bad_food_eye → LA 연결 강화

        Garcia Effect: 나쁜 음식 섭취 시 bad_food_eye의 활성화 패턴 기반으로
        시각-공포 연합 학습. 학습 후 나쁜 음식을 보기만 해도 LA 활성화 →
        CEA → Fear → Pain Push-Pull 회피 반응.

        Δw = η × pre_activity (활성화된 bad_food_eye 뉴런의 가중치 강화)
        """
        if not self.config.taste_aversion_learning_enabled:
            return None
        if not self.config.amygdala_enabled or not self.config.perceptual_learning_enabled:
            return None
        if not hasattr(self, 'bad_food_to_la_left'):
            return None

        eta = self.config.taste_aversion_hebbian_eta
        w_max = self.config.taste_aversion_hebbian_w_max
        n_pre = self.config.n_bad_food_eye // 2
        n_post = self.config.n_lateral_amygdala

        results = {}
        # 이전 스텝의 활성도 사용 (음식 섭취 시 현재 obs에서 음식이 이미 사라짐)
        sides = [
            ("left", self.bad_food_to_la_left, getattr(self, 'prev_bad_food_activity_left', 0.0)),
            ("right", self.bad_food_to_la_right, getattr(self, 'prev_bad_food_activity_right', 0.0)),
        ]

        for side, syn, activity in sides:
            syn.vars["g"].pull_from_device()
            weights = syn.vars["g"].view.copy().reshape(n_pre, n_post)

            # activity is scalar (mean of rays) — broadcast to all pre neurons
            # 나쁜 음식이 보이는 방향의 뉴런만 강화
            if activity > 0.05:  # 최소 활성도 임계값
                delta_w = eta * activity
                weights += delta_w
                weights = np.clip(weights, 0.0, w_max)
                n_strengthened = n_pre
            else:
                n_strengthened = 0

            syn.vars["g"].view[:] = weights.flatten()
            syn.vars["g"].push_to_device()

            results[f"n_strengthened_{side}"] = n_strengthened
            results[f"avg_w_{side}"] = float(np.mean(weights))
            results[f"max_w_{side}"] = float(np.max(weights))

        # Cache for real-time graph
        avg_left = results.get("avg_w_left", 0.0)
        avg_right = results.get("avg_w_right", 0.0)
        self._last_garcia_avg_w = (avg_left + avg_right) / 2.0

        return results

    def add_experience(self, pos_x: float, pos_y: float, food_type: int,
                       step: int, reward: float, tagged: bool = False):
        """Phase L11 + C0.5: 경험 버퍼에 이벤트 저장 (selective replay용 tag)"""
        if not self.config.swr_replay_enabled:
            return
        # 음식 섭취 이벤트는 자동 tag (보상 시점 SWR 태깅 — Yang et al. Science 2024)
        if reward != 0:
            tagged = True
        self.experience_buffer.append((pos_x, pos_y, food_type, step, reward, tagged))
        if len(self.experience_buffer) > self.config.swr_experience_max:
            self.experience_buffer = self.experience_buffer[-self.config.swr_experience_max:]

    def replay_swr(self):
        """Phase L11: SWR Replay — 에피소드 간 오프라인 기억 재생

        experience_buffer에서 좋은 음식 경험을 샘플링하여
        Place Cell 전류 주입 → learn_food_location() Hebbian 학습 강화.
        생물학적 근거: Buzsáki 2015, Foster & Wilson 2006
        """
        if not self.config.swr_replay_enabled or not self.config.hippocampus_enabled:
            return None
        if len(self.experience_buffer) == 0:
            return {"replayed_count": 0}

        # C0.5: Selective Replay — tagged 경험 80% 우선 (Yang et al. Science 2024)
        tagged_exp = [e for e in self.experience_buffer if len(e) > 5 and e[5]]
        untagged_exp = [e for e in self.experience_buffer if len(e) <= 5 or not e[5]]
        # fallback: 기존 5-tuple 경험은 good food만 선택
        if not tagged_exp:
            tagged_exp = [e for e in self.experience_buffer if e[2] == 0]
        if not tagged_exp:
            tagged_exp = self.experience_buffer

        stats_before = self.get_hippocampus_stats()
        avg_w_before = stats_before.get("avg_weight", 0.0) if stats_before else 0.0

        n_replay = min(self.config.swr_replay_count, len(self.experience_buffer))
        # 80% tagged, 20% random
        n_tagged = min(int(n_replay * 0.8), len(tagged_exp))
        n_random = n_replay - n_tagged

        replay_pool = []
        if n_tagged > 0 and tagged_exp:
            idxs = np.random.choice(len(tagged_exp), size=n_tagged, replace=False)
            replay_pool.extend([tagged_exp[i] for i in idxs])
        if n_random > 0 and untagged_exp:
            idxs = np.random.choice(len(untagged_exp), size=min(n_random, len(untagged_exp)), replace=False)
            replay_pool.extend([untagged_exp[i] for i in idxs])
        if not replay_pool:
            replay_pool = self.experience_buffer[:n_replay]

        replayed = 0
        for exp in replay_pool:
            pos_x, pos_y, food_type = exp[0], exp[1], exp[2]
            step, reward = exp[3], exp[4]

            # 1. SWR Gate ON → replay_inhibitory → Motor 억제
            self.swr_gate.vars["I_input"].view[:] = 30.0
            self.swr_gate.vars["I_input"].push_to_device()

            # 2. Place Cell 전류 주입 (스케일 다운)
            place_currents = self._compute_place_cell_input(pos_x, pos_y)
            place_currents *= self.config.swr_place_current_scale
            self.place_cells.vars["I_input"].view[:] = place_currents
            self.place_cells.vars["I_input"].push_to_device()

            # 3. 다른 감각 입력 제로화
            for pop in [self.food_eye_left, self.food_eye_right,
                        self.wall_eye_left, self.wall_eye_right]:
                pop.vars["I_input"].view[:] = 0.0
                pop.vars["I_input"].push_to_device()

            # 4. 시뮬레이션 스텝 (뉴런 활성화 전파)
            for _ in range(self.config.swr_replay_steps):
                self.model.step_time()

            # 5. Hebbian 학습 (기존 함수 재사용)
            # C0.5: negative 경험 (bad food)은 anti-learning (가중치 약화)
            if reward < 0:
                self.learn_food_location(food_position=(pos_x, pos_y), anti_learn=True)
            else:
                self.learn_food_location(food_position=(pos_x, pos_y))
            replayed += 1

        # M3: Reverse replay — value backup through transition graph
        reverse_replayed = 0
        if (self.config.place_transition_enabled
                and hasattr(self, 'w_pp')
                and len(self.transition_buffer) > 0):
            n_pc = self.config.n_place_cells
            w_pp_2d = self.w_pp  # CPU numpy array, no GPU pull needed

            self.place_to_value.vars["g"].pull_from_device()
            w_pv = self.place_to_value.vars["g"].view.copy()
            n_val = self.config.n_place_value
            w_pv_2d = w_pv.reshape(n_pc, n_val)

            eta_pv = self.config.place_to_value_eta
            w_max_pv = self.config.place_to_value_w_max
            rev_steps = self.config.reverse_replay_steps

            # 최근 rewarded transitions에서 역방향 전파
            recent_rewards = self.transition_buffer[-5:]
            total_preds_found = 0
            total_value_updates = 0
            max_wpp = float(np.max(w_pp_2d)) if w_pp_2d.size > 0 else 0.0

            for prev_cells, curr_cells, reward in recent_rewards:
                if reward <= 0 or not curr_cells:
                    continue
                # Start from rewarded cells, go backward through W_pp
                active_set = set(curr_cells[:10])  # 시작: rewarded place cells
                for step in range(rev_steps):
                    # 현재 active set의 value 강화
                    discount = 0.9 ** step
                    for cell_idx in active_set:
                        if cell_idx < n_pc:
                            w_pv_2d[cell_idx, :] += eta_pv * discount * reward
                            total_value_updates += 1
                    # W_pp 역방향: top-k adaptive predecessor selection
                    predecessors = set()
                    for target in active_set:
                        if target < n_pc:
                            incoming = w_pp_2d[:, target]
                            mx = incoming.max()
                            if mx < 0.02:  # transition graph가 아직 안 배웠음
                                continue
                            # Top-5 predecessors (adaptive threshold)
                            top_k = min(5, n_pc)
                            cand = np.argpartition(incoming, -top_k)[-top_k:]
                            threshold = max(0.02, 0.3 * mx)
                            preds = [int(p) for p in cand if incoming[p] >= threshold and p != target]
                            predecessors.update(preds)
                            total_preds_found += len(preds)
                    if not predecessors:
                        break
                    active_set = predecessors
                reverse_replayed += 1

            # 디버그 로그
            if reverse_replayed > 0 or len(self.transition_buffer) > 0:
                print(f"    [REVERSE] buffer={len(self.transition_buffer)}, "
                      f"max_wpp={max_wpp:.4f}, preds={total_preds_found}, "
                      f"updates={total_value_updates}")

            # Budget normalization on PC→Value
            for j in range(n_val):
                col_sum = np.sum(np.maximum(w_pv_2d[:, j], 0.0))
                budget = w_max_pv * n_pc * 0.1
                if col_sum > budget:
                    w_pv_2d[:, j] *= budget / col_sum
            np.clip(w_pv_2d, 0.0, w_max_pv, out=w_pv_2d)
            self.place_to_value.vars["g"].view[:] = w_pv_2d.reshape(-1)
            self.place_to_value.vars["g"].push_to_device()

        # 리플레이 후 정리
        self.swr_gate.vars["I_input"].view[:] = 0.0
        self.swr_gate.vars["I_input"].push_to_device()
        self.place_cells.vars["I_input"].view[:] = 0.0
        self.place_cells.vars["I_input"].push_to_device()

        stats_after = self.get_hippocampus_stats()
        avg_w_after = stats_after.get("avg_weight", 0.0) if stats_after else 0.0

        return {
            "replayed_count": replayed,
            "reverse_replayed": reverse_replayed,
            "buffer_size": len(self.experience_buffer),
            "avg_w_before": avg_w_before,
            "avg_w_after": avg_w_after,
        }

    def process(self, observation: Dict, debug: bool = False) -> Tuple[float, Dict]:
        """
        관찰을 받아 행동 출력

        Args:
            observation: ForagerGym observation dict
            debug: 상세 로그 출력 여부

        Returns:
            angle_delta, debug_info
        """
        # Phase L3: step counter for homeostatic decay batching
        if self.config.basal_ganglia_enabled:
            self._rstdp_step += 1

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

        # Obstacle Eye 입력 (wall에서 분리된 약한 회피)
        if self.config.obstacle_eye_enabled:
            obs_l = np.mean(observation.get("obstacle_rays_left", np.zeros(8)))
            obs_r = np.mean(observation.get("obstacle_rays_right", np.zeros(8)))
            obstacle_sensitivity = 40.0
            self.obstacle_eye_left.vars["I_input"].view[:] = obs_l * obstacle_sensitivity
            self.obstacle_eye_right.vars["I_input"].view[:] = obs_r * obstacle_sensitivity
            self.obstacle_eye_left.vars["I_input"].push_to_device()
            self.obstacle_eye_right.vars["I_input"].push_to_device()

        # === Phase L5: Good/Bad Food Eye 입력 ===
        good_food_l = 0.0
        good_food_r = 0.0
        bad_food_l = 0.0
        bad_food_r = 0.0
        if self.config.perceptual_learning_enabled:
            good_food_l = np.mean(observation.get("good_food_rays_left", np.zeros(8)))
            good_food_r = np.mean(observation.get("good_food_rays_right", np.zeros(8)))
            bad_food_l = np.mean(observation.get("bad_food_rays_left", np.zeros(8)))
            bad_food_r = np.mean(observation.get("bad_food_rays_right", np.zeros(8)))

            gs = self.config.good_food_eye_sensitivity
            self.good_food_eye_left.vars["I_input"].view[:] = good_food_l * gs
            self.good_food_eye_right.vars["I_input"].view[:] = good_food_r * gs
            self.good_food_eye_left.vars["I_input"].push_to_device()
            self.good_food_eye_right.vars["I_input"].push_to_device()

            bs = self.config.bad_food_eye_sensitivity
            self.bad_food_eye_left.vars["I_input"].view[:] = bad_food_l * bs
            self.bad_food_eye_right.vars["I_input"].view[:] = bad_food_r * bs
            self.bad_food_eye_left.vars["I_input"].push_to_device()
            self.bad_food_eye_right.vars["I_input"].push_to_device()

            # Phase L13: 맛 혐오 학습용 활성도 저장
            # 이전 스텝의 활성도 보존 (음식 섭취 시 현재 obs에서 음식이 사라지므로)
            self.prev_bad_food_activity_left = getattr(self, 'last_bad_food_activity_left', 0.0)
            self.prev_bad_food_activity_right = getattr(self, 'last_bad_food_activity_right', 0.0)
            self.last_bad_food_activity_left = bad_food_l
            self.last_bad_food_activity_right = bad_food_r

            self._cortical_step += 1
            # 맛 혐오 Ioffset 클리어 (이전 스텝에서 설정된 경우)
            self._clear_taste_aversion()

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

        # === Phase 17: Language Circuit 감각 입력 ===
        npc_call_food_l = 0.0
        npc_call_food_r = 0.0
        npc_call_danger_l = 0.0
        npc_call_danger_r = 0.0

        if self.config.language_enabled:
            npc_call_food_l = observation.get("npc_call_food_left", 0.0)
            npc_call_food_r = observation.get("npc_call_food_right", 0.0)
            npc_call_danger_l = observation.get("npc_call_danger_left", 0.0)
            npc_call_danger_r = observation.get("npc_call_danger_right", 0.0)

            call_sensitivity = 45.0
            self.call_food_input_left.vars["I_input"].view[:] = npc_call_food_l * call_sensitivity
            self.call_food_input_left.vars["I_input"].push_to_device()
            self.call_food_input_right.vars["I_input"].view[:] = npc_call_food_r * call_sensitivity
            self.call_food_input_right.vars["I_input"].push_to_device()
            self.call_danger_input_left.vars["I_input"].view[:] = npc_call_danger_l * call_sensitivity
            self.call_danger_input_left.vars["I_input"].push_to_device()
            self.call_danger_input_right.vars["I_input"].view[:] = npc_call_danger_r * call_sensitivity
            self.call_danger_input_right.vars["I_input"].push_to_device()

            # Vocal Gate: Fear 억제 (I_input 직접 주입, 이전 스텝 fear_rate 사용)
            fear_inhibition = -self.last_fear_rate * 8.0 if self.config.amygdala_enabled else 0.0
            self.vocal_gate.vars["I_input"].view[:] = fear_inhibition
            self.vocal_gate.vars["I_input"].push_to_device()

        # === Phase 18: WM Update Gate I_input 주입 ===
        if self.config.wm_expansion_enabled:
            gate_signal = (
                self.last_dopamine_rate * self.config.wm_gate_dopamine_scale +
                self.last_novelty_rate * self.config.wm_gate_novelty_scale +
                self.last_acc_conflict_rate * self.config.wm_gate_conflict_scale
            )
            self.wm_update_gate.vars["I_input"].view[:] = gate_signal
            self.wm_update_gate.vars["I_input"].push_to_device()

        # === Phase 19: Meta Evaluate I_input 주입 ===
        if self.config.metacognition_enabled:
            meta_eval_signal = (
                self.last_meta_uncertainty_rate * self.config.meta_eval_uncertainty_scale +
                self.last_meta_confidence_rate * self.config.meta_eval_confidence_scale +
                self.last_dopamine_rate * self.config.meta_eval_dopamine_scale
            )
            self.meta_evaluate.vars["I_input"].view[:] = meta_eval_signal
            self.meta_evaluate.vars["I_input"].push_to_device()

        # === Phase 20: Self-Model I_input 주입 ===
        if self.config.self_model_enabled:
            # Self_Body: 내수용감각 (에너지/배고픔/포만)
            self_body_signal = (
                energy * self.config.self_body_energy_scale +
                self.last_hunger_rate * self.config.self_body_hunger_scale +
                self.last_satiety_rate * self.config.self_body_satiety_scale
            )
            self.self_body.vars["I_input"].view[:] = self_body_signal
            self.self_body.vars["I_input"].push_to_device()

            # Self_Predict: 순행 모델
            # L14: 순수 efference 예측 (food_eye 제거 → Agency PE가 prediction error 계산)
            if self.config.agency_detection_enabled:
                self_predict_signal = (
                    self.last_self_efference_rate * self.config.self_predict_efference_scale
                )
            else:
                food_eye_signal = (food_l + food_r) * 0.5
                self_predict_signal = (
                    self.last_self_efference_rate * self.config.self_predict_efference_scale +
                    food_eye_signal * self.config.self_predict_food_eye_scale
                )
            self.self_predict.vars["I_input"].view[:] = self_predict_signal
            self.self_predict.vars["I_input"].push_to_device()

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

        # Phase 4 스파이크 카운트 (Phase L2: D1/D2 MSN)
        striatum_spikes = 0  # 호환용 (D1+D2 합산)
        d1_l_spikes = 0
        d1_r_spikes = 0
        d2_l_spikes = 0
        d2_r_spikes = 0
        direct_spikes = 0
        direct_l_spikes = 0
        direct_r_spikes = 0
        indirect_spikes = 0
        indirect_l_spikes = 0
        indirect_r_spikes = 0
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

        # Phase 17 스파이크 카운트 (Language Circuit)
        wernicke_food_spikes = 0
        wernicke_danger_spikes = 0
        wernicke_social_spikes = 0
        wernicke_context_spikes = 0
        broca_food_spikes = 0
        broca_danger_spikes = 0
        broca_social_spikes = 0
        broca_sequence_spikes = 0
        vocal_gate_spikes = 0
        call_mirror_spikes = 0
        call_binding_spikes = 0

        # Phase 18 스파이크 카운트 (WM Expansion)
        wm_thalamic_spikes = 0
        wm_update_gate_spikes = 0
        temporal_recent_spikes = 0
        temporal_prior_spikes = 0
        goal_pending_spikes = 0
        goal_switch_spikes = 0
        wm_context_binding_spikes = 0
        wm_inhibitory_spikes = 0

        # Phase 19 스파이크 카운트 (Metacognition)
        meta_confidence_spikes = 0
        meta_uncertainty_spikes = 0
        meta_evaluate_spikes = 0
        meta_arousal_mod_spikes = 0
        meta_inhibitory_spikes = 0

        # Phase 20 스파이크 카운트 (Self-Model)
        self_body_spikes = 0
        self_efference_spikes = 0
        self_predict_spikes = 0
        self_agency_spikes = 0
        self_narrative_spikes = 0
        self_inhibitory_sm_spikes = 0

        # Phase L14 스파이크 카운트 (Agency PE)
        agency_pe_spikes = 0

        # Phase L6 스파이크 카운트 (Prediction Error)
        pe_food_l_spikes = 0
        pe_food_r_spikes = 0
        pe_danger_l_spikes = 0
        pe_danger_r_spikes = 0

        # Phase L10 스파이크 카운트
        nac_value_spikes = 0

        # Phase L12: GW 스파이크 카운트
        gw_food_l_spikes = 0
        gw_food_r_spikes = 0
        gw_safety_spikes = 0

        # Phase C4: Prediction 스파이크 카운트
        pred_food_spikes = 0

        # Phase C5: Curiosity 스파이크 카운트
        curiosity_spikes = 0

        # M4: Context Gate 스파이크 카운트
        ctx_a_spikes = 0
        ctx_b_spikes = 0

        # M3: Uncertainty Gate 스파이크 카운트
        surprise_spikes = 0

        # === Phase 11: 청각 입력 (Sound → A1) — sensitivity 절반으로 재활성화 ===
        if self.config.auditory_enabled and hasattr(self, 'sound_danger_left'):
            sound_sensitivity = 20.0

            sd_l = np.mean(observation.get("sound_danger_left", np.zeros(4)))
            sd_r = np.mean(observation.get("sound_danger_right", np.zeros(4)))
            sf_l = np.mean(observation.get("sound_food_left", np.zeros(4)))
            sf_r = np.mean(observation.get("sound_food_right", np.zeros(4)))
            self.sound_danger_left.vars["I_input"].view[:] = sd_l * sound_sensitivity
            self.sound_danger_right.vars["I_input"].view[:] = sd_r * sound_sensitivity
            self.sound_food_left.vars["I_input"].view[:] = sf_l * sound_sensitivity
            self.sound_food_right.vars["I_input"].view[:] = sf_r * sound_sensitivity
            self.sound_danger_left.vars["I_input"].push_to_device()
            self.sound_danger_right.vars["I_input"].push_to_device()
            self.sound_food_left.vars["I_input"].push_to_device()
            self.sound_food_right.vars["I_input"].push_to_device()

            # C1: Food sound cues → A1 (고음=음식 안전, 저음=음식 위험)
            food_sound_high = observation.get("food_sound_high", 0.0)
            food_sound_low = observation.get("food_sound_low", 0.0)
            if food_sound_high > 0 or food_sound_low > 0:
                food_sound_sens = 30.0
                self.a1_food.vars["I_input"].view[:] += food_sound_high * food_sound_sens
                self.a1_food.vars["I_input"].push_to_device()
                self.a1_danger.vars["I_input"].view[:] += food_sound_low * food_sound_sens
                self.a1_danger.vars["I_input"].push_to_device()

        # === 시뮬레이션 10스텝 실행 (spike_recording으로 배치 수집) ===
        for _ in range(10):
            self.model.step_time()

        # 한 번의 GPU→CPU 전송으로 모든 population의 스파이크 데이터 수집
        self.model.pull_recording_buffers_from_device()

        # 스파이크 카운팅 (Phase 2a) — spike_recording_data[0] = times array
        # DEBUG: spike recording 검증 (첫 5스텝만)
        motor_left_spikes = len(self.motor_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
        motor_right_spikes = len(self.motor_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
        hunger_spikes = len(self.hunger_drive.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
        satiety_spikes = len(self.satiety_drive.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
        low_energy_spikes = len(self.low_energy_sensor.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
        high_energy_spikes = len(self.high_energy_sensor.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 2b 스파이크 카운팅
        if self.config.amygdala_enabled:
            la_spikes = len(self.lateral_amygdala.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            cea_spikes = len(self.central_amygdala.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            fear_spikes = len(self.fear_response.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 3 스파이크 카운팅
        if self.config.hippocampus_enabled:
            place_cell_spikes = len(self.place_cells.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

            if self.config.directional_food_memory:
                food_memory_spikes = (len(self.food_memory_left.spike_recording_data[0][0])
                                      + len(self.food_memory_right.spike_recording_data[0][0]))
            elif self.food_memory is not None:
                food_memory_spikes = len(self.food_memory.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 4 스파이크 카운팅 (Phase L2: D1/D2 MSN)
        if self.config.basal_ganglia_enabled:
            d1_l_spikes = len(self.d1_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            d1_r_spikes = len(self.d1_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            d2_l_spikes = len(self.d2_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            d2_r_spikes = len(self.d2_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            striatum_spikes = d1_l_spikes + d1_r_spikes + d2_l_spikes + d2_r_spikes  # 호환용
            direct_l_spikes = len(self.direct_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            direct_r_spikes = len(self.direct_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            direct_spikes = direct_l_spikes + direct_r_spikes
            indirect_l_spikes = len(self.indirect_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            indirect_r_spikes = len(self.indirect_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            indirect_spikes = indirect_l_spikes + indirect_r_spikes
            dopamine_spikes = len(self.dopamine_neurons.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

            # Phase L10: NAc spike counting
            if self.config.td_learning_enabled:
                nac_value_spikes = len(self.nac_value.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

            # Phase L12: GW spike counting
            if self.config.gw_enabled:
                gw_food_l_spikes = len(self.gw_food_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
                gw_food_r_spikes = len(self.gw_food_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
                gw_safety_spikes = len(self.gw_safety.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

            # Phase C4: Prediction spike counting
            if self.config.contextual_prediction_enabled and hasattr(self, 'pred_food_soon'):
                pred_food_spikes = len(self.pred_food_soon.spike_recording_data[0][0])

            # Phase C5: Curiosity spike counting
            if self.config.curiosity_enabled and hasattr(self, 'curiosity_gate'):
                curiosity_spikes = len(self.curiosity_gate.spike_recording_data[0][0])

            # M4: Context Gate spike counting
            if self.config.context_gate_enabled and hasattr(self, 'ctx_a'):
                ctx_a_spikes = len(self.ctx_a.spike_recording_data[0][0])
                ctx_b_spikes = len(self.ctx_b.spike_recording_data[0][0])

            # M3: Uncertainty Gate spike counting
            if self.config.uncertainty_gate_enabled and hasattr(self, 'surprise_accum'):
                surprise_spikes = len(self.surprise_accum.spike_recording_data[0][0])

        # Phase 5 스파이크 카운팅
        if self.config.prefrontal_enabled:
            working_memory_spikes = len(self.working_memory.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            goal_food_spikes = len(self.goal_food.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            goal_safety_spikes = len(self.goal_safety.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            inhibitory_spikes = len(self.inhibitory_control.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 6a 스파이크 카운팅
        if self.config.cerebellum_enabled:
            granule_spikes = len(self.granule_cells.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            purkinje_spikes = len(self.purkinje_cells.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            deep_nuclei_spikes = len(self.deep_nuclei.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            error_spikes = len(self.error_signal.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 6b 스파이크 카운팅
        if self.config.thalamus_enabled:
            food_relay_spikes = len(self.food_relay.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            danger_relay_spikes = len(self.danger_relay.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            trn_spikes = len(self.trn.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            arousal_spikes = len(self.arousal.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 8 스파이크 카운팅 (V1)
        if self.config.v1_enabled:
            v1_food_left_spikes = len(self.v1_food_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            v1_food_right_spikes = len(self.v1_food_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            v1_danger_left_spikes = len(self.v1_danger_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            v1_danger_right_spikes = len(self.v1_danger_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 9 스파이크 카운팅 (V2/V4)
        if self.config.v2v4_enabled and self.config.v1_enabled:
            v2_edge_food_spikes = len(self.v2_edge_food.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            v2_edge_danger_spikes = len(self.v2_edge_danger.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            v4_food_object_spikes = len(self.v4_food_object.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            v4_danger_object_spikes = len(self.v4_danger_object.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            v4_novel_object_spikes = len(self.v4_novel_object.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 10 스파이크 카운팅 (IT Cortex)
        if self.config.it_enabled and self.config.v2v4_enabled:
            it_food_category_spikes = len(self.it_food_category.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

            # Phase L9: IT_Food 활성도 캐싱 (trace 누적용)
            if self.config.it_bg_enabled:
                n_it_f = self.config.n_it_food_category
                self._it_food_active = 1.0 if (it_food_category_spikes / max(n_it_f, 1)) > 0.05 else 0.0

            it_danger_category_spikes = len(self.it_danger_category.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            it_neutral_category_spikes = len(self.it_neutral_category.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            it_association_spikes = len(self.it_association.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            it_memory_buffer_spikes = len(self.it_memory_buffer.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 11 스파이크 카운팅 (Auditory Cortex)
        if self.config.auditory_enabled:
            a1_danger_spikes = len(self.a1_danger.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            a1_food_spikes = len(self.a1_food.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            a2_association_spikes = len(self.a2_association.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 12 스파이크 카운팅 (Multimodal Integration)
        if self.config.multimodal_enabled:
            sts_food_spikes = len(self.sts_food.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            sts_danger_spikes = len(self.sts_danger.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            sts_congruence_spikes = len(self.sts_congruence.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            sts_mismatch_spikes = len(self.sts_mismatch.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 13 스파이크 카운팅 (Parietal Cortex)
        if self.config.parietal_enabled:
            ppc_space_left_spikes = len(self.ppc_space_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            ppc_space_right_spikes = len(self.ppc_space_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            ppc_goal_food_spikes = len(self.ppc_goal_food.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            ppc_goal_safety_spikes = len(self.ppc_goal_safety.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            ppc_attention_spikes = len(self.ppc_attention.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            ppc_path_buffer_spikes = len(self.ppc_path_buffer.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 14 스파이크 카운팅 (Premotor Cortex)
        if self.config.premotor_enabled:
            pmd_left_spikes = len(self.pmd_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            pmd_right_spikes = len(self.pmd_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            pmv_approach_spikes = len(self.pmv_approach.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            pmv_avoid_spikes = len(self.pmv_avoid.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            sma_sequence_spikes = len(self.sma_sequence.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            motor_prep_spikes = len(self.motor_preparation.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 15 스파이크 카운팅 (Social Brain)
        if self.config.social_brain_enabled:
            sts_social_spikes = len(self.sts_social.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            tpj_self_spikes = len(self.tpj_self.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            tpj_other_spikes = len(self.tpj_other.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            tpj_compare_spikes = len(self.tpj_compare.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            acc_conflict_spikes = len(self.acc_conflict.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            acc_monitor_spikes = len(self.acc_monitor.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            social_approach_spikes = len(self.social_approach.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            social_avoid_spikes = len(self.social_avoid.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

            # Phase 15b 스파이크 카운팅 (Mirror Neurons)
            if self.config.mirror_enabled:
                social_obs_spikes = len(self.social_observation.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
                mirror_food_spikes = len(self.mirror_food.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
                vicarious_reward_spikes = len(self.vicarious_reward.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
                social_memory_spikes = len(self.social_memory.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

            # Phase 15c 스파이크 카운팅 (Theory of Mind)
            if self.config.tom_enabled:
                tom_intention_spikes = len(self.tom_intention.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
                tom_belief_spikes = len(self.tom_belief.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
                tom_prediction_spikes = len(self.tom_prediction.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
                tom_surprise_spikes = len(self.tom_surprise.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
                coop_spikes = len(self.coop_compete_coop.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
                compete_spikes = len(self.coop_compete_compete.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 16 스파이크 카운팅 (Association Cortex)
        if self.config.association_cortex_enabled:
            assoc_edible_spikes = len(self.assoc_edible.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            assoc_threatening_spikes = len(self.assoc_threatening.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            assoc_animate_spikes = len(self.assoc_animate.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            assoc_context_spikes = len(self.assoc_context.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            assoc_valence_spikes = len(self.assoc_valence.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            assoc_binding_spikes = len(self.assoc_binding.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            assoc_novelty_spikes = len(self.assoc_novelty.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 17 스파이크 카운팅 (Language Circuit)
        if self.config.language_enabled:
            wernicke_food_spikes = len(self.wernicke_food.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            wernicke_danger_spikes = len(self.wernicke_danger.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            wernicke_social_spikes = len(self.wernicke_social.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            wernicke_context_spikes = len(self.wernicke_context.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            broca_food_spikes = len(self.broca_food.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            broca_danger_spikes = len(self.broca_danger.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            broca_social_spikes = len(self.broca_social.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            broca_sequence_spikes = len(self.broca_sequence.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            vocal_gate_spikes = len(self.vocal_gate.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            call_mirror_spikes = len(self.call_mirror.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            call_binding_spikes = len(self.call_binding.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 18 스파이크 카운팅 (WM Expansion)
        if self.config.wm_expansion_enabled:
            wm_thalamic_spikes = len(self.wm_thalamic.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            wm_update_gate_spikes = len(self.wm_update_gate.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            temporal_recent_spikes = len(self.temporal_recent.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            temporal_prior_spikes = len(self.temporal_prior.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            goal_pending_spikes = len(self.goal_pending.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            goal_switch_spikes = len(self.goal_switch.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            wm_context_binding_spikes = len(self.wm_context_binding.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            wm_inhibitory_spikes = len(self.wm_inhibitory.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 19: Metacognition 스파이크 카운팅
        if self.config.metacognition_enabled:
            meta_confidence_spikes = len(self.meta_confidence.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            meta_uncertainty_spikes = len(self.meta_uncertainty.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            meta_evaluate_spikes = len(self.meta_evaluate.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            meta_arousal_mod_spikes = len(self.meta_arousal_mod.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            meta_inhibitory_spikes = len(self.meta_inhibitory_pop.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase 20: Self-Model 스파이크 카운팅
        if self.config.self_model_enabled:
            self_body_spikes = len(self.self_body.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            self_efference_spikes = len(self.self_efference.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            self_predict_spikes = len(self.self_predict.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            self_agency_spikes = len(self.self_agency.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            self_narrative_spikes = len(self.self_narrative.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            self_inhibitory_sm_spikes = len(self.self_inhibitory_sm.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase L14 스파이크 카운팅 (Agency PE)
        if self.config.agency_detection_enabled and hasattr(self, 'agency_pe'):
            agency_pe_spikes = len(self.agency_pe.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

        # Phase L16: KC 스파이크 카운팅 (single KC)
        kc_l_spikes = kc_r_spikes = 0
        if self.config.sparse_expansion_enabled and hasattr(self, 'kc_left'):
            kc_l_spikes = len(self.kc_left.spike_recording_data[0][0])
            kc_r_spikes = len(self.kc_right.spike_recording_data[0][0])

        # Phase L6 스파이크 카운팅 (Prediction Error)
        if self.config.prediction_error_enabled and self.config.v1_enabled and self.config.it_enabled:
            pe_food_l_spikes = len(self.pe_food_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            pe_food_r_spikes = len(self.pe_food_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            pe_danger_l_spikes = len(self.pe_danger_left.spike_recording_data[0][0])  # [0]=first batch, [0]=times array
            pe_danger_r_spikes = len(self.pe_danger_right.spike_recording_data[0][0])  # [0]=first batch, [0]=times array

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
        self.last_hunger_rate = hunger_rate
        self.last_satiety_rate = satiety_rate

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
            self.last_fear_rate = fear_rate

        # Phase 3 스파이크율
        place_cell_rate = 0.0
        food_memory_rate = 0.0
        if self.config.hippocampus_enabled:
            max_spikes_place = self.config.n_place_cells * 5
            max_spikes_food_memory = self.config.n_food_memory * 5

            place_cell_rate = place_cell_spikes / max_spikes_place
            food_memory_rate = food_memory_spikes / max_spikes_food_memory

        # Phase 4 스파이크율 (Phase L2: D1/D2 MSN)
        striatum_rate = 0.0
        d1_l_rate = 0.0
        d1_r_rate = 0.0
        d2_l_rate = 0.0
        d2_r_rate = 0.0
        direct_rate = 0.0
        direct_l_rate = 0.0
        direct_r_rate = 0.0
        indirect_rate = 0.0
        dopamine_rate = 0.0
        if self.config.basal_ganglia_enabled:
            n_d1_half = self.config.n_d1_msn // 2
            n_d2_half = self.config.n_d2_msn // 2
            n_dir_half = self.config.n_direct_pathway // 2
            n_ind_half = self.config.n_indirect_pathway // 2
            n_msn_total = self.config.n_d1_msn + self.config.n_d2_msn
            max_spikes_direct = self.config.n_direct_pathway * 5
            max_spikes_indirect = self.config.n_indirect_pathway * 5
            max_spikes_dopamine = self.config.n_dopamine * 5

            d1_l_rate = d1_l_spikes / (n_d1_half * 5)
            d1_r_rate = d1_r_spikes / (n_d1_half * 5)
            d2_l_rate = d2_l_spikes / (n_d2_half * 5)
            d2_r_rate = d2_r_spikes / (n_d2_half * 5)
            striatum_rate = striatum_spikes / (n_msn_total * 5)  # 호환용
            direct_rate = direct_spikes / max_spikes_direct
            direct_l_rate = direct_l_spikes / (n_dir_half * 5)
            direct_r_rate = direct_r_spikes / (n_dir_half * 5)
            indirect_rate = indirect_spikes / max_spikes_indirect
            dopamine_rate = dopamine_spikes / max_spikes_dopamine
            self.last_dopamine_rate = dopamine_rate
            self.last_d1_l_rate = d1_l_rate
            self.last_d1_r_rate = d1_r_rate

            # Phase L3: R-STDP 적격 추적 업데이트 (D1: 강화, D2: 약화)
            food_l_active = 1.0 if food_l > 0.05 else 0.0
            food_r_active = 1.0 if food_r > 0.05 else 0.0
            d1_l_active = 1.0 if d1_l_rate > 0.05 else 0.0
            d1_r_active = 1.0 if d1_r_rate > 0.05 else 0.0
            trace_max = self.config.rstdp_trace_max
            self.rstdp_trace_l = min(self.rstdp_trace_l * self.config.rstdp_trace_decay + food_l_active * d1_l_active, trace_max)
            self.rstdp_trace_r = min(self.rstdp_trace_r * self.config.rstdp_trace_decay + food_r_active * d1_r_active, trace_max)

            # Phase L4: D2 Anti-Hebbian 적격 추적
            # 생물학적 근거: D2 LTD는 pre-synaptic(food_eye) 활동 + 도파민으로 발생
            # D1↔D2 경쟁으로 D2가 억제되어도, food_eye가 활성이면 trace 누적
            self.rstdp_d2_trace_l = min(self.rstdp_d2_trace_l * self.config.rstdp_trace_decay + food_l_active, trace_max)
            self.rstdp_d2_trace_r = min(self.rstdp_d2_trace_r * self.config.rstdp_trace_decay + food_r_active, trace_max)

            # Phase L7: d1_active 캐싱 (L7 trace에서 BG 블록 밖에서 사용)
            self._d1_l_active = d1_l_active
            self._d1_r_active = d1_r_active

            # Phase L10: NAc rate 계산 + trace 업데이트
            if self.config.td_learning_enabled:
                nac_value_rate = nac_value_spikes / max(self.config.n_nac_value * 5, 1)
                self._nac_value_rate = nac_value_rate

                nac_active = 1.0 if nac_value_rate > 0.05 else 0.0
                td = self.config.rstdp_trace_decay
                tm = self.config.rstdp_trace_max
                self.nac_trace_l = min(self.nac_trace_l * td + food_l_active * nac_active, tm)
                self.nac_trace_r = min(self.nac_trace_r * td + food_r_active * nac_active, tm)

            # Phase L16: KC rate & trace (single KC)
            if self.config.sparse_expansion_enabled and hasattr(self, 'kc_left'):
                n_kc = self.config.n_kc_per_side

                kc_l_rate = kc_l_spikes / max(n_kc * 10, 1)
                kc_r_rate = kc_r_spikes / max(n_kc * 10, 1)
                self.last_kc_l_rate = kc_l_rate
                self.last_kc_r_rate = kc_r_rate

                # Homeostatic KC inhibition: PI control (target ~5%)
                kc_target = 0.05
                avg_rate = (kc_l_rate + kc_r_rate) / 2.0
                error = avg_rate - kc_target
                integral = getattr(self, '_kc_inh_integral', 0.0) + error * 0.1
                integral = max(0.0, min(integral, 50.0))
                self._kc_inh_integral = integral
                drive = max(0.0, error * 100.0 + integral)
                self.kc_inh_left.vars["I_input"].view[:] = drive
                self.kc_inh_left.vars["I_input"].push_to_device()
                self.kc_inh_right.vars["I_input"].view[:] = drive
                self.kc_inh_right.vars["I_input"].push_to_device()

                # Traces
                d1_l_active_kc = 1.0 if d1_l_rate > 0.05 else 0.0
                d1_r_active_kc = 1.0 if d1_r_rate > 0.05 else 0.0
                trace_decay = self.config.rstdp_trace_decay
                trace_max = self.config.rstdp_trace_max

                kc_l_active = 1.0 if kc_l_rate > 0.03 else 0.0
                kc_r_active = 1.0 if kc_r_rate > 0.03 else 0.0
                # D1 trace (pre×post)
                self.kc_d1_trace_l = min(self.kc_d1_trace_l * trace_decay + kc_l_active * d1_l_active_kc, trace_max)
                self.kc_d1_trace_r = min(self.kc_d1_trace_r * trace_decay + kc_r_active * d1_r_active_kc, trace_max)
                # D2 trace (pre-synaptic only)
                self.kc_d2_trace_l = min(self.kc_d2_trace_l * trace_decay + kc_l_active, trace_max)
                self.kc_d2_trace_r = min(self.kc_d2_trace_r * trace_decay + kc_r_active, trace_max)

                # C1: Eligibility bridge — sound onset → slow-decay tag → KC D1 trace
                sound_food_l = observation.get("sound_food_left", 0.0)
                sound_food_r = observation.get("sound_food_right", 0.0)
                food_sound_high = observation.get("food_sound_high", 0.0)
                if isinstance(sound_food_l, np.ndarray):
                    sound_food_l = float(np.mean(sound_food_l))
                if isinstance(sound_food_r, np.ndarray):
                    sound_food_r = float(np.mean(sound_food_r))
                sound_on = max(sound_food_l, sound_food_r, food_sound_high)
                if sound_on > 0.2:
                    tag_l = getattr(self, '_sound_elig_tag_l', 0.0)
                    tag_r = getattr(self, '_sound_elig_tag_r', 0.0)
                    self._sound_elig_tag_l = min(tag_l * 0.995 + sound_food_l * 0.5, trace_max)
                    self._sound_elig_tag_r = min(tag_r * 0.995 + sound_food_r * 0.5, trace_max)
                else:
                    self._sound_elig_tag_l = getattr(self, '_sound_elig_tag_l', 0.0) * 0.995
                    self._sound_elig_tag_r = getattr(self, '_sound_elig_tag_r', 0.0) * 0.995
                # Tag를 KC D1 trace에 주입 (도파민이 올 때 학습됨)
                self.kc_d1_trace_l = min(self.kc_d1_trace_l + self._sound_elig_tag_l * 0.1, trace_max)
                self.kc_d1_trace_r = min(self.kc_d1_trace_r + self._sound_elig_tag_r * 0.1, trace_max)

        # Phase L12: GW rate + broadcast
        gw_food_l_rate = gw_food_r_rate = gw_safety_rate = 0.0
        gw_broadcast = "neutral"
        if self.config.gw_enabled:
            gw_food_l_rate = gw_food_l_spikes / (self.config.n_gw_food * 10)
            gw_food_r_rate = gw_food_r_spikes / (self.config.n_gw_food * 10)
            gw_safety_rate = gw_safety_spikes / (self.config.n_gw_safety * 10)
            gw_food_rate = (gw_food_l_rate + gw_food_r_rate) / 2
            if gw_safety_rate > 0.15 and gw_safety_rate > gw_food_rate:
                gw_broadcast = "safety"
            elif gw_food_rate > 0.08:
                gw_broadcast = "food"
            self.last_gw_food_rate = gw_food_rate
            self.last_gw_safety_rate = gw_safety_rate
            self.last_gw_broadcast = gw_broadcast

        # Phase C4: Prediction rate + food teacher signal
        if self.config.contextual_prediction_enabled and hasattr(self, 'pred_food_soon'):
            pred_rate = pred_food_spikes / max(self.config.n_pred_food_soon * 10, 1)
            self.last_pred_food_rate = pred_rate

            # Food teacher signal: food is visible (food_rays > threshold)
            food_l = np.mean(observation.get("food_rays_left", np.zeros(8)))
            food_r = np.mean(observation.get("food_rays_right", np.zeros(8)))
            self._pred_food_teacher_active = (food_l > 0.15 or food_r > 0.15)

        # M3: Place→Place transition learning (online STDP)
        if self.config.place_transition_enabled and hasattr(self, 'w_pp'):
            current_place = self.last_active_place_cells  # shape: (n_place_cells,)
            if self.prev_place_activation is not None and len(current_place) == self.config.n_place_cells:
                # STDP: active_t → active_t+1 연결 강화
                prev = self.prev_place_activation
                prev_active = np.where(prev > 0.1)[0]  # 이전 활성 place cells
                curr_active = np.where(current_place > 0.1)[0]  # 현재 활성 place cells
                if len(prev_active) > 0 and len(curr_active) > 0:
                    # 매 50스텝마다 배치 업데이트 (GPU 전송 최소화)
                    if not hasattr(self, '_transition_update_counter'):
                        self._transition_update_counter = 0
                        self._transition_delta = np.zeros((self.config.n_place_cells, self.config.n_place_cells))
                    # 누적
                    eta = self.config.place_transition_eta
                    for pi in prev_active[:5]:  # 상위 5개만 (비용 제한)
                        for ci in curr_active[:5]:
                            if pi != ci:
                                self._transition_delta[pi, ci] += eta
                    self._transition_update_counter += 1
                    if self._transition_update_counter >= 50:
                        # CPU only — no GPU pull/push needed
                        self.w_pp += self._transition_delta
                        self.w_pp -= 0.001 * self.w_pp  # LTD
                        np.clip(self.w_pp, 0.0, self.config.place_transition_w_max, out=self.w_pp)
                        self._transition_delta[:] = 0.0
                        self._transition_update_counter = 0

                # Transition buffer 기록 (음식이 보일 때)
                food_vis_l = np.mean(observation.get("food_rays_left", np.zeros(8)))
                food_vis_r = np.mean(observation.get("food_rays_right", np.zeros(8)))
                if food_vis_l > 0.2 or food_vis_r > 0.2:
                    self.transition_buffer.append((prev_active.tolist(), curr_active.tolist(), 1.0))
                    if len(self.transition_buffer) > 100:
                        self.transition_buffer = self.transition_buffer[-100:]

            self.prev_place_activation = current_place.copy() if len(current_place) == self.config.n_place_cells else None

        # Phase C5: Curiosity rate
        if self.config.curiosity_enabled and hasattr(self, 'curiosity_gate'):
            curiosity_rate = curiosity_spikes / max(self.config.n_curiosity_gate * 10, 1)
            self.last_curiosity_rate = curiosity_rate

        # M4: Context Gate — I_input from agent position + rate computation
        if self.config.context_gate_enabled and hasattr(self, 'ctx_a'):
            # position_x (0~1) → zone 판별 → I_input
            pos_x = observation.get("position_x", 0.5)
            # 왼쪽(0)이면 CtxA, 오른쪽(1)이면 CtxB
            left_signal = max(0.0, 1.0 - pos_x * 2.0) * 30.0   # pos<0.5 → 0~30
            right_signal = max(0.0, pos_x * 2.0 - 1.0) * 30.0  # pos>0.5 → 0~30
            self.ctx_a.vars["I_input"].view[:] = left_signal
            self.ctx_a.vars["I_input"].push_to_device()
            self.ctx_b.vars["I_input"].view[:] = right_signal
            self.ctx_b.vars["I_input"].push_to_device()

            self.last_ctx_a_rate = ctx_a_spikes / max(self.config.n_ctx_a * 10, 1)
            self.last_ctx_b_rate = ctx_b_spikes / max(self.config.n_ctx_b * 10, 1)

            # Context tracking + DA-gated ctxval learning
            self._current_ctx = "a" if pos_x < 0.5 else "b"

            # Food eye × context weight → CtxVal I_input (CPU computation)
            food_l = np.mean(observation.get("food_rays_left", np.zeros(8)))
            food_r = np.mean(observation.get("food_rays_right", np.zeros(8)))
            ctx = self._current_ctx
            # 현재 context의 CtxVal만 food signal 받음 (반대쪽은 0)
            ctx_a_gate = 1.0 if ctx == "a" else 0.0
            ctx_b_gate = 1.0 if ctx == "b" else 0.0
            w_a_l = float(np.mean(self._ctxval_w["a_l"]))
            w_a_r = float(np.mean(self._ctxval_w["a_r"]))
            w_b_l = float(np.mean(self._ctxval_w["b_l"]))
            w_b_r = float(np.mean(self._ctxval_w["b_r"]))
            self.ctx_val_a_l.vars["I_input"].view[:] = food_l * w_a_l * ctx_a_gate
            self.ctx_val_a_l.vars["I_input"].push_to_device()
            self.ctx_val_a_r.vars["I_input"].view[:] = food_r * w_a_r * ctx_a_gate
            self.ctx_val_a_r.vars["I_input"].push_to_device()
            self.ctx_val_b_l.vars["I_input"].view[:] = food_l * w_b_l * ctx_b_gate
            self.ctx_val_b_l.vars["I_input"].push_to_device()
            self.ctx_val_b_r.vars["I_input"].view[:] = food_r * w_b_r * ctx_b_gate
            self.ctx_val_b_r.vars["I_input"].push_to_device()

            # Context-specific food value learning (food 근처에서 매 스텝)
            # 좋은 음식 근처(good_food_rays 높음) → weight 증가
            # 나쁜 음식 근처(bad_food_rays 높음) → weight 감소
            good_l = np.mean(observation.get("good_food_rays_left", np.zeros(8)))
            good_r = np.mean(observation.get("good_food_rays_right", np.zeros(8)))
            bad_l = np.mean(observation.get("bad_food_rays_left", np.zeros(8)))
            bad_r = np.mean(observation.get("bad_food_rays_right", np.zeros(8)))
            eta_ctx = 0.0005  # 느린 학습 (매 스텝)
            food_signal_l = good_l - bad_l  # +면 좋은 음식 우세, -면 나쁜 음식 우세
            food_signal_r = good_r - bad_r
            if abs(food_signal_l) > 0.05 or abs(food_signal_r) > 0.05:
                self._ctxval_w[f"{ctx}_l"] += eta_ctx * food_signal_l
                self._ctxval_w[f"{ctx}_r"] += eta_ctx * food_signal_r
                np.clip(self._ctxval_w[f"{ctx}_l"], 0.1, 8.0, out=self._ctxval_w[f"{ctx}_l"])
                np.clip(self._ctxval_w[f"{ctx}_r"], 0.1, 8.0, out=self._ctxval_w[f"{ctx}_r"])

        # M3: Uncertainty Gate rate → eta modulation
        if self.config.uncertainty_gate_enabled and hasattr(self, 'surprise_accum'):
            surprise_rate = surprise_spikes / max(self.config.n_surprise_accum * 10, 1)
            self.last_surprise_rate = surprise_rate
            # ACh-like modulation: high surprise → boost learning rate (1.0 ~ 2.0x)
            # Low surprise (stable) → normal learning (1.0x)
            self.uncertainty_eta_mod = 1.0 + min(1.0, surprise_rate * 5.0)

        # Phase L5: 피질 R-STDP 적격 추적 (좋은/나쁜 음식 활성도 기반)
        if self.config.perceptual_learning_enabled:
            good_food_l_active = 1.0 if good_food_l > 0.05 else 0.0
            good_food_r_active = 1.0 if good_food_r > 0.05 else 0.0
            bad_food_l_active = 1.0 if bad_food_l > 0.05 else 0.0
            bad_food_r_active = 1.0 if bad_food_r > 0.05 else 0.0
            ct_decay = self.config.cortical_rstdp_trace_decay
            ct_max = self.config.cortical_rstdp_trace_max
            self.cortical_trace_good_l = min(self.cortical_trace_good_l * ct_decay + good_food_l_active, ct_max)
            self.cortical_trace_good_r = min(self.cortical_trace_good_r * ct_decay + good_food_r_active, ct_max)
            self.cortical_trace_bad_l = min(self.cortical_trace_bad_l * ct_decay + bad_food_l_active, ct_max)
            self.cortical_trace_bad_r = min(self.cortical_trace_bad_r * ct_decay + bad_food_r_active, ct_max)

            # 항상성 감쇠: 50 스텝마다 (BG R-STDP와 동일 패턴)
            if self.config.cortical_rstdp_weight_decay > 0 and self._cortical_step % 50 == 0:
                c_decay = self.config.cortical_rstdp_weight_decay
                c_rest = self.config.cortical_rstdp_w_rest
                c_w_max = self.config.cortical_rstdp_w_max
                c_w_min = self.config.cortical_rstdp_w_min
                for syn in [self.good_food_to_it_food_l, self.good_food_to_it_food_r,
                            self.good_food_to_it_danger_l, self.good_food_to_it_danger_r,
                            self.bad_food_to_it_danger_l, self.bad_food_to_it_danger_r,
                            self.bad_food_to_it_food_l, self.bad_food_to_it_food_r]:
                    syn.vars["g"].pull_from_device()
                    w = syn.vars["g"].values
                    w[:] -= (c_decay * 50) * (w - c_rest)
                    w[:] = np.clip(w, c_w_min, c_w_max)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()

        # Phase L7: Discriminative BG 적격 추적 (good/bad food → D1/D2)
        if (self.config.discriminative_bg_enabled and self.config.perceptual_learning_enabled
                and self.config.basal_ganglia_enabled):
            td = self.config.rstdp_trace_decay
            tm = self.config.rstdp_trace_max
            _d1_l = getattr(self, '_d1_l_active', 0.0)
            _d1_r = getattr(self, '_d1_r_active', 0.0)
            # L5 블록에서 계산된 good/bad food active 변수 사용
            _gfl = good_food_l_active if self.config.perceptual_learning_enabled else 0.0
            _gfr = good_food_r_active if self.config.perceptual_learning_enabled else 0.0
            _bfl = bad_food_l_active if self.config.perceptual_learning_enabled else 0.0
            _bfr = bad_food_r_active if self.config.perceptual_learning_enabled else 0.0
            # D1 traces: pre(typed food) × post(D1)
            self.typed_d1_trace_good_l = min(self.typed_d1_trace_good_l * td + _gfl * _d1_l, tm)
            self.typed_d1_trace_good_r = min(self.typed_d1_trace_good_r * td + _gfr * _d1_r, tm)
            self.typed_d1_trace_bad_l = min(self.typed_d1_trace_bad_l * td + _bfl * _d1_l, tm)
            self.typed_d1_trace_bad_r = min(self.typed_d1_trace_bad_r * td + _bfr * _d1_r, tm)
            # D2 traces: pre-only (D1↔D2 경쟁으로 D2 발화 억제되므로)
            self.typed_d2_trace_good_l = min(self.typed_d2_trace_good_l * td + _gfl, tm)
            self.typed_d2_trace_good_r = min(self.typed_d2_trace_good_r * td + _gfr, tm)
            self.typed_d2_trace_bad_l = min(self.typed_d2_trace_bad_l * td + _bfl, tm)
            self.typed_d2_trace_bad_r = min(self.typed_d2_trace_bad_r * td + _bfr, tm)

        # Phase L9: IT→D1/D2 적격 추적
        if (self.config.it_bg_enabled and self.config.it_enabled
                and self.config.basal_ganglia_enabled):
            td = self.config.rstdp_trace_decay
            tm = self.config.rstdp_trace_max
            _d1_l = getattr(self, '_d1_l_active', 0.0)
            _d1_r = getattr(self, '_d1_r_active', 0.0)
            _it_f = getattr(self, '_it_food_active', 0.0)

            # D1: pre(IT_Food) × post(D1) → 방향성 있는 trace
            self.it_food_d1_trace_l = min(self.it_food_d1_trace_l * td + _it_f * _d1_l, tm)
            self.it_food_d1_trace_r = min(self.it_food_d1_trace_r * td + _it_f * _d1_r, tm)

            # D2: pre-synaptic only (L4 Anti-Hebbian 패턴)
            self.it_food_d2_trace_l = min(self.it_food_d2_trace_l * td + _it_f, tm)
            self.it_food_d2_trace_r = min(self.it_food_d2_trace_r * td + _it_f, tm)

        # Phase L6: 예측 오차 스파이크율 + 적격 추적
        pe_food_l_rate = 0.0
        pe_food_r_rate = 0.0
        pe_danger_l_rate = 0.0
        pe_danger_r_rate = 0.0
        if self.config.prediction_error_enabled and self.config.v1_enabled and self.config.it_enabled:
            n_pe_food_half = self.config.n_pe_food // 2
            n_pe_danger_half = self.config.n_pe_danger // 2
            pe_food_l_rate = pe_food_l_spikes / (n_pe_food_half * 5)
            pe_food_r_rate = pe_food_r_spikes / (n_pe_food_half * 5)
            pe_danger_l_rate = pe_danger_l_spikes / (n_pe_danger_half * 5)
            pe_danger_r_rate = pe_danger_r_spikes / (n_pe_danger_half * 5)

            self._pe_step += 1

            # PE 적격 추적: PE가 발화하면 trace 누적
            pe_food_l_active = 1.0 if pe_food_l_rate > 0.05 else 0.0
            pe_food_r_active = 1.0 if pe_food_r_rate > 0.05 else 0.0
            pe_danger_l_active = 1.0 if pe_danger_l_rate > 0.05 else 0.0
            pe_danger_r_active = 1.0 if pe_danger_r_rate > 0.05 else 0.0
            pe_td = self.config.pe_trace_decay
            pe_tm = self.config.pe_trace_max
            self.pe_trace_food_l = min(self.pe_trace_food_l * pe_td + pe_food_l_active, pe_tm)
            self.pe_trace_food_r = min(self.pe_trace_food_r * pe_td + pe_food_r_active, pe_tm)
            self.pe_trace_danger_l = min(self.pe_trace_danger_l * pe_td + pe_danger_l_active, pe_tm)
            self.pe_trace_danger_r = min(self.pe_trace_danger_r * pe_td + pe_danger_r_active, pe_tm)

            # 항상성 감쇠: 50 스텝마다
            if self.config.pe_weight_decay > 0 and self._pe_step % 50 == 0:
                pe_decay = self.config.pe_weight_decay
                pe_rest = self.config.pe_w_rest
                pe_wmax = self.config.pe_to_it_w_max
                pe_wmin = self.config.pe_to_it_w_min
                for syn in [self.pe_food_to_it_food_l, self.pe_food_to_it_food_r,
                            self.pe_danger_to_it_danger_l, self.pe_danger_to_it_danger_r]:
                    syn.vars["g"].pull_from_device()
                    w = syn.vars["g"].values
                    w[:] -= (pe_decay * 50) * (w - pe_rest)
                    w[:] = np.clip(w, pe_wmin, pe_wmax)
                    syn.vars["g"].values = w
                    syn.vars["g"].push_to_device()

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
            self.last_acc_conflict_rate = acc_conflict_rate

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
            self.last_novelty_rate = assoc_novelty_rate

        # Phase 17 스파이크율 (Language Circuit)
        wernicke_food_rate = 0.0
        wernicke_danger_rate = 0.0
        wernicke_social_rate = 0.0
        wernicke_context_rate = 0.0
        broca_food_rate = 0.0
        broca_danger_rate = 0.0
        broca_social_rate = 0.0
        broca_sequence_rate = 0.0
        vocal_gate_rate = 0.0
        call_mirror_rate = 0.0
        call_binding_rate = 0.0
        if self.config.language_enabled:
            wernicke_food_rate = wernicke_food_spikes / (self.config.n_wernicke_food * 5)
            wernicke_danger_rate = wernicke_danger_spikes / (self.config.n_wernicke_danger * 5)
            wernicke_social_rate = wernicke_social_spikes / (self.config.n_wernicke_social * 5)
            wernicke_context_rate = wernicke_context_spikes / (self.config.n_wernicke_context * 5)
            broca_food_rate = broca_food_spikes / (self.config.n_broca_food * 5)
            broca_danger_rate = broca_danger_spikes / (self.config.n_broca_danger * 5)
            broca_social_rate = broca_social_spikes / (self.config.n_broca_social * 5)
            broca_sequence_rate = broca_sequence_spikes / (self.config.n_broca_sequence * 5)
            vocal_gate_rate = vocal_gate_spikes / (self.config.n_vocal_gate * 5)
            call_mirror_rate = call_mirror_spikes / (self.config.n_call_mirror * 5)
            call_binding_rate = call_binding_spikes / (self.config.n_call_binding * 5)

            self.last_wernicke_food_rate = wernicke_food_rate
            self.last_wernicke_danger_rate = wernicke_danger_rate
            self.last_broca_food_rate = broca_food_rate
            self.last_broca_danger_rate = broca_danger_rate
            self.last_vocal_gate_rate = vocal_gate_rate
            self.last_call_binding_rate = call_binding_rate

            # Vocalize type 결정: Broca + Vocal Gate
            self.vocalize_type = 0
            if vocal_gate_rate > 0.05:
                if broca_food_rate > broca_danger_rate and broca_food_rate > 0.05:
                    self.vocalize_type = 1  # food call
                elif broca_danger_rate > 0.05:
                    self.vocalize_type = 2  # danger call

        # Phase 18 스파이크율 (WM Expansion)
        wm_thalamic_rate = 0.0
        wm_update_gate_rate = 0.0
        temporal_recent_rate = 0.0
        temporal_prior_rate = 0.0
        goal_pending_rate = 0.0
        goal_switch_rate = 0.0
        wm_context_binding_rate = 0.0
        wm_inhibitory_rate = 0.0
        if self.config.wm_expansion_enabled:
            wm_thalamic_rate = wm_thalamic_spikes / (self.config.n_wm_thalamic * 5)
            wm_update_gate_rate = wm_update_gate_spikes / (self.config.n_wm_update_gate * 5)
            temporal_recent_rate = temporal_recent_spikes / (self.config.n_temporal_recent * 5)
            temporal_prior_rate = temporal_prior_spikes / (self.config.n_temporal_prior * 5)
            goal_pending_rate = goal_pending_spikes / (self.config.n_goal_pending * 5)
            goal_switch_rate = goal_switch_spikes / (self.config.n_goal_switch * 5)
            wm_context_binding_rate = wm_context_binding_spikes / (self.config.n_wm_context_binding * 5)
            wm_inhibitory_rate = wm_inhibitory_spikes / (self.config.n_wm_inhibitory * 5)

            self.last_wm_thalamic_rate = wm_thalamic_rate
            self.last_wm_update_gate_rate = wm_update_gate_rate
            self.last_temporal_recent_rate = temporal_recent_rate
            self.last_temporal_prior_rate = temporal_prior_rate
            self.last_goal_pending_rate = goal_pending_rate
            self.last_goal_switch_rate = goal_switch_rate
            self.last_wm_context_binding_rate = wm_context_binding_rate
            self.last_wm_inhibitory_rate = wm_inhibitory_rate

        # Phase 19 스파이크율 (Metacognition)
        meta_confidence_rate = 0.0
        meta_uncertainty_rate = 0.0
        meta_evaluate_rate = 0.0
        meta_arousal_mod_rate = 0.0
        meta_inhibitory_rate = 0.0
        if self.config.metacognition_enabled:
            meta_confidence_rate = meta_confidence_spikes / (self.config.n_meta_confidence * 5)
            meta_uncertainty_rate = meta_uncertainty_spikes / (self.config.n_meta_uncertainty * 5)
            meta_evaluate_rate = meta_evaluate_spikes / (self.config.n_meta_evaluate * 5)
            meta_arousal_mod_rate = meta_arousal_mod_spikes / (self.config.n_meta_arousal_mod * 5)
            meta_inhibitory_rate = meta_inhibitory_spikes / (self.config.n_meta_inhibitory * 5)

            self.last_meta_confidence_rate = meta_confidence_rate
            self.last_meta_uncertainty_rate = meta_uncertainty_rate
            self.last_meta_evaluate_rate = meta_evaluate_rate
            self.last_meta_arousal_mod_rate = meta_arousal_mod_rate
            self.last_meta_inhibitory_rate = meta_inhibitory_rate

        # Phase 20 스파이크율 (Self-Model)
        self_body_rate = 0.0
        self_efference_rate = 0.0
        self_predict_rate = 0.0
        self_agency_rate = 0.0
        self_narrative_rate = 0.0
        self_inhibitory_sm_rate = 0.0
        if self.config.self_model_enabled:
            self_body_rate = self_body_spikes / (self.config.n_self_body * 5)
            self_efference_rate = self_efference_spikes / (self.config.n_self_efference * 5)
            self_predict_rate = self_predict_spikes / (self.config.n_self_predict * 5)
            self_agency_rate = self_agency_spikes / (self.config.n_self_agency * 5)
            self_narrative_rate = self_narrative_spikes / (self.config.n_self_narrative * 5)
            self_inhibitory_sm_rate = self_inhibitory_sm_spikes / (self.config.n_self_inhibitory * 5)

            # Phase L15: Track previous body rate for Δbody change detection
            self.prev_self_body_rate = self.last_self_body_rate
            self.last_self_body_rate = self_body_rate
            self.last_self_efference_rate = self_efference_rate
            self.last_self_predict_rate = self_predict_rate
            self.last_self_agency_rate = self_agency_rate
            self.last_self_narrative_rate = self_narrative_rate
            self.last_self_inhibitory_rate = self_inhibitory_sm_rate

        # Phase L14 Agency PE rate
        agency_pe_rate = 0.0
        if self.config.agency_detection_enabled and hasattr(self, 'agency_pe'):
            agency_pe_rate = agency_pe_spikes / (self.config.n_agency_pe * 5)
            self.last_agency_pe_rate = agency_pe_rate

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

            # Phase 4 뉴런 활성화 (Phase L2: D1/D2 MSN)
            "striatum_rate": striatum_rate,  # 호환용 (D1+D2 평균)
            "d1_rate": (d1_l_rate + d1_r_rate) / 2,
            "d1_l_rate": d1_l_rate,
            "d1_r_rate": d1_r_rate,
            "d2_rate": (d2_l_rate + d2_r_rate) / 2,
            "d2_l_rate": d2_l_rate,
            "d2_r_rate": d2_r_rate,
            "direct_rate": direct_rate,
            "direct_l_rate": direct_l_rate,
            "direct_r_rate": direct_r_rate,
            "indirect_rate": indirect_rate,
            "dopamine_rate": dopamine_rate,
            "rstdp_trace_l": getattr(self, 'rstdp_trace_l', 0.0),
            "rstdp_trace_r": getattr(self, 'rstdp_trace_r', 0.0),
            "rstdp_d2_trace_l": getattr(self, 'rstdp_d2_trace_l', 0.0),
            "rstdp_d2_trace_r": getattr(self, 'rstdp_d2_trace_r', 0.0),
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

            # Phase 17 입력 (Language Circuit)
            "npc_call_food_l": npc_call_food_l,
            "npc_call_food_r": npc_call_food_r,
            "npc_call_danger_l": npc_call_danger_l,
            "npc_call_danger_r": npc_call_danger_r,

            # Phase 17 뉴런 활성화 (Language Circuit)
            "wernicke_food_rate": wernicke_food_rate,
            "wernicke_danger_rate": wernicke_danger_rate,
            "wernicke_social_rate": wernicke_social_rate,
            "wernicke_context_rate": wernicke_context_rate,
            "broca_food_rate": broca_food_rate,
            "broca_danger_rate": broca_danger_rate,
            "broca_social_rate": broca_social_rate,
            "broca_sequence_rate": broca_sequence_rate,
            "vocal_gate_rate": vocal_gate_rate,
            "call_mirror_rate": call_mirror_rate,
            "call_binding_rate": call_binding_rate,
            "vocalize_type": self.vocalize_type if self.config.language_enabled else 0,

            # Phase 18: WM Expansion
            "wm_thalamic_rate": wm_thalamic_rate,
            "wm_update_gate_rate": wm_update_gate_rate,
            "temporal_recent_rate": temporal_recent_rate,
            "temporal_prior_rate": temporal_prior_rate,
            "goal_pending_rate": goal_pending_rate,
            "goal_switch_rate": goal_switch_rate,
            "wm_context_binding_rate": wm_context_binding_rate,
            "wm_inhibitory_rate": wm_inhibitory_rate,

            # Phase 19: Metacognition
            "meta_confidence_rate": meta_confidence_rate,
            "meta_uncertainty_rate": meta_uncertainty_rate,
            "meta_evaluate_rate": meta_evaluate_rate,
            "meta_arousal_mod_rate": meta_arousal_mod_rate,
            "meta_inhibitory_rate": meta_inhibitory_rate,

            # Phase 20: Self-Model
            "self_body_rate": self_body_rate,
            "self_efference_rate": self_efference_rate,
            "self_predict_rate": self_predict_rate,
            "self_agency_rate": self_agency_rate,
            "self_narrative_rate": self_narrative_rate,
            "self_inhibitory_sm_rate": self_inhibitory_sm_rate,

            # Phase L14: Agency Detection
            "agency_pe_rate": agency_pe_rate,

            # Phase L12: Global Workspace
            "gw_food_l_rate": gw_food_l_rate,
            "gw_food_r_rate": gw_food_r_rate,
            "gw_safety_rate": gw_safety_rate,
            "gw_broadcast": gw_broadcast,

            # Phase L6: Prediction Error
            "pe_food_l_rate": pe_food_l_rate,
            "pe_food_r_rate": pe_food_r_rate,
            "pe_danger_l_rate": pe_danger_l_rate,
            "pe_danger_r_rate": pe_danger_r_rate,

            # 에이전트 위치 (Place Cell 시각화용)
            "agent_grid_x": int(observation.get("position_x", 0.5) * 10),  # 0~10 그리드
            "agent_grid_y": int(observation.get("position_y", 0.5) * 10),

            # 출력
            "angle_delta": angle_delta,

            # 학습 가중치 (실시간 그래프용)
            "learning_weights": {
                "D1_RSTDP": (self._last_rstdp_results.get("rstdp_avg_w_left", 0.0)
                             + self._last_rstdp_results.get("rstdp_avg_w_right", 0.0)) / 2.0
                             if self._last_rstdp_results else 0.0,
                "Hippo": self._last_hippo_avg_w,
                "Garcia": self._last_garcia_avg_w,
                "KC_D1": (self._last_rstdp_results.get("kc_d1_l", 0.0)
                          + self._last_rstdp_results.get("kc_d1_r", 0.0)) / 2.0
                          if self._last_rstdp_results else 0.0,
                "Pred_Place": self._last_rstdp_results.get("pred_place_w", 0.0)
                              if self._last_rstdp_results else 0.0,
            },
            "pred_food_rate": self.last_pred_food_rate,
            "curiosity_rate": self.last_curiosity_rate,
            "surprise_rate": self.last_surprise_rate,
            "eta_mod": self.uncertainty_eta_mod,
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

            # Hunger와 Fear 모두 높음 (경쟁 — 로그 비활성, 불필요한 스팸 방지)

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

        # Phase L11: SWR Gate (SensoryLIF — I_input 있음)
        if self.config.swr_replay_enabled and self.config.hippocampus_enabled:
            sensory_pops.append(self.swr_gate)

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

        # Phase L11: CA3 + Replay Inhibitory
        if self.config.swr_replay_enabled and self.config.hippocampus_enabled:
            lif_pops.extend([self.ca3_sequence, self.replay_inhibitory])

        # Phase L12: GW populations
        if self.config.gw_enabled:
            lif_pops.extend([self.gw_food_left, self.gw_food_right, self.gw_safety])

        # Phase 4: Basal Ganglia 추가 (Phase L2: D1/D2 MSN)
        if self.config.basal_ganglia_enabled:
            lif_pops.extend([self.d1_left, self.d1_right,
                           self.d2_left, self.d2_right,
                           self.direct_left, self.direct_right,
                           self.indirect_left, self.indirect_right])

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
                no_association: bool = False, no_language: bool = False,
                no_wm_expansion: bool = False, no_metacognition: bool = False,
                no_self_model: bool = False,
                no_predator: bool = False,
                no_agency: bool = False,
                no_narrative_self: bool = False,
                no_sparse_expansion: bool = False,
                no_prediction: bool = False,
                no_curiosity: bool = False,
                log_data: bool = False, log_dir: str = None,
                log_sample_rate: int = 5,
                save_weights: str = None, load_weights: str = None):
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
    if no_language:
        brain_config.language_enabled = False
        env_config.npc_vocalization_enabled = False
        print("  [!] Phase 17 (Language Circuit) DISABLED")
    if no_wm_expansion:
        brain_config.wm_expansion_enabled = False
        print("  [!] Phase 18 (WM Expansion) DISABLED")
    if no_metacognition:
        brain_config.metacognition_enabled = False
        print("  [!] Phase 19 (Metacognition) DISABLED")
    if no_self_model:
        brain_config.self_model_enabled = False
        print("  [!] Phase 20 (Self-Model) DISABLED")
    if no_predator:
        env_config.predator_enabled = False
        print("  [!] Predators DISABLED")
    if no_agency:
        brain_config.agency_detection_enabled = False
        env_config.motor_noise_enabled = False
        env_config.sensor_jitter_enabled = False
        print("  [!] Phase L14 (Agency Detection) DISABLED")
    if no_narrative_self:
        brain_config.narrative_self_enabled = False
        print("  [!] Phase L15 (Narrative Self) DISABLED")
    if no_sparse_expansion:
        brain_config.sparse_expansion_enabled = False
        print("  [!] Phase L16 (Sparse Expansion) DISABLED")
    if no_prediction:
        brain_config.contextual_prediction_enabled = False
        print("  [!] Phase C4 (Contextual Prediction) DISABLED")
    if no_curiosity:
        brain_config.curiosity_enabled = False
        print("  [!] Phase C5 (Curiosity) DISABLED")
    if food_patch:
        env_config.food_patch_enabled = True
        print(f"      Patches: {env_config.n_patches}, radius={env_config.patch_radius}")
        print(f"      Spawn in patch: {env_config.food_spawn_in_patch_prob*100:.0f}%")

    env = ForagerGym(env_config, render_mode=render_mode)
    env.render_fps = fps  # FPS 설정 (시각화 속도 조절)
    brain = ForagerBrain(brain_config)

    # Data logging for dashboard
    logger = None
    if log_data:
        logger = DataLogger(log_dir=log_dir, sample_rate=log_sample_rate)
        logger.log_config(brain_config, env_config, episodes)
        print(f"  [LOG] Data logging enabled → {logger.log_dir}")

    # 가중치 로드 (학습된 모델 시각화용)
    if load_weights:
        brain.load_all_weights(load_weights)

    # 학습 비활성화 옵션
    if no_learning:
        brain.food_learning_enabled = False

    # 통계
    all_steps = []
    all_food = []
    all_homeostasis = []
    all_pain_visits = []
    all_pain_steps = []
    all_wall_bounces_in_pain = []
    all_avg_dist_to_pain = []
    all_max_pain_dwell = []
    all_avg_pain_dwell = []
    all_pain_approach_pct = []
    death_causes = {"starve": 0, "timeout": 0, "pain": 0, "predator": 0}

    # Phase 3b: 학습 통계
    all_learn_events = []  # 총 학습 이벤트 수

    # Food Patch 통계
    all_patch_visits = []   # 에피소드별 [patch0_visits, patch1_visits, ...]
    all_patch_food = []     # 에피소드별 [patch0_food, patch1_food, ...]

    # 행동 진단 집계
    all_straight_pct = []           # 에피소드별 직진 비율
    all_max_straight_streak = []    # 에피소드별 최대 연속 직진
    all_food_correct_pct = []       # 에피소드별 음식 올바른 전환율
    all_pain_escape_pct = []        # 에피소드별 pain 올바른 탈출율
    all_angle_std = []              # 에피소드별 angle_delta 표준편차

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

        # === 행동 진단 (Behavior Diagnostics) ===
        ep_angle_deltas = []          # 매 스텝 angle_delta 기록
        ep_motor_left_rates = []      # Motor L 발화율
        ep_motor_right_rates = []     # Motor R 발화율
        ep_food_detect_count = 0      # 음식 감지 횟수 (food_l > 0.05 or food_r > 0.05)
        ep_food_correct_turn = 0      # 음식 방향으로 올바른 전환 횟수
        ep_food_wrong_turn = 0        # 음식 반대 방향 전환
        ep_food_no_turn = 0           # 음식 감지했으나 전환 없음
        ep_pain_in_steps = 0          # pain zone 내 총 스텝
        ep_pain_correct_escape = 0    # pain zone에서 올바른 탈출 방향 전환
        ep_pain_wrong_dir = 0         # pain zone에서 잘못된 방향 (더 깊이 진입)
        ep_turn_left = 0              # 좌회전 횟수 (angle_delta < -0.02)
        ep_turn_right = 0             # 우회전 횟수 (angle_delta > 0.02)
        ep_straight = 0               # 직진 횟수 (|angle_delta| <= 0.02)
        ep_max_consecutive_straight = 0  # 최대 연속 직진 스텝
        ep_current_straight_streak = 0   # 현재 연속 직진 카운터

        while not done:
            # 뇌 처리
            action_delta, info = brain.process(obs, debug=debug)
            action = (action_delta,)

            # Phase 4: Dopamine 감쇠 (매 스텝)
            brain.decay_dopamine()

            # Phase 17: 발성 타입 전달
            if brain_config.language_enabled:
                env._agent_call_type = brain.vocalize_type

            # 시각화를 위해 뇌 정보 전달 (render 전에 설정)
            env.set_brain_info(info)

            # 환경 스텝
            obs, reward, done, env_info = env.step(action)
            total_reward += reward

            # Phase L1: 도파민 셰이핑 리워드 (음식 접근 시)
            approach_signal = env_info.get("food_approach_signal", 0.0)
            if approach_signal > 0.01 and brain_config.basal_ganglia_enabled:
                brain.release_dopamine(reward_magnitude=0.1 * approach_signal)

            # C1: Food sound incentive salience (소리 자체가 작은 도파민)
            # 생물학적 근거: incentive salience — 음식 관련 감각 단서가 도파민 유발
            food_sound_high = obs.get("food_sound_high", 0.0)
            if food_sound_high > 0.3 and brain_config.basal_ganglia_enabled:
                brain.release_dopamine(reward_magnitude=0.05 * food_sound_high)  # 보조만, 메인 해법 아님

            # 통계 수집
            ep_hunger_rates.append(info["hunger_rate"])
            ep_satiety_rates.append(info["satiety_rate"])
            if brain_config.amygdala_enabled:
                ep_fear_rates.append(info["fear_rate"])

            # === 행동 진단: 매 스텝 추적 ===
            ml_rate = info["motor_left_rate"]
            mr_rate = info["motor_right_rate"]
            ad = action_delta  # angle_delta
            ep_angle_deltas.append(ad)
            ep_motor_left_rates.append(ml_rate)
            ep_motor_right_rates.append(mr_rate)

            # 방향 전환 분류 (threshold: 0.02)
            if ad < -0.02:
                ep_turn_left += 1
                ep_current_straight_streak = 0
            elif ad > 0.02:
                ep_turn_right += 1
                ep_current_straight_streak = 0
            else:
                ep_straight += 1
                ep_current_straight_streak += 1
                ep_max_consecutive_straight = max(ep_max_consecutive_straight, ep_current_straight_streak)

            # 음식 감지 반응 분석
            fl = info.get("food_l", 0)
            fr = info.get("food_r", 0)
            food_threshold = 0.05
            if fl > food_threshold or fr > food_threshold:
                ep_food_detect_count += 1
                food_side = "LEFT" if fl > fr else "RIGHT"
                # 올바른 반응: 음식이 왼쪽이면 좌회전(angle_delta < 0), 오른쪽이면 우회전(angle_delta > 0)
                if food_side == "LEFT" and ad < -0.01:
                    ep_food_correct_turn += 1
                elif food_side == "RIGHT" and ad > 0.01:
                    ep_food_correct_turn += 1
                elif abs(ad) <= 0.01:
                    ep_food_no_turn += 1
                else:
                    ep_food_wrong_turn += 1

                # 음식 감지 이벤트 로그 (첫 50회만)
                if ep_food_detect_count <= 50 and log_level in ["debug", "verbose"]:
                    print(f"  [FOOD_DETECT] step={env.steps} food_L={fl:.3f} food_R={fr:.3f} "
                          f"→ M_L={ml_rate:.3f} M_R={mr_rate:.3f} angle={ad:.4f} "
                          f"({'CORRECT' if (food_side=='LEFT' and ad<-0.01) or (food_side=='RIGHT' and ad>0.01) else 'MISS'})")

            # Pain zone 반응 분석
            if env_info.get('in_pain', False):
                ep_pain_in_steps += 1
                pl = info.get("pain_l", 0)
                pr = info.get("pain_r", 0)
                # Pain Push-Pull: pain_L → Motor_R(push) + Motor_L(pull)
                # 올바른 탈출: pain_L > pain_R이면 우회전(angle_delta > 0), 반대도 마찬가지
                if pl > pr and ad > 0.01:
                    ep_pain_correct_escape += 1
                elif pr > pl and ad < -0.01:
                    ep_pain_correct_escape += 1
                elif (pl > pr and ad < -0.01) or (pr > pl and ad > 0.01):
                    ep_pain_wrong_dir += 1

                # Pain 반응 로그 (첫 진입 40스텝만)
                if ep_pain_in_steps <= 40 and log_level in ["normal", "debug", "verbose"]:
                    print(f"  [PAIN_RESPONSE] step={env.steps} pain_L={pl:.3f} pain_R={pr:.3f} "
                          f"→ M_L={ml_rate:.3f} M_R={mr_rate:.3f} angle={ad:.4f} "
                          f"({'ESCAPE' if (pl>pr and ad>0.01) or (pr>pl and ad<-0.01) else 'STUCK'})")

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
                eaten_food_type = env_info.get("food_type", 0)
                learn_info = None
                dopamine_info = None
                assoc_learn = None
                call_learn = None
                wm_ctx_learn = None
                meta_learn = None
                sm_learn = None

                # M4: Context-specific food value update (reward-time)
                if brain_config.context_gate_enabled and hasattr(brain, '_ctxval_w'):
                    ctx = brain._current_ctx
                    reward_sign = 1.0 if eaten_food_type == 0 else -1.0  # good=+1, bad=-1
                    eta_ctx = 0.15  # 매우 강한 학습 (음식 먹은 순간)
                    for side in ["l", "r"]:
                        key = f"{ctx}_{side}"
                        brain._ctxval_w[key] += eta_ctx * reward_sign
                        np.clip(brain._ctxval_w[key], 0.1, 8.0, out=brain._ctxval_w[key])

                if eaten_food_type == 0:  # === 좋은 음식: 도파민 + 기존 학습 ===
                    # Phase 3b/3c: Hebbian 학습 실행
                    food_pos = (obs["position_x"], obs["position_y"])
                    learn_info = brain.learn_food_location(food_position=food_pos)
                    if learn_info:
                        ep_learn_events += 1

                    # Phase 4: Dopamine 방출 (보상) - Phase L1: 0.5→1.0, Phase L10: RPE 모듈레이션
                    dopamine_info = brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)

                    # Phase L5: 피질 R-STDP (좋은 음식 학습)
                    if brain_config.perceptual_learning_enabled and brain_config.it_enabled:
                        cortical_learn = brain.update_cortical_rstdp("good_food")

                    # Phase L6: PE R-STDP (음식 예측 오차 → IT 정교화)
                    if brain_config.prediction_error_enabled:
                        brain.update_prediction_error_rstdp("food")

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

                    # Phase 17: Call Binding 학습 (food call 듣고 음식 찾기 = 강한 학습)
                    if brain_config.language_enabled:
                        heard_food_call = info.get("npc_call_food_l", 0) > 0.01 or info.get("npc_call_food_r", 0) > 0.01
                        call_learn = brain.learn_call_binding(reward_context=heard_food_call)

                    # Phase 18: WM Context 학습 (음식 먹기 = 강한 학습)
                    if brain_config.wm_expansion_enabled:
                        wm_ctx_learn = brain.learn_wm_context(reward_context=True)

                    # Phase 19: Metacognitive Confidence 학습 (음식 먹기 = 강한 학습)
                    if brain_config.metacognition_enabled:
                        meta_learn = brain.learn_metacognitive_confidence(reward_context=True)

                    # Phase 20: Self-Narrative 학습 (음식 먹기 = 강한 학습)
                    if brain_config.self_model_enabled:
                        sm_learn = brain.learn_self_narrative(reward_context=True)

                    # Phase L14: Forward Model 학습 (음식 먹기 = 강한 학습)
                    if brain_config.agency_detection_enabled:
                        brain.learn_forward_model(reward_context=True)

                    # Phase L15: Agency→Narrative 학습 (음식 먹기 = 강한 학습)
                    if brain_config.narrative_self_enabled:
                        brain.learn_agency_narrative(reward_context=True)

                    # Phase L11: SWR 경험 버퍼에 좋은 음식 저장
                    if brain_config.swr_replay_enabled and brain_config.hippocampus_enabled:
                        brain.add_experience(food_pos[0], food_pos[1], 0, env.steps, 25.0)

                elif eaten_food_type == 1:  # === 나쁜 음식: 도파민 딥 + 맛 혐오 + 피질 약화 ===
                    # Phase L8: 도파민 딥 → D1 약화 (LTD) + D2 강화 (LTP)
                    if brain_config.dopamine_dip_enabled and brain_config.basal_ganglia_enabled:
                        dopamine_info = brain.release_dopamine(
                            reward_magnitude=-brain_config.dopamine_dip_magnitude)

                    # Phase L5: 피질 R-STDP (나쁜 음식 학습) — 도파민 비의존, 유지
                    if brain_config.perceptual_learning_enabled and brain_config.it_enabled:
                        cortical_learn = brain.update_cortical_rstdp("bad_food")

                    # Phase L5: 맛 혐오 → Amygdala (Garcia Effect) — 편도체 경로, 유지
                    brain.trigger_taste_aversion(0.5)

                    # Phase L13: 조건화된 맛 혐오 Hebbian 학습
                    if brain_config.taste_aversion_learning_enabled:
                        ta_learn = brain.learn_taste_aversion()
                        if ta_learn and log_level in ["debug", "verbose"]:
                            print(f"    [L13] Taste Aversion: L avg_w={ta_learn['avg_w_left']:.3f}, "
                                  f"R avg_w={ta_learn['avg_w_right']:.3f}")

                    # Phase L14: Forward Model 학습 (나쁜 음식 = 강한 학습)
                    if brain_config.agency_detection_enabled:
                        brain.learn_forward_model(reward_context=True)

                    # Phase L15: Agency→Narrative 학습 (나쁜 음식 = 강한 학습)
                    if brain_config.narrative_self_enabled:
                        brain.learn_agency_narrative(reward_context=True)

                    # Phase L11: SWR 경험 버퍼에 나쁜 음식 저장
                    if brain_config.swr_replay_enabled and brain_config.hippocampus_enabled:
                        bad_food_pos = (obs["position_x"], obs["position_y"])
                        brain.add_experience(bad_food_pos[0], bad_food_pos[1], 1, env.steps, -5.0)

                # 공통 로그
                if log_level in ["normal", "debug", "verbose"]:
                    type_str = "GOOD" if eaten_food_type == 0 else "BAD"
                    da_str = f", DA={dopamine_info['dopamine_level']:.2f}" if dopamine_info else ""
                    if learn_info:
                        side_str = f", side={learn_info.get('side', 'N/A')}" if 'side' in learn_info else ""
                        print(f"  [!] {type_str} FOOD at step {env.steps}, Energy: {env_info['energy']:.1f} "
                              f"[LEARN: {learn_info['n_strengthened']} cells, avg_w={learn_info['avg_weight']:.2f}{side_str}{da_str}]")
                    else:
                        print(f"  [!] {type_str} FOOD at step {env.steps}, Energy: {env_info['energy']:.1f}{da_str}")

                # Hebbian logging (food context)
                if logger:
                    if learn_info:
                        logger.log_hebbian(ep, env.steps, "hippo", learn_info.get('avg_weight', 0), "food")
                    if brain_config.association_cortex_enabled and assoc_learn:
                        logger.log_hebbian(ep, env.steps, "assoc_binding", assoc_learn.get('avg_w_edible', 0), "food")
                    if brain_config.language_enabled and call_learn:
                        logger.log_hebbian(ep, env.steps, "call_binding", call_learn.get('avg_w', call_learn.get('avg_weight', 0)), "food")
                    if brain_config.wm_expansion_enabled and wm_ctx_learn:
                        logger.log_hebbian(ep, env.steps, "wm_context", wm_ctx_learn.get('avg_w', 0), "food")
                    if brain_config.metacognition_enabled and meta_learn:
                        logger.log_hebbian(ep, env.steps, "meta_confidence", meta_learn.get('avg_w', 0), "food")
                    if brain_config.self_model_enabled and sm_learn:
                        logger.log_hebbian(ep, env.steps, "self_narrative", sm_learn.get('avg_w', 0), "food")

            # Phase 15b: NPC 먹기 관찰 → 사회적 학습
            if brain_config.social_brain_enabled and brain_config.mirror_enabled:
                npc_events = env_info.get("npc_eating_events", [])
                for npc_x, npc_y, npc_step in npc_events:
                    npc_pos = (npc_x / env_config.width, npc_y / env_config.height)
                    social_learn = brain.learn_social_food_location(npc_pos)
                    if social_learn and log_level in ["debug", "verbose"]:
                        print(f"  [SOCIAL] NPC ate at ({npc_x:.0f},{npc_y:.0f}), "
                              f"avg_w={social_learn['avg_weight']:.2f}, surprise={social_learn['surprise']:.2f}")

            # Phase L6: Pain zone → PE danger 학습 (위험 예측 오차 → IT_Danger 정교화)
            if brain_config.prediction_error_enabled and env_info.get('in_pain', False):
                brain.update_prediction_error_rstdp("danger")

            # Phase 17: Pain zone + danger call → 강한 학습
            if brain_config.language_enabled and env_info.get('in_pain', False):
                heard_danger_call = info.get("npc_call_danger_l", 0) > 0.01 or info.get("npc_call_danger_r", 0) > 0.01
                brain.learn_call_binding(reward_context=heard_danger_call)

            # Phase 18: Pain zone → WM Context 강한 학습
            if brain_config.wm_expansion_enabled and env_info.get('in_pain', False):
                brain.learn_wm_context(reward_context=True)

            # Phase 19: Pain zone → Metacognitive 강한 학습
            if brain_config.metacognition_enabled and env_info.get('in_pain', False):
                brain.learn_metacognitive_confidence(reward_context=True)

            # Phase 20: Pain zone → Self-Narrative 강한 학습
            if brain_config.self_model_enabled and env_info.get('in_pain', False):
                brain.learn_self_narrative(reward_context=True)

            # Phase L14: Pain zone → Forward Model 강한 학습
            if brain_config.agency_detection_enabled and env_info.get('in_pain', False):
                brain.learn_forward_model(reward_context=True)

            # Phase L15: Pain zone → Agency→Narrative 강한 학습
            if brain_config.narrative_self_enabled and env_info.get('in_pain', False):
                brain.learn_agency_narrative(reward_context=True)

            # Pain Zone 진입 이벤트
            if log_level in ["normal", "debug", "verbose"]:
                if env_info.get('in_pain', False) and env.pain_zone_visits == 1 and env.pain_zone_steps == 1:
                    print(f"  [!] ENTERED Pain Zone at step {env.steps}!")

            # Phase 17: 배경 학습 (약한 학습 = 항상)
            if brain_config.language_enabled and env.steps % 5 == 0:
                brain.learn_call_binding(reward_context=False)

            # Phase 18: WM Context 배경 학습 (약한 학습 = 매 5스텝)
            if brain_config.wm_expansion_enabled and env.steps % 5 == 0:
                brain.learn_wm_context(reward_context=False)

            # Phase 19: Metacognitive 배경 학습 (약한 학습 = 매 5스텝)
            if brain_config.metacognition_enabled and env.steps % 5 == 0:
                brain.learn_metacognitive_confidence(reward_context=False)

            # Phase 20: Self-Narrative 배경 학습 (약한 학습 = 매 5스텝)
            if brain_config.self_model_enabled and env.steps % 5 == 0:
                brain.learn_self_narrative(reward_context=False)

            # Phase L14: Forward Model 배경 학습 (약한 학습 = 매 5스텝)
            if brain_config.agency_detection_enabled and env.steps % 5 == 0:
                brain.learn_forward_model(reward_context=False)

            # Phase L15: Agency→Narrative 배경 학습 (약한 학습 = 매 10스텝)
            if brain_config.narrative_self_enabled and env.steps % 10 == 0:
                brain.learn_agency_narrative(reward_context=False)

            # Data logging (sampled every N steps)
            if logger:
                logger.log_step(ep, env.steps, info, env_info)

        # 에피소드 종료
        total_steps_ep = max(1, env.steps)
        std_ad = np.std(ep_angle_deltas) if ep_angle_deltas else 0
        all_steps.append(env.steps)
        all_food.append(env.total_food_eaten)
        all_homeostasis.append(env_info["homeostasis_ratio"])
        all_pain_visits.append(env_info.get("pain_visits", 0))
        all_pain_steps.append(env_info.get("pain_steps", 0))
        all_wall_bounces_in_pain.append(env_info.get("wall_bounces_in_pain", 0))
        all_avg_dist_to_pain.append(env_info.get("avg_dist_to_pain", 0))
        all_max_pain_dwell.append(env_info.get("max_pain_dwell", 0))
        all_avg_pain_dwell.append(env_info.get("avg_pain_dwell", 0))
        _approach = env_info.get("pain_approach_steps", 0)
        all_pain_approach_pct.append(_approach / max(1, env.steps) * 100)
        all_learn_events.append(ep_learn_events)  # Phase 3b

        # 행동 진단 집계
        all_straight_pct.append(100 * ep_straight / total_steps_ep)
        all_max_straight_streak.append(ep_max_consecutive_straight)
        all_food_correct_pct.append(100 * ep_food_correct_turn / max(1, ep_food_detect_count))
        all_pain_escape_pct.append(100 * ep_pain_correct_escape / max(1, ep_pain_in_steps))
        all_angle_std.append(std_ad)

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

        # Phase L11 + M3: SWR Replay (surprise-GATED: suppress when uncertain)
        if brain_config.swr_replay_enabled and brain_config.hippocampus_enabled:
            # M3 수정: surprise 높으면 replay 억제 (stale memory 강화 방지)
            # 생물학적 근거: ACh가 높으면 hippocampal replay 억제, 안정 시에만 replay
            base_replay = brain_config.swr_replay_count  # 기본 5회
            surprise_gate = 1.0  # 1.0 = full replay, 0.0 = no replay
            if brain_config.uncertainty_gate_enabled:
                surprise = brain.last_surprise_rate
                # surprise > 0.1이면 replay 억제 (세계가 바뀌었으니 옛 기억 강화 금지)
                surprise_gate = max(0.0, 1.0 - surprise * 5.0)  # 0.2 rate → 0% replay
            total_replay = max(0, int(base_replay * surprise_gate))
            old_count = brain_config.swr_replay_count
            brain_config.swr_replay_count = total_replay

            replay_info = brain.replay_swr()

            brain_config.swr_replay_count = old_count  # 복원
            if replay_info and replay_info["replayed_count"] > 0:
                print(f"  [SWR] Replayed {replay_info['replayed_count']} experiences "
                      f"(buffer: {replay_info['buffer_size']}, "
                      f"surprise_gate: {surprise_gate:.0%})")
                print(f"  [SWR] Hebbian w: {replay_info['avg_w_before']:.3f} → "
                      f"{replay_info['avg_w_after']:.3f}")

        # 에피소드 요약
        avg_hunger = np.mean(ep_hunger_rates) if ep_hunger_rates else 0
        avg_satiety = np.mean(ep_satiety_rates) if ep_satiety_rates else 0
        avg_fear = np.mean(ep_fear_rates) if ep_fear_rates else 0

        print(f"\n{'='*60}")
        print(f"Episode {ep + 1}/{episodes} Summary")
        print(f"{'='*60}")
        print(f"  Steps:        {env.steps}")
        print(f"  Final Energy: {env_info['energy']:.1f}")
        print(f"  Food Eaten:   {env.total_food_eaten} (Good: {env.good_food_eaten}, Bad: {env.bad_food_eaten})")
        _selectivity = env.good_food_eaten / max(1, env.total_food_eaten)
        print(f"  Selectivity:  {_selectivity:.2f} (good/total)")
        print(f"  Death Cause:  {env_info['death_cause']}")
        print(f"  Reward:       {total_reward:.2f}")
        print(f"  Homeostasis:  {env_info['homeostasis_ratio']*100:.1f}%")
        print(f"  Avg Hunger:   {avg_hunger:.3f}")
        print(f"  Avg Satiety:  {avg_satiety:.3f}")

        if brain_config.amygdala_enabled:
            print(f"  --- Phase 2b: Pain ---")
            print(f"  Avg Fear:     {avg_fear:.3f}")
            print(f"  Pain Visits:  {env_info.get('pain_visits', 0)}")
            print(f"  Pain Time:    {env_info.get('pain_steps', 0)} steps")
            print(f"  Wall Bounce(pain): {env_info.get('wall_bounces_in_pain', 0)}/{env_info.get('wall_bounces_total', 0)}")
            print(f"  Avg Dist→Pain: {env_info.get('avg_dist_to_pain', 0):.1f}px")
            print(f"  Max Dwell:    {env_info.get('max_pain_dwell', 0)} steps")

        if env_config.predator_enabled:
            print(f"  --- Predator ---")
            print(f"  Contact: {env_info.get('predator_contact_steps', 0)} steps")
            print(f"  Damage:  {env_info.get('predator_damage_total', 0):.1f}")

        # === 행동 진단 요약 ===
        print(f"  --- Behavior Diagnostics ---")
        # 모터 출력 분석
        avg_ml = np.mean(ep_motor_left_rates) if ep_motor_left_rates else 0
        avg_mr = np.mean(ep_motor_right_rates) if ep_motor_right_rates else 0
        avg_ad = np.mean(ep_angle_deltas) if ep_angle_deltas else 0
        print(f"  Motor: avg_L={avg_ml:.4f} avg_R={avg_mr:.4f} diff={avg_mr-avg_ml:.4f}")
        print(f"  Angle: avg={avg_ad:.4f} std={std_ad:.4f}")
        print(f"  Turns: LEFT={ep_turn_left} ({100*ep_turn_left/total_steps_ep:.1f}%) "
              f"RIGHT={ep_turn_right} ({100*ep_turn_right/total_steps_ep:.1f}%) "
              f"STRAIGHT={ep_straight} ({100*ep_straight/total_steps_ep:.1f}%)")
        print(f"  Max Consecutive Straight: {ep_max_consecutive_straight} steps")
        # 음식 반응 분석
        if ep_food_detect_count > 0:
            print(f"  Food Detection: {ep_food_detect_count} events")
            print(f"    Correct Turn: {ep_food_correct_turn} ({100*ep_food_correct_turn/ep_food_detect_count:.1f}%)")
            print(f"    Wrong Turn:   {ep_food_wrong_turn} ({100*ep_food_wrong_turn/ep_food_detect_count:.1f}%)")
            print(f"    No Turn:      {ep_food_no_turn} ({100*ep_food_no_turn/ep_food_detect_count:.1f}%)")
        else:
            print(f"  Food Detection: 0 events (NEVER SAW FOOD?)")
        # Pain 반응 분석
        if ep_pain_in_steps > 0:
            print(f"  Pain Response: {ep_pain_in_steps} steps in pain zone")
            print(f"    Correct Escape: {ep_pain_correct_escape} ({100*ep_pain_correct_escape/ep_pain_in_steps:.1f}%)")
            print(f"    Wrong Direction: {ep_pain_wrong_dir} ({100*ep_pain_wrong_dir/ep_pain_in_steps:.1f}%)")
            neutral_pain = ep_pain_in_steps - ep_pain_correct_escape - ep_pain_wrong_dir
            print(f"    Neutral/Equal: {neutral_pain} ({100*neutral_pain/ep_pain_in_steps:.1f}%)")
        else:
            print(f"  Pain Response: 0 steps in pain zone")

        # Phase L4: R-STDP D1/D2 가중치 현황
        if brain_config.basal_ganglia_enabled:
            brain.food_to_d1_l.vars["g"].pull_from_device()
            brain.food_to_d1_r.vars["g"].pull_from_device()
            rstdp_w_l = float(np.nanmean(brain.food_to_d1_l.vars["g"].values))
            rstdp_w_r = float(np.nanmean(brain.food_to_d1_r.vars["g"].values))
            brain.food_to_d2_l.vars["g"].pull_from_device()
            brain.food_to_d2_r.vars["g"].pull_from_device()
            d2_w_l = float(np.nanmean(brain.food_to_d2_l.vars["g"].values))
            d2_w_r = float(np.nanmean(brain.food_to_d2_r.vars["g"].values))
            print(f"  D1 R-STDP: L={rstdp_w_l:.3f} R={rstdp_w_r:.3f} "
                  f"(init={brain_config.food_to_d1_init_weight}, max={brain_config.rstdp_w_max})")
            print(f"  D2 Anti-H: L={d2_w_l:.3f} R={d2_w_r:.3f} "
                  f"(init={brain_config.food_to_d2_weight}, min={brain_config.rstdp_d2_w_min})")

        # Phase L7: Discriminative BG 가중치 현황
        if brain_config.discriminative_bg_enabled and brain_config.perceptual_learning_enabled:
            print(f"  --- Phase L7: Discriminative BG ---")
            for label, syn in [
                ("Good→D1_L", brain.good_food_to_d1_l),
                ("Good→D1_R", brain.good_food_to_d1_r),
                ("Bad→D1_L", brain.bad_food_to_d1_l),
                ("Bad→D1_R", brain.bad_food_to_d1_r),
                ("Good→D2_L", brain.good_food_to_d2_l),
                ("Good→D2_R", brain.good_food_to_d2_r),
                ("Bad→D2_L", brain.bad_food_to_d2_l),
                ("Bad→D2_R", brain.bad_food_to_d2_r),
            ]:
                syn.vars["g"].pull_from_device()
                avg_w = float(np.nanmean(syn.vars["g"].values))
                print(f"    {label}: {avg_w:.3f}")

        # Phase L9: IT→BG 가중치 현황
        if brain_config.it_bg_enabled and brain_config.it_enabled:
            print(f"  --- Phase L9: IT_Food→BG ---")
            for label, syn in [
                ("IT_Food→D1_L", brain.it_food_to_d1_l),
                ("IT_Food→D1_R", brain.it_food_to_d1_r),
                ("IT_Food→D2_L", brain.it_food_to_d2_l),
                ("IT_Food→D2_R", brain.it_food_to_d2_r),
            ]:
                syn.vars["g"].pull_from_device()
                avg_w = float(np.nanmean(syn.vars["g"].values))
                print(f"    {label}: {avg_w:.3f}")

        # Phase L10: NAc Critic 가중치 + RPE 현황
        if brain_config.td_learning_enabled and brain_config.basal_ganglia_enabled:
            print(f"  --- Phase L10: NAc Critic (TD Learning) ---")
            for label, syn in [
                ("Food_Eye→NAc_L", brain.food_to_nac_l),
                ("Food_Eye→NAc_R", brain.food_to_nac_r),
            ]:
                syn.vars["g"].pull_from_device()
                avg_w = float(np.nanmean(syn.vars["g"].values))
                print(f"    {label}: {avg_w:.3f}")
            print(f"    NAc rate: {brain._nac_value_rate:.3f}")

        # Phase L12: Global Workspace 현황
        if brain_config.gw_enabled:
            print(f"  --- Phase L12: Global Workspace ---")
            print(f"    GW Food rate: {brain.last_gw_food_rate:.3f}")
            print(f"    GW Safety rate: {brain.last_gw_safety_rate:.3f}")
            print(f"    Broadcast: {brain.last_gw_broadcast}")

        # Phase L11: SWR Replay 현황
        if brain_config.swr_replay_enabled and brain_config.hippocampus_enabled:
            print(f"  --- Phase L11: SWR Replay ---")
            print(f"    Experience buffer: {len(brain.experience_buffer)} events")
            stats = brain.get_hippocampus_stats()
            if stats:
                print(f"    Hippocampal avg_w: {stats['avg_weight']:.3f}")
                print(f"    Strong connections: {stats['n_strong_connections']}")

        # Phase L13: Taste Aversion Hebbian 가중치 현황
        if brain_config.taste_aversion_learning_enabled and hasattr(brain, 'bad_food_to_la_left'):
            print(f"  --- Phase L13: Taste Aversion (BadFood→LA) ---")
            brain.bad_food_to_la_left.vars["g"].pull_from_device()
            brain.bad_food_to_la_right.vars["g"].pull_from_device()
            ta_l_avg = float(np.mean(brain.bad_food_to_la_left.vars["g"].view))
            ta_r_avg = float(np.mean(brain.bad_food_to_la_right.vars["g"].view))
            ta_l_max = float(np.max(brain.bad_food_to_la_left.vars["g"].view))
            ta_r_max = float(np.max(brain.bad_food_to_la_right.vars["g"].view))
            print(f"    BadFood→LA Left:  avg_w={ta_l_avg:.3f}, max_w={ta_l_max:.3f}")
            print(f"    BadFood→LA Right: avg_w={ta_r_avg:.3f}, max_w={ta_r_max:.3f}")

        # Phase L14: Agency Detection 현황
        if brain_config.agency_detection_enabled and hasattr(brain, 'efference_to_predict_hebbian'):
            print(f"  --- Phase L14: Agency Detection (Forward Model) ---")
            brain.efference_to_predict_hebbian.vars["g"].pull_from_device()
            fm_avg = float(np.mean(brain.efference_to_predict_hebbian.vars["g"].view))
            fm_max = float(np.max(brain.efference_to_predict_hebbian.vars["g"].view))
            print(f"    Forward Model avg_w: {fm_avg:.3f}, max_w: {fm_max:.3f}")
            print(f"    Agency_PE rate: {brain.last_agency_pe_rate:.3f}")
            print(f"    Self_Agency rate: {brain.last_self_agency_rate:.3f}")
            print(f"    Self_Predict rate: {brain.last_self_predict_rate:.3f}")

        # Phase L15: Narrative Self 현황
        if brain_config.narrative_self_enabled and hasattr(brain, 'agency_to_narrative_hebbian'):
            print(f"  --- Phase L15: Narrative Self (Agency-Gated) ---")
            brain.agency_to_narrative_hebbian.vars["g"].pull_from_device()
            an_avg = float(np.mean(brain.agency_to_narrative_hebbian.vars["g"].view))
            an_max = float(np.max(brain.agency_to_narrative_hebbian.vars["g"].view))
            brain.body_to_narrative_hebbian.vars["g"].pull_from_device()
            bn_avg = float(np.mean(brain.body_to_narrative_hebbian.vars["g"].view))
            bn_max = float(np.max(brain.body_to_narrative_hebbian.vars["g"].view))
            print(f"    Agency→Narrative avg_w: {an_avg:.3f}, max_w: {an_max:.3f}")
            print(f"    Body→Narrative avg_w: {bn_avg:.3f}, max_w: {bn_max:.3f}")
            print(f"    Self_Narrative rate: {brain.last_self_narrative_rate:.3f}")

        # Phase L16: Sparse Expansion (KC) 현황 — single KC
        if brain_config.sparse_expansion_enabled and hasattr(brain, 'kc_to_d1_l'):
            print(f"  --- Phase L16: Sparse Expansion (KC, single) ---")
            print(f"    KC rate: L={brain.last_kc_l_rate:.3f} R={brain.last_kc_r_rate:.3f}")
            for label, syn in [
                ("KC→D1_L", brain.kc_to_d1_l),
                ("KC→D1_R", brain.kc_to_d1_r),
                ("KC→D2_L", brain.kc_to_d2_l),
                ("KC→D2_R", brain.kc_to_d2_r),
            ]:
                syn.vars["g"].pull_from_device()
                avg_w = float(np.nanmean(syn.vars["g"].values))
                print(f"    {label} avg_w: {avg_w:.3f}")

        # Phase C4: Contextual Prediction 가중치 현황
        if brain_config.contextual_prediction_enabled and hasattr(brain, 'place_to_pred'):
            print(f"  --- Phase C4: Predictive Plasticity ---")
            print(f"    Pred_FoodSoon rate: {brain.last_pred_food_rate:.3f}")
            brain.place_to_pred.vars["g"].pull_from_device()
            pp_w = brain.place_to_pred.vars["g"].view.copy()
            at_max = np.sum(pp_w >= brain_config.place_to_pred_w_max * 0.95)
            at_zero = np.sum(pp_w < 0.01)
            print(f"    Place→Pred: avg={np.mean(pp_w):.3f}, std={np.std(pp_w):.3f}, "
                  f"max={np.max(pp_w):.3f}, at_ceil={at_max}, at_zero={at_zero}/{len(pp_w)}")
            if hasattr(brain, 'wmcb_to_pred'):
                brain.wmcb_to_pred.vars["g"].pull_from_device()
                wc_w = brain.wmcb_to_pred.vars["g"].view.copy()
                print(f"    WMCB→Pred: avg={np.mean(wc_w):.3f}, std={np.std(wc_w):.3f}")

        # Phase C5: Curiosity 현황
        if brain_config.curiosity_enabled and hasattr(brain, 'curiosity_gate'):
            print(f"  --- Phase C5: Curiosity ---")
            print(f"    Curiosity_Gate rate: {brain.last_curiosity_rate:.3f}")

        # Phase L5: 피질 R-STDP 가중치 현황
        if brain_config.perceptual_learning_enabled and brain_config.it_enabled:
            print(f"  --- Phase L5: Cortical R-STDP ---")
            for label, syn in [
                ("Good→IT_Food_L", brain.good_food_to_it_food_l),
                ("Good→IT_Food_R", brain.good_food_to_it_food_r),
                ("Good→IT_Danger_L", brain.good_food_to_it_danger_l),
                ("Good→IT_Danger_R", brain.good_food_to_it_danger_r),
                ("Bad→IT_Danger_L", brain.bad_food_to_it_danger_l),
                ("Bad→IT_Danger_R", brain.bad_food_to_it_danger_r),
                ("Bad→IT_Food_L", brain.bad_food_to_it_food_l),
                ("Bad→IT_Food_R", brain.bad_food_to_it_food_r),
            ]:
                syn.vars["g"].pull_from_device()
                avg_w = float(np.nanmean(syn.vars["g"].values))
                print(f"    {label}: {avg_w:.3f}")

        # Phase L6: PE→IT 가중치 현황
        if brain_config.prediction_error_enabled and brain_config.v1_enabled and brain_config.it_enabled:
            print(f"  --- Phase L6: Prediction Error R-STDP ---")
            for label, syn in [
                ("PE_Food→IT_Food_L", brain.pe_food_to_it_food_l),
                ("PE_Food→IT_Food_R", brain.pe_food_to_it_food_r),
                ("PE_Danger→IT_Danger_L", brain.pe_danger_to_it_danger_l),
                ("PE_Danger→IT_Danger_R", brain.pe_danger_to_it_danger_r),
            ]:
                syn.vars["g"].pull_from_device()
                avg_w = float(np.nanmean(syn.vars["g"].values))
                print(f"    {label}: {avg_w:.3f}")

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

        # 중간 체크포인트 저장 (50ep마다 + 강제 종료 대비)
        if save_weights and (ep + 1) % 50 == 0:
            base, ext = os.path.splitext(save_weights)
            checkpoint_name = f"{base}_ep{ep+1}{ext}"
            brain.save_all_weights(checkpoint_name)
            print(f"  [CHECKPOINT] Saved at episode {ep+1}: {checkpoint_name}")

        # Episode data logging
        if logger:
            logger.log_episode(ep, {
                "steps": env.steps,
                "food_eaten": env.total_food_eaten,
                "death_cause": env_info["death_cause"],
                "homeostasis": env_info["homeostasis_ratio"],
                "pain_visits": env_info.get("pain_visits", 0),
                "pain_steps": env_info.get("pain_steps", 0),
                "wall_bounces_in_pain": env_info.get("wall_bounces_in_pain", 0),
                "avg_dist_to_pain": round(env_info.get("avg_dist_to_pain", 0), 1),
                "max_pain_dwell": env_info.get("max_pain_dwell", 0),
                "avg_pain_dwell": round(env_info.get("avg_pain_dwell", 0), 1),
                "pain_approach_pct": round(env_info.get("pain_approach_steps", 0) / max(1, env.steps) * 100, 1),
                "avg_hunger": round(avg_hunger, 4),
                "avg_satiety": round(avg_satiety, 4),
                "avg_fear": round(avg_fear, 4),
            })

    # === 최종 요약 ===
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - Final Statistics")
    print("=" * 70)
    print(f"  Episodes:       {episodes}")
    print(f"  Avg Steps:      {np.mean(all_steps):.1f}")
    print(f"  Avg Food:       {np.mean(all_food):.1f}")
    print(f"  Avg Homeostasis:{np.mean(all_homeostasis)*100:.1f}%")
    print(f"  Reward Freq:    {np.sum(all_food) / np.sum(all_steps) * 100:.2f}%")

    # Phase L5: Food Selectivity
    if env_config.n_food_types >= 2:
        total_good = sum(1 for _ in [])  # 에피소드 단위가 아니라 최종 env 기준
        # 최종 에피소드의 good/bad는 env에 남아있음, 전체는 에피소드별 누적 필요
        # 간단히 최종 에피소드 selectivity만 출력
        final_good = env.good_food_eaten
        final_bad = env.bad_food_eaten
        final_total = final_good + final_bad
        final_selectivity = final_good / max(1, final_total)
        print(f"  (Last ep) Food Selectivity: {final_selectivity:.2f} "
              f"(good={final_good}, bad={final_bad})")

    if env_config.pain_zone_enabled:
        pain_pct = np.sum(all_pain_steps) / np.sum(all_steps) * 100
        pain_death_pct = death_causes.get("pain", 0) / episodes * 100
        avg_visits = np.mean(all_pain_visits)
        avg_dist = np.mean(all_avg_dist_to_pain) if all_avg_dist_to_pain else 0
        avg_bounce_in_pain = np.mean(all_wall_bounces_in_pain) if all_wall_bounces_in_pain else 0
        avg_max_dwell = np.mean(all_max_pain_dwell) if all_max_pain_dwell else 0
        avg_approach = np.mean(all_pain_approach_pct) if all_pain_approach_pct else 0

        print(f"\n  === Phase 2b: Pain Zone (Honest Metrics) ===")
        print(f"  Pain Death Rate:    {pain_death_pct:.0f}% ({death_causes.get('pain', 0)}/{episodes})")
        print(f"  Pain Time Ratio:    {pain_pct:.1f}%")
        print(f"  Avg Pain Entries:   {avg_visits:.1f}/ep")
        print(f"  Avg Dist to Pain:   {avg_dist:.1f}px (zone radius: {env_config.pain_zone_radius}px, map: {env_config.width}px)")
        print(f"  Wall Bounce in Pain:{avg_bounce_in_pain:.1f}/ep (exit by wall, not by brain)")
        print(f"  Avg Max Dwell:      {avg_max_dwell:.0f} steps (longest single pain visit)")
        print(f"  Approach Ratio:     {avg_approach:.1f}% of steps moving toward pain")

        if env_config.predator_enabled:
            pred_deaths = death_causes.get("predator", 0)
            print(f"\n  === Predator ===")
            print(f"  Predator Death Rate: {pred_deaths/episodes*100:.0f}% ({pred_deaths}/{episodes})")

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
        pain_death_pct = death_causes.get("pain", 0) / episodes * 100
        avg_visits = np.mean(all_pain_visits)

        # Pain 종합 판정: 3개 지표 교차 검증
        pain_time_ok = pain_pct < 15
        pain_death_ok = pain_death_pct < 20
        pain_entry_ok = avg_visits < 10
        pain_pass = pain_time_ok and pain_death_ok and pain_entry_ok

        print(f"  Pain Composite: {'✓ PASS' if pain_pass else '✗ FAIL'}")
        print(f"    Time in Pain:  {pain_pct:.1f}% {'✓' if pain_time_ok else '✗'} (target: <15%)")
        print(f"    Pain Deaths:   {pain_death_pct:.0f}% {'✓' if pain_death_ok else '✗'} (target: <20%)")
        print(f"    Pain Entries:  {avg_visits:.1f}/ep {'✓' if pain_entry_ok else '✗'} (target: <10/ep)")

        # === 모순 탐지 (Contradiction Alerts) ===
        contradictions = []
        if pain_time_ok and not pain_death_ok:
            contradictions.append(
                f"LOW PAIN TIME ({pain_pct:.1f}%) BUT HIGH PAIN DEATH ({pain_death_pct:.0f}%)"
                f" → Agent enters pain zone briefly but repeatedly, accumulating lethal damage")
        if pain_time_ok and avg_visits > 10:
            contradictions.append(
                f"LOW PAIN TIME ({pain_pct:.1f}%) BUT HIGH ENTRIES ({avg_visits:.0f}/ep)"
                f" → Wall bounce hides repeated entries; brain is NOT avoiding")
        if avg_bounce_in_pain > avg_visits * 0.5 and avg_visits > 3:
            contradictions.append(
                f"WALL BOUNCE EXITS ({avg_bounce_in_pain:.0f}) ≈ PAIN VISITS ({avg_visits:.0f})"
                f" → Most 'escapes' are wall bounces, not learned avoidance")
        if avg_dist < env_config.pain_zone_radius * 2 and pain_time_ok:
            contradictions.append(
                f"LOW DIST TO PAIN ({avg_dist:.0f}px) BUT LOW PAIN TIME"
                f" → Agent hugs pain boundary, doesn't actively avoid")

        if contradictions:
            print(f"\n  *** CONTRADICTION ALERTS ({len(contradictions)}) ***")
            for i, c in enumerate(contradictions, 1):
                print(f"  [{i}] {c}")
        else:
            print(f"\n  No contradictions detected - metrics are consistent.")

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

    # === 행동 진단 종합 ===
    print(f"\n  === Behavior Diagnostics (Aggregate) ===")
    print(f"  Straight Line: avg {np.mean(all_straight_pct):.1f}% of steps")
    print(f"  Max Straight Streak: avg {np.mean(all_max_straight_streak):.0f} steps, "
          f"worst {max(all_max_straight_streak):.0f} steps")
    print(f"  Angle StdDev: avg {np.mean(all_angle_std):.4f} "
          f"({'DIVERSE' if np.mean(all_angle_std) > 0.02 else 'MONOTONE - PROBLEM!'})")
    if any(p > 0 for p in all_food_correct_pct):
        print(f"  Food Response: avg {np.mean(all_food_correct_pct):.1f}% correct turns "
              f"({'ACTIVE' if np.mean(all_food_correct_pct) > 40 else 'PASSIVE - PROBLEM!'})")
    else:
        print(f"  Food Response: NO food detections across all episodes!")
    if any(p > 0 for p in all_pain_escape_pct):
        print(f"  Pain Escape: avg {np.mean(all_pain_escape_pct):.1f}% correct escapes "
              f"({'ACTIVE' if np.mean(all_pain_escape_pct) > 40 else 'PASSIVE - PROBLEM!'})")
    else:
        print(f"  Pain Escape: NO pain zone entries across all episodes!")

    # 초반 vs 후반 행동 변화 (학습 효과)
    if episodes >= 10:
        n5 = min(5, episodes // 2)
        early_straight = np.mean(all_straight_pct[:n5])
        late_straight = np.mean(all_straight_pct[-n5:])
        early_food = np.mean(all_food_correct_pct[:n5])
        late_food = np.mean(all_food_correct_pct[-n5:])
        print(f"\n  --- Learning Effect (Early vs Late) ---")
        print(f"  Straight %: early {early_straight:.1f}% → late {late_straight:.1f}% "
              f"({late_straight - early_straight:+.1f}pp)")
        print(f"  Food Correct %: early {early_food:.1f}% → late {late_food:.1f}% "
              f"({late_food - early_food:+.1f}pp)")

    print("=" * 70)

    # 가중치 저장
    if save_weights:
        brain.save_all_weights(save_weights)

    env.close()
    if logger:
        logger.close()
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
    parser.add_argument("--save-weights", type=str, default=None,
                       help="Save all Hebbian weights after training (e.g. brain_20ep.npz)")
    parser.add_argument("--load-weights", type=str, default=None,
                       help="Load Hebbian weights before running (for visualization)")
    parser.add_argument("--no-learning", action="store_true",
                       help="Disable Hebbian learning (for baseline comparison)")
    parser.add_argument("--fps", type=int, default=10,
                       help="Render FPS (default: 10, slower=easier to observe)")
    parser.add_argument("--render-from", type=int, default=1,
                       help="Start rendering from episode N (default: 1, use with --render pygame)")
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
    parser.add_argument("--no-language", action="store_true",
                       help="Disable Phase 17 (Language Circuit)")
    parser.add_argument("--no-wm-expansion", action="store_true",
                       help="Disable Phase 18 (WM Expansion)")
    parser.add_argument("--no-metacognition", action="store_true",
                       help="Disable Phase 19 (Metacognition)")
    parser.add_argument("--no-self-model", action="store_true",
                       help="Disable Phase 20 (Self-Model)")
    parser.add_argument("--no-predator", action="store_true",
                       help="Disable predators in environment")
    parser.add_argument("--no-agency", action="store_true",
                       help="Disable Phase L14 (Agency Detection)")
    parser.add_argument("--no-narrative-self", action="store_true",
                       help="Disable Phase L15 (Narrative Self)")
    parser.add_argument("--no-sparse-expansion", action="store_true",
                       help="Disable Phase L16 (Sparse Expansion Layer)")
    parser.add_argument("--no-prediction", action="store_true",
                       help="Disable Phase C4 (Contextual Prediction)")
    parser.add_argument("--no-curiosity", action="store_true",
                       help="Disable Phase C5 (Curiosity-Driven Exploration)")
    # Data logging for dashboard
    parser.add_argument("--log-data", action="store_true",
                       help="Enable data logging for dashboard visualization")
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Custom log directory (default: logs/run_TIMESTAMP)")
    parser.add_argument("--log-sample-rate", type=int, default=5,
                       help="Log every N steps (default: 5)")
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
        no_association=args.no_association,
        no_language=args.no_language,
        no_wm_expansion=args.no_wm_expansion,
        no_metacognition=args.no_metacognition,
        no_self_model=args.no_self_model,
        no_predator=args.no_predator,
        no_agency=args.no_agency,
        no_narrative_self=args.no_narrative_self,
        no_sparse_expansion=args.no_sparse_expansion,
        no_prediction=args.no_prediction,
        no_curiosity=args.no_curiosity,
        log_data=args.log_data,
        log_dir=args.log_dir,
        log_sample_rate=args.log_sample_rate,
        save_weights=args.save_weights,
        load_weights=args.load_weights,
    )
