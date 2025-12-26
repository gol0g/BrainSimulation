"""
Test Scenarios - 근본 원리 검증을 위한 특수 상황 테스트

목적:
- Genesis가 "원리의 힘"으로 작동하는지, "우연"인지 검증
- risk vs ambiguity vs complexity 우세가 상황에 따라 자연스럽게 바뀌는지
- 모델이 틀렸을 때 회복되는지 (고착/진동 없는지)

시나리오:
1. conflict: 음식과 위험이 같은 방향 (갈등)
2. deadend: 막다른 길 (코너)
3. temptation: 위험이 먹이 근처에 붙어 있음 (유혹-위협)
4. sensor_noise: 방향 dx/dy 10~20% 플립
5. partial_obs: danger_dx/dy 숨김 (부분관측)
6. slip: 10% 확률로 다른 방향 이동 (전이 불확실)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum


class ScenarioType(Enum):
    NORMAL = "normal"
    CONFLICT = "conflict"           # 음식-위험 같은 방향
    DEADEND = "deadend"             # 막다른 길
    TEMPTATION = "temptation"       # 위험이 음식 근처
    SENSOR_NOISE = "sensor_noise"   # 센서 노이즈
    PARTIAL_OBS = "partial_obs"     # 부분 관측
    SLIP = "slip"                   # 미끄러짐 (전이 불확실)
    DRIFT = "drift"                 # G1 Gate: 환경 dynamics 변화


@dataclass
class ScenarioConfig:
    """시나리오 설정"""
    type: ScenarioType
    duration: int = 200  # 스텝 수
    params: Dict = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """시나리오 실행 결과"""
    scenario_type: str
    total_steps: int
    food_eaten: int
    danger_hits: int
    deaths: int
    avg_F: float
    avg_risk: float
    avg_ambiguity: float
    avg_complexity: float
    dominant_factor_counts: Dict[str, int]
    recovery_events: int  # 위험 후 회복 횟수
    oscillation_count: int  # 진동 횟수 (같은 행동 반복)

    # 핵심 지표
    risk_ambiguity_ratio: float  # risk/ambiguity 비율
    adaptation_score: float  # 상황 적응 점수


@dataclass
class G1GateResult:
    """
    G1 Gate (Generalization Test) 결과 - v4.3 Enhanced Metrics

    핵심 질문: 에이전트가 일반화하는가, 암기하는가?

    통과 기준:
    1. drift 후 transition_std 증가 (새 상황 인지)
    2. adaptation_steps < threshold (빠른 적응)
    3. with_ltm의 적응이 without_ltm보다 빠름 (LTM이 도움)

    v4.3 신규 지표:
    - std_auc_shock: shock window에서 std 면적 (적분)
    - peak_std_ratio: max(std) / pre_std (스파이크 강도)
    - time_to_recovery: 회복까지 걸린 스텝
    - volatility_auc: shock window에서 volatility 면적
    """
    # 설정
    drift_type: str
    drift_after: int

    # Phase 1: Training (drift 전)
    pre_drift_food: int
    pre_drift_danger: int
    pre_drift_avg_G: float
    pre_drift_transition_std: float

    # Phase 2: Post-drift (drift 후)
    post_drift_food: int
    post_drift_danger: int
    post_drift_avg_G: float
    post_drift_transition_std: float

    # === v4.3 Enhanced Metrics ===
    # Shock 구간 지표
    std_auc_shock: float  # Σ (std_t - std_baseline)+ over shock window
    peak_std_ratio: float  # max(std) / pre_std - 스파이크 강도
    volatility_auc: float  # shock window에서 volatility 면적

    # Recovery 지표
    time_to_recovery: int  # 회복까지 걸린 스텝 (volatility < ε and food rate recovers)
    food_rate_pre: float  # pre-drift food rate (per step)
    food_rate_shock: float  # shock phase (first 20 steps) food rate
    food_rate_adapt: float  # adapt phase (after 20 steps) food rate

    # 기존 메트릭
    std_increase_ratio: float  # (post - pre) / pre, 양수면 상승 감지
    adaptation_steps: int  # G가 pre-drift 수준으로 돌아오는 스텝 수
    recovery_ratio: float  # post_food / pre_food 비율

    # Gate 결과
    passed: bool
    reasons: List[str]


class ScenarioManager:
    """
    테스트 시나리오 관리자

    핵심 역할:
    - 특수 상황 설정 및 실행
    - 결과 수집 및 분석
    - "원리의 힘" vs "우연" 판별
    """

    def __init__(self, world, agent):
        self.world = world
        self.agent = agent
        self.current_scenario: Optional[ScenarioConfig] = None
        self.results: List[ScenarioResult] = []

        # 시나리오별 설정 함수
        self.setup_functions = {
            ScenarioType.CONFLICT: self._setup_conflict,
            ScenarioType.DEADEND: self._setup_deadend,
            ScenarioType.TEMPTATION: self._setup_temptation,
            ScenarioType.SENSOR_NOISE: self._setup_sensor_noise,
            ScenarioType.PARTIAL_OBS: self._setup_partial_obs,
            ScenarioType.SLIP: self._setup_slip,
            ScenarioType.DRIFT: self._setup_drift,
        }

        # DRIFT 시나리오용 상태
        self._drift_active = False
        self._drift_step_count = 0

        # 로깅
        self._step_logs: List[Dict] = []
        self._action_history: List[int] = []

    def start_scenario(self, scenario_type: ScenarioType, duration: int = 200, **params):
        """시나리오 시작"""
        self.current_scenario = ScenarioConfig(
            type=scenario_type,
            duration=duration,
            params=params
        )
        self._step_logs = []
        self._action_history = []

        # 월드 리셋
        self.world.reset()
        self.agent.reset()

        # 시나리오별 초기 설정
        if scenario_type in self.setup_functions:
            self.setup_functions[scenario_type](**params)

        return {
            "status": "started",
            "scenario": scenario_type.value,
            "duration": duration
        }

    def _setup_conflict(self, **params):
        """
        갈등 시나리오: 음식과 위험이 같은 방향
        에이전트가 음식을 얻으려면 위험 방향으로 가야 함
        """
        # 에이전트를 중앙에
        self.world.agent_pos = [5, 5]

        # 음식과 위험을 같은 방향에 배치 (예: 오른쪽)
        self.world.food_pos = [8, 5]   # 오른쪽
        self.world.danger_pos = [7, 5]  # 음식 바로 앞

    def _setup_deadend(self, **params):
        """
        막다른 길 시나리오: 코너에서 탈출
        """
        # 에이전트를 코너에
        self.world.agent_pos = [0, 0]

        # 음식은 반대편
        self.world.food_pos = [9, 9]

        # 위험은 탈출 경로 근처
        self.world.danger_pos = [1, 1]

    def _setup_temptation(self, **params):
        """
        유혹-위협 시나리오: 위험이 음식 바로 옆에
        음식을 먹으면 위험에 노출됨
        """
        self.world.agent_pos = [5, 5]
        self.world.food_pos = [7, 5]
        self.world.danger_pos = [7, 6]  # 음식 바로 아래

    def _setup_sensor_noise(self, noise_rate: float = 0.15, **params):
        """
        센서 노이즈 시나리오: 방향 정보가 가끔 틀림

        Args:
            noise_rate: 노이즈 확률 (0.1~0.2 권장)
        """
        self.current_scenario.params['noise_rate'] = noise_rate
        # 일반 배치
        self.world.agent_pos = [5, 5]
        self.world.food_pos = [7, 7]
        self.world.danger_pos = [2, 2]

    def _setup_partial_obs(self, hidden_dims: List[int] = None, **params):
        """
        부분 관측 시나리오: 일부 관측이 숨겨짐

        Args:
            hidden_dims: 숨길 차원 인덱스 (기본: [4, 5] = danger_dx, danger_dy)
        """
        if hidden_dims is None:
            hidden_dims = [4, 5]  # danger 방향 숨김
        self.current_scenario.params['hidden_dims'] = hidden_dims

        self.world.agent_pos = [5, 5]
        self.world.food_pos = [7, 7]
        self.world.danger_pos = [3, 3]

    def _setup_slip(self, slip_rate: float = 0.1, **params):
        """
        미끄러짐 시나리오: 행동이 가끔 실패

        Args:
            slip_rate: 미끄러질 확률 (0.1 권장)
        """
        self.current_scenario.params['slip_rate'] = slip_rate

        self.world.agent_pos = [5, 5]
        self.world.food_pos = [8, 5]
        self.world.danger_pos = [2, 5]

    def _setup_drift(self, drift_after: int = 100, drift_type: str = "rotate", **params):
        """
        G1 Gate - 환경 Drift 시나리오: dynamics가 중간에 변함

        Args:
            drift_after: 몇 스텝 후에 drift 시작 (기본 100)
            drift_type: drift 종류
                기본 타입:
                - "rotate": 행동 회전 (UP→RIGHT, RIGHT→DOWN, ...)
                - "flip_x": 좌우 반전 (LEFT↔RIGHT)
                - "flip_y": 상하 반전 (UP↔DOWN)
                - "reverse": 전체 반전 (UP→DOWN, LEFT→RIGHT)

                v4.3 신규 타입 (암기형 죽이기):
                - "partial": internal만 변경 (energy decay 2배)
                - "delayed": drift 후 20스텝 지나서야 적용
                - "probabilistic": 70% 기존, 30% 반전 (노이즈 섞인 변화)
        """
        self.current_scenario.params['drift_after'] = drift_after
        self.current_scenario.params['drift_type'] = drift_type
        self._drift_active = False
        self._drift_step_count = 0

        # v4.3: 새 drift 타입용 상태
        self._drift_delay_counter = 0  # delayed drift용
        self._drift_delay_threshold = params.get('delay_steps', 20)
        self._probabilistic_ratio = params.get('prob_ratio', 0.3)  # 30% 반전

        # 일반 배치
        self.world.agent_pos = [5, 5]
        self.world.food_pos = [7, 7]
        self.world.danger_pos = [2, 2]

    def check_and_activate_drift(self):
        """
        DRIFT 시나리오에서 drift 시작 조건 체크
        매 스텝마다 호출
        """
        if self.current_scenario is None:
            return False

        if self.current_scenario.type != ScenarioType.DRIFT:
            return False

        self._drift_step_count += 1
        drift_after = self.current_scenario.params.get('drift_after', 100)

        if not self._drift_active and self._drift_step_count >= drift_after:
            self._drift_active = True
            return True  # drift 방금 활성화됨

        return False

    def apply_drift_to_action(self, action: int) -> int:
        """
        DRIFT 시나리오에서 행동 효과 변경
        환경이 행동을 다르게 해석 (전이 모델과 불일치 발생)

        v4.3: 새 drift 타입 추가
        - partial: 행동 변경 없음 (internal만 변경)
        - delayed: drift_delay_threshold 후에야 적용
        - probabilistic: prob_ratio 확률로만 반전
        """
        if not self._drift_active:
            return action

        drift_type = self.current_scenario.params.get('drift_type', 'rotate')

        # === v4.3 신규 타입 처리 ===

        # partial: 행동 변경 없음 (internal drift는 get_internal_drift_modifier()에서 처리)
        if drift_type == "partial":
            return action

        # delayed: 일정 스텝 후에야 적용
        if drift_type == "delayed":
            self._drift_delay_counter += 1
            if self._drift_delay_counter < self._drift_delay_threshold:
                return action  # 아직 delay 중
            # delay 지나면 reverse 적용
            mapping = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}
            return mapping.get(action, action)

        # probabilistic: 일정 확률로만 반전
        if drift_type == "probabilistic":
            import random
            if random.random() < self._probabilistic_ratio:
                # 30% 확률로 반전
                mapping = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}
                return mapping.get(action, action)
            return action  # 70%는 정상

        # === 기존 타입 ===

        # 행동 매핑: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        if drift_type == "rotate":
            # UP→RIGHT, RIGHT→DOWN, DOWN→LEFT, LEFT→UP
            mapping = {0: 0, 1: 4, 2: 3, 3: 1, 4: 2}
            return mapping.get(action, action)

        elif drift_type == "flip_x":
            # LEFT↔RIGHT
            mapping = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3}
            return mapping.get(action, action)

        elif drift_type == "flip_y":
            # UP↔DOWN
            mapping = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
            return mapping.get(action, action)

        elif drift_type == "reverse":
            # 전체 반전
            mapping = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}
            return mapping.get(action, action)

        return action

    def get_internal_drift_modifier(self) -> dict:
        """
        v4.3: partial drift용 - internal state 변경 파라미터 반환

        Returns:
            dict with modifiers:
            - energy_decay_mult: energy 감소 배율 (기본 1.0)
            - pain_increase_mult: pain 증가 배율 (기본 1.0)
        """
        if not self._drift_active:
            return {'energy_decay_mult': 1.0, 'pain_increase_mult': 1.0}

        drift_type = self.current_scenario.params.get('drift_type', 'rotate')

        if drift_type == "partial":
            # internal만 변경: energy decay 2배
            return {'energy_decay_mult': 2.0, 'pain_increase_mult': 1.0}

        return {'energy_decay_mult': 1.0, 'pain_increase_mult': 1.0}

    def modify_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        시나리오에 따라 관측 수정

        센서 노이즈, 부분 관측 등 적용
        """
        if self.current_scenario is None:
            return obs

        modified_obs = obs.copy()
        scenario_type = self.current_scenario.type

        if scenario_type == ScenarioType.SENSOR_NOISE:
            noise_rate = self.current_scenario.params.get('noise_rate', 0.15)
            # 방향 정보 (인덱스 2-5)에 노이즈 추가
            for i in [2, 3, 4, 5]:
                if np.random.random() < noise_rate:
                    # 방향 플립: -1 <-> 1, 0은 유지
                    if modified_obs[i] != 0:
                        modified_obs[i] = -modified_obs[i]

        elif scenario_type == ScenarioType.PARTIAL_OBS:
            hidden_dims = self.current_scenario.params.get('hidden_dims', [4, 5])
            for dim in hidden_dims:
                if dim < len(modified_obs):
                    modified_obs[dim] = 0.0  # 숨긴 차원은 0으로

        return modified_obs

    def modify_action(self, intended_action: int) -> int:
        """
        시나리오에 따라 행동 수정 (미끄러짐, Drift 등)
        """
        if self.current_scenario is None:
            return intended_action

        if self.current_scenario.type == ScenarioType.SLIP:
            slip_rate = self.current_scenario.params.get('slip_rate', 0.1)
            if np.random.random() < slip_rate and intended_action != 0:
                # 다른 랜덤 행동으로 변경 (stay 제외)
                other_actions = [a for a in [1, 2, 3, 4] if a != intended_action]
                return np.random.choice(other_actions)

        elif self.current_scenario.type == ScenarioType.DRIFT:
            # DRIFT: 환경이 행동을 다르게 해석
            return self.apply_drift_to_action(intended_action)

        return intended_action

    def log_step(self, state, action: int, outcome: Dict):
        """스텝 결과 로깅"""
        self._step_logs.append({
            'F': state.F,
            'risk': state.risk,
            'ambiguity': state.ambiguity,
            'complexity': state.complexity,
            'action': action,
            'dominant_factor': state.dominant_factor,
            'ate_food': outcome.get('ate_food', False),
            'hit_danger': outcome.get('hit_danger', False),
            'died': outcome.get('died', False),
        })
        self._action_history.append(action)

    def is_scenario_complete(self) -> bool:
        """시나리오 완료 여부"""
        if self.current_scenario is None:
            return True
        return len(self._step_logs) >= self.current_scenario.duration

    def get_result(self) -> Optional[ScenarioResult]:
        """시나리오 결과 분석"""
        if not self._step_logs:
            return None

        logs = self._step_logs
        n = len(logs)

        # 기본 통계
        food_eaten = sum(1 for l in logs if l['ate_food'])
        danger_hits = sum(1 for l in logs if l['hit_danger'])
        deaths = sum(1 for l in logs if l['died'])

        avg_F = np.mean([l['F'] for l in logs])
        avg_risk = np.mean([l['risk'] for l in logs])
        avg_ambiguity = np.mean([l['ambiguity'] for l in logs])
        avg_complexity = np.mean([l['complexity'] for l in logs])

        # dominant factor 카운트
        dominant_counts = {}
        for l in logs:
            df = l['dominant_factor']
            dominant_counts[df] = dominant_counts.get(df, 0) + 1

        # 회복 이벤트: 위험 충돌 후 10스텝 내 음식 획득
        recovery_events = 0
        for i, l in enumerate(logs):
            if l['hit_danger']:
                # 이후 10스텝 내 음식 획득 확인
                for j in range(i+1, min(i+11, n)):
                    if logs[j]['ate_food']:
                        recovery_events += 1
                        break

        # 진동 카운트: 연속 4스텝 내 같은 2개 행동 반복
        oscillation_count = 0
        for i in range(len(self._action_history) - 3):
            window = self._action_history[i:i+4]
            if len(set(window)) == 2:
                # 두 행동 번갈아 (예: [1,2,1,2])
                if window[0] == window[2] and window[1] == window[3]:
                    oscillation_count += 1

        # 핵심 지표
        risk_ambiguity_ratio = avg_risk / (avg_ambiguity + 1e-10)

        # 적응 점수: 음식 획득 - 위험 충돌 - 진동 페널티
        adaptation_score = food_eaten - danger_hits * 2 - oscillation_count * 0.5
        adaptation_score = max(0, adaptation_score) / max(1, n / 100)

        return ScenarioResult(
            scenario_type=self.current_scenario.type.value if self.current_scenario else "unknown",
            total_steps=n,
            food_eaten=food_eaten,
            danger_hits=danger_hits,
            deaths=deaths,
            avg_F=avg_F,
            avg_risk=avg_risk,
            avg_ambiguity=avg_ambiguity,
            avg_complexity=avg_complexity,
            dominant_factor_counts=dominant_counts,
            recovery_events=recovery_events,
            oscillation_count=oscillation_count,
            risk_ambiguity_ratio=risk_ambiguity_ratio,
            adaptation_score=adaptation_score
        )

    def end_scenario(self) -> Dict:
        """시나리오 종료 및 결과 반환"""
        result = self.get_result()
        if result:
            self.results.append(result)

        self.current_scenario = None

        if result:
            return {
                "status": "completed",
                "scenario": result.scenario_type,
                "summary": {
                    "steps": result.total_steps,
                    "food_eaten": result.food_eaten,
                    "danger_hits": result.danger_hits,
                    "deaths": result.deaths,
                    "avg_F": round(result.avg_F, 3),
                    "avg_risk": round(result.avg_risk, 3),
                    "avg_ambiguity": round(result.avg_ambiguity, 3),
                    "avg_complexity": round(result.avg_complexity, 3),
                    "dominant_factors": result.dominant_factor_counts,
                    "recovery_events": result.recovery_events,
                    "oscillation_count": result.oscillation_count,
                    "risk_ambiguity_ratio": round(result.risk_ambiguity_ratio, 3),
                    "adaptation_score": round(result.adaptation_score, 3),
                },
                "analysis": self._analyze_result(result)
            }

        return {"status": "no_data"}

    def _analyze_result(self, result: ScenarioResult) -> Dict:
        """결과 분석 및 해석"""
        analysis = {
            "is_principle_driven": True,
            "concerns": [],
            "positives": []
        }

        # 진동 체크
        if result.oscillation_count > result.total_steps * 0.1:
            analysis["concerns"].append("높은 진동률 - 의사결정 불안정")
            analysis["is_principle_driven"] = False

        # 회복력 체크
        if result.danger_hits > 0:
            recovery_rate = result.recovery_events / result.danger_hits
            if recovery_rate > 0.5:
                analysis["positives"].append(f"좋은 회복력 ({recovery_rate:.0%})")
            else:
                analysis["concerns"].append(f"낮은 회복력 ({recovery_rate:.0%})")

        # dominant factor 다양성 체크
        if len(result.dominant_factor_counts) >= 2:
            analysis["positives"].append("상황에 따른 요인 전환 확인")
        else:
            analysis["concerns"].append("단일 요인에 고착")

        # Risk/Ambiguity 균형 체크
        ratio = result.risk_ambiguity_ratio
        if 0.3 < ratio < 3.0:
            analysis["positives"].append(f"Risk-Ambiguity 균형 양호 (비율: {ratio:.2f})")
        elif ratio > 10:
            analysis["concerns"].append("Risk 과잉 - 과도한 회피?")
        elif ratio < 0.1:
            analysis["concerns"].append("Ambiguity 과잉 - 과도한 탐색?")

        return analysis

    def run_all_scenarios(self) -> Dict:
        """모든 시나리오 자동 실행 (별도 호출용)"""
        all_results = {}

        scenarios = [
            (ScenarioType.CONFLICT, {}),
            (ScenarioType.DEADEND, {}),
            (ScenarioType.TEMPTATION, {}),
            (ScenarioType.SENSOR_NOISE, {"noise_rate": 0.15}),
            (ScenarioType.PARTIAL_OBS, {"hidden_dims": [4, 5]}),
            (ScenarioType.SLIP, {"slip_rate": 0.1}),
        ]

        for scenario_type, params in scenarios:
            # 각 시나리오 실행 필요 (step 루프는 외부에서)
            all_results[scenario_type.value] = {
                "ready": True,
                "params": params
            }

        return all_results

    # ========== G1 Gate (Generalization Test) ==========

    def log_step_g1(self, state, action: int, outcome: Dict, transition_std: float, G_value: float,
                     volatility_ratio: float = 0.0, std_change_pct: float = 0.0):
        """
        G1 Gate용 확장 로깅 - transition_std와 G 추적

        v4.3: volatility_ratio, std_change_pct 추가
        """
        log_entry = {
            'F': state.F,
            'G': G_value,
            'risk': state.risk,
            'ambiguity': state.ambiguity,
            'complexity': state.complexity,
            'action': action,
            'dominant_factor': state.dominant_factor,
            'ate_food': outcome.get('ate_food', False),
            'hit_danger': outcome.get('hit_danger', False),
            'died': outcome.get('died', False),
            'transition_std': transition_std,
            'drift_active': self._drift_active,
            # v4.3 Enhanced
            'volatility_ratio': volatility_ratio,
            'std_change_pct': std_change_pct,
        }
        self._step_logs.append(log_entry)
        self._action_history.append(action)

    def get_g1_gate_result(self) -> Optional[G1GateResult]:
        """
        G1 Gate 결과 분석 - v4.3 Enhanced Metrics

        통과 기준:
        1. drift 후 transition_std 증가 (새 상황 인지)
        2. post-drift에서도 생존 (food 획득)
        3. 적응 시간 측정 (G가 pre-drift 수준으로 복귀)

        v4.3 신규 지표:
        - std_auc_shock: shock window에서 (std - baseline)의 적분
        - peak_std_ratio: max(std) / pre_std - 스파이크 강도
        - time_to_recovery: 회복까지 걸린 스텝
        - volatility_auc: shock window에서 volatility 면적
        """
        if self.current_scenario is None or self.current_scenario.type != ScenarioType.DRIFT:
            return None

        if len(self._step_logs) < 10:
            return None

        drift_after = self.current_scenario.params.get('drift_after', 100)
        drift_type = self.current_scenario.params.get('drift_type', 'rotate')

        # Phase 분리
        pre_logs = [l for l in self._step_logs if not l.get('drift_active', False)]
        post_logs = [l for l in self._step_logs if l.get('drift_active', False)]

        if len(pre_logs) < 5 or len(post_logs) < 5:
            return None

        # === Pre-drift 통계 ===
        pre_drift_food = sum(1 for l in pre_logs if l['ate_food'])
        pre_drift_danger = sum(1 for l in pre_logs if l['hit_danger'])
        pre_drift_avg_G = np.mean([l['G'] for l in pre_logs])
        pre_drift_transition_std = np.mean([l['transition_std'] for l in pre_logs[-10:]])
        std_baseline = pre_drift_transition_std

        # === Post-drift 통계 ===
        post_drift_food = sum(1 for l in post_logs if l['ate_food'])
        post_drift_danger = sum(1 for l in post_logs if l['hit_danger'])
        post_drift_avg_G = np.mean([l['G'] for l in post_logs])
        post_drift_transition_std = np.mean([l['transition_std'] for l in post_logs[:10]])

        # === v4.3 Enhanced Metrics ===

        # 1. std_auc_shock: Σ (std_t - std_baseline)+ over shock window (first 20 steps)
        shock_window = min(20, len(post_logs))
        std_auc_shock = 0.0
        for l in post_logs[:shock_window]:
            delta = l['transition_std'] - std_baseline
            if delta > 0:
                std_auc_shock += delta

        # 2. peak_std_ratio: max(std) / pre_std
        all_stds = [l['transition_std'] for l in post_logs]
        peak_std = max(all_stds) if all_stds else std_baseline
        peak_std_ratio = peak_std / (std_baseline + 1e-6)

        # 3. volatility_auc: Σ volatility_ratio over shock window
        volatility_auc = 0.0
        for l in post_logs[:shock_window]:
            volatility_auc += l.get('volatility_ratio', 0.0)

        # 4. time_to_recovery: first step where volatility < 0.2 AND rolling food rate > 50% of pre
        food_rate_pre = pre_drift_food / len(pre_logs) if len(pre_logs) > 0 else 0
        time_to_recovery = len(post_logs)  # default: never recovered
        recovery_threshold_vol = 0.2
        recovery_threshold_food = food_rate_pre * 0.5

        for i in range(5, len(post_logs)):
            vol = post_logs[i].get('volatility_ratio', 1.0)
            # Rolling food rate over last 10 steps
            window_start = max(0, i - 10)
            window_food = sum(1 for l in post_logs[window_start:i] if l['ate_food'])
            window_rate = window_food / (i - window_start) if i > window_start else 0

            if vol < recovery_threshold_vol and window_rate >= recovery_threshold_food:
                time_to_recovery = i
                break

        # 5. Food rates by phase
        shock_phase_end = min(20, len(post_logs))
        shock_logs = post_logs[:shock_phase_end]
        adapt_logs = post_logs[shock_phase_end:]

        food_rate_shock = sum(1 for l in shock_logs if l['ate_food']) / len(shock_logs) if shock_logs else 0
        food_rate_adapt = sum(1 for l in adapt_logs if l['ate_food']) / len(adapt_logs) if adapt_logs else 0

        # === 기존 메트릭 ===
        if pre_drift_transition_std > 0:
            std_increase_ratio = (post_drift_transition_std - pre_drift_transition_std) / pre_drift_transition_std
        else:
            std_increase_ratio = 0.0

        # 적응 시간: post-drift에서 G가 pre-drift 수준으로 돌아오는 스텝
        adaptation_steps = len(post_logs)
        pre_G_threshold = pre_drift_avg_G * 1.1

        for i, l in enumerate(post_logs):
            if l['G'] <= pre_G_threshold:
                if i + 5 <= len(post_logs):
                    window_G = np.mean([post_logs[j]['G'] for j in range(i, min(i+5, len(post_logs)))])
                    if window_G <= pre_G_threshold:
                        adaptation_steps = i
                        break

        if pre_drift_food > 0:
            recovery_ratio = post_drift_food / pre_drift_food
        else:
            recovery_ratio = 1.0 if post_drift_food > 0 else 0.0

        # === Gate 통과 판정 (v4.3 강화) ===
        reasons = []
        passed = True

        # 기준 1: drift 감지 - peak_std_ratio 사용 (더 민감)
        if peak_std_ratio < 1.05:
            passed = False
            reasons.append(f"FAIL: peak_std 미미 ({peak_std_ratio:.2f}x) - drift 감지 안됨")
        else:
            reasons.append(f"PASS: peak_std 스파이크 ({peak_std_ratio:.2f}x)")

        # 기준 2: 생존 (post-drift food > 0)
        if post_drift_food == 0:
            passed = False
            reasons.append("FAIL: post-drift food=0 - 적응 실패")
        else:
            reasons.append(f"PASS: post-drift food={post_drift_food}")

        # 기준 3: 회복 (time_to_recovery < 50)
        if time_to_recovery >= len(post_logs):
            reasons.append("WARN: 회복 안됨")
        elif time_to_recovery > 50:
            reasons.append(f"WARN: 회복 느림 ({time_to_recovery} steps)")
        else:
            reasons.append(f"PASS: 빠른 회복 ({time_to_recovery} steps)")

        # 기준 4: 적응 후 성능 회복 (adapt phase food rate >= 50% of pre)
        if food_rate_adapt >= food_rate_pre * 0.5:
            reasons.append(f"PASS: 적응 후 성능 회복 ({food_rate_adapt:.2f} vs {food_rate_pre:.2f})")
        else:
            reasons.append(f"WARN: 적응 후 성능 저하 ({food_rate_adapt:.2f} < {food_rate_pre*0.5:.2f})")

        return G1GateResult(
            drift_type=drift_type,
            drift_after=drift_after,
            pre_drift_food=pre_drift_food,
            pre_drift_danger=pre_drift_danger,
            pre_drift_avg_G=round(pre_drift_avg_G, 3),
            pre_drift_transition_std=round(pre_drift_transition_std, 4),
            post_drift_food=post_drift_food,
            post_drift_danger=post_drift_danger,
            post_drift_avg_G=round(post_drift_avg_G, 3),
            post_drift_transition_std=round(post_drift_transition_std, 4),
            # v4.3 Enhanced
            std_auc_shock=round(std_auc_shock, 4),
            peak_std_ratio=round(peak_std_ratio, 3),
            volatility_auc=round(volatility_auc, 3),
            time_to_recovery=time_to_recovery,
            food_rate_pre=round(food_rate_pre, 3),
            food_rate_shock=round(food_rate_shock, 3),
            food_rate_adapt=round(food_rate_adapt, 3),
            # Original
            std_increase_ratio=round(std_increase_ratio, 3),
            adaptation_steps=adaptation_steps,
            recovery_ratio=round(recovery_ratio, 3),
            passed=passed,
            reasons=reasons
        )
