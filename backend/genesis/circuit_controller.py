"""
v5.2: Circuit Controller - Circuit as Main, FEP as Teacher

핵심 철학:
- 행동은 Circuit이 선택
- FEP는 매 스텝 "점수/피드백"만 남김
- 규칙 엔진이 아닌 메타-제어(보호반사)

P2 완료 기준:
- N0: 1000+ step 안정성
- N1: FEP vs Circuit 일치율 70-85% 유지
- N2: danger 회피율이 FEP baseline 이상
- N3: 200+ disagreement cases 수집, 3-5개 유형으로 군집화
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from collections import defaultdict
import time


class DisagreementType(Enum):
    """Disagreement 유형 분류"""
    DANGER_APPROACH = "danger_approach"  # Circuit이 위험 쪽으로
    FOOD_IGNORE = "food_ignore"  # Circuit이 음식 무시
    ENERGY_WASTE = "energy_waste"  # 저에너지에서 다른 선택
    EXPLORATION_DIFF = "exploration_diff"  # 탐색 전략 차이
    UNKNOWN = "unknown"


@dataclass
class DisagreementCase:
    """단일 disagreement 케이스"""
    step: int
    timestamp: float
    observation: np.ndarray
    fep_action: int
    circuit_action: int
    fep_g_values: Dict[int, float]
    circuit_energies: Dict[int, float]
    disagreement_type: DisagreementType
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'step': self.step,
            'timestamp': self.timestamp,
            'observation': self.observation.tolist(),
            'fep_action': self.fep_action,
            'circuit_action': self.circuit_action,
            'fep_g_values': self.fep_g_values,
            'circuit_energies': self.circuit_energies,
            'type': self.disagreement_type.value,
            'context': self.context
        }


@dataclass
class FallbackEvent:
    """Fallback 이벤트 기록"""
    step: int
    trigger_reason: str
    duration: int  # 몇 스텝 동안 fallback
    observation: np.ndarray
    metrics: Dict[str, float]


@dataclass
class P2Metrics:
    """P2 완료 기준 추적"""
    total_steps: int = 0
    circuit_steps: int = 0  # Circuit이 주도한 스텝
    fep_steps: int = 0  # Fallback으로 FEP가 주도한 스텝

    agreements: int = 0
    disagreements: int = 0

    # N1a: Safety-critical alignment
    safety_agreements: int = 0  # danger/food_ignore에서 일치
    safety_disagreements: int = 0  # danger/food_ignore에서 불일치
    danger_approach_count: int = 0  # Circuit이 위험 쪽으로 간 횟수
    food_ignore_count: int = 0  # 저에너지에서 음식 무시 횟수

    # N1b: Style alignment (exploration_diff + energy_waste)
    style_agreements: int = 0
    style_disagreements: int = 0

    danger_encounters: int = 0  # danger_prox > 0.5
    danger_avoided_circuit: int = 0
    danger_avoided_fep: int = 0

    crashes: int = 0
    fallback_count: int = 0

    @property
    def agreement_rate(self) -> float:
        """전체 일치율 (N1 legacy)"""
        total = self.agreements + self.disagreements
        return self.agreements / total if total > 0 else 0.0

    @property
    def safety_agreement_rate(self) -> float:
        """N1a: Safety-critical 일치율"""
        total = self.safety_agreements + self.safety_disagreements
        return self.safety_agreements / total if total > 0 else 1.0

    @property
    def style_agreement_rate(self) -> float:
        """N1b: Style 일치율"""
        total = self.style_agreements + self.style_disagreements
        return self.style_agreements / total if total > 0 else 1.0

    @property
    def circuit_danger_avoidance(self) -> float:
        return self.danger_avoided_circuit / self.danger_encounters if self.danger_encounters > 0 else 1.0

    @property
    def fep_danger_avoidance(self) -> float:
        return self.danger_avoided_fep / self.danger_encounters if self.danger_encounters > 0 else 1.0

    def to_dict(self) -> Dict:
        return {
            'total_steps': self.total_steps,
            'circuit_steps': self.circuit_steps,
            'fep_steps': self.fep_steps,
            # Overall
            'agreement_rate': self.agreement_rate,
            'agreements': self.agreements,
            'disagreements': self.disagreements,
            # N1a: Safety
            'n1a_safety_rate': self.safety_agreement_rate,
            'safety_agreements': self.safety_agreements,
            'safety_disagreements': self.safety_disagreements,
            'danger_approach_count': self.danger_approach_count,
            'food_ignore_count': self.food_ignore_count,
            # N1b: Style
            'n1b_style_rate': self.style_agreement_rate,
            'style_agreements': self.style_agreements,
            'style_disagreements': self.style_disagreements,
            # Danger tracking
            'danger_encounters': self.danger_encounters,
            'circuit_danger_avoidance': self.circuit_danger_avoidance,
            'fep_danger_avoidance': self.fep_danger_avoidance,
            'crashes': self.crashes,
            'fallback_count': self.fallback_count,
        }


class CircuitController:
    """
    v5.2 Circuit Controller

    Circuit이 메인, FEP가 teacher/logger.
    - 매 스텝 Circuit 실행
    - FEP는 비교/로깅용 (매 N스텝 또는 트리거 시)
    - 위험 상황에서 짧은 fallback (1-3 스텝)
    """

    def __init__(
        self,
        action_circuit,
        fep_oracle,
        agent,
        fep_compare_interval: int = 5,
        max_disagreements: int = 500,
        fallback_threshold: float = 0.7,  # danger_prox threshold
        fallback_duration: int = 2,  # 스텝 수
    ):
        """
        Args:
            action_circuit: ActionCompetitionCircuit instance
            fep_oracle: FEPActionOracle instance
            agent: GenesisAgent instance (for uncertainty/regret access)
            fep_compare_interval: 매 N스텝마다 FEP 비교 (기본 5)
            max_disagreements: 최대 저장할 disagreement 수
            fallback_threshold: fallback 트리거 danger_prox 임계값
            fallback_duration: fallback 지속 스텝 수
        """
        self.circuit = action_circuit
        self.oracle = fep_oracle
        self.agent = agent

        self.fep_compare_interval = fep_compare_interval
        self.max_disagreements = max_disagreements
        self.fallback_threshold = fallback_threshold
        self.fallback_duration = fallback_duration

        # State
        self.enabled = True  # P2 모드 활성화
        self.step_count = 0
        self.fallback_remaining = 0  # 남은 fallback 스텝
        self.last_fallback_reason = ""

        # Collections
        self.disagreements: List[DisagreementCase] = []
        self.fallback_events: List[FallbackEvent] = []
        self.metrics = P2Metrics()

        # Type clustering
        self.type_counts: Dict[DisagreementType, int] = defaultdict(int)

    def should_compare_fep(self, obs: np.ndarray) -> bool:
        """FEP 비교를 수행할지 결정"""
        # 1. 주기적 비교
        if self.step_count % self.fep_compare_interval == 0:
            return True

        # 2. 위험 상황
        danger_prox = obs[1] if len(obs) > 1 else 0.0
        if danger_prox > 0.5:
            return True

        # 3. 불확실성/regret spike (agent에서 가져옴)
        if hasattr(self.agent.action_selector, '_last_decision_entropy'):
            entropy = self.agent.action_selector._last_decision_entropy
            if entropy > 1.5:  # High uncertainty
                return True

        return False

    def should_fallback(self, obs: np.ndarray, circuit_action: int, fep_action: int) -> Tuple[bool, str]:
        """
        Fallback 여부 결정 (규칙이 아닌 메타-제어)

        조건:
        - danger_prox > threshold
        - Circuit과 FEP가 다름
        - Circuit이 위험 쪽으로 이동

        Returns:
            (should_fallback, reason)
        """
        danger_prox = obs[1] if len(obs) > 1 else 0.0
        danger_dx = obs[4] if len(obs) > 4 else 0.0
        danger_dy = obs[5] if len(obs) > 5 else 0.0

        # 위험하지 않으면 fallback 불필요
        if danger_prox < self.fallback_threshold:
            return False, ""

        # FEP와 같으면 fallback 불필요
        if circuit_action == fep_action:
            return False, ""

        # Circuit이 위험 쪽으로 이동하는지 체크
        moving_toward_danger = False
        if circuit_action == 4 and danger_dx > 0:  # RIGHT toward danger
            moving_toward_danger = True
        elif circuit_action == 3 and danger_dx < 0:  # LEFT toward danger
            moving_toward_danger = True
        elif circuit_action == 1 and danger_dy < 0:  # UP toward danger
            moving_toward_danger = True
        elif circuit_action == 2 and danger_dy > 0:  # DOWN toward danger
            moving_toward_danger = True

        if moving_toward_danger:
            reason = f"danger_prox={danger_prox:.2f}, circuit→danger ({self._action_name(circuit_action)})"
            return True, reason

        return False, ""

    def select_action(
        self,
        obs: np.ndarray,
        uncertainty: float = 0.0
    ) -> Tuple[int, Dict[str, Any]]:
        """
        메인 action 선택 함수

        Returns:
            (action, info_dict)
        """
        self.step_count += 1
        self.metrics.total_steps += 1

        info = {
            'controller': 'circuit',
            'fallback': False,
            'compared_fep': False,
            'disagreement': False,
        }

        # === Circuit 실행 ===
        self.circuit.pc_core.reset()
        circuit_action, circuit_result = self.circuit.select_action(obs, uncertainty)

        # === FEP 비교 (조건부) ===
        fep_action = None
        fep_g_values = None

        if self.should_compare_fep(obs):
            info['compared_fep'] = True
            fep_g_values = self.oracle.get_g_values(obs)
            fep_action = min(fep_g_values.keys(), key=lambda a: fep_g_values[a])

            # Agreement 체크
            if fep_action == circuit_action:
                self.metrics.agreements += 1
                # N1a/N1b: Agreement도 safety/style 구분
                is_safety_context = self._is_safety_context(obs)
                if is_safety_context:
                    self.metrics.safety_agreements += 1
                else:
                    self.metrics.style_agreements += 1
            else:
                self.metrics.disagreements += 1
                info['disagreement'] = True
                self._record_disagreement(obs, fep_action, circuit_action, fep_g_values, circuit_result)

        # === Fallback 체크 ===
        if self.fallback_remaining > 0:
            # 이미 fallback 중
            self.fallback_remaining -= 1
            self.metrics.fep_steps += 1
            info['controller'] = 'fep_fallback'
            info['fallback'] = True
            info['fallback_remaining'] = self.fallback_remaining

            # Fallback 중에는 FEP action 사용
            if fep_action is None:
                fep_g_values = self.oracle.get_g_values(obs)
                fep_action = min(fep_g_values.keys(), key=lambda a: fep_g_values[a])

            final_action = fep_action
        else:
            # Fallback 필요 여부 체크
            if fep_action is not None:
                should_fb, reason = self.should_fallback(obs, circuit_action, fep_action)
                if should_fb:
                    self.fallback_remaining = self.fallback_duration
                    self.last_fallback_reason = reason
                    self.metrics.fallback_count += 1

                    self.fallback_events.append(FallbackEvent(
                        step=self.step_count,
                        trigger_reason=reason,
                        duration=self.fallback_duration,
                        observation=obs.copy(),
                        metrics={
                            'danger_prox': float(obs[1]) if len(obs) > 1 else 0.0,
                            'circuit_action': circuit_action,
                            'fep_action': fep_action,
                        }
                    ))

                    info['controller'] = 'fep_fallback'
                    info['fallback'] = True
                    info['fallback_reason'] = reason
                    final_action = fep_action
                    self.metrics.fep_steps += 1
                else:
                    final_action = circuit_action
                    self.metrics.circuit_steps += 1
            else:
                final_action = circuit_action
                self.metrics.circuit_steps += 1

        # === Danger 회피 추적 ===
        danger_prox = obs[1] if len(obs) > 1 else 0.0
        if danger_prox > 0.5:
            self.metrics.danger_encounters += 1
            worst = self._get_worst_action(obs)
            if circuit_action != worst:
                self.metrics.danger_avoided_circuit += 1
            if fep_action is not None and fep_action != worst:
                self.metrics.danger_avoided_fep += 1

        # === Info 추가 ===
        info['circuit_action'] = circuit_action
        info['fep_action'] = fep_action
        info['final_action'] = final_action
        info['circuit_energies'] = {ae.action: ae.energy for ae in circuit_result.action_energies} if circuit_result else {}
        info['circuit_margin'] = circuit_result.margin if circuit_result else 0.0

        return final_action, info

    def _record_disagreement(
        self,
        obs: np.ndarray,
        fep_action: int,
        circuit_action: int,
        fep_g_values: Dict[int, float],
        circuit_result
    ):
        """Disagreement 케이스 기록"""
        # 유형 분류
        dtype = self._classify_disagreement(obs, fep_action, circuit_action)
        self.type_counts[dtype] += 1

        # N1a/N1b: Safety vs Style 분류
        is_safety = dtype in (DisagreementType.DANGER_APPROACH, DisagreementType.FOOD_IGNORE)
        if is_safety:
            self.metrics.safety_disagreements += 1
            if dtype == DisagreementType.DANGER_APPROACH:
                self.metrics.danger_approach_count += 1
            elif dtype == DisagreementType.FOOD_IGNORE:
                self.metrics.food_ignore_count += 1
        else:
            self.metrics.style_disagreements += 1

        case = DisagreementCase(
            step=self.step_count,
            timestamp=time.time(),
            observation=obs.copy(),
            fep_action=fep_action,
            circuit_action=circuit_action,
            fep_g_values=fep_g_values,
            circuit_energies={ae.action: ae.energy for ae in circuit_result.action_energies} if circuit_result else {},
            disagreement_type=dtype,
            context={
                'danger_prox': float(obs[1]) if len(obs) > 1 else 0.0,
                'food_prox': float(obs[0]) if len(obs) > 0 else 0.0,
                'energy': float(obs[6]) if len(obs) > 6 else 0.5,
            }
        )

        self.disagreements.append(case)

        # 최대 개수 제한
        if len(self.disagreements) > self.max_disagreements:
            self.disagreements.pop(0)

    def _classify_disagreement(
        self,
        obs: np.ndarray,
        fep_action: int,
        circuit_action: int
    ) -> DisagreementType:
        """Disagreement 유형 분류"""
        danger_prox = obs[1] if len(obs) > 1 else 0.0
        food_prox = obs[0] if len(obs) > 0 else 0.0
        energy = obs[6] if len(obs) > 6 else 0.5
        danger_dx = obs[4] if len(obs) > 4 else 0.0
        danger_dy = obs[5] if len(obs) > 5 else 0.0
        food_dx = obs[2] if len(obs) > 2 else 0.0
        food_dy = obs[3] if len(obs) > 3 else 0.0

        # 1. Danger approach: Circuit이 위험 쪽으로
        if danger_prox > 0.5:
            moving_toward = False
            if circuit_action == 4 and danger_dx > 0:
                moving_toward = True
            elif circuit_action == 3 and danger_dx < 0:
                moving_toward = True
            elif circuit_action == 1 and danger_dy < 0:
                moving_toward = True
            elif circuit_action == 2 and danger_dy > 0:
                moving_toward = True

            if moving_toward:
                return DisagreementType.DANGER_APPROACH

        # 2. Food ignore: 음식 가깝고 에너지 낮은데 다른 방향
        if food_prox > 0.5 and energy < 0.4:
            # FEP가 음식 쪽으로 가는데 Circuit이 안 가면
            fep_toward_food = (
                (fep_action == 4 and food_dx > 0) or
                (fep_action == 3 and food_dx < 0) or
                (fep_action == 1 and food_dy < 0) or
                (fep_action == 2 and food_dy > 0)
            )
            if fep_toward_food and circuit_action != fep_action:
                return DisagreementType.FOOD_IGNORE

        # 3. Energy waste: 저에너지에서 FEP와 다른 선택
        if energy < 0.3:
            return DisagreementType.ENERGY_WASTE

        # 4. 그 외는 탐색 전략 차이
        return DisagreementType.EXPLORATION_DIFF

    def _get_worst_action(self, obs: np.ndarray) -> Optional[int]:
        """위험 쪽으로 이동하는 action 반환"""
        danger_dx = obs[4] if len(obs) > 4 else 0.0
        danger_dy = obs[5] if len(obs) > 5 else 0.0

        if abs(danger_dx) > abs(danger_dy):
            return 4 if danger_dx > 0 else 3
        else:
            return 2 if danger_dy > 0 else 1

    def _action_name(self, action: int) -> str:
        names = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'THINK']
        return names[action] if 0 <= action < len(names) else f'ACTION_{action}'

    def _is_safety_context(self, obs: np.ndarray) -> bool:
        """
        Safety-critical 상황인지 판단 (N1a 분류용)

        Safety contexts:
        - danger_prox > 0.5 (위험 근접)
        - food_prox > 0.5 AND energy < 0.4 (저에너지 음식 상황)
        """
        danger_prox = obs[1] if len(obs) > 1 else 0.0
        food_prox = obs[0] if len(obs) > 0 else 0.0
        energy = obs[6] if len(obs) > 6 else 0.5

        if danger_prox > 0.5:
            return True
        if food_prox > 0.5 and energy < 0.4:
            return True
        return False

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        total_comparisons = self.metrics.agreements + self.metrics.disagreements
        safety_total = self.metrics.safety_agreements + self.metrics.safety_disagreements
        style_total = self.metrics.style_agreements + self.metrics.style_disagreements

        # N1a: Safety-critical alignment (hard gate)
        # ONLY danger_approach matters for true safety
        n1a_passed = None
        if self.metrics.total_steps > 50:
            # 핵심: danger_approach는 0이어야 함 (위험 쪽으로 이동 금지)
            n1a_passed = self.metrics.danger_approach_count == 0

        # N1b: Style alignment (soft target, 50-80% is acceptable)
        n1b_passed = None
        if style_total > 30:
            n1b_passed = self.metrics.style_agreement_rate >= 0.50

        return {
            'enabled': self.enabled,
            'step_count': self.step_count,
            'metrics': self.metrics.to_dict(),
            'disagreement_count': len(self.disagreements),
            'type_distribution': {k.value: v for k, v in self.type_counts.items()},
            'fallback_count': len(self.fallback_events),
            'fallback_remaining': self.fallback_remaining,
            'last_fallback_reason': self.last_fallback_reason,
            # Gates
            'n0_passed': self.metrics.total_steps >= 1000 and self.metrics.crashes == 0,
            'n1_passed': 0.70 <= self.metrics.agreement_rate <= 0.95 if total_comparisons > 50 else None,  # Legacy
            'n1a_passed': n1a_passed,  # Safety-critical (hard gate)
            'n1b_passed': n1b_passed,  # Style (soft gate)
            'n2_passed': self.metrics.circuit_danger_avoidance >= self.metrics.fep_danger_avoidance * 0.95 if self.metrics.danger_encounters > 10 else None,
            'n3_passed': len(self.disagreements) >= 200 and len(self.type_counts) >= 3,
        }

    def get_disagreements(self, limit: int = 50, dtype: Optional[str] = None) -> List[Dict]:
        """Disagreement 케이스 조회"""
        cases = self.disagreements
        if dtype:
            try:
                target_type = DisagreementType(dtype)
                cases = [c for c in cases if c.disagreement_type == target_type]
            except ValueError:
                pass
        return [c.to_dict() for c in cases[-limit:]]

    def get_fallback_events(self, limit: int = 20) -> List[Dict]:
        """Fallback 이벤트 조회"""
        events = self.fallback_events[-limit:]
        return [{
            'step': e.step,
            'reason': e.trigger_reason,
            'duration': e.duration,
            'metrics': e.metrics,
        } for e in events]

    def reset_metrics(self):
        """메트릭 리셋"""
        self.metrics = P2Metrics()
        self.disagreements.clear()
        self.fallback_events.clear()
        self.type_counts.clear()
        self.step_count = 0
        self.fallback_remaining = 0
