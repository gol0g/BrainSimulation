"""
Temporal Depth - Multi-step Imagination/Rollout

핵심 개념:
- 현재: G(a) = 1-step expected free energy
- 확장: G(a) = E[Σ_{t=1}^{H} γ^{t-1} * G_t(a)]

여기서:
- H = rollout horizon (몇 스텝 앞까지 상상)
- γ = discount factor (미래 가치 할인)
- G_t = t스텝 후의 expected free energy
- E[...] = Monte Carlo 샘플링으로 기댓값 추정

왜 필요한가:
- 1-step만 보면 근시안적 결정
- n-step 보면 "지금은 나빠도 나중에 좋아지는" 경로 발견
- 예: 위험을 돌아가는 경로 (단기 손해, 장기 이득)

FEP 관점:
- 더 깊은 temporal depth = 더 정교한 생성 모델
- 시간적 추상화 = 계층적 예측
- **불확실성을 유지하며 미래를 적분** (Monte Carlo)

v2.4.1 변경:
- Stochastic rollout: P(o'|o,a) 분포에서 샘플링
- Complexity decay: 미래로 갈수록 complexity 가중치 감소
- Monte Carlo averaging: n_samples개 경로 평균
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RolloutStep:
    """한 스텝의 rollout 결과"""
    obs: np.ndarray       # 예측 관측
    G: float              # 이 스텝의 G
    risk: float
    ambiguity: float
    complexity: float
    action: int           # 이 스텝에서 선택할 행동


@dataclass
class RolloutResult:
    """전체 rollout 결과"""
    total_G: float                    # 할인된 총 G (Monte Carlo 평균)
    steps: List[RolloutStep]          # 대표 경로의 각 스텝 상세
    best_first_action: int            # 첫 번째로 취할 최적 행동
    horizon: int                      # rollout depth
    discount: float                   # 사용된 할인율
    n_samples: int = 1                # Monte Carlo 샘플 수
    G_std: float = 0.0                # G의 표준편차 (불확실성)


class TemporalPlanner:
    """
    Multi-step Temporal Planning

    핵심 역할:
    1. 여러 스텝 앞의 관측 예측
    2. 각 스텝에서 G 계산
    3. 할인된 총 G로 행동 선택

    주의:
    - 이것은 "상상" - 실제 환경 변화 없음
    - 재귀적 예측은 불확실성 누적
    - horizon이 길수록 ambiguity 증가
    """

    def __init__(self,
                 action_selector,  # ActionSelector 인스턴스
                 horizon: int = 3,
                 discount: float = 0.9,
                 n_samples: int = 3,
                 complexity_decay: float = 0.7):
        """
        Args:
            action_selector: G 계산에 사용할 ActionSelector
            horizon: 몇 스텝 앞까지 볼 것인가 (1-5 권장)
            discount: 미래 가치 할인율 (0.8-0.99)
            n_samples: Monte Carlo 샘플 수 (2-5 권장)
            complexity_decay: 미래 complexity 감쇠율 (0.5-0.9)
                - 1.0 = 모든 스텝 동일 가중치
                - 0.7 = step t에서 complexity *= 0.7^t
                - 이유: 먼 미래 믿음 이탈은 덜 확신 → 덜 페널티
        """
        self.action_selector = action_selector
        self.horizon = horizon
        self.discount = discount
        self.n_samples = n_samples
        self.complexity_decay = complexity_decay

        # 통계
        self.rollout_count = 0

    def predict_next_obs(self,
                         current_obs: np.ndarray,
                         action: int,
                         stochastic: bool = True) -> Tuple[np.ndarray, float]:
        """
        행동 후 다음 관측 예측 (stochastic sampling from P(o'|o,a))

        Args:
            current_obs: 현재 관측
            action: 수행할 행동
            stochastic: True면 분포에서 샘플링, False면 기댓값

        Returns:
            (predicted_obs, uncertainty)
        """
        if len(current_obs) < 6:
            return current_obs.copy(), 1.0

        pred_obs = current_obs.copy()
        delta_prox_base = 0.1

        food_dx, food_dy = current_obs[2], current_obs[3]
        danger_dx, danger_dy = current_obs[4], current_obs[5]
        food_prox, danger_prox = current_obs[0], current_obs[1]

        # 전이 모델에서 std 가져오기
        delta_std = self.action_selector.transition_model['delta_std'][action]
        food_std = delta_std[0] if len(delta_std) > 0 else 0.05
        danger_std = delta_std[1] if len(delta_std) > 1 else 0.05

        # Food proximity 변화 예측 (기댓값)
        delta_food_mean = 0.0
        if action == 1:  # UP
            if food_dy < 0: delta_food_mean = delta_prox_base
            elif food_dy > 0: delta_food_mean = -delta_prox_base
        elif action == 2:  # DOWN
            if food_dy > 0: delta_food_mean = delta_prox_base
            elif food_dy < 0: delta_food_mean = -delta_prox_base
        elif action == 3:  # LEFT
            if food_dx < 0: delta_food_mean = delta_prox_base
            elif food_dx > 0: delta_food_mean = -delta_prox_base
        elif action == 4:  # RIGHT
            if food_dx > 0: delta_food_mean = delta_prox_base
            elif food_dx < 0: delta_food_mean = -delta_prox_base

        # Danger proximity 변화 예측 (기댓값)
        danger_scale = danger_prox * 0.4
        delta_danger_mean = 0.0
        if action == 1:  # UP
            if danger_dy < 0: delta_danger_mean = danger_scale
            elif danger_dy > 0: delta_danger_mean = -danger_scale
        elif action == 2:  # DOWN
            if danger_dy > 0: delta_danger_mean = danger_scale
            elif danger_dy < 0: delta_danger_mean = -danger_scale
        elif action == 3:  # LEFT
            if danger_dx < 0: delta_danger_mean = danger_scale
            elif danger_dx > 0: delta_danger_mean = -danger_scale
        elif action == 4:  # RIGHT
            if danger_dx > 0: delta_danger_mean = danger_scale
            elif danger_dx < 0: delta_danger_mean = -danger_scale

        # Stochastic sampling: 분포에서 샘플링
        if stochastic:
            delta_food = delta_food_mean + np.random.normal(0, food_std)
            delta_danger = delta_danger_mean + np.random.normal(0, danger_std)
        else:
            delta_food = delta_food_mean
            delta_danger = delta_danger_mean

        pred_obs[0] = np.clip(food_prox + delta_food, 0.0, 1.0)
        pred_obs[1] = np.clip(danger_prox + delta_danger, 0.0, 1.0)

        # 불확실성: 전이 모델의 std
        uncertainty = float(np.mean([food_std, danger_std]))

        return pred_obs, uncertainty

    def _single_rollout(self,
                        start_obs: np.ndarray,
                        first_action: int,
                        stochastic: bool = True) -> Tuple[float, List[RolloutStep]]:
        """
        단일 rollout 경로 수행 (Monte Carlo 샘플 하나)

        Returns:
            (total_G, steps)
        """
        steps = []
        current_obs = start_obs.copy()
        total_G = 0.0
        cumulative_uncertainty = 0.0

        for t in range(self.horizon):
            # 첫 스텝은 지정된 행동, 이후는 softmax 샘플링 (stochastic policy)
            if t == 0:
                action = first_action
            else:
                G_decomp = self.action_selector.compute_G(current_obs=current_obs)
                if stochastic:
                    # Softmax sampling: G가 낮을수록 선택 확률 높음
                    G_values = np.array([G_decomp[a].G for a in range(len(G_decomp))])
                    # Temperature=1.0, negative G for softmax (lower is better)
                    probs = np.exp(-G_values) / np.sum(np.exp(-G_values))
                    action = np.random.choice(len(G_decomp), p=probs)
                else:
                    action = min(G_decomp.keys(), key=lambda a: G_decomp[a].G)

            # 이 행동의 G 계산
            G_decomp = self.action_selector.compute_G(current_obs=current_obs)
            g = G_decomp[action]

            # 불확실성 누적 → ambiguity 증가
            uncertainty_penalty = cumulative_uncertainty * 0.1
            adjusted_ambiguity = g.ambiguity + uncertainty_penalty

            # Complexity decay: 먼 미래의 complexity는 덜 페널티
            # 이유: 미래 믿음 이탈은 불확실 → 현재에만 강하게 적용
            complexity_weight = self.complexity_decay ** t
            adjusted_complexity = g.complexity * complexity_weight

            # 조정된 G
            adjusted_G = g.risk + adjusted_ambiguity + adjusted_complexity

            # 스텝 기록
            step = RolloutStep(
                obs=current_obs.copy(),
                G=adjusted_G,
                risk=g.risk,
                ambiguity=adjusted_ambiguity,
                complexity=adjusted_complexity,
                action=action
            )
            steps.append(step)

            # 할인된 G 누적
            discount_factor = self.discount ** t
            total_G += discount_factor * adjusted_G

            # 다음 관측 예측 (stochastic)
            next_obs, uncertainty = self.predict_next_obs(current_obs, action, stochastic=stochastic)
            cumulative_uncertainty += uncertainty
            current_obs = next_obs

        return total_G, steps

    def rollout_action(self,
                       start_obs: np.ndarray,
                       first_action: int) -> RolloutResult:
        """
        특정 첫 행동에 대한 Monte Carlo rollout 수행

        n_samples개의 stochastic 경로를 샘플링하고 평균 G 계산

        Args:
            start_obs: 시작 관측
            first_action: 첫 번째 행동

        Returns:
            RolloutResult with Monte Carlo averaged G
        """
        all_Gs = []
        all_steps = []

        for _ in range(self.n_samples):
            total_G, steps = self._single_rollout(start_obs, first_action, stochastic=True)
            all_Gs.append(total_G)
            all_steps.append(steps)

        self.rollout_count += 1

        # Monte Carlo 평균 및 표준편차
        mean_G = np.mean(all_Gs)
        std_G = np.std(all_Gs) if len(all_Gs) > 1 else 0.0

        # 대표 경로: 평균에 가장 가까운 것
        best_idx = np.argmin([abs(g - mean_G) for g in all_Gs])
        representative_steps = all_steps[best_idx]

        return RolloutResult(
            total_G=mean_G,
            steps=representative_steps,
            best_first_action=first_action,
            horizon=self.horizon,
            discount=self.discount,
            n_samples=self.n_samples,
            G_std=std_G
        )

    def compute_G_with_rollout(self,
                                current_obs: np.ndarray
                                ) -> Dict[int, RolloutResult]:
        """
        모든 물리 행동에 대해 rollout 수행하고 결과 반환

        Args:
            current_obs: 현재 관측

        Returns:
            Dict[action, RolloutResult]
        """
        results = {}

        # v3.4: THINK (action 5)는 rollout에서 제외 - 물리 행동만 평가
        n_physical = getattr(self.action_selector, 'N_PHYSICAL_ACTIONS', self.action_selector.n_actions)
        for action in range(n_physical):
            results[action] = self.rollout_action(current_obs, action)

        return results

    def select_action_with_rollout(self,
                                    current_obs: np.ndarray
                                    ) -> Tuple[int, Dict[int, RolloutResult]]:
        """
        Rollout 기반 행동 선택

        Returns:
            (best_action, all_rollout_results)
        """
        rollouts = self.compute_G_with_rollout(current_obs)

        # 총 G가 가장 낮은 행동 선택
        best_action = min(rollouts.keys(), key=lambda a: rollouts[a].total_G)

        return best_action, rollouts

    def set_horizon(self, horizon: int):
        """Rollout 깊이 설정"""
        self.horizon = max(1, min(10, horizon))

    def set_discount(self, discount: float):
        """할인율 설정"""
        self.discount = max(0.5, min(0.99, discount))

    def get_rollout_summary(self, rollouts: Dict[int, RolloutResult]) -> Dict:
        """Rollout 결과 요약 (Monte Carlo 정보 포함)"""
        action_labels = {0: 'STAY', 1: 'UP', 2: 'DOWN', 3: 'LEFT', 4: 'RIGHT'}

        summary = {
            'horizon': self.horizon,
            'discount': self.discount,
            'n_samples': self.n_samples,
            'complexity_decay': self.complexity_decay,
            'actions': {}
        }

        for action, result in rollouts.items():
            # 각 스텝의 주도 요인 추적
            dominant_per_step = []
            for step in result.steps:
                factors = {'risk': step.risk, 'ambiguity': step.ambiguity, 'complexity': step.complexity}
                dominant = max(factors, key=factors.get)
                dominant_per_step.append(dominant)

            summary['actions'][action_labels[action]] = {
                'total_G': round(result.total_G, 3),
                'G_std': round(result.G_std, 3),  # Monte Carlo 불확실성
                'step_Gs': [round(s.G, 3) for s in result.steps],
                'dominant_factors': dominant_per_step,
                'trajectory': [s.action for s in result.steps]
            }

        # 최적 행동
        best = min(rollouts.keys(), key=lambda a: rollouts[a].total_G)
        summary['best_action'] = action_labels[best]
        summary['best_total_G'] = round(rollouts[best].total_G, 3)

        return summary
