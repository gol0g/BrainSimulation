"""
E8-Hard-TTC*: Approach-Gated TTC Ablation

E8-Hard에서 TTC가 실패한 이유:
- 랜덤 워크 위협에서 "우연히 가까워짐"을 진짜 접근으로 오인
- 과방어 → starvation

TTC* (Approach-gated TTC):
- 기존: if ttc < τ: defensive_on
- 개선: if ttc < τ AND closing_speed > 0 AND approach_streak >= m: defensive_on

핵심 변경:
1. closing_speed: 실제로 가까워지는 중인가?
2. approach_streak: 연속 m step 접근하는가?
3. 더 타이트한 τ_on

실험군:
- FULL+RF (현재 최강)
- FULL+RF+TTC (기존)
- FULL+RF+TTC* (approach-gated)

Usage:
    python test_e8_ttc_star.py --episodes 200
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# Environment (from E8-Hard)
# ============================================================================

@dataclass
class HardEnvConfig:
    grid_size: int = 10
    max_steps: int = 500
    energy_start: float = 1.0
    energy_decay: float = 0.015  # H3 level
    energy_from_food: float = 0.3
    energy_loss_danger: float = 0.3
    pain_from_danger: float = 0.5
    pain_decay: float = 0.02
    danger_move_prob: float = 0.9  # H3 level


class HardSurvivalEnv:
    def __init__(self, config: HardEnvConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.agent_pos = None
        self.food_pos = None
        self.danger_pos = None
        self.energy = None
        self.pain = None
        self.step_count = 0
        self.total_food = 0
        self.total_danger_hits = 0
        self.death_cause = None

    def _random_pos(self) -> List[int]:
        return [self.rng.integers(0, self.config.grid_size),
                self.rng.integers(0, self.config.grid_size)]

    def _nearby_pos(self, center: List[int], radius: int = 3) -> List[int]:
        for _ in range(50):
            dx = self.rng.integers(-radius, radius + 1)
            dy = self.rng.integers(-radius, radius + 1)
            if dx == 0 and dy == 0:
                continue
            x = max(0, min(self.config.grid_size - 1, center[0] + dx))
            y = max(0, min(self.config.grid_size - 1, center[1] + dy))
            return [x, y]
        return self._random_pos()

    def _far_pos(self, center: List[int], min_dist: int = 4) -> List[int]:
        for _ in range(50):
            pos = self._random_pos()
            dist = abs(pos[0] - center[0]) + abs(pos[1] - center[1])
            if dist >= min_dist:
                return pos
        return self._random_pos()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agent_pos = [self.config.grid_size // 2, self.config.grid_size // 2]
        self.food_pos = self._nearby_pos(self.agent_pos, radius=3)
        self.danger_pos = self._far_pos(self.agent_pos, min_dist=4)
        self.energy = self.config.energy_start
        self.pain = 0.0
        self.step_count = 0
        self.total_food = 0
        self.total_danger_hits = 0
        self.death_cause = None
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(8, dtype=np.float32)
        dx_food = self.food_pos[0] - self.agent_pos[0]
        dy_food = self.food_pos[1] - self.agent_pos[1]
        dist_food = abs(dx_food) + abs(dy_food)
        obs[0] = max(0, 1 - dist_food / self.config.grid_size)
        obs[2] = np.sign(dx_food) if dx_food != 0 else 0
        obs[3] = np.sign(dy_food) if dy_food != 0 else 0

        dx_danger = self.danger_pos[0] - self.agent_pos[0]
        dy_danger = self.danger_pos[1] - self.agent_pos[1]
        dist_danger = abs(dx_danger) + abs(dy_danger)
        obs[1] = max(0, 1 - dist_danger / 5.0)
        obs[4] = np.sign(dx_danger) if dx_danger != 0 else 0
        obs[5] = np.sign(dy_danger) if dy_danger != 0 else 0
        obs[6] = self.energy
        obs[7] = self.pain
        return obs

    def get_danger_distance(self) -> int:
        return abs(self.agent_pos[0] - self.danger_pos[0]) + \
               abs(self.agent_pos[1] - self.danger_pos[1])

    def step(self, action: int) -> Tuple[np.ndarray, bool, Dict]:
        dx, dy = 0, 0
        if action == 1:
            dy = -1
        elif action == 2:
            dy = 1
        elif action == 3:
            dx = -1
        elif action == 4:
            dx = 1

        new_x = max(0, min(self.config.grid_size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.config.grid_size - 1, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]

        info = {'ate_food': False, 'hit_danger': False, 'died': False, 'death_cause': None}

        if self.agent_pos == self.food_pos:
            self.energy = min(1.0, self.energy + self.config.energy_from_food)
            self.total_food += 1
            info['ate_food'] = True
            self.food_pos = self._nearby_pos(self.agent_pos, radius=3)

        if self.agent_pos == self.danger_pos:
            self.energy = max(0.0, self.energy - self.config.energy_loss_danger)
            self.pain = min(1.0, self.pain + self.config.pain_from_danger)
            self.total_danger_hits += 1
            info['hit_danger'] = True

        self.energy = max(0.0, self.energy - self.config.energy_decay)
        self.pain = max(0.0, self.pain - self.config.pain_decay)

        if self.rng.random() < self.config.danger_move_prob:
            ddx = self.rng.choice([-1, 0, 1])
            ddy = self.rng.choice([-1, 0, 1])
            new_dx = max(0, min(self.config.grid_size - 1, self.danger_pos[0] + ddx))
            new_dy = max(0, min(self.config.grid_size - 1, self.danger_pos[1] + ddy))
            self.danger_pos = [new_dx, new_dy]

        self.step_count += 1

        died = False
        if self.energy <= 0:
            died = True
            self.death_cause = 'danger' if info['hit_danger'] else 'starvation'
            info['death_cause'] = self.death_cause
            info['died'] = True
        elif self.step_count >= self.config.max_steps:
            died = True

        info['energy'] = self.energy
        info['step'] = self.step_count
        return self._get_obs(), died, info


# ============================================================================
# Agents
# ============================================================================

class RiskFilterAgent:
    """FULL+RF: 현재 H3 최강"""
    def __init__(self, risk_on: float = 0.5, risk_off: float = 0.2, risk_ema_alpha: float = 0.5):
        self.name = "FULL+RF"
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.risk_ema = None
        self.in_defense_mode = False

    def act(self, obs: np.ndarray, env=None) -> int:
        food_dx, food_dy = obs[2], obs[3]
        danger_prox = obs[1]
        danger_dx, danger_dy = obs[4], obs[5]

        raw_risk = danger_prox
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        if self.in_defense_mode:
            if self.risk_ema < self.risk_off:
                self.in_defense_mode = False
        else:
            if self.risk_ema > self.risk_on:
                self.in_defense_mode = True

        if self.in_defense_mode:
            return self._avoid_danger(danger_dx, danger_dy)
        else:
            return self._toward_food(food_dx, food_dy)

    def _toward_food(self, dx, dy):
        if abs(dx) >= abs(dy):
            return 4 if dx > 0 else 3 if dx < 0 else 0
        return 2 if dy > 0 else 1 if dy < 0 else 0

    def _avoid_danger(self, dx, dy):
        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 4 if dx < 0 else 0
        return 1 if dy > 0 else 2 if dy < 0 else 0

    def reset(self):
        self.risk_ema = None
        self.in_defense_mode = False


class RiskFilterTTCAgent:
    """FULL+RF+TTC: 기존 TTC (과반응 문제)"""
    def __init__(self, risk_on: float = 0.5, risk_off: float = 0.2,
                 risk_ema_alpha: float = 0.5, ttc_dist_threshold: int = 3):
        self.name = "FULL+RF+TTC"
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.ttc_dist_threshold = ttc_dist_threshold
        self.risk_ema = None
        self.in_defense_mode = False
        self.ttc_triggered = False
        self.prev_danger_dist = None

    def act(self, obs: np.ndarray, env=None) -> int:
        food_dx, food_dy = obs[2], obs[3]
        danger_prox = obs[1]
        danger_dx, danger_dy = obs[4], obs[5]

        current_danger_dist = int(5 * (1 - danger_prox))

        danger_approaching = False
        if self.prev_danger_dist is not None:
            if current_danger_dist < self.prev_danger_dist:
                danger_approaching = True
        self.prev_danger_dist = current_danger_dist

        raw_risk = danger_prox
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        # 기존 TTC: 한 번 접근하면 바로 트리거
        if current_danger_dist <= self.ttc_dist_threshold and danger_approaching:
            self.ttc_triggered = True
            self.in_defense_mode = True
        elif current_danger_dist > self.ttc_dist_threshold + 1:
            self.ttc_triggered = False

        if not self.ttc_triggered:
            if self.in_defense_mode:
                if self.risk_ema < self.risk_off:
                    self.in_defense_mode = False
            else:
                if self.risk_ema > self.risk_on:
                    self.in_defense_mode = True

        if self.in_defense_mode:
            return self._avoid_danger(danger_dx, danger_dy)
        else:
            return self._toward_food(food_dx, food_dy)

    def _toward_food(self, dx, dy):
        if abs(dx) >= abs(dy):
            return 4 if dx > 0 else 3 if dx < 0 else 0
        return 2 if dy > 0 else 1 if dy < 0 else 0

    def _avoid_danger(self, dx, dy):
        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 4 if dx < 0 else 0
        return 1 if dy > 0 else 2 if dy < 0 else 0

    def reset(self):
        self.risk_ema = None
        self.in_defense_mode = False
        self.ttc_triggered = False
        self.prev_danger_dist = None


class RiskFilterTTCStarAgent:
    """
    FULL+RF+TTC*: Approach-Gated TTC

    핵심 변경:
    1. approach_streak >= m: 연속 m step 접근해야 트리거
    2. 더 타이트한 ttc_dist_threshold
    3. closing_speed 기반 확인
    """
    def __init__(
        self,
        risk_on: float = 0.5,
        risk_off: float = 0.2,
        risk_ema_alpha: float = 0.5,
        ttc_dist_threshold: int = 2,   # 더 타이트하게 (기존 3)
        approach_streak_min: int = 2,  # 연속 2 step 접근 필요
        ttc_off_dist: int = 4,         # 4 이상 멀어지면 해제
    ):
        self.name = "FULL+RF+TTC*"
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.ttc_dist_threshold = ttc_dist_threshold
        self.approach_streak_min = approach_streak_min
        self.ttc_off_dist = ttc_off_dist

        self.risk_ema = None
        self.in_defense_mode = False
        self.ttc_triggered = False
        self.prev_danger_dist = None
        self.approach_streak = 0       # 연속 접근 카운터

    def act(self, obs: np.ndarray, env=None) -> int:
        food_dx, food_dy = obs[2], obs[3]
        danger_prox = obs[1]
        danger_dx, danger_dy = obs[4], obs[5]

        # 현재 danger 거리
        current_danger_dist = int(5 * (1 - danger_prox))

        # Closing speed 계산 (양수 = 가까워지는 중)
        closing = 0
        if self.prev_danger_dist is not None:
            closing = self.prev_danger_dist - current_danger_dist

        # Approach streak 업데이트
        if closing > 0:
            self.approach_streak += 1
        else:
            self.approach_streak = 0  # 멀어지거나 유지되면 리셋

        self.prev_danger_dist = current_danger_dist

        # Risk EMA
        raw_risk = danger_prox
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        # TTC* 트리거: 3중 조건!
        # 1. 거리 < threshold
        # 2. 실제로 가까워지는 중 (closing > 0)
        # 3. 연속 m step 이상 접근
        ttc_condition = (
            current_danger_dist <= self.ttc_dist_threshold and
            closing > 0 and
            self.approach_streak >= self.approach_streak_min
        )

        if ttc_condition:
            self.ttc_triggered = True
            self.in_defense_mode = True
        elif current_danger_dist >= self.ttc_off_dist or self.approach_streak == 0:
            # 멀어졌거나 접근이 끊기면 TTC 해제
            self.ttc_triggered = False

        # 기존 risk hysteresis (TTC 트리거 아닐 때만)
        if not self.ttc_triggered:
            if self.in_defense_mode:
                if self.risk_ema < self.risk_off:
                    self.in_defense_mode = False
            else:
                if self.risk_ema > self.risk_on:
                    self.in_defense_mode = True

        if self.in_defense_mode:
            return self._avoid_danger(danger_dx, danger_dy)
        else:
            return self._toward_food(food_dx, food_dy)

    def _toward_food(self, dx, dy):
        if abs(dx) >= abs(dy):
            return 4 if dx > 0 else 3 if dx < 0 else 0
        return 2 if dy > 0 else 1 if dy < 0 else 0

    def _avoid_danger(self, dx, dy):
        if abs(dx) >= abs(dy):
            return 3 if dx > 0 else 4 if dx < 0 else 0
        return 1 if dy > 0 else 2 if dy < 0 else 0

    def reset(self):
        self.risk_ema = None
        self.in_defense_mode = False
        self.ttc_triggered = False
        self.prev_danger_dist = None
        self.approach_streak = 0


# ============================================================================
# Test Runner
# ============================================================================

def run_ttc_star_test(n_episodes: int = 200, base_seed: int = 42) -> Dict:
    """TTC* Ablation Test on H3"""
    print("\n" + "=" * 70)
    print("  E8-Hard-TTC*: Approach-Gated TTC Ablation")
    print("  Level: H3 (move=0.9, decay=0.015)")
    print("=" * 70)

    start_time = time.time()

    agents = {
        'FULL+RF': RiskFilterAgent,
        'FULL+RF+TTC': RiskFilterTTCAgent,
        'FULL+RF+TTC*': RiskFilterTTCStarAgent,
    }

    config = HardEnvConfig()  # H3 level
    results = {}

    for agent_name, agent_class in agents.items():
        print(f"\n  Testing {agent_name}...")

        all_survival = []
        all_food = []
        all_danger = []
        all_deaths = []
        death_by_danger = []
        death_by_starvation = []

        for ep in range(n_episodes):
            seed = base_seed + ep
            env = HardSurvivalEnv(config, seed=seed)
            agent = agent_class()
            agent.reset()

            obs = env.reset(seed=seed)

            for _ in range(config.max_steps):
                action = agent.act(obs, env)
                obs, done, info = env.step(action)
                if done:
                    break

            all_survival.append(env.step_count)
            all_food.append(env.total_food)
            all_danger.append(env.total_danger_hits)

            died = info.get('died', False) and env.energy <= 0
            all_deaths.append(1 if died else 0)

            if died:
                if env.death_cause == 'danger':
                    death_by_danger.append(1)
                    death_by_starvation.append(0)
                else:
                    death_by_danger.append(0)
                    death_by_starvation.append(1)
            else:
                death_by_danger.append(0)
                death_by_starvation.append(0)

        results[agent_name] = {
            'survival': np.mean(all_survival),
            'food': np.mean(all_food),
            'danger': np.mean(all_danger),
            'death_rate': np.mean(all_deaths),
            'death_by_danger': np.sum(death_by_danger),
            'death_by_starvation': np.sum(death_by_starvation),
        }

        r = results[agent_name]
        print(f"    Survival: {r['survival']:>5.0f} | Food: {r['food']:>5.1f} | "
              f"Danger: {r['danger']:>4.2f} | Death: {r['death_rate']:>5.1%} "
              f"(D:{r['death_by_danger']:>3}, S:{r['death_by_starvation']:>3})")

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print("  Comparison Summary")
    print(f"{'='*70}")

    rf = results['FULL+RF']
    ttc = results['FULL+RF+TTC']
    ttc_star = results['FULL+RF+TTC*']

    print(f"\n  {'Metric':<20} | {'FULL+RF':>12} | {'TTC':>12} | {'TTC*':>12}")
    print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    print(f"  {'Survival':<20} | {rf['survival']:>12.0f} | {ttc['survival']:>12.0f} | {ttc_star['survival']:>12.0f}")
    print(f"  {'Food':<20} | {rf['food']:>12.1f} | {ttc['food']:>12.1f} | {ttc_star['food']:>12.1f}")
    print(f"  {'Danger Hits':<20} | {rf['danger']:>12.2f} | {ttc['danger']:>12.2f} | {ttc_star['danger']:>12.2f}")
    print(f"  {'Death Rate':<20} | {rf['death_rate']:>11.1%} | {ttc['death_rate']:>11.1%} | {ttc_star['death_rate']:>11.1%}")
    print(f"  {'Death by Danger':<20} | {rf['death_by_danger']:>12} | {ttc['death_by_danger']:>12} | {ttc_star['death_by_danger']:>12}")
    print(f"  {'Death by Starvation':<20} | {rf['death_by_starvation']:>12} | {ttc['death_by_starvation']:>12} | {ttc_star['death_by_starvation']:>12}")

    # Analysis
    print(f"\n{'='*70}")
    print("  TTC* Effectiveness Analysis")
    print(f"{'='*70}")

    # TTC* vs TTC
    starvation_reduction = ttc['death_by_starvation'] - ttc_star['death_by_starvation']
    danger_change = ttc_star['danger'] - ttc['danger']
    death_rate_improvement = ttc['death_rate'] - ttc_star['death_rate']

    print(f"\n  TTC* vs TTC:")
    print(f"    Starvation 감소: {starvation_reduction:+d} ({ttc['death_by_starvation']} → {ttc_star['death_by_starvation']})")
    print(f"    Danger hits 변화: {danger_change:+.2f} ({ttc['danger']:.2f} → {ttc_star['danger']:.2f})")
    print(f"    Death rate 개선: {death_rate_improvement:+.1%}")

    # TTC* vs FULL+RF
    vs_rf_death = rf['death_rate'] - ttc_star['death_rate']
    vs_rf_danger = rf['danger'] - ttc_star['danger']

    print(f"\n  TTC* vs FULL+RF:")
    print(f"    Death rate: {vs_rf_death:+.1%} ({'TTC* better' if vs_rf_death > 0 else 'RF better'})")
    print(f"    Danger hits: {vs_rf_danger:+.2f} ({'TTC* safer' if vs_rf_danger > 0 else 'RF safer'})")

    # Verdict
    print(f"\n{'='*70}")
    print("  Verdict")
    print(f"{'='*70}")

    ttc_star_improved = starvation_reduction > 20 and ttc_star['death_rate'] < ttc['death_rate']
    ttc_star_competitive = ttc_star['death_rate'] <= rf['death_rate'] + 0.05

    if ttc_star_improved and ttc_star_competitive:
        verdict = "SUCCESS"
        explanation = """
  [SUCCESS] TTC* approach-gating이 과반응 문제 해결

  핵심:
  - Starvation 크게 감소 (approach 확인으로 가짜 트리거 제거)
  - Death rate가 FULL+RF에 근접 또는 능가
  - Danger hits는 여전히 낮게 유지

  결론:
  "TTC 개념은 유효하나, 트리거 조건이 환경 특성에 맞아야 한다."
  "랜덤 워크 위협에는 approach-gating이 필수."
        """
    elif ttc_star_improved:
        verdict = "PARTIAL"
        explanation = """
  [PARTIAL] TTC* 개선 효과 있으나 불완전

  관찰:
  - Starvation 감소 확인
  - 그러나 FULL+RF 대비 아직 불리

  해석:
  "이 레짐에서는 TTC 자체가 에너지 경제와 구조적 충돌"
        """
    else:
        verdict = "FAILED"
        explanation = """
  [FAILED] TTC* approach-gating 효과 미미

  관찰:
  - Starvation 감소 불충분
  - 환경 구조상 TTC가 맞지 않음

  결론:
  "이 레짐(랜덤 위협 + 에너지 경제 중심)에서는 RF만이 최적"
        """

    print(explanation)
    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    return {
        'verdict': verdict,
        'results': results,
        'starvation_reduction': starvation_reduction,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    result = run_ttc_star_test(args.episodes, args.seed)
    exit(0 if result['verdict'] == 'SUCCESS' else 1)
