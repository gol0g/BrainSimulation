"""
E8-TrackCurve: TTC Justification Phase Diagram

Tracking bias sweep으로 "TTC가 정당화되는 임계점" 찾기

Tracking bias 정의:
- bias=0.0: 완전 랜덤 (현재 E8)
- bias=0.3: 약추적 (30% 확률로 agent 방향 이동)
- bias=0.6: 중추적 (60%)
- bias=0.9: 강추적 (90%)

핵심 질문:
- 어느 bias(b*)에서 TTC가 FULL+RF를 역전하는가?
- 그 임계점이 "TTC 정당화 조건"의 정량적 경계

비교군 (3개):
- FULL+RF: 에너지 경제 최적
- FULL+RF+TTC*: 랜덤 위협 완화형
- FULL+RF+TTC: 예측 가능 위협 최적

Usage:
    python test_e8_track_curve.py --episodes 150
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# Environment with Tracking Bias
# ============================================================================

@dataclass
class TrackingEnvConfig:
    grid_size: int = 10
    max_steps: int = 500
    energy_start: float = 1.0
    energy_decay: float = 0.012  # H2 level (중간 난이도)
    energy_from_food: float = 0.3
    energy_loss_danger: float = 0.3
    pain_from_danger: float = 0.5
    pain_decay: float = 0.02
    danger_move_prob: float = 0.7  # H2 level

    # Tracking bias (핵심!)
    tracking_bias: float = 0.0  # 0.0=random, 1.0=perfect tracking


class TrackingSurvivalEnv:
    """Tracking bias가 있는 생존 환경"""

    def __init__(self, config: TrackingEnvConfig, seed: Optional[int] = None):
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
        self.defense_steps = 0  # 방어 모드 step 수 추적

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
        self.defense_steps = 0
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

    def _move_danger_with_tracking(self):
        """Tracking bias에 따라 danger 이동"""
        if self.rng.random() >= self.config.danger_move_prob:
            return  # 이동 안 함

        if self.rng.random() < self.config.tracking_bias:
            # Tracking: agent 방향으로 이동
            dx = np.sign(self.agent_pos[0] - self.danger_pos[0])
            dy = np.sign(self.agent_pos[1] - self.danger_pos[1])

            # 한 방향만 이동 (대각선 이동 방지)
            if abs(dx) > 0 and abs(dy) > 0:
                if self.rng.random() < 0.5:
                    dy = 0
                else:
                    dx = 0
        else:
            # Random: 기존 랜덤 이동
            dx = self.rng.choice([-1, 0, 1])
            dy = self.rng.choice([-1, 0, 1])

        new_x = max(0, min(self.config.grid_size - 1, self.danger_pos[0] + dx))
        new_y = max(0, min(self.config.grid_size - 1, self.danger_pos[1] + dy))
        self.danger_pos = [new_x, new_y]

    def step(self, action: int, in_defense_mode: bool = False) -> Tuple[np.ndarray, bool, Dict]:
        # 방어 모드 추적
        if in_defense_mode:
            self.defense_steps += 1

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

        # Tracking bias 적용된 danger 이동
        self._move_danger_with_tracking()

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
# Agents (from previous tests)
# ============================================================================

class RiskFilterAgent:
    def __init__(self, risk_on=0.5, risk_off=0.2, risk_ema_alpha=0.5):
        self.name = "FULL+RF"
        self.risk_on, self.risk_off = risk_on, risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.risk_ema = None
        self.in_defense_mode = False

    def act(self, obs, env=None):
        food_dx, food_dy = obs[2], obs[3]
        danger_prox, danger_dx, danger_dy = obs[1], obs[4], obs[5]

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
            if abs(danger_dx) >= abs(danger_dy):
                return 3 if danger_dx > 0 else 4 if danger_dx < 0 else 0
            return 1 if danger_dy > 0 else 2 if danger_dy < 0 else 0
        else:
            if abs(food_dx) >= abs(food_dy):
                return 4 if food_dx > 0 else 3 if food_dx < 0 else 0
            return 2 if food_dy > 0 else 1 if food_dy < 0 else 0

    def reset(self):
        self.risk_ema = None
        self.in_defense_mode = False

    def get_defense_mode(self):
        return self.in_defense_mode


class RiskFilterTTCAgent:
    def __init__(self, risk_on=0.5, risk_off=0.2, risk_ema_alpha=0.5, ttc_dist_threshold=3):
        self.name = "FULL+RF+TTC"
        self.risk_on, self.risk_off = risk_on, risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.ttc_dist_threshold = ttc_dist_threshold
        self.risk_ema = None
        self.in_defense_mode = False
        self.ttc_triggered = False
        self.prev_danger_dist = None

    def act(self, obs, env=None):
        food_dx, food_dy = obs[2], obs[3]
        danger_prox, danger_dx, danger_dy = obs[1], obs[4], obs[5]

        current_danger_dist = int(5 * (1 - danger_prox))

        danger_approaching = False
        if self.prev_danger_dist is not None and current_danger_dist < self.prev_danger_dist:
            danger_approaching = True
        self.prev_danger_dist = current_danger_dist

        raw_risk = danger_prox
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        if current_danger_dist <= self.ttc_dist_threshold and danger_approaching:
            self.ttc_triggered = True
            self.in_defense_mode = True
        elif current_danger_dist > self.ttc_dist_threshold + 1:
            self.ttc_triggered = False

        if not self.ttc_triggered:
            if self.in_defense_mode and self.risk_ema < self.risk_off:
                self.in_defense_mode = False
            elif not self.in_defense_mode and self.risk_ema > self.risk_on:
                self.in_defense_mode = True

        if self.in_defense_mode:
            if abs(danger_dx) >= abs(danger_dy):
                return 3 if danger_dx > 0 else 4 if danger_dx < 0 else 0
            return 1 if danger_dy > 0 else 2 if danger_dy < 0 else 0
        else:
            if abs(food_dx) >= abs(food_dy):
                return 4 if food_dx > 0 else 3 if food_dx < 0 else 0
            return 2 if food_dy > 0 else 1 if food_dy < 0 else 0

    def reset(self):
        self.risk_ema = None
        self.in_defense_mode = False
        self.ttc_triggered = False
        self.prev_danger_dist = None

    def get_defense_mode(self):
        return self.in_defense_mode


class RiskFilterTTCStarAgent:
    def __init__(self, risk_on=0.5, risk_off=0.2, risk_ema_alpha=0.5,
                 ttc_dist_threshold=2, approach_streak_min=2, ttc_off_dist=4):
        self.name = "FULL+RF+TTC*"
        self.risk_on, self.risk_off = risk_on, risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.ttc_dist_threshold = ttc_dist_threshold
        self.approach_streak_min = approach_streak_min
        self.ttc_off_dist = ttc_off_dist
        self.risk_ema = None
        self.in_defense_mode = False
        self.ttc_triggered = False
        self.prev_danger_dist = None
        self.approach_streak = 0

    def act(self, obs, env=None):
        food_dx, food_dy = obs[2], obs[3]
        danger_prox, danger_dx, danger_dy = obs[1], obs[4], obs[5]

        current_danger_dist = int(5 * (1 - danger_prox))

        closing = 0
        if self.prev_danger_dist is not None:
            closing = self.prev_danger_dist - current_danger_dist

        if closing > 0:
            self.approach_streak += 1
        else:
            self.approach_streak = 0

        self.prev_danger_dist = current_danger_dist

        raw_risk = danger_prox
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = self.risk_ema_alpha * raw_risk + (1 - self.risk_ema_alpha) * self.risk_ema

        ttc_condition = (
            current_danger_dist <= self.ttc_dist_threshold and
            closing > 0 and
            self.approach_streak >= self.approach_streak_min
        )

        if ttc_condition:
            self.ttc_triggered = True
            self.in_defense_mode = True
        elif current_danger_dist >= self.ttc_off_dist or self.approach_streak == 0:
            self.ttc_triggered = False

        if not self.ttc_triggered:
            if self.in_defense_mode and self.risk_ema < self.risk_off:
                self.in_defense_mode = False
            elif not self.in_defense_mode and self.risk_ema > self.risk_on:
                self.in_defense_mode = True

        if self.in_defense_mode:
            if abs(danger_dx) >= abs(danger_dy):
                return 3 if danger_dx > 0 else 4 if danger_dx < 0 else 0
            return 1 if danger_dy > 0 else 2 if danger_dy < 0 else 0
        else:
            if abs(food_dx) >= abs(food_dy):
                return 4 if food_dx > 0 else 3 if food_dx < 0 else 0
            return 2 if food_dy > 0 else 1 if food_dy < 0 else 0

    def reset(self):
        self.risk_ema = None
        self.in_defense_mode = False
        self.ttc_triggered = False
        self.prev_danger_dist = None
        self.approach_streak = 0

    def get_defense_mode(self):
        return self.in_defense_mode


# ============================================================================
# Test Runner
# ============================================================================

def run_single_bias_test(bias: float, agents: Dict, n_episodes: int, base_seed: int) -> Dict:
    """단일 tracking bias 레벨 테스트"""
    results = {}

    for agent_name, agent_class in agents.items():
        all_survival = []
        all_food = []
        all_danger = []
        all_deaths = []
        all_starvation = []
        all_defense_steps = []

        for ep in range(n_episodes):
            seed = base_seed + ep
            config = TrackingEnvConfig(tracking_bias=bias)
            env = TrackingSurvivalEnv(config, seed=seed)
            agent = agent_class()
            agent.reset()

            obs = env.reset(seed=seed)

            for _ in range(config.max_steps):
                action = agent.act(obs, env)
                obs, done, info = env.step(action, agent.get_defense_mode())
                if done:
                    break

            all_survival.append(env.step_count)
            all_food.append(env.total_food)
            all_danger.append(env.total_danger_hits)
            all_defense_steps.append(env.defense_steps)

            died = info.get('died', False) and env.energy <= 0
            all_deaths.append(1 if died else 0)
            all_starvation.append(1 if env.death_cause == 'starvation' else 0)

        # Net energy margin: 0.3*food - 0.3*danger - 0.012*survival (energy decay cost)
        mean_food = np.mean(all_food)
        mean_danger = np.mean(all_danger)
        mean_survival = np.mean(all_survival)
        net_margin = 0.3 * mean_food - 0.3 * mean_danger - 0.012 * mean_survival

        results[agent_name] = {
            'survival': mean_survival,
            'food': mean_food,
            'danger': mean_danger,
            'death_rate': np.mean(all_deaths),
            'starvation_rate': np.mean(all_starvation),
            'defense_ratio': np.mean(all_defense_steps) / max(mean_survival, 1),
            'net_margin': net_margin,
        }

    return results


def run_tracking_curve(n_episodes: int = 150, base_seed: int = 42) -> Dict:
    """Tracking bias sweep - TTC 정당화 곡선"""
    print("\n" + "=" * 75)
    print("  E8-TrackCurve: TTC Justification Phase Diagram")
    print("  Sweep: tracking_bias = [0.0, 0.3, 0.6, 0.9]")
    print("=" * 75)

    start_time = time.time()

    biases = [0.0, 0.3, 0.6, 0.9]
    agents = {
        'FULL+RF': RiskFilterAgent,
        'FULL+RF+TTC*': RiskFilterTTCStarAgent,
        'FULL+RF+TTC': RiskFilterTTCAgent,
    }

    all_results = {}

    for bias in biases:
        print(f"\n  === Tracking Bias = {bias:.1f} ===")
        results = run_single_bias_test(bias, agents, n_episodes, base_seed)
        all_results[bias] = results

        for name, r in results.items():
            print(f"    {name:<14} | Death: {r['death_rate']:>5.1%} | "
                  f"Danger: {r['danger']:>4.1f} | Food: {r['food']:>5.1f} | "
                  f"Net: {r['net_margin']:>+5.1f}")

    elapsed = time.time() - start_time

    # Phase Diagram Summary
    print(f"\n{'='*75}")
    print("  Phase Diagram: Death Rate by Tracking Bias")
    print(f"{'='*75}")

    print(f"\n  {'Bias':<8} | {'FULL+RF':>12} | {'TTC*':>12} | {'TTC':>12} | {'Winner':>14}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}")

    crossover_bias = None
    for bias in biases:
        rf = all_results[bias]['FULL+RF']['death_rate']
        ttc_star = all_results[bias]['FULL+RF+TTC*']['death_rate']
        ttc = all_results[bias]['FULL+RF+TTC']['death_rate']

        # Winner
        rates = {'RF': rf, 'TTC*': ttc_star, 'TTC': ttc}
        winner = min(rates, key=rates.get)

        print(f"  {bias:<8.1f} | {rf:>11.1%} | {ttc_star:>11.1%} | {ttc:>11.1%} | {winner:>14}")

        # Crossover detection (TTC or TTC* beats RF)
        if crossover_bias is None and (ttc < rf or ttc_star < rf):
            crossover_bias = bias

    # Danger Hits comparison
    print(f"\n{'='*75}")
    print("  Danger Hits by Tracking Bias")
    print(f"{'='*75}")

    print(f"\n  {'Bias':<8} | {'FULL+RF':>12} | {'TTC*':>12} | {'TTC':>12}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for bias in biases:
        rf = all_results[bias]['FULL+RF']['danger']
        ttc_star = all_results[bias]['FULL+RF+TTC*']['danger']
        ttc = all_results[bias]['FULL+RF+TTC']['danger']
        print(f"  {bias:<8.1f} | {rf:>12.2f} | {ttc_star:>12.2f} | {ttc:>12.2f}")

    # Net Energy Margin
    print(f"\n{'='*75}")
    print("  Net Energy Margin by Tracking Bias")
    print("  (0.3*food - 0.3*danger - 0.012*survival)")
    print(f"{'='*75}")

    print(f"\n  {'Bias':<8} | {'FULL+RF':>12} | {'TTC*':>12} | {'TTC':>12}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for bias in biases:
        rf = all_results[bias]['FULL+RF']['net_margin']
        ttc_star = all_results[bias]['FULL+RF+TTC*']['net_margin']
        ttc = all_results[bias]['FULL+RF+TTC']['net_margin']
        print(f"  {bias:<8.1f} | {rf:>+12.1f} | {ttc_star:>+12.1f} | {ttc:>+12.1f}")

    # Final Analysis
    print(f"\n{'='*75}")
    print("  TTC Justification Analysis")
    print(f"{'='*75}")

    if crossover_bias is not None:
        print(f"\n  [CROSSOVER DETECTED] TTC 정당화 임계점: bias ≈ {crossover_bias:.1f}")
        print(f"""
  해석:
  - tracking_bias < {crossover_bias:.1f}: 랜덤 위협 → FULL+RF 최적 (방어 비용 > 이득)
  - tracking_bias ≥ {crossover_bias:.1f}: 추적 위협 → TTC 정당화 (예측적 방어 가치↑)

  결론:
  "TTC 선제 방어는 위협의 예측 가능성(tracking bias)이
   약 {crossover_bias:.0%} 이상일 때 경제적으로 정당화된다."
        """)
    else:
        print(f"\n  [NO CROSSOVER] 테스트 범위 내 TTC 역전 없음")
        print(f"""
  해석:
  - 현재 난이도/설정에서 FULL+RF가 전 구간 최적
  - 더 높은 tracking bias 또는 다른 파라미터 필요

  참고: 위협이 강하게 추적해도 에너지 경제가 여전히 지배적일 수 있음
        """)

    # Gradient analysis
    print(f"\n{'='*75}")
    print("  Gradient Analysis: Death Rate Change")
    print(f"{'='*75}")

    for agent_name in agents.keys():
        d0 = all_results[0.0][agent_name]['death_rate']
        d9 = all_results[0.9][agent_name]['death_rate']
        gradient = (d9 - d0) / 0.9

        trend = "↗ 취약" if gradient > 0.1 else "↘ 강건" if gradient < -0.1 else "→ 안정"
        print(f"  {agent_name:<14}: {d0:.1%} → {d9:.1%} (Δ={d9-d0:+.1%}) {trend}")

    print(f"\n  Time: {elapsed:.1f}s")
    print("=" * 75 + "\n")

    return {
        'crossover_bias': crossover_bias,
        'results': all_results,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150, help="Episodes per bias level")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    result = run_tracking_curve(args.episodes, args.seed)
    exit(0 if result['crossover_bias'] is not None else 1)
