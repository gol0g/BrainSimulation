"""
E8-LazyTrack: TTC Justification Phase Diagram (Fixed)

기존 문제: tracking_bias가 너무 강해서 세계 붕괴
해결: "게으른 추적" - 추적 빈도와 강도 분리

파라미터:
- p_chase: 추적을 "시도하는 빈도" (스윕 대상)
- p_bias: 추적 시도 시 agent 방향 이동 확률 (0.6 고정)

스윕: p_chase = [0.0, 0.05, 0.10, 0.20, 0.30]

비교군:
- FULL+RF
- FULL+RF+TTC

Usage:
    python test_e8_lazy_track.py --episodes 150
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# Environment with Lazy Tracking
# ============================================================================

@dataclass
class LazyTrackConfig:
    grid_size: int = 10
    max_steps: int = 500
    energy_start: float = 1.0
    energy_decay: float = 0.010  # H1 level (안정적인 base)
    energy_from_food: float = 0.3
    energy_loss_danger: float = 0.3
    pain_from_danger: float = 0.5
    pain_decay: float = 0.02
    danger_move_prob: float = 0.7  # 기본 이동 확률 유지

    # Lazy tracking (핵심!)
    p_chase: float = 0.0   # 추적 시도 빈도 (스윕 대상)
    p_bias: float = 0.6    # 추적 시도 시 agent 방향 이동 확률 (고정)


class LazyTrackEnv:
    """게으른 추적이 있는 생존 환경"""

    def __init__(self, config: LazyTrackConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.agent_pos = None
        self.food_pos = None
        self.danger_pos = None
        self.energy = None
        self.step_count = 0
        self.total_food = 0
        self.total_danger_hits = 0
        self.death_cause = None
        self.defense_steps = 0
        self.chase_events = 0  # 추적 이벤트 횟수

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
        self.step_count = 0
        self.total_food = 0
        self.total_danger_hits = 0
        self.death_cause = None
        self.defense_steps = 0
        self.chase_events = 0
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
        obs[7] = 0.0  # pain
        return obs

    def _move_danger_lazy_tracking(self):
        """게으른 추적: p_chase 확률로만 추적 시도"""
        if self.rng.random() >= self.config.danger_move_prob:
            return  # 이동 안 함

        # Step 1: 추적을 시도할지 결정
        if self.rng.random() < self.config.p_chase:
            # 추적 모드!
            self.chase_events += 1

            # Step 2: p_bias 확률로 agent 방향 이동
            if self.rng.random() < self.config.p_bias:
                # Agent 방향으로 이동
                dx = np.sign(self.agent_pos[0] - self.danger_pos[0])
                dy = np.sign(self.agent_pos[1] - self.danger_pos[1])

                # 한 방향만 이동
                if dx != 0 and dy != 0:
                    if self.rng.random() < 0.5:
                        dy = 0
                    else:
                        dx = 0
            else:
                # 추적 시도했지만 실패 → 랜덤
                dx = self.rng.choice([-1, 0, 1])
                dy = self.rng.choice([-1, 0, 1])
        else:
            # 추적 안 함 → 완전 랜덤
            dx = self.rng.choice([-1, 0, 1])
            dy = self.rng.choice([-1, 0, 1])

        new_x = max(0, min(self.config.grid_size - 1, self.danger_pos[0] + dx))
        new_y = max(0, min(self.config.grid_size - 1, self.danger_pos[1] + dy))
        self.danger_pos = [new_x, new_y]

    def step(self, action: int, in_defense_mode: bool = False) -> Tuple[np.ndarray, bool, Dict]:
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

        info = {'ate_food': False, 'hit_danger': False, 'died': False}

        if self.agent_pos == self.food_pos:
            self.energy = min(1.0, self.energy + self.config.energy_from_food)
            self.total_food += 1
            info['ate_food'] = True
            self.food_pos = self._nearby_pos(self.agent_pos, radius=3)

        if self.agent_pos == self.danger_pos:
            self.energy = max(0.0, self.energy - self.config.energy_loss_danger)
            self.total_danger_hits += 1
            info['hit_danger'] = True

        self.energy = max(0.0, self.energy - self.config.energy_decay)

        # 게으른 추적 적용
        self._move_danger_lazy_tracking()

        self.step_count += 1

        died = False
        if self.energy <= 0:
            died = True
            self.death_cause = 'danger' if info['hit_danger'] else 'starvation'
            info['died'] = True
        elif self.step_count >= self.config.max_steps:
            died = True

        info['energy'] = self.energy
        return self._get_obs(), died, info


# ============================================================================
# Agents
# ============================================================================

class RiskFilterAgent:
    def __init__(self, risk_on=0.5, risk_off=0.2, risk_ema_alpha=0.5):
        self.name = "FULL+RF"
        self.risk_on, self.risk_off = risk_on, risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.risk_ema = None
        self.in_defense_mode = False

    def act(self, obs):
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

    def act(self, obs):
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


# ============================================================================
# Test Runner
# ============================================================================

def run_single_pchase_test(p_chase: float, agents: Dict, n_episodes: int, base_seed: int) -> Dict:
    """단일 p_chase 레벨 테스트"""
    results = {}

    for agent_name, agent_class in agents.items():
        all_survival = []
        all_food = []
        all_danger = []
        all_deaths = []
        all_starvation = []
        all_defense_ratio = []

        for ep in range(n_episodes):
            seed = base_seed + ep
            config = LazyTrackConfig(p_chase=p_chase)
            env = LazyTrackEnv(config, seed=seed)
            agent = agent_class()
            agent.reset()

            obs = env.reset(seed=seed)

            for _ in range(config.max_steps):
                action = agent.act(obs)
                obs, done, info = env.step(action, agent.get_defense_mode())
                if done:
                    break

            all_survival.append(env.step_count)
            all_food.append(env.total_food)
            all_danger.append(env.total_danger_hits)
            all_defense_ratio.append(env.defense_steps / max(env.step_count, 1))

            died = info.get('died', False) and env.energy <= 0
            all_deaths.append(1 if died else 0)
            all_starvation.append(1 if env.death_cause == 'starvation' else 0)

        results[agent_name] = {
            'survival': np.mean(all_survival),
            'food': np.mean(all_food),
            'danger': np.mean(all_danger),
            'death_rate': np.mean(all_deaths),
            'starvation_rate': np.mean(all_starvation),
            'defense_ratio': np.mean(all_defense_ratio),
        }

    return results


def run_lazy_track_curve(n_episodes: int = 150, base_seed: int = 42) -> Dict:
    """Lazy tracking p_chase sweep"""
    print("\n" + "=" * 75)
    print("  E8-LazyTrack: TTC Justification Phase Diagram")
    print("  Sweep: p_chase = [0.0, 0.05, 0.10, 0.20, 0.30]")
    print("  Fixed: p_bias = 0.6, danger_move_prob = 0.7")
    print("=" * 75)

    start_time = time.time()

    p_chase_values = [0.0, 0.05, 0.10, 0.20, 0.30]
    agents = {
        'FULL+RF': RiskFilterAgent,
        'FULL+RF+TTC': RiskFilterTTCAgent,
    }

    all_results = {}

    for p_chase in p_chase_values:
        print(f"\n  === p_chase = {p_chase:.2f} ===")
        results = run_single_pchase_test(p_chase, agents, n_episodes, base_seed)
        all_results[p_chase] = results

        for name, r in results.items():
            print(f"    {name:<14} | Death: {r['death_rate']:>5.1%} | "
                  f"Danger: {r['danger']:>4.1f} | Food: {r['food']:>5.1f} | "
                  f"Def: {r['defense_ratio']:>4.1%}")

    elapsed = time.time() - start_time

    # Phase Diagram
    print(f"\n{'='*75}")
    print("  Phase Diagram: Death Rate by p_chase")
    print(f"{'='*75}")

    print(f"\n  {'p_chase':<10} | {'FULL+RF':>12} | {'TTC':>12} | {'Delta':>10} | {'Winner':>10}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

    crossover_point = None
    prev_winner = None

    for p_chase in p_chase_values:
        rf = all_results[p_chase]['FULL+RF']['death_rate']
        ttc = all_results[p_chase]['FULL+RF+TTC']['death_rate']
        delta = rf - ttc  # 양수면 TTC 승
        winner = "TTC" if delta > 0.02 else "RF" if delta < -0.02 else "TIE"

        print(f"  {p_chase:<10.2f} | {rf:>11.1%} | {ttc:>11.1%} | {delta:>+9.1%} | {winner:>10}")

        # Crossover detection
        if prev_winner == "RF" and winner == "TTC":
            crossover_point = p_chase

        prev_winner = winner

    # Danger Hits
    print(f"\n{'='*75}")
    print("  Danger Hits by p_chase")
    print(f"{'='*75}")

    print(f"\n  {'p_chase':<10} | {'FULL+RF':>12} | {'TTC':>12} | {'TTC Saved':>12}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for p_chase in p_chase_values:
        rf = all_results[p_chase]['FULL+RF']['danger']
        ttc = all_results[p_chase]['FULL+RF+TTC']['danger']
        saved = rf - ttc
        print(f"  {p_chase:<10.2f} | {rf:>12.2f} | {ttc:>12.2f} | {saved:>+12.2f}")

    # Defense Ratio
    print(f"\n{'='*75}")
    print("  Defense Ratio (time spent in defense mode)")
    print(f"{'='*75}")

    print(f"\n  {'p_chase':<10} | {'FULL+RF':>12} | {'TTC':>12}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}")

    for p_chase in p_chase_values:
        rf = all_results[p_chase]['FULL+RF']['defense_ratio']
        ttc = all_results[p_chase]['FULL+RF+TTC']['defense_ratio']
        print(f"  {p_chase:<10.2f} | {rf:>11.1%} | {ttc:>11.1%}")

    # Food Collection
    print(f"\n{'='*75}")
    print("  Food Collection by p_chase")
    print(f"{'='*75}")

    print(f"\n  {'p_chase':<10} | {'FULL+RF':>12} | {'TTC':>12}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*12}")

    for p_chase in p_chase_values:
        rf = all_results[p_chase]['FULL+RF']['food']
        ttc = all_results[p_chase]['FULL+RF+TTC']['food']
        print(f"  {p_chase:<10.2f} | {rf:>12.1f} | {ttc:>12.1f}")

    # Analysis
    print(f"\n{'='*75}")
    print("  TTC Justification Analysis")
    print(f"{'='*75}")

    # Check if TTC ever beats RF
    ttc_wins = []
    for p_chase in p_chase_values:
        rf = all_results[p_chase]['FULL+RF']['death_rate']
        ttc = all_results[p_chase]['FULL+RF+TTC']['death_rate']
        if ttc < rf - 0.02:
            ttc_wins.append(p_chase)

    if ttc_wins:
        print(f"\n  TTC wins at p_chase: {ttc_wins}")
        if crossover_point:
            print(f"  Crossover point (RF->TTC): p_chase = {crossover_point:.2f}")
        print(f"""
  Interpretation:
  - As tracking increases, danger becomes more predictable
  - TTC's preemptive defense becomes more valuable
  - Justification threshold: p_chase >= {min(ttc_wins):.2f}
        """)
    else:
        # Check trend
        rf_trend = all_results[0.30]['FULL+RF']['death_rate'] - all_results[0.0]['FULL+RF']['death_rate']
        ttc_trend = all_results[0.30]['FULL+RF+TTC']['death_rate'] - all_results[0.0]['FULL+RF+TTC']['death_rate']

        print(f"\n  No clear crossover in test range")
        print(f"  Death rate trend (0.0 -> 0.30):")
        print(f"    FULL+RF: {rf_trend:+.1%}")
        print(f"    TTC:     {ttc_trend:+.1%}")

        if ttc_trend < rf_trend:
            print(f"""
  Interpretation:
  - TTC degrades more slowly than RF as tracking increases
  - Crossover likely at higher p_chase values
  - TTC's value increases with threat predictability
            """)

    print(f"\n  Time: {elapsed:.1f}s")
    print("=" * 75 + "\n")

    return {
        'crossover_point': crossover_point,
        'ttc_wins': ttc_wins,
        'results': all_results,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=150, help="Episodes per level")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    result = run_lazy_track_curve(args.episodes, args.seed)
    exit(0 if result['ttc_wins'] else 1)
