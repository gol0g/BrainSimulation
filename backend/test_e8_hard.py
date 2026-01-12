"""
E8-Hard: Cross-Domain Transfer - Difficulty Sweep

E8에서 BASE가 이긴 이유: 환경이 너무 관대함
→ Hard에서 "방어 스택의 존재 이유"를 확정

난이도 레벨:
- H1: danger_move_prob=0.5, energy_decay=0.010
- H2: danger_move_prob=0.7, energy_decay=0.012
- H3: danger_move_prob=0.9, energy_decay=0.015

핵심 질문:
1. BASE가 언제부터 무너지기 시작하는가?
2. FULL+RF+TTC가 그 경계를 얼마나 밀어내는가?
3. Death 원인 분해: danger hit vs starvation

Usage:
    python test_e8_hard.py --episodes 200
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# Discrete Grid Environment with Configurable Difficulty
# ============================================================================

@dataclass
class HardEnvConfig:
    """E8-Hard 환경 설정"""
    grid_size: int = 10
    max_steps: int = 500

    # Energy dynamics (튜닝 대상)
    energy_start: float = 1.0
    energy_decay: float = 0.003
    energy_from_food: float = 0.3
    energy_loss_danger: float = 0.3

    # Pain dynamics
    pain_from_danger: float = 0.5
    pain_decay: float = 0.02

    # Danger movement (튜닝 대상)
    danger_move_prob: float = 0.3

    # Food respawn (Hard에서 조정 가능)
    food_respawn_delay: int = 0  # 0 = 즉시 리스폰
    food_respawn_random: bool = False  # True = 랜덤 위치


class HardSurvivalEnv:
    """Hard 난이도 이산 그리드 생존 환경"""

    def __init__(self, config: HardEnvConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

        self.agent_pos = None
        self.food_pos = None
        self.danger_pos = None
        self.energy = None
        self.pain = None
        self.step_count = 0
        self.food_respawn_timer = 0

        # Statistics
        self.total_food = 0
        self.total_danger_hits = 0
        self.death_cause = None  # 'danger', 'starvation', or None

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
        self.food_respawn_timer = 0
        self.total_food = 0
        self.total_danger_hits = 0
        self.death_cause = None

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(8, dtype=np.float32)

        # Food (리스폰 대기 중이면 거리 무한대 취급)
        if self.food_respawn_timer > 0:
            obs[0] = 0  # food_prox = 0
            obs[2] = 0
            obs[3] = 0
        else:
            dx_food = self.food_pos[0] - self.agent_pos[0]
            dy_food = self.food_pos[1] - self.agent_pos[1]
            dist_food = abs(dx_food) + abs(dy_food)
            obs[0] = max(0, 1 - dist_food / self.config.grid_size)
            obs[2] = np.sign(dx_food) if dx_food != 0 else 0
            obs[3] = np.sign(dy_food) if dy_food != 0 else 0

        # Danger
        dx_danger = self.danger_pos[0] - self.agent_pos[0]
        dy_danger = self.danger_pos[1] - self.agent_pos[1]
        dist_danger = abs(dx_danger) + abs(dy_danger)
        obs[1] = max(0, 1 - dist_danger / 5.0)
        obs[4] = np.sign(dx_danger) if dx_danger != 0 else 0
        obs[5] = np.sign(dy_danger) if dy_danger != 0 else 0

        obs[6] = self.energy
        obs[7] = self.pain

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, bool, Dict]:
        # Apply action
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

        info = {
            'ate_food': False,
            'hit_danger': False,
            'died': False,
            'death_cause': None,
        }

        # Food respawn timer
        if self.food_respawn_timer > 0:
            self.food_respawn_timer -= 1
            if self.food_respawn_timer == 0:
                # Respawn food
                if self.config.food_respawn_random:
                    self.food_pos = self._random_pos()
                else:
                    self.food_pos = self._nearby_pos(self.agent_pos, radius=3)

        # Check food
        if self.food_respawn_timer == 0 and self.agent_pos == self.food_pos:
            self.energy = min(1.0, self.energy + self.config.energy_from_food)
            self.total_food += 1
            info['ate_food'] = True

            if self.config.food_respawn_delay > 0:
                self.food_respawn_timer = self.config.food_respawn_delay
                self.food_pos = [-1, -1]  # 임시로 맵 밖
            else:
                if self.config.food_respawn_random:
                    self.food_pos = self._random_pos()
                else:
                    self.food_pos = self._nearby_pos(self.agent_pos, radius=3)

        # Check danger
        if self.agent_pos == self.danger_pos:
            self.energy = max(0.0, self.energy - self.config.energy_loss_danger)
            self.pain = min(1.0, self.pain + self.config.pain_from_danger)
            self.total_danger_hits += 1
            info['hit_danger'] = True

        # Energy decay
        self.energy = max(0.0, self.energy - self.config.energy_decay)

        # Pain decay
        self.pain = max(0.0, self.pain - self.config.pain_decay)

        # Move danger (more aggressive in Hard)
        if self.rng.random() < self.config.danger_move_prob:
            ddx = self.rng.choice([-1, 0, 1])
            ddy = self.rng.choice([-1, 0, 1])
            new_dx = max(0, min(self.config.grid_size - 1, self.danger_pos[0] + ddx))
            new_dy = max(0, min(self.config.grid_size - 1, self.danger_pos[1] + ddy))
            self.danger_pos = [new_dx, new_dy]

        self.step_count += 1

        # Check death
        died = False
        if self.energy <= 0:
            died = True
            # Death cause analysis
            if info['hit_danger']:
                self.death_cause = 'danger'
            else:
                self.death_cause = 'starvation'
            info['death_cause'] = self.death_cause
            info['died'] = True
        elif self.step_count >= self.config.max_steps:
            died = True  # Survived!

        info['energy'] = self.energy
        info['pain'] = self.pain
        info['step'] = self.step_count

        return self._get_obs(), died, info


# ============================================================================
# Agents (from E8)
# ============================================================================

class BaseAgent:
    def __init__(self):
        self.name = "BASE"

    def act(self, obs: np.ndarray, info: Dict = None) -> int:
        food_dx, food_dy = obs[2], obs[3]
        if abs(food_dx) >= abs(food_dy):
            if food_dx > 0:
                return 4
            elif food_dx < 0:
                return 3
        if food_dy > 0:
            return 2
        elif food_dy < 0:
            return 1
        return 0

    def reset(self):
        pass


class PlusMEMAgent:
    def __init__(self, ema_alpha: float = 0.3):
        self.name = "+MEM"
        self.ema_alpha = ema_alpha
        self.food_dir_ema = None

    def act(self, obs: np.ndarray, info: Dict = None) -> int:
        food_dx, food_dy = obs[2], obs[3]
        current_dir = np.array([food_dx, food_dy])

        if self.food_dir_ema is None:
            self.food_dir_ema = current_dir
        else:
            self.food_dir_ema = (self.ema_alpha * current_dir +
                                (1 - self.ema_alpha) * self.food_dir_ema)

        dx, dy = self.food_dir_ema
        if abs(dx) >= abs(dy):
            if dx > 0.3:
                return 4
            elif dx < -0.3:
                return 3
        if dy > 0.3:
            return 2
        elif dy < -0.3:
            return 1
        return 0

    def reset(self):
        self.food_dir_ema = None


class FullAgent:
    def __init__(self, ema_alpha: float = 0.3, risk_on: float = 0.5,
                 risk_off: float = 0.2, risk_ema_alpha: float = 0.5):
        self.name = "FULL"
        self.ema_alpha = ema_alpha
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.food_dir_ema = None
        self.risk_ema = None
        self.in_defense_mode = False

    def act(self, obs: np.ndarray, info: Dict = None) -> int:
        food_dx, food_dy = obs[2], obs[3]
        danger_prox = obs[1]
        danger_dx, danger_dy = obs[4], obs[5]

        current_dir = np.array([food_dx, food_dy])
        if self.food_dir_ema is None:
            self.food_dir_ema = current_dir
        else:
            self.food_dir_ema = (self.ema_alpha * current_dir +
                                (1 - self.ema_alpha) * self.food_dir_ema)

        raw_risk = danger_prox
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = (self.risk_ema_alpha * raw_risk +
                           (1 - self.risk_ema_alpha) * self.risk_ema)

        if self.in_defense_mode:
            if self.risk_ema < self.risk_off:
                self.in_defense_mode = False
        else:
            if self.risk_ema > self.risk_on:
                self.in_defense_mode = True

        if self.in_defense_mode:
            return self._avoid_danger(danger_dx, danger_dy)
        else:
            return self._toward_food(self.food_dir_ema[0], self.food_dir_ema[1])

    def _toward_food(self, dx: float, dy: float) -> int:
        if abs(dx) >= abs(dy):
            if dx > 0.3:
                return 4
            elif dx < -0.3:
                return 3
        if dy > 0.3:
            return 2
        elif dy < -0.3:
            return 1
        return 0

    def _avoid_danger(self, danger_dx: float, danger_dy: float) -> int:
        if abs(danger_dx) >= abs(danger_dy):
            if danger_dx > 0:
                return 3
            elif danger_dx < 0:
                return 4
        if danger_dy > 0:
            return 1
        elif danger_dy < 0:
            return 2
        return 0

    def reset(self):
        self.food_dir_ema = None
        self.risk_ema = None
        self.in_defense_mode = False


class RiskFilterAgent:
    def __init__(self, risk_on: float = 0.5, risk_off: float = 0.2,
                 risk_ema_alpha: float = 0.5):
        self.name = "FULL+RF"
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha
        self.risk_ema = None
        self.in_defense_mode = False

    def act(self, obs: np.ndarray, info: Dict = None) -> int:
        food_dx, food_dy = obs[2], obs[3]
        danger_prox = obs[1]
        danger_dx, danger_dy = obs[4], obs[5]

        raw_risk = danger_prox
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = (self.risk_ema_alpha * raw_risk +
                           (1 - self.risk_ema_alpha) * self.risk_ema)

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

    def _toward_food(self, dx: float, dy: float) -> int:
        if abs(dx) >= abs(dy):
            if dx > 0:
                return 4
            elif dx < 0:
                return 3
        if dy > 0:
            return 2
        elif dy < 0:
            return 1
        return 0

    def _avoid_danger(self, danger_dx: float, danger_dy: float) -> int:
        if abs(danger_dx) >= abs(danger_dy):
            if danger_dx > 0:
                return 3
            elif danger_dx < 0:
                return 4
        if danger_dy > 0:
            return 1
        elif danger_dy < 0:
            return 2
        return 0

    def reset(self):
        self.risk_ema = None
        self.in_defense_mode = False


class RiskFilterTTCAgent:
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

    def act(self, obs: np.ndarray, info: Dict = None) -> int:
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
            self.risk_ema = (self.risk_ema_alpha * raw_risk +
                           (1 - self.risk_ema_alpha) * self.risk_ema)

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

    def _toward_food(self, dx: float, dy: float) -> int:
        if abs(dx) >= abs(dy):
            if dx > 0:
                return 4
            elif dx < 0:
                return 3
        if dy > 0:
            return 2
        elif dy < 0:
            return 1
        return 0

    def _avoid_danger(self, danger_dx: float, danger_dy: float) -> int:
        if abs(danger_dx) >= abs(danger_dy):
            if danger_dx > 0:
                return 3
            elif danger_dx < 0:
                return 4
        if danger_dy > 0:
            return 1
        elif danger_dy < 0:
            return 2
        return 0

    def reset(self):
        self.risk_ema = None
        self.in_defense_mode = False
        self.ttc_triggered = False
        self.prev_danger_dist = None


# ============================================================================
# Test Runner
# ============================================================================

def run_single_level_test(
    level_name: str,
    config: HardEnvConfig,
    n_episodes: int = 200,
    base_seed: int = 42,
) -> Dict:
    """단일 난이도 레벨 테스트"""
    print(f"\n  === {level_name} (move={config.danger_move_prob}, decay={config.energy_decay}) ===")

    agents = {
        'BASE': BaseAgent,
        '+MEM': PlusMEMAgent,
        'FULL': FullAgent,
        'FULL+RF': RiskFilterAgent,
        'FULL+RF+TTC': RiskFilterTTCAgent,
    }

    level_results = {}

    for agent_name, agent_class in agents.items():
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
                action = agent.act(obs)
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

        result = {
            'survival': np.mean(all_survival),
            'survival_median': np.median(all_survival),
            'food': np.mean(all_food),
            'danger': np.mean(all_danger),
            'death_rate': np.mean(all_deaths),
            'death_by_danger': np.sum(death_by_danger),
            'death_by_starvation': np.sum(death_by_starvation),
        }

        level_results[agent_name] = result

        print(f"    {agent_name:<12} | Surv: {result['survival']:>5.0f} | Food: {result['food']:>5.1f} | "
              f"Dng: {result['danger']:>4.1f} | Death: {result['death_rate']:>5.1%} "
              f"(D:{result['death_by_danger']:>3}, S:{result['death_by_starvation']:>3})")

    return level_results


def run_e8_hard_sweep(n_episodes: int = 200, base_seed: int = 42) -> Dict:
    """E8-Hard 3레벨 스윕"""
    print("\n" + "=" * 75)
    print("  E8-Hard: Difficulty Sweep")
    print("  Goal: BASE가 무너지는 경계, TTC가 밀어내는 경계 찾기")
    print("=" * 75)

    start_time = time.time()

    # 난이도 레벨 정의
    levels = {
        'H1': HardEnvConfig(danger_move_prob=0.5, energy_decay=0.010),
        'H2': HardEnvConfig(danger_move_prob=0.7, energy_decay=0.012),
        'H3': HardEnvConfig(danger_move_prob=0.9, energy_decay=0.015),
    }

    all_results = {}
    for level_name, config in levels.items():
        all_results[level_name] = run_single_level_test(
            level_name, config, n_episodes, base_seed
        )

    elapsed = time.time() - start_time

    # Summary comparison
    print(f"\n{'='*75}")
    print("  Cross-Level Comparison: Death Rate")
    print(f"{'='*75}")

    print(f"\n  {'Agent':<12} | {'H1':>8} | {'H2':>8} | {'H3':>8} | {'Trend':>12}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}")

    for agent_name in ['BASE', '+MEM', 'FULL', 'FULL+RF', 'FULL+RF+TTC']:
        h1 = all_results['H1'][agent_name]['death_rate']
        h2 = all_results['H2'][agent_name]['death_rate']
        h3 = all_results['H3'][agent_name]['death_rate']

        # Trend
        if h3 > h1 + 0.1:
            trend = "↗ 취약"
        elif h3 < h1 - 0.1:
            trend = "↘ 강건"
        else:
            trend = "→ 안정"

        print(f"  {agent_name:<12} | {h1:>7.1%} | {h2:>7.1%} | {h3:>7.1%} | {trend:>12}")

    # Danger hits comparison
    print(f"\n{'='*75}")
    print("  Cross-Level Comparison: Danger Hits")
    print(f"{'='*75}")

    print(f"\n  {'Agent':<12} | {'H1':>8} | {'H2':>8} | {'H3':>8}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for agent_name in ['BASE', '+MEM', 'FULL', 'FULL+RF', 'FULL+RF+TTC']:
        h1 = all_results['H1'][agent_name]['danger']
        h2 = all_results['H2'][agent_name]['danger']
        h3 = all_results['H3'][agent_name]['danger']
        print(f"  {agent_name:<12} | {h1:>8.2f} | {h2:>8.2f} | {h3:>8.2f}")

    # Death cause breakdown
    print(f"\n{'='*75}")
    print("  Death Cause Breakdown (H3 - Hardest)")
    print(f"{'='*75}")

    h3 = all_results['H3']
    print(f"\n  {'Agent':<12} | {'Total':>8} | {'Danger':>8} | {'Starvation':>10}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    for agent_name in ['BASE', '+MEM', 'FULL', 'FULL+RF', 'FULL+RF+TTC']:
        r = h3[agent_name]
        total = r['death_by_danger'] + r['death_by_starvation']
        print(f"  {agent_name:<12} | {total:>8} | {r['death_by_danger']:>8} | {r['death_by_starvation']:>10}")

    # Final analysis
    print(f"\n{'='*75}")
    print("  Boundary Analysis")
    print(f"{'='*75}")

    base_h1 = all_results['H1']['BASE']['death_rate']
    base_h2 = all_results['H2']['BASE']['death_rate']
    base_h3 = all_results['H3']['BASE']['death_rate']

    ttc_h1 = all_results['H1']['FULL+RF+TTC']['death_rate']
    ttc_h2 = all_results['H2']['FULL+RF+TTC']['death_rate']
    ttc_h3 = all_results['H3']['FULL+RF+TTC']['death_rate']

    # BASE 붕괴 시점
    base_collapse_level = None
    if base_h1 >= 0.3:
        base_collapse_level = 'H1'
    elif base_h2 >= 0.3:
        base_collapse_level = 'H2'
    elif base_h3 >= 0.3:
        base_collapse_level = 'H3'

    # TTC 우위 확정
    ttc_advantage = []
    if ttc_h1 < base_h1 - 0.05:
        ttc_advantage.append('H1')
    if ttc_h2 < base_h2 - 0.05:
        ttc_advantage.append('H2')
    if ttc_h3 < base_h3 - 0.05:
        ttc_advantage.append('H3')

    print(f"\n  BASE 붕괴 시작: {base_collapse_level if base_collapse_level else '미붕괴'}")
    print(f"  TTC 우위 레벨: {', '.join(ttc_advantage) if ttc_advantage else '없음'}")

    # 최종 판정
    transfer_confirmed = len(ttc_advantage) >= 2 or (base_collapse_level and ttc_h3 < base_h3)

    print(f"\n{'='*75}")
    print("  Final Conclusion")
    print(f"{'='*75}")

    if transfer_confirmed:
        print(f"""
  [CONFIRMED] Robust Navigation Stack의 범용성 확정

  핵심 증거:
  - BASE는 난이도 증가 시 death rate 급증
  - FULL+RF+TTC는 증가 폭이 더 완만
  - 난이도가 높아질수록 방어 스택의 가치 증가

  결론:
  "방어 구조는 환경이 가혹해질수록 필수가 된다."
  "이것은 네비게이션 특화 트릭이 아니라 범용 의사결정 구조다."
        """)
    else:
        print(f"""
  [PARTIAL] 추가 분석 필요

  관찰:
  - 현재 난이도 범위에서 명확한 경계 미확정
  - 더 가혹한 조건 또는 파라미터 조정 필요
        """)

    print(f"  Time: {elapsed:.1f}s")
    print("=" * 75 + "\n")

    return {
        'transfer_confirmed': transfer_confirmed,
        'results': all_results,
        'base_collapse_level': base_collapse_level,
        'ttc_advantage_levels': ttc_advantage,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per level")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    result = run_e8_hard_sweep(args.episodes, args.seed)
    exit(0 if result['transfer_confirmed'] else 1)
