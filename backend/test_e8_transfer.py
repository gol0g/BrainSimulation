"""
E8: Cross-Domain Transfer Test

E6→E7에서 확립된 Robust Navigation Stack을 원래 "먹이 찾기/생존" 환경에 이식.

핵심 질문:
- 이 방어 구조가 '네비게이션 트릭'인가, '범용 의사결정 구조'인가?

환경 차이:
- E6/E7: 연속 2D 물리, 장애물 회피
- E8: 이산 10x10 그리드, food/danger 엔티티

적응:
- Risk estimation: danger_prox 직접 사용
- Risk hysteresis: 동일
- TTC: 거리 변화율 기반 (danger 접근 여부)
- Goal EMA: food 방향 스무딩 (이산 방향)

지표:
- Survival time (죽기 전까지 step)
- Food collected
- Danger collisions
- Death count per N episodes

Usage:
    python test_e8_transfer.py --episodes 500
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# Discrete Grid Environment (Simplified from main_genesis.py)
# ============================================================================

@dataclass
class SurvivalEnvConfig:
    """E8 환경 설정"""
    grid_size: int = 10
    max_steps: int = 500

    # Energy dynamics
    energy_start: float = 1.0
    energy_decay: float = 0.003  # per step
    energy_from_food: float = 0.3
    energy_loss_danger: float = 0.3

    # Pain dynamics
    pain_from_danger: float = 0.5
    pain_decay: float = 0.02

    # Danger movement
    danger_move_prob: float = 0.3


class SurvivalGridEnv:
    """이산 그리드 생존 환경"""

    def __init__(self, config: SurvivalEnvConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.default_rng(seed)

        self.agent_pos = None
        self.food_pos = None
        self.danger_pos = None
        self.energy = None
        self.pain = None
        self.step_count = 0

        # Statistics
        self.total_food = 0
        self.total_danger_hits = 0
        self.total_deaths = 0
        self.survival_times = []

    def _random_pos(self) -> List[int]:
        return [self.rng.integers(0, self.config.grid_size),
                self.rng.integers(0, self.config.grid_size)]

    def _nearby_pos(self, center: List[int], radius: int = 3) -> List[int]:
        """center 근처에 위치 생성"""
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
        """center에서 멀리 위치 생성"""
        for _ in range(50):
            pos = self._random_pos()
            dist = abs(pos[0] - center[0]) + abs(pos[1] - center[1])
            if dist >= min_dist:
                return pos
        return self._random_pos()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """환경 리셋"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agent_pos = [self.config.grid_size // 2, self.config.grid_size // 2]
        self.food_pos = self._nearby_pos(self.agent_pos, radius=3)
        self.danger_pos = self._far_pos(self.agent_pos, min_dist=4)
        self.energy = self.config.energy_start
        self.pain = 0.0
        self.step_count = 0

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """관측 생성 [food_prox, danger_prox, food_dx, food_dy, danger_dx, danger_dy, energy, pain]"""
        obs = np.zeros(8, dtype=np.float32)

        # Food
        dx_food = self.food_pos[0] - self.agent_pos[0]
        dy_food = self.food_pos[1] - self.agent_pos[1]
        dist_food = abs(dx_food) + abs(dy_food)
        obs[0] = max(0, 1 - dist_food / self.config.grid_size)  # food_prox
        obs[2] = np.sign(dx_food) if dx_food != 0 else 0  # food_dx
        obs[3] = np.sign(dy_food) if dy_food != 0 else 0  # food_dy

        # Danger
        dx_danger = self.danger_pos[0] - self.agent_pos[0]
        dy_danger = self.danger_pos[1] - self.agent_pos[1]
        dist_danger = abs(dx_danger) + abs(dy_danger)
        obs[1] = max(0, 1 - dist_danger / 5.0)  # danger_prox (더 민감하게)
        obs[4] = np.sign(dx_danger) if dx_danger != 0 else 0  # danger_dx
        obs[5] = np.sign(dy_danger) if dy_danger != 0 else 0  # danger_dy

        # Internal
        obs[6] = self.energy
        obs[7] = self.pain

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, Dict]:
        """
        환경 스텝

        Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
        """
        # Apply action
        dx, dy = 0, 0
        if action == 1:  # up
            dy = -1
        elif action == 2:  # down
            dy = 1
        elif action == 3:  # left
            dx = -1
        elif action == 4:  # right
            dx = 1

        new_x = max(0, min(self.config.grid_size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.config.grid_size - 1, self.agent_pos[1] + dy))
        self.agent_pos = [new_x, new_y]

        info = {
            'ate_food': False,
            'hit_danger': False,
            'died': False,
        }

        # Check food
        if self.agent_pos == self.food_pos:
            self.energy = min(1.0, self.energy + self.config.energy_from_food)
            self.total_food += 1
            info['ate_food'] = True
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

        # Move danger
        if self.rng.random() < self.config.danger_move_prob:
            ddx = self.rng.choice([-1, 0, 1])
            ddy = self.rng.choice([-1, 0, 1])
            new_dx = max(0, min(self.config.grid_size - 1, self.danger_pos[0] + ddx))
            new_dy = max(0, min(self.config.grid_size - 1, self.danger_pos[1] + ddy))
            self.danger_pos = [new_dx, new_dy]

        self.step_count += 1

        # Check death
        died = self.energy <= 0 or self.step_count >= self.config.max_steps
        if self.energy <= 0:
            self.total_deaths += 1
            info['died'] = True

        if died:
            self.survival_times.append(self.step_count)

        info['energy'] = self.energy
        info['pain'] = self.pain
        info['step'] = self.step_count

        return self._get_obs(), died, info

    def get_danger_distance(self) -> int:
        """현재 danger까지 맨해튼 거리"""
        return abs(self.agent_pos[0] - self.danger_pos[0]) + \
               abs(self.agent_pos[1] - self.danger_pos[1])


# ============================================================================
# Discrete Agents with Defense Stack
# ============================================================================

class BaseAgent:
    """BASE: 탐욕적 food 추구, danger 회피 없음"""

    def __init__(self):
        self.name = "BASE"

    def act(self, obs: np.ndarray, info: Dict = None) -> int:
        """food 방향으로 이동"""
        food_dx, food_dy = obs[2], obs[3]

        # 우선순위: 더 큰 방향 성분
        if abs(food_dx) >= abs(food_dy):
            if food_dx > 0:
                return 4  # right
            elif food_dx < 0:
                return 3  # left

        if food_dy > 0:
            return 2  # down
        elif food_dy < 0:
            return 1  # up

        return 0  # stay (on food)

    def reset(self):
        pass

    def get_defense_mode(self) -> bool:
        return False


class PlusMEMAgent:
    """
    +MEM: food 방향 EMA 스무딩

    E6에서 확인된 lag 병리가 여기서도 나타나는지 확인
    """

    def __init__(self, ema_alpha: float = 0.3):
        self.name = "+MEM"
        self.ema_alpha = ema_alpha
        self.food_dir_ema = None

    def act(self, obs: np.ndarray, info: Dict = None) -> int:
        """EMA 스무딩된 food 방향으로 이동"""
        food_dx, food_dy = obs[2], obs[3]
        current_dir = np.array([food_dx, food_dy])

        # EMA smoothing
        if self.food_dir_ema is None:
            self.food_dir_ema = current_dir
        else:
            self.food_dir_ema = (self.ema_alpha * current_dir +
                                (1 - self.ema_alpha) * self.food_dir_ema)

        # Smoothed direction to action
        dx, dy = self.food_dir_ema

        if abs(dx) >= abs(dy):
            if dx > 0.3:
                return 4  # right
            elif dx < -0.3:
                return 3  # left

        if dy > 0.3:
            return 2  # down
        elif dy < -0.3:
            return 1  # up

        return 0  # stay

    def reset(self):
        self.food_dir_ema = None

    def get_defense_mode(self) -> bool:
        return False


class FullAgent:
    """
    FULL: food EMA + risk hysteresis + 방어 모드

    E6/E7의 FULL 구조를 이산 환경에 적응
    """

    def __init__(
        self,
        ema_alpha: float = 0.3,
        risk_on: float = 0.5,
        risk_off: float = 0.2,
        risk_ema_alpha: float = 0.5,
    ):
        self.name = "FULL"
        self.ema_alpha = ema_alpha
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha

        self.food_dir_ema = None
        self.risk_ema = None
        self.in_defense_mode = False

    def act(self, obs: np.ndarray, info: Dict = None) -> int:
        """방어 모드에 따라 food 추구 또는 danger 회피"""
        food_dx, food_dy = obs[2], obs[3]
        danger_prox = obs[1]
        danger_dx, danger_dy = obs[4], obs[5]

        # Food EMA
        current_dir = np.array([food_dx, food_dy])
        if self.food_dir_ema is None:
            self.food_dir_ema = current_dir
        else:
            self.food_dir_ema = (self.ema_alpha * current_dir +
                                (1 - self.ema_alpha) * self.food_dir_ema)

        # Risk EMA + hysteresis
        raw_risk = danger_prox
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = (self.risk_ema_alpha * raw_risk +
                           (1 - self.risk_ema_alpha) * self.risk_ema)

        # Defense mode hysteresis
        if self.in_defense_mode:
            if self.risk_ema < self.risk_off:
                self.in_defense_mode = False
        else:
            if self.risk_ema > self.risk_on:
                self.in_defense_mode = True

        # Action selection
        if self.in_defense_mode:
            # 방어 모드: danger 반대 방향으로 이동
            return self._avoid_danger(danger_dx, danger_dy)
        else:
            # 정상 모드: food 방향으로 이동 (EMA smoothed)
            return self._toward_food(self.food_dir_ema[0], self.food_dir_ema[1])

    def _toward_food(self, dx: float, dy: float) -> int:
        """food 방향 행동"""
        if abs(dx) >= abs(dy):
            if dx > 0.3:
                return 4  # right
            elif dx < -0.3:
                return 3  # left

        if dy > 0.3:
            return 2  # down
        elif dy < -0.3:
            return 1  # up

        return 0  # stay

    def _avoid_danger(self, danger_dx: float, danger_dy: float) -> int:
        """danger 회피 행동 (반대 방향)"""
        # danger_dx > 0 = danger가 오른쪽 → 왼쪽으로 이동
        if abs(danger_dx) >= abs(danger_dy):
            if danger_dx > 0:
                return 3  # left (away from danger)
            elif danger_dx < 0:
                return 4  # right (away from danger)

        if danger_dy > 0:
            return 1  # up (away from danger)
        elif danger_dy < 0:
            return 2  # down (away from danger)

        return 0  # stay (danger on same cell - shouldn't happen)

    def reset(self):
        self.food_dir_ema = None
        self.risk_ema = None
        self.in_defense_mode = False

    def get_defense_mode(self) -> bool:
        return self.in_defense_mode


class RiskFilterAgent:
    """
    FULL+RF: risk hysteresis만, NO food EMA

    E6-4b에서 최고 성능을 보인 구조
    """

    def __init__(
        self,
        risk_on: float = 0.5,
        risk_off: float = 0.2,
        risk_ema_alpha: float = 0.5,
    ):
        self.name = "FULL+RF"
        self.risk_on = risk_on
        self.risk_off = risk_off
        self.risk_ema_alpha = risk_ema_alpha

        self.risk_ema = None
        self.in_defense_mode = False

    def act(self, obs: np.ndarray, info: Dict = None) -> int:
        """방어 모드에 따라 food 추구 또는 danger 회피 (NO food EMA)"""
        food_dx, food_dy = obs[2], obs[3]
        danger_prox = obs[1]
        danger_dx, danger_dy = obs[4], obs[5]

        # Risk EMA + hysteresis (NO food EMA)
        raw_risk = danger_prox
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = (self.risk_ema_alpha * raw_risk +
                           (1 - self.risk_ema_alpha) * self.risk_ema)

        # Defense mode hysteresis
        if self.in_defense_mode:
            if self.risk_ema < self.risk_off:
                self.in_defense_mode = False
        else:
            if self.risk_ema > self.risk_on:
                self.in_defense_mode = True

        # Action selection
        if self.in_defense_mode:
            return self._avoid_danger(danger_dx, danger_dy)
        else:
            return self._toward_food(food_dx, food_dy)  # Raw, no EMA

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

    def get_defense_mode(self) -> bool:
        return self.in_defense_mode


class RiskFilterTTCAgent:
    """
    FULL+RF+TTC: risk hysteresis + TTC 선제 방어

    E7-D1에서 최고 성능을 보인 구조
    이산 환경 적응: TTC = danger 접근 여부 + 거리 기반
    """

    def __init__(
        self,
        risk_on: float = 0.5,
        risk_off: float = 0.2,
        risk_ema_alpha: float = 0.5,
        ttc_dist_threshold: int = 3,  # 거리 3 이하면 선제 방어
    ):
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
        """방어 모드에 따라 food 추구 또는 danger 회피 + TTC 선제 방어"""
        food_dx, food_dy = obs[2], obs[3]
        danger_prox = obs[1]
        danger_dx, danger_dy = obs[4], obs[5]

        # 현재 danger 거리 계산
        current_danger_dist = int(abs(danger_dx) + abs(danger_dy) + 0.5)  # 맨해튼 거리 추정
        # 더 정확하게: danger_prox = 1 - dist/5 → dist = 5 * (1 - danger_prox)
        current_danger_dist = int(5 * (1 - danger_prox))

        # TTC 계산: danger가 접근 중인지?
        danger_approaching = False
        if self.prev_danger_dist is not None:
            if current_danger_dist < self.prev_danger_dist:
                danger_approaching = True
        self.prev_danger_dist = current_danger_dist

        # Risk EMA + hysteresis
        raw_risk = danger_prox
        if self.risk_ema is None:
            self.risk_ema = raw_risk
        else:
            self.risk_ema = (self.risk_ema_alpha * raw_risk +
                           (1 - self.risk_ema_alpha) * self.risk_ema)

        # TTC 선제 방어 트리거 (핵심!)
        if current_danger_dist <= self.ttc_dist_threshold and danger_approaching:
            self.ttc_triggered = True
            self.in_defense_mode = True
        elif current_danger_dist > self.ttc_dist_threshold + 1:
            self.ttc_triggered = False

        # 기존 risk hysteresis (TTC 트리거 아닐 때만)
        if not self.ttc_triggered:
            if self.in_defense_mode:
                if self.risk_ema < self.risk_off:
                    self.in_defense_mode = False
            else:
                if self.risk_ema > self.risk_on:
                    self.in_defense_mode = True

        # Action selection
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

    def get_defense_mode(self) -> bool:
        return self.in_defense_mode


# ============================================================================
# Test Runner
# ============================================================================

def run_single_agent_test(
    agent_class,
    agent_name: str,
    n_episodes: int = 500,
    base_seed: int = 42,
) -> Dict:
    """단일 에이전트 테스트"""
    print(f"  Testing {agent_name}...")

    all_survival_times = []
    all_food_collected = []
    all_danger_hits = []
    all_deaths = []

    for ep in range(n_episodes):
        seed = base_seed + ep
        config = SurvivalEnvConfig()
        env = SurvivalGridEnv(config, seed=seed)
        agent = agent_class()
        agent.reset()

        obs = env.reset(seed=seed)

        food_this_ep = 0
        danger_hits_this_ep = 0

        for _ in range(config.max_steps):
            action = agent.act(obs)
            obs, done, info = env.step(action)

            if info.get('ate_food'):
                food_this_ep += 1
            if info.get('hit_danger'):
                danger_hits_this_ep += 1

            if done:
                break

        all_survival_times.append(env.step_count)
        all_food_collected.append(food_this_ep)
        all_danger_hits.append(danger_hits_this_ep)
        all_deaths.append(1 if info.get('died') else 0)

    mean_survival = np.mean(all_survival_times)
    mean_food = np.mean(all_food_collected)
    mean_danger = np.mean(all_danger_hits)
    death_rate = np.mean(all_deaths)

    # 효율성: food per survival time
    efficiency = mean_food / max(mean_survival, 1) * 100

    print(f"    Survival: {mean_survival:.1f} | Food: {mean_food:.1f} | "
          f"Danger: {mean_danger:.2f} | Death: {death_rate:.1%} | "
          f"Eff: {efficiency:.2f}")

    return {
        'mean_survival_time': mean_survival,
        'mean_food_collected': mean_food,
        'mean_danger_hits': mean_danger,
        'death_rate': death_rate,
        'efficiency': efficiency,
    }


def run_e8_transfer_test(n_episodes: int = 500, base_seed: int = 42) -> Dict:
    """E8 Cross-Domain Transfer Test"""
    print("\n" + "=" * 70)
    print("  E8: Cross-Domain Transfer Test")
    print("  Goal: Robust Navigation Stack이 생존 환경에서도 작동하는가?")
    print("=" * 70)

    start_time = time.time()

    agents = {
        'BASE': BaseAgent,
        '+MEM': PlusMEMAgent,
        'FULL': FullAgent,
        'FULL+RF': RiskFilterAgent,
        'FULL+RF+TTC': RiskFilterTTCAgent,
    }

    results = {}
    for name, agent_class in agents.items():
        results[name] = run_single_agent_test(agent_class, name, n_episodes, base_seed)

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*70}")
    print("  Summary Table")
    print(f"{'='*70}")
    print(f"\n  {'Agent':<15} | {'Survival':>10} | {'Food':>8} | {'Danger':>8} | {'Death':>8} | {'Eff':>8}")
    print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for name, r in results.items():
        print(f"  {name:<15} | {r['mean_survival_time']:>10.1f} | {r['mean_food_collected']:>8.1f} | "
              f"{r['mean_danger_hits']:>8.2f} | {r['death_rate']:>7.1%} | {r['efficiency']:>8.2f}")

    # Transfer verification
    print(f"\n{'='*70}")
    print("  Transfer Verification")
    print(f"{'='*70}")

    base = results['BASE']
    mem = results['+MEM']
    full = results['FULL']
    rf = results['FULL+RF']
    ttc = results['FULL+RF+TTC']

    print(f"\n  E6/E7 예상 패턴:")
    print(f"    - +MEM lag 병리: danger_hits 증가 예상")
    print(f"    - FULL > BASE: risk hysteresis 효과")
    print(f"    - FULL+RF ≥ FULL: action EMA 불필요 확인")
    print(f"    - FULL+RF+TTC 최고: TTC 선제 방어 효과")

    # 검증
    mem_pathology = mem['mean_danger_hits'] > base['mean_danger_hits']
    full_better_than_base = full['mean_danger_hits'] < base['mean_danger_hits']
    rf_as_good_as_full = rf['mean_danger_hits'] <= full['mean_danger_hits'] + 0.1
    ttc_best = ttc['mean_danger_hits'] <= min(full['mean_danger_hits'], rf['mean_danger_hits'])

    ttc_best_survival = ttc['mean_survival_time'] >= max(
        full['mean_survival_time'], rf['mean_survival_time']) - 10

    print(f"\n  검증 결과:")
    print(f"    +MEM lag 병리 확인: {'YES' if mem_pathology else 'NO'} "
          f"({mem['mean_danger_hits']:.2f} vs BASE {base['mean_danger_hits']:.2f})")
    print(f"    FULL > BASE: {'YES' if full_better_than_base else 'NO'} "
          f"({full['mean_danger_hits']:.2f} vs {base['mean_danger_hits']:.2f})")
    print(f"    FULL+RF ≥ FULL: {'YES' if rf_as_good_as_full else 'NO'} "
          f"({rf['mean_danger_hits']:.2f} vs {full['mean_danger_hits']:.2f})")
    print(f"    FULL+RF+TTC 최저 danger: {'YES' if ttc_best else 'NO'} "
          f"({ttc['mean_danger_hits']:.2f})")
    print(f"    FULL+RF+TTC 최고/동급 survival: {'YES' if ttc_best_survival else 'NO'} "
          f"({ttc['mean_survival_time']:.1f})")

    # Transfer 성공 판정
    transfer_success = sum([
        full_better_than_base,
        rf_as_good_as_full,
        ttc_best or ttc_best_survival,
    ]) >= 2

    print(f"\n{'='*70}")
    print("  Final Conclusion")
    print(f"{'='*70}")

    if transfer_success:
        print(f"""
  [TRANSFER SUCCESS] Robust Navigation Stack이 생존 환경에서도 작동

  핵심 확인:
  - Risk hysteresis: 이산 환경에서도 방어 모드 안정화
  - TTC 선제 방어: 거리 기반 선제 트리거가 접근하는 danger에 효과적
  - 구조의 범용성: 연속/이산, 물리/그리드 모두에서 동일 패턴

  결론: "네비게이션 트릭"이 아니라 "범용 의사결정 구조"
        """)
    else:
        print(f"""
  [PARTIAL TRANSFER] 일부 패턴만 전이됨

  분석 필요:
  - 이산 환경의 특수성 (danger 랜덤 이동)
  - 관측 구조 차이 (방향 신호 vs 연속 좌표)
  - 추가 적응 필요 가능성
        """)

    print(f"  Time: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    return {
        'transfer_success': transfer_success,
        'results': results,
        'elapsed_sec': elapsed,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    args = parser.parse_args()

    result = run_e8_transfer_test(args.episodes, args.seed)
    exit(0 if result['transfer_success'] else 1)
