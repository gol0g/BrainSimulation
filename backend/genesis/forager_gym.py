"""
ForagerGym - Phase 2a+2b: 항상성 + 공포 조건화 환경

단순한 2D 환경에서 에이전트가 내부 상태(Energy)를 유지하며 생존하는 환경.
Slither.io와 달리 내부 감각(Interoception)이 핵심.

Phase 2a 특징:
- 400x400 2D 공간
- Nest (중앙, 안전) + Field (외곽, 음식)
- Energy 시스템 (항상성)

Phase 2b 특징 (신규):
- Pain Zone (가장자리 위험 영역)
- Pain 신호 + Energy 감소
- Danger Cue (위험 거리 신호)
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import time


@dataclass
class ForagerConfig:
    """Phase 2a+2b 환경 설정"""
    # 맵
    width: int = 400
    height: int = 400
    nest_size: int = 100  # 중앙 둥지 크기

    # 에이전트
    agent_speed: float = 3.0
    agent_radius: float = 10.0

    # 음식 (Phase 7: Reward Freq 개선)
    n_food: int = 35          # 25 → 35 (음식 추가 증가)
    food_radius: float = 8.0  # 더 쉽게 찾기
    food_value: float = 25.0  # 음식 가치 유지

    # === Food Patch 설정 (Hebbian 학습 검증용) ===
    food_patch_enabled: bool = False           # Food Patch 모드 활성화
    n_patches: int = 2                         # Patch 개수
    patch_radius: float = 50.0                 # Patch 반경
    food_spawn_in_patch_prob: float = 0.8      # Patch 내 음식 생성 확률 (80%)

    # 에너지 (항상성)
    energy_start: float = 50.0  # 기본 시작 에너지
    energy_max: float = 100.0
    energy_decay_field: float = 0.12   # Phase 7: 0.15 → 0.12 (약간 완화)
    energy_decay_nest: float = 0.04    # Phase 7: 0.05 → 0.04

    # 감각
    n_rays: int = 16  # 음식/벽 감지 레이 수
    view_range: float = 150.0  # 시야 거리 (Phase 7: 120 → 150)

    # 보상
    reward_food: float = 1.0
    reward_starve: float = -10.0
    reward_homeostasis: float = 0.01  # Energy 30-70 유지 시

    # === Phase 2b: Pain Zone (신규) ===
    pain_zone_enabled: bool = True      # Pain Zone 활성화 여부
    pain_zone_thickness: float = 25.0   # Pain Zone 두께 (가장자리)
    pain_intensity: float = 1.0         # 고통 강도 (0~1)
    pain_damage: float = 0.3            # Pain Zone에서 매 스텝 Energy 감소
    pain_max_damage: float = 80.0       # 누적 Pain 데미지 한계 (초과 시 사망)
    danger_range: float = 50.0          # Danger Cue 감지 거리

    # Phase 2b 보상
    reward_pain: float = -0.5           # Pain Zone에서 매 스텝
    reward_escape: float = 0.1          # Pain Zone 탈출 시

    # === Phase 11: Sound System (청각) ===
    sound_enabled: bool = True          # 소리 시스템 활성화
    danger_sound_range: float = 100.0   # Pain Zone 소리 범위
    food_sound_range: float = 80.0      # 음식 소리 범위
    sound_decay: float = 1.5            # 거리에 따른 감쇠 지수
    food_cluster_bonus: float = 0.3     # 음식 클러스터 보너스

    # === Phase 15: Social (다중 에이전트) ===
    social_enabled: bool = True             # 다중 에이전트 활성화
    n_npc_agents: int = 2                   # NPC 수
    npc_speed: float = 2.5                  # NPC 이동 속도 (플레이어보다 약간 느림)
    npc_radius: float = 10.0               # NPC 충돌 반경
    npc_behavior: str = "forager"           # "forager" (음식 탐색) or "predator" (추적)
    agent_view_range: float = 120.0         # 에이전트 감지 거리
    agent_sound_range: float = 100.0        # 에이전트 소리 감지 거리
    npc_food_eat_enabled: bool = True       # NPC가 음식을 먹을 수 있는지
    social_proximity_range: float = 60.0    # 사회적 상호작용 거리

    # === Phase 15b: Mirror Neuron 관찰 채널 ===
    npc_eating_signal_duration: int = 5         # NPC 먹기 이벤트 지속 시간 (steps)
    npc_food_observation_range: float = 120.0   # NPC 먹기 관찰 가능 거리

    # === Phase 15c: Theory of Mind ===
    npc_intention_observation_range: float = 120.0    # 의도 추론 관찰 거리
    npc_competition_range: float = 80.0               # 경쟁 감지 거리
    npc_food_seek_threshold: int = 20                  # 음식 추구 확신 threshold (steps)

    # 시뮬레이션
    max_steps: int = 3000


class NPCAgent:
    """Phase 15: 간단한 NPC 에이전트 (반사적 행동)"""

    def __init__(self, x: float, y: float, config: 'ForagerConfig'):
        self.x = x
        self.y = y
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.speed = config.npc_speed
        self.radius = config.npc_radius
        self.behavior = config.npc_behavior
        self.food_eaten = 0
        self._wander_timer = 0  # 랜덤 방향 전환 타이머
        # Phase 15b: Mirror neuron support
        self.target_food = None           # 현재 추적 중인 음식 위치
        self.last_eat_step = -100         # 마지막 먹기 시점
        self.heading_toward_food = False  # 목표지향 움직임 플래그
        # Phase 15c: Theory of Mind tracking
        self.last_heading = self.angle            # 이전 heading (방향 변화 감지)
        self.heading_changed = False              # heading 크게 변했는지
        self.consecutive_food_seeks = 0           # 연속 음식 추구 스텝 수

    def step(self, foods: list, player_pos: Tuple[float, float],
             map_width: float, map_height: float, config: 'ForagerConfig'):
        """NPC 행동 업데이트

        forager: 가장 가까운 음식 방향으로 이동 (70%), 랜덤 탐색 (30%)
        predator: 플레이어 추적
        """
        if self.behavior == "forager":
            self._forager_step(foods, map_width, map_height, config)
        elif self.behavior == "predator":
            self._predator_step(player_pos, map_width, map_height, config)

    def _forager_step(self, foods: list, map_w: float, map_h: float,
                      config: 'ForagerConfig'):
        """음식 탐색 행동"""
        # 가장 가까운 음식 찾기
        best_dist = float('inf')
        best_food = None
        for fx, fy in foods:
            dist = math.sqrt((self.x - fx)**2 + (self.y - fy)**2)
            if dist < config.view_range and dist < best_dist:
                best_dist = dist
                best_food = (fx, fy)

        if best_food and np.random.random() < 0.7:
            # 음식 방향으로 이동
            self.target_food = best_food
            self.heading_toward_food = True
            target_angle = math.atan2(best_food[1] - self.y,
                                       best_food[0] - self.x)
            # 부드러운 회전 (최대 0.2 rad/step)
            angle_diff = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            self.angle += np.clip(angle_diff, -0.2, 0.2)
        else:
            # 랜덤 탐색
            self.target_food = None
            self.heading_toward_food = False
            self._wander_timer -= 1
            if self._wander_timer <= 0:
                self.angle += np.random.uniform(-0.5, 0.5)
                self._wander_timer = np.random.randint(20, 60)

        # Phase 15c: 방향 변화 및 음식 추구 일관성 추적
        heading_diff = abs((self.angle - self.last_heading + math.pi) % (2 * math.pi) - math.pi)
        self.heading_changed = heading_diff > 0.1  # >~5.7도
        self.last_heading = self.angle
        if self.heading_toward_food:
            self.consecutive_food_seeks += 1
        else:
            self.consecutive_food_seeks = 0

        # 이동
        new_x = self.x + math.cos(self.angle) * self.speed
        new_y = self.y + math.sin(self.angle) * self.speed

        # 벽 충돌
        if new_x < self.radius:
            new_x = self.radius
            self.angle = math.pi - self.angle
        elif new_x > map_w - self.radius:
            new_x = map_w - self.radius
            self.angle = math.pi - self.angle
        if new_y < self.radius:
            new_y = self.radius
            self.angle = -self.angle
        elif new_y > map_h - self.radius:
            new_y = map_h - self.radius
            self.angle = -self.angle

        self.x = new_x
        self.y = new_y

    def _predator_step(self, player_pos: Tuple[float, float],
                       map_w: float, map_h: float, config: 'ForagerConfig'):
        """플레이어 추적 행동"""
        px, py = player_pos
        dist = math.sqrt((self.x - px)**2 + (self.y - py)**2)

        if dist < config.view_range:
            # 플레이어 방향으로 회전
            target_angle = math.atan2(py - self.y, px - self.x)
            angle_diff = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            self.angle += np.clip(angle_diff, -0.15, 0.15)
        else:
            # 랜덤 이동
            self._wander_timer -= 1
            if self._wander_timer <= 0:
                self.angle += np.random.uniform(-0.5, 0.5)
                self._wander_timer = np.random.randint(30, 80)

        # 이동
        new_x = self.x + math.cos(self.angle) * self.speed
        new_y = self.y + math.sin(self.angle) * self.speed

        # 벽 충돌
        if new_x < self.radius:
            new_x = self.radius
            self.angle = math.pi - self.angle
        elif new_x > map_w - self.radius:
            new_x = map_w - self.radius
            self.angle = math.pi - self.angle
        if new_y < self.radius:
            new_y = self.radius
            self.angle = -self.angle
        elif new_y > map_h - self.radius:
            new_y = map_h - self.radius
            self.angle = -self.angle

        self.x = new_x
        self.y = new_y

    def check_food_collision(self, foods: list, config: 'ForagerConfig',
                             current_step: int = 0) -> bool:
        """NPC 음식 충돌 확인 (먹으면 제거 + 새 음식 생성)"""
        if not config.npc_food_eat_enabled:
            return False

        collision_dist = self.radius + config.food_radius
        for i, (fx, fy) in enumerate(foods):
            dist = math.sqrt((self.x - fx)**2 + (self.y - fy)**2)
            if dist < collision_dist:
                foods.pop(i)
                self.food_eaten += 1
                self.last_eat_step = current_step  # Phase 15b
                return True
        return False


class ForagerGym:
    """
    Phase 2a+2b: 항상성 + 공포 조건화 환경

    핵심 차이점 (vs Slither.io):
    - 내부 상태(Energy)가 행동에 영향
    - 복잡한 뱀 구조 대신 단순한 원형 에이전트
    - Phase 2a: 순수 항상성 검증
    - Phase 2b: Pain Zone 추가 (공포 조건화)
    """

    def __init__(self, config: Optional[ForagerConfig] = None, render_mode: str = "none"):
        self.config = config or ForagerConfig()
        self.render_mode = render_mode  # "none", "pygame"

        # 상태
        self.agent_x: float = 0
        self.agent_y: float = 0
        self.agent_angle: float = 0
        self.energy: float = 0
        self.foods: List[Tuple[float, float]] = []
        self.steps: int = 0

        # 통계
        self.total_food_eaten: int = 0
        self.min_energy: float = 100
        self.max_energy: float = 0
        self.homeostasis_steps: int = 0  # 30-70 범위 유지 시간
        self.energy_history: List[float] = []
        self.position_history: List[Tuple[float, float]] = []

        # === Phase 2b: Pain Zone 통계 ===
        self.pain_damage_accumulated: float = 0.0  # 누적 Pain 데미지
        self.pain_zone_visits: int = 0             # Pain Zone 진입 횟수
        self.pain_zone_steps: int = 0              # Pain Zone 내 체류 시간
        self.was_in_pain: bool = False             # 이전 스텝 Pain Zone 여부 (탈출 감지)

        # === Food Patch 설정 및 통계 ===
        self.patches: List[Tuple[float, float]] = []  # Patch 중심 좌표 (실제 픽셀)
        self.patch_visits: List[int] = []             # 각 Patch 방문 횟수
        self.patch_food_eaten: List[int] = []         # 각 Patch에서 먹은 음식 수
        self.was_in_patch: List[bool] = []            # 이전 스텝 Patch 내 여부
        self._init_patches()

        # === Phase 15: NPC 에이전트 ===
        self.npc_agents: List[NPCAgent] = []
        self.npc_food_stolen: int = 0                 # NPC가 빼앗은 음식 수

        # Pygame (lazy init)
        self.screen = None
        self.clock = None
        self.font = None

        self.reset()

    def reset(self) -> Dict:
        """환경 초기화"""
        # 에이전트를 중앙(둥지)에 배치
        self.agent_x = self.config.width / 2
        self.agent_y = self.config.height / 2
        self.agent_angle = np.random.uniform(0, 2 * np.pi)
        self.energy = self.config.energy_start

        # 음식 생성 (Field에만)
        self.foods = []
        self._spawn_foods(self.config.n_food)

        # 통계 초기화
        self.steps = 0
        self.total_food_eaten = 0
        self.min_energy = self.config.energy_start
        self.max_energy = self.config.energy_start
        self.homeostasis_steps = 0
        self.energy_history = [self.energy]
        self.position_history = [(self.agent_x, self.agent_y)]

        # Phase 2b: Pain Zone 통계 초기화
        self.pain_damage_accumulated = 0.0
        self.pain_zone_visits = 0
        self.pain_zone_steps = 0
        self.was_in_pain = False

        # Food Patch 통계 초기화
        self.patch_visits = [0] * self.config.n_patches
        self.patch_food_eaten = [0] * self.config.n_patches
        self.was_in_patch = [False] * self.config.n_patches

        # Phase 15: NPC 초기화 (맵 사분면에 분산 배치)
        self.npc_agents = []
        self.npc_food_stolen = 0
        if self.config.social_enabled:
            spawn_positions = [
                (self.config.width * 0.25, self.config.height * 0.25),
                (self.config.width * 0.75, self.config.height * 0.75),
                (self.config.width * 0.25, self.config.height * 0.75),
                (self.config.width * 0.75, self.config.height * 0.25),
            ]
            for i in range(self.config.n_npc_agents):
                sx, sy = spawn_positions[i % len(spawn_positions)]
                self.npc_agents.append(NPCAgent(sx, sy, self.config))

        return self._get_observation()

    def step(self, action: Tuple[float, ...]) -> Tuple[Dict, float, bool, Dict]:
        """
        한 스텝 실행

        Args:
            action: (angle_delta,) - 회전량 (-1 ~ +1)

        Returns:
            observation, reward, done, info
        """
        angle_delta = np.clip(action[0], -1, 1) * 0.3  # 최대 회전각 ~17도

        # 1. 이동
        self.agent_angle += angle_delta
        self.agent_angle = self.agent_angle % (2 * np.pi)  # 0 ~ 2π 범위 유지

        new_x = self.agent_x + np.cos(self.agent_angle) * self.config.agent_speed
        new_y = self.agent_y + np.sin(self.agent_angle) * self.config.agent_speed

        # 벽 충돌 처리 (부드러운 바운스)
        if new_x < self.config.agent_radius:
            new_x = self.config.agent_radius
            self.agent_angle = np.pi - self.agent_angle
        elif new_x > self.config.width - self.config.agent_radius:
            new_x = self.config.width - self.config.agent_radius
            self.agent_angle = np.pi - self.agent_angle

        if new_y < self.config.agent_radius:
            new_y = self.config.agent_radius
            self.agent_angle = -self.agent_angle
        elif new_y > self.config.height - self.config.agent_radius:
            new_y = self.config.height - self.config.agent_radius
            self.agent_angle = -self.agent_angle

        self.agent_x = new_x
        self.agent_y = new_y

        # 2. 에너지 감소 (위치에 따라 다름)
        if self._in_nest():
            self.energy -= self.config.energy_decay_nest
        else:
            self.energy -= self.config.energy_decay_field

        self.energy = max(0, self.energy)

        # 3. 음식 섭취
        reward = 0.0
        food_eaten = self._check_food_collision()
        if food_eaten:
            self.energy = min(self.config.energy_max,
                            self.energy + self.config.food_value)
            reward += self.config.reward_food
            self.total_food_eaten += 1

        # 4. 항상성 보상 (30-70 범위 유지)
        if 30 <= self.energy <= 70:
            reward += self.config.reward_homeostasis
            self.homeostasis_steps += 1

        # === Phase 2b: Pain Zone 처리 ===
        in_pain = self._in_pain_zone() if self.config.pain_zone_enabled else False
        pain_signal = 0.0

        if in_pain:
            pain_signal = self.config.pain_intensity
            # Pain Zone 진입 시 Energy 추가 감소
            self.energy -= self.config.pain_damage
            self.energy = max(0, self.energy)
            # 누적 데미지
            self.pain_damage_accumulated += self.config.pain_damage
            # Pain Zone 체류 시간
            self.pain_zone_steps += 1
            # Pain 보상 (부정적)
            reward += self.config.reward_pain

            # 새로 진입한 경우 방문 횟수 증가
            if not self.was_in_pain:
                self.pain_zone_visits += 1

        # Pain Zone 탈출 보상
        if self.was_in_pain and not in_pain:
            reward += self.config.reward_escape

        self.was_in_pain = in_pain

        # === Phase 15: NPC 업데이트 ===
        npc_eating_events = []
        if self.config.social_enabled:
            for npc in self.npc_agents:
                npc.step(self.foods, (self.agent_x, self.agent_y),
                         self.config.width, self.config.height, self.config)
                # NPC 음식 경쟁
                if npc.check_food_collision(self.foods, self.config,
                                            current_step=self.steps):
                    self.npc_food_stolen += 1
                    npc_eating_events.append((npc.x, npc.y, self.steps))
                    self._spawn_foods(1)  # 새 음식 생성 (총 개수 유지)

        # === Food Patch 방문 추적 ===
        if self.config.food_patch_enabled:
            current_patch = self._get_current_patch()
            for i in range(len(self.patches)):
                in_this_patch = (current_patch == i)
                # 새로 진입한 경우 방문 횟수 증가
                if in_this_patch and not self.was_in_patch[i]:
                    self.patch_visits[i] += 1
                self.was_in_patch[i] = in_this_patch

        # 5. 통계 업데이트
        self.min_energy = min(self.min_energy, self.energy)
        self.max_energy = max(self.max_energy, self.energy)
        self.energy_history.append(self.energy)
        self.position_history.append((self.agent_x, self.agent_y))

        # 6. 종료 조건
        self.steps += 1
        done = False
        death_cause = None

        if self.energy <= 0:
            done = True
            death_cause = "starve"
            reward += self.config.reward_starve
        elif self.config.pain_zone_enabled and self.pain_damage_accumulated >= self.config.pain_max_damage:
            done = True
            death_cause = "pain"
            reward += self.config.reward_starve  # Pain 사망도 큰 페널티
        elif self.steps >= self.config.max_steps:
            done = True
            death_cause = "timeout"

        info = {
            "energy": self.energy,
            "in_nest": self._in_nest(),
            "food_eaten": food_eaten,
            "total_food": self.total_food_eaten,
            "death_cause": death_cause,
            "homeostasis_ratio": self.homeostasis_steps / max(1, self.steps),
            "min_energy": self.min_energy,
            "max_energy": self.max_energy,
            # Phase 2b: Pain 정보
            "in_pain": in_pain,
            "pain_signal": pain_signal,
            "pain_damage": self.pain_damage_accumulated,
            "pain_visits": self.pain_zone_visits,
            "pain_steps": self.pain_zone_steps,
            # Food Patch 정보
            "patch_visits": self.patch_visits.copy() if self.config.food_patch_enabled else [],
            "patch_food_eaten": self.patch_food_eaten.copy() if self.config.food_patch_enabled else [],
            "current_patch": self._get_current_patch() if self.config.food_patch_enabled else -1,
            # Phase 15: NPC 정보
            "npc_food_stolen": self.npc_food_stolen,
            "npc_positions": [(npc.x, npc.y) for npc in self.npc_agents] if self.config.social_enabled else [],
            # Phase 15b: NPC 먹기 이벤트
            "npc_eating_events": npc_eating_events,
        }

        # 렌더링
        if self.render_mode == "pygame":
            self.render()

        return self._get_observation(), reward, done, info

    def _in_nest(self) -> bool:
        """둥지 안에 있는지 확인"""
        cx, cy = self.config.width / 2, self.config.height / 2
        half = self.config.nest_size / 2
        return (cx - half <= self.agent_x <= cx + half and
                cy - half <= self.agent_y <= cy + half)

    def _in_pain_zone(self) -> bool:
        """Pain Zone (가장자리) 내부인지 확인"""
        t = self.config.pain_zone_thickness
        return (self.agent_x < t or
                self.agent_x > self.config.width - t or
                self.agent_y < t or
                self.agent_y > self.config.height - t)

    def _init_patches(self):
        """Food Patch 초기화 (고정 위치)"""
        if not self.config.food_patch_enabled:
            self.patches = []
            return

        # 기본 Patch 위치: 좌상단(100, 100)과 우하단(300, 300)
        # Nest(200, 200) 중심을 피해 대칭 배치
        default_positions = [
            (0.25, 0.25),  # 좌상단 (100, 100)
            (0.75, 0.75),  # 우하단 (300, 300)
        ]

        self.patches = []
        for i in range(min(self.config.n_patches, len(default_positions))):
            px = default_positions[i][0] * self.config.width
            py = default_positions[i][1] * self.config.height
            self.patches.append((px, py))

    def _in_patch(self, x: float, y: float) -> int:
        """주어진 좌표가 어느 Patch 내에 있는지 반환 (없으면 -1)"""
        for i, (px, py) in enumerate(self.patches):
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist <= self.config.patch_radius:
                return i
        return -1

    def _get_current_patch(self) -> int:
        """에이전트가 현재 어느 Patch에 있는지 반환 (-1: 없음)"""
        return self._in_patch(self.agent_x, self.agent_y)

    def _distance_to_pain_zone(self) -> float:
        """Pain Zone까지 최단 거리 (내부면 음수)"""
        t = self.config.pain_zone_thickness
        # 각 가장자리까지 거리
        dist_left = self.agent_x - t
        dist_right = (self.config.width - t) - self.agent_x
        dist_top = self.agent_y - t
        dist_bottom = (self.config.height - t) - self.agent_y
        # 최소 거리 (가장 가까운 Pain Zone 경계)
        return min(dist_left, dist_right, dist_top, dist_bottom)

    def _get_danger_signal(self) -> float:
        """Pain Zone까지 거리 기반 위험 신호 (0=안전, 1=위험)"""
        if not self.config.pain_zone_enabled:
            return 0.0

        dist = self._distance_to_pain_zone()
        if dist <= 0:  # Pain Zone 내부
            return 1.0
        elif dist < self.config.danger_range:  # 접근 중
            return 1.0 - (dist / self.config.danger_range)
        else:
            return 0.0

    def _compute_danger_sound(self) -> Tuple[float, float]:
        """
        Phase 11: Pain Zone에서 발생하는 위험 소리 계산

        Returns:
            (left_sound, right_sound): 좌우 귀에 들리는 소리 강도 (0~1)
        """
        if not self.config.sound_enabled or not self.config.pain_zone_enabled:
            return 0.0, 0.0

        # Pain Zone 중심 (맵 가장자리 평균)
        center_x = self.config.width / 2
        center_y = self.config.height / 2

        # 가장 가까운 Pain Zone 경계까지 거리
        dist_to_pain = self._distance_to_pain_zone()

        if dist_to_pain > self.config.danger_sound_range:
            return 0.0, 0.0

        # 거리에 따른 기본 강도 (Pain Zone 내부면 최대)
        if dist_to_pain <= 0:
            intensity = 1.0
        else:
            intensity = 1.0 - (dist_to_pain / self.config.danger_sound_range) ** self.config.sound_decay
        intensity = max(0.0, min(1.0, intensity))

        # 가장 가까운 경계 방향 계산
        dist_left = self.agent_x
        dist_right = self.config.width - self.agent_x
        dist_top = self.agent_y
        dist_bottom = self.config.height - self.agent_y

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        # 가장 가까운 경계 방향으로의 각도
        if min_dist == dist_left:
            danger_angle = np.pi  # 왼쪽
        elif min_dist == dist_right:
            danger_angle = 0  # 오른쪽
        elif min_dist == dist_top:
            danger_angle = -np.pi / 2  # 위
        else:
            danger_angle = np.pi / 2  # 아래

        # 에이전트 방향 기준 상대 각도
        rel_angle = danger_angle - self.agent_angle
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi  # -π ~ π 정규화

        # 좌우 분리 (왼쪽 귀: 음의 각도, 오른쪽 귀: 양의 각도)
        left_factor = max(0.0, -np.sin(rel_angle))  # 왼쪽에서 오는 소리
        right_factor = max(0.0, np.sin(rel_angle))  # 오른쪽에서 오는 소리

        # 정면/후면에서도 양쪽에 들리도록 기본값 추가
        base = 0.3
        left_sound = intensity * (base + (1 - base) * left_factor)
        right_sound = intensity * (base + (1 - base) * right_factor)

        return left_sound, right_sound

    def _compute_food_sound(self) -> Tuple[float, float]:
        """
        Phase 11: 음식에서 발생하는 소리 계산

        여러 음식이 가까이 있으면 더 강한 소리 (클러스터 효과)

        Returns:
            (left_sound, right_sound): 좌우 귀에 들리는 소리 강도 (0~1)
        """
        if not self.config.sound_enabled or len(self.foods) == 0:
            return 0.0, 0.0

        left_total = 0.0
        right_total = 0.0
        food_count_nearby = 0

        for food_x, food_y in self.foods:
            dx = food_x - self.agent_x
            dy = food_y - self.agent_y
            dist = np.sqrt(dx * dx + dy * dy)

            if dist > self.config.food_sound_range:
                continue

            food_count_nearby += 1

            # 거리에 따른 강도
            intensity = 1.0 - (dist / self.config.food_sound_range) ** self.config.sound_decay
            intensity = max(0.0, min(1.0, intensity)) * 0.5  # 개별 음식은 약한 소리

            # 음식 방향
            food_angle = np.arctan2(dy, dx)
            rel_angle = food_angle - self.agent_angle
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

            # 좌우 분리
            left_factor = max(0.0, -np.sin(rel_angle))
            right_factor = max(0.0, np.sin(rel_angle))

            base = 0.3
            left_total += intensity * (base + (1 - base) * left_factor)
            right_total += intensity * (base + (1 - base) * right_factor)

        # 클러스터 보너스 (여러 음식이 가까이 있으면 더 강한 소리)
        if food_count_nearby > 1:
            cluster_bonus = 1.0 + self.config.food_cluster_bonus * (food_count_nearby - 1)
            left_total *= cluster_bonus
            right_total *= cluster_bonus

        return min(1.0, left_total), min(1.0, right_total)

    def _get_observation(self) -> Dict:
        """관찰 반환"""
        food_rays_l, food_rays_r = self._cast_food_rays()
        wall_rays_l, wall_rays_r = self._cast_wall_rays()

        # Phase 2b: Pain 관련 관찰
        if self.config.pain_zone_enabled:
            pain_rays_l, pain_rays_r = self._cast_pain_rays()
            danger_signal = self._get_danger_signal()
        else:
            pain_rays_l = np.zeros(self.config.n_rays // 2)
            pain_rays_r = np.zeros(self.config.n_rays // 2)
            danger_signal = 0.0

        # Phase 11: Sound 관련 관찰
        if self.config.sound_enabled:
            sound_danger_l, sound_danger_r = self._compute_danger_sound()
            sound_food_l, sound_food_r = self._compute_food_sound()
        else:
            sound_danger_l, sound_danger_r = 0.0, 0.0
            sound_food_l, sound_food_r = 0.0, 0.0

        # Phase 15: Agent detection
        if self.config.social_enabled and self.npc_agents:
            agent_rays_l, agent_rays_r = self._cast_agent_rays()
            agent_sound_l, agent_sound_r = self._compute_agent_sound()
            social_proximity = self._get_social_proximity()
            # Phase 15b: Mirror neuron observation channels
            npc_eating_l, npc_eating_r = self._detect_npc_eating()
            npc_food_dir_l, npc_food_dir_r = self._compute_npc_food_direction()
            npc_near_food = self._compute_npc_near_food()
            # Phase 15c: Theory of Mind observation channels
            npc_intention_food = self._compute_npc_intention_food()
            npc_heading_change = self._compute_npc_heading_change()
            npc_competition = self._compute_npc_competition()
        else:
            n_half = self.config.n_rays // 2
            agent_rays_l = np.zeros(n_half)
            agent_rays_r = np.zeros(n_half)
            agent_sound_l, agent_sound_r = 0.0, 0.0
            social_proximity = 0.0
            npc_eating_l, npc_eating_r = 0.0, 0.0
            npc_food_dir_l, npc_food_dir_r = 0.0, 0.0
            npc_near_food = 0.0
            npc_intention_food = 0.0
            npc_heading_change = 0.0
            npc_competition = 0.0

        return {
            # 외부 감각 (L/R 분리 - Phase 1 호환)
            "food_rays_left": food_rays_l,
            "food_rays_right": food_rays_r,
            "wall_rays_left": wall_rays_l,
            "wall_rays_right": wall_rays_r,

            # 내부 감각 (Phase 2a)
            "energy": self.energy / self.config.energy_max,  # 0~1 정규화
            "in_nest": float(self._in_nest()),

            # Phase 2b: Pain 감각 (신규)
            "pain_rays_left": pain_rays_l,
            "pain_rays_right": pain_rays_r,
            "danger_signal": danger_signal,  # 0=안전, 1=위험

            # Phase 3: 위치 정보 (신규)
            "position_x": self.agent_x / self.config.width,   # 0~1 정규화
            "position_y": self.agent_y / self.config.height,  # 0~1 정규화

            # Phase 11: Sound 감각 (신규)
            "sound_danger_left": sound_danger_l,
            "sound_danger_right": sound_danger_r,
            "sound_food_left": sound_food_l,
            "sound_food_right": sound_food_r,

            # Phase 15: Agent 감각 (신규)
            "agent_rays_left": agent_rays_l,
            "agent_rays_right": agent_rays_r,
            "agent_sound_left": agent_sound_l,
            "agent_sound_right": agent_sound_r,
            "social_proximity": social_proximity,

            # Phase 15b: Mirror neuron 관찰 채널
            "npc_eating_left": npc_eating_l,
            "npc_eating_right": npc_eating_r,
            "npc_food_direction_left": npc_food_dir_l,
            "npc_food_direction_right": npc_food_dir_r,
            "npc_near_food": npc_near_food,

            # Phase 15c: Theory of Mind 관찰 채널
            "npc_intention_food": npc_intention_food,
            "npc_heading_change": npc_heading_change,
            "npc_competition": npc_competition,
        }

    def _cast_food_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        음식 방향 레이캐스트 (L/R 분리)

        Returns:
            (left_rays, right_rays): 각각 n_rays//2 크기
        """
        n_half = self.config.n_rays // 2
        rays_left = np.zeros(n_half)
        rays_right = np.zeros(n_half)

        # 레이 방향 계산 (에이전트 기준 상대 각도)
        # Left: -90° ~ 0°, Right: 0° ~ +90°
        for i in range(n_half):
            # Left rays (-π/2 ~ 0)
            angle_l = self.agent_angle - np.pi/2 + (i / n_half) * np.pi/2
            rays_left[i] = self._cast_single_food_ray(angle_l)

            # Right rays (0 ~ π/2)
            angle_r = self.agent_angle + (i / n_half) * np.pi/2
            rays_right[i] = self._cast_single_food_ray(angle_r)

        return rays_left, rays_right

    def _cast_single_food_ray(self, angle: float) -> float:
        """단일 레이로 가장 가까운 음식 감지 (0=없음, 1=매우가까움)"""
        best_dist = self.config.view_range

        for food_x, food_y in self.foods:
            # 음식까지 벡터
            dx = food_x - self.agent_x
            dy = food_y - self.agent_y
            dist = np.sqrt(dx*dx + dy*dy)

            if dist > self.config.view_range:
                continue

            # 레이 방향과 음식 방향 사이 각도
            food_angle = np.arctan2(dy, dx)
            angle_diff = abs((food_angle - angle + np.pi) % (2*np.pi) - np.pi)

            # 레이 폭 (~15도)
            if angle_diff < 0.26:  # ~15 degrees
                if dist < best_dist:
                    best_dist = dist

        if best_dist >= self.config.view_range:
            return 0.0
        return 1.0 - (best_dist / self.config.view_range)

    def _cast_wall_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        벽 방향 레이캐스트 (L/R 분리)

        Returns:
            (left_rays, right_rays): 각각 n_rays//2 크기
        """
        n_half = self.config.n_rays // 2
        rays_left = np.zeros(n_half)
        rays_right = np.zeros(n_half)

        for i in range(n_half):
            # Left rays
            angle_l = self.agent_angle - np.pi/2 + (i / n_half) * np.pi/2
            rays_left[i] = self._cast_single_wall_ray(angle_l)

            # Right rays
            angle_r = self.agent_angle + (i / n_half) * np.pi/2
            rays_right[i] = self._cast_single_wall_ray(angle_r)

        return rays_left, rays_right

    def _cast_single_wall_ray(self, angle: float) -> float:
        """단일 레이로 벽까지 거리 (0=멀리, 1=매우가까움)"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # 각 벽까지 거리 계산
        distances = []

        # 오른쪽 벽
        if cos_a > 0.001:
            d = (self.config.width - self.agent_x) / cos_a
            if d > 0:
                distances.append(d)
        # 왼쪽 벽
        if cos_a < -0.001:
            d = -self.agent_x / cos_a
            if d > 0:
                distances.append(d)
        # 위쪽 벽
        if sin_a > 0.001:
            d = (self.config.height - self.agent_y) / sin_a
            if d > 0:
                distances.append(d)
        # 아래쪽 벽
        if sin_a < -0.001:
            d = -self.agent_y / sin_a
            if d > 0:
                distances.append(d)

        if not distances:
            return 0.0

        min_dist = min(distances)
        if min_dist >= self.config.view_range:
            return 0.0
        return 1.0 - (min_dist / self.config.view_range)

    def _cast_pain_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pain Zone 방향 레이캐스트 (L/R 분리)
        Pain Zone = 가장자리 띠 영역

        Returns:
            (left_rays, right_rays): 각각 n_rays//2 크기
            0=Pain Zone 멀리, 1=Pain Zone 가까이
        """
        n_half = self.config.n_rays // 2
        rays_left = np.zeros(n_half)
        rays_right = np.zeros(n_half)

        for i in range(n_half):
            # Left rays (-π/2 ~ 0)
            angle_l = self.agent_angle - np.pi/2 + (i / n_half) * np.pi/2
            rays_left[i] = self._cast_single_pain_ray(angle_l)

            # Right rays (0 ~ π/2)
            angle_r = self.agent_angle + (i / n_half) * np.pi/2
            rays_right[i] = self._cast_single_pain_ray(angle_r)

        return rays_left, rays_right

    def _cast_single_pain_ray(self, angle: float) -> float:
        """
        단일 레이로 Pain Zone까지 거리
        Pain Zone = 가장자리 thickness 영역

        Returns:
            0=Pain Zone 멀리/없음, 1=Pain Zone 매우 가까움
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        t = self.config.pain_zone_thickness

        # Pain Zone 경계까지 거리 계산
        # Pain Zone은 0~t와 (width-t)~width 영역
        distances = []

        # 오른쪽 Pain Zone 경계 (x = width - t)
        if cos_a > 0.001:
            target_x = self.config.width - t
            if self.agent_x < target_x:
                d = (target_x - self.agent_x) / cos_a
                if d > 0:
                    distances.append(d)

        # 왼쪽 Pain Zone 경계 (x = t)
        if cos_a < -0.001:
            target_x = t
            if self.agent_x > target_x:
                d = (target_x - self.agent_x) / cos_a
                if d > 0:
                    distances.append(d)

        # 위쪽 Pain Zone 경계 (y = height - t)
        if sin_a > 0.001:
            target_y = self.config.height - t
            if self.agent_y < target_y:
                d = (target_y - self.agent_y) / sin_a
                if d > 0:
                    distances.append(d)

        # 아래쪽 Pain Zone 경계 (y = t)
        if sin_a < -0.001:
            target_y = t
            if self.agent_y > target_y:
                d = (target_y - self.agent_y) / sin_a
                if d > 0:
                    distances.append(d)

        if not distances:
            # 이미 Pain Zone 내부
            return 1.0

        min_dist = min(distances)
        if min_dist >= self.config.view_range:
            return 0.0
        return 1.0 - (min_dist / self.config.view_range)

    def _check_food_collision(self) -> bool:
        """음식 충돌 확인"""
        collision_dist = self.config.agent_radius + self.config.food_radius

        for i, (food_x, food_y) in enumerate(self.foods):
            dist = np.sqrt((self.agent_x - food_x)**2 +
                          (self.agent_y - food_y)**2)
            if dist < collision_dist:
                # Food Patch 통계: 어느 Patch에서 먹었는지 기록
                if self.config.food_patch_enabled:
                    patch_idx = self._in_patch(food_x, food_y)
                    if patch_idx >= 0:
                        self.patch_food_eaten[patch_idx] += 1
                self.foods.pop(i)
                self._spawn_foods(1)  # 새 음식 생성
                return True
        return False

    def _spawn_foods(self, n: int):
        """Field에 음식 생성 (Nest 외부, Pain Zone 외부)

        Food Patch 모드 활성화 시:
        - 80%의 음식이 Patch 내에 생성
        - 20%는 랜덤 위치에 생성
        """
        cx, cy = self.config.width / 2, self.config.height / 2
        half = self.config.nest_size / 2
        # Phase 7: Pain Zone 밖에 음식 생성 (pain_zone_thickness + margin)
        pain_margin = self.config.pain_zone_thickness + self.config.food_radius * 2
        margin = max(pain_margin, self.config.food_radius * 2)

        for _ in range(n):
            # Food Patch 모드: 80% Patch 내, 20% 랜덤
            if self.config.food_patch_enabled and self.patches:
                if np.random.random() < self.config.food_spawn_in_patch_prob:
                    # Patch 내에 음식 생성
                    patch = self.patches[np.random.randint(len(self.patches))]
                    for _ in range(100):
                        # Patch 중심 주변 랜덤 위치 (균등 분포)
                        angle = np.random.uniform(0, 2 * np.pi)
                        r = np.sqrt(np.random.uniform(0, 1)) * self.config.patch_radius
                        x = patch[0] + r * np.cos(angle)
                        y = patch[1] + r * np.sin(angle)

                        # 경계 체크: Pain Zone, Nest 외부 확인
                        if (margin <= x <= self.config.width - margin and
                            margin <= y <= self.config.height - margin and
                            not (cx - half <= x <= cx + half and
                                 cy - half <= y <= cy + half)):
                            self.foods.append((x, y))
                            break
                    continue  # 다음 음식으로

            # 기존 방식: 랜덤 위치 (Patch 모드 OFF 또는 20% 확률)
            for _ in range(100):  # 최대 100번 시도
                x = np.random.uniform(margin, self.config.width - margin)
                y = np.random.uniform(margin, self.config.height - margin)

                # Nest 내부면 다시
                if not (cx - half <= x <= cx + half and
                       cy - half <= y <= cy + half):
                    self.foods.append((x, y))
                    break

    # === Phase 15: Agent Sensing Methods ===

    def _cast_agent_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """에이전트 방향 레이캐스트 (L/R 분리) - food_rays와 동일 구조"""
        n_half = self.config.n_rays // 2
        rays_left = np.zeros(n_half)
        rays_right = np.zeros(n_half)

        for i in range(n_half):
            angle_l = self.agent_angle - np.pi/2 + (i / n_half) * np.pi/2
            rays_left[i] = self._cast_single_agent_ray(angle_l)
            angle_r = self.agent_angle + (i / n_half) * np.pi/2
            rays_right[i] = self._cast_single_agent_ray(angle_r)

        return rays_left, rays_right

    def _cast_single_agent_ray(self, angle: float) -> float:
        """단일 레이로 가장 가까운 NPC 감지 (0=없음, 1=매우가까움)"""
        best_dist = self.config.agent_view_range

        for npc in self.npc_agents:
            dx = npc.x - self.agent_x
            dy = npc.y - self.agent_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > self.config.agent_view_range:
                continue

            npc_angle = math.atan2(dy, dx)
            angle_diff = abs((npc_angle - angle + math.pi) % (2*math.pi) - math.pi)

            if angle_diff < 0.26:  # ~15 degrees (food_rays와 동일)
                if dist < best_dist:
                    best_dist = dist

        if best_dist >= self.config.agent_view_range:
            return 0.0
        return 1.0 - (best_dist / self.config.agent_view_range)

    def _compute_agent_sound(self) -> Tuple[float, float]:
        """NPC 에이전트 소리 (이동 중인 NPC가 소리 발생)"""
        left_total = 0.0
        right_total = 0.0

        for npc in self.npc_agents:
            dx = npc.x - self.agent_x
            dy = npc.y - self.agent_y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > self.config.agent_sound_range or dist < 1.0:
                continue

            # 거리 감쇠
            intensity = 1.0 - (dist / self.config.agent_sound_range) ** self.config.sound_decay
            intensity = max(0.0, intensity) * 0.6  # 에이전트 소리는 음식보다 약간 약함

            # 좌우 분리
            npc_angle = math.atan2(dy, dx)
            rel_angle = npc_angle - self.agent_angle
            direction = math.sin(rel_angle)

            base = intensity * 0.3  # 무방향 성분
            directional = intensity * 0.7  # 방향성 성분

            if direction < 0:  # 왼쪽
                left_total += base + directional * abs(direction)
                right_total += base
            else:  # 오른쪽
                right_total += base + directional * abs(direction)
                left_total += base

        return min(1.0, left_total), min(1.0, right_total)

    def _get_social_proximity(self) -> float:
        """가장 가까운 NPC까지의 근접도 (0=멀리, 1=매우가까움)"""
        min_dist = float('inf')
        for npc in self.npc_agents:
            dist = math.sqrt((self.agent_x - npc.x)**2 + (self.agent_y - npc.y)**2)
            if dist < min_dist:
                min_dist = dist

        if min_dist >= self.config.social_proximity_range:
            return 0.0
        return 1.0 - (min_dist / self.config.social_proximity_range)

    # === Phase 15b: Mirror Neuron 관찰 메서드 ===

    def _detect_npc_eating(self) -> Tuple[float, float]:
        """
        NPC가 음식 먹는 이벤트 감지 (좌우 분리)

        NPC가 최근 npc_eating_signal_duration 스텝 내에 먹었고,
        플레이어 관찰 범위 내에 있으면 신호 발생.

        Returns:
            (eating_left, eating_right): 0~1 좌우 먹기 신호
        """
        if not self.npc_agents:
            return 0.0, 0.0

        left_signal = 0.0
        right_signal = 0.0
        duration = self.config.npc_eating_signal_duration
        obs_range = self.config.npc_food_observation_range

        for npc in self.npc_agents:
            # 최근에 먹었는지 확인
            steps_since_eat = self.steps - npc.last_eat_step
            if steps_since_eat > duration or steps_since_eat < 0:
                continue

            # 관찰 범위 내인지 확인
            dx = npc.x - self.agent_x
            dy = npc.y - self.agent_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > obs_range or dist < 1.0:
                continue

            # 먹기 신호 강도 (거리 감쇠 + 시간 감쇠)
            dist_factor = 1.0 - (dist / obs_range)
            time_factor = 1.0 - (steps_since_eat / duration)
            intensity = dist_factor * time_factor

            # 좌우 분리
            npc_angle = math.atan2(dy, dx)
            rel_angle = npc_angle - self.agent_angle
            direction = math.sin(rel_angle)

            base = intensity * 0.3
            directional = intensity * 0.7

            if direction < 0:  # 왼쪽
                left_signal = max(left_signal, base + directional * abs(direction))
                right_signal = max(right_signal, base)
            else:  # 오른쪽
                right_signal = max(right_signal, base + directional * abs(direction))
                left_signal = max(left_signal, base)

        return min(1.0, left_signal), min(1.0, right_signal)

    def _compute_npc_food_direction(self) -> Tuple[float, float]:
        """
        NPC가 음식을 향해 이동 중인지 감지 (좌우 분리)

        NPC가 heading_toward_food이고 관찰 범위 내이면 신호 발생.

        Returns:
            (food_dir_left, food_dir_right): 0~1 좌우 방향 신호
        """
        if not self.npc_agents:
            return 0.0, 0.0

        left_signal = 0.0
        right_signal = 0.0
        obs_range = self.config.npc_food_observation_range

        for npc in self.npc_agents:
            if not npc.heading_toward_food or npc.target_food is None:
                continue

            dx = npc.x - self.agent_x
            dy = npc.y - self.agent_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > obs_range or dist < 1.0:
                continue

            # NPC의 target_food 방향 = NPC가 향하는 음식의 방향
            # 플레이어 기준으로 NPC target_food의 방향을 계산
            tfx, tfy = npc.target_food
            tdx = tfx - self.agent_x
            tdy = tfy - self.agent_y
            target_dist = math.sqrt(tdx * tdx + tdy * tdy)

            # 음식이 너무 멀면 무시
            if target_dist > obs_range * 1.5:
                continue

            # NPC까지 거리 기반 강도
            intensity = 1.0 - (dist / obs_range)

            # 음식 방향의 좌우 분리 (플레이어 기준)
            food_angle = math.atan2(tdy, tdx)
            rel_angle = food_angle - self.agent_angle
            direction = math.sin(rel_angle)

            base = intensity * 0.3
            directional = intensity * 0.7

            if direction < 0:  # 왼쪽
                left_signal = max(left_signal, base + directional * abs(direction))
                right_signal = max(right_signal, base)
            else:  # 오른쪽
                right_signal = max(right_signal, base + directional * abs(direction))
                left_signal = max(left_signal, base)

        return min(1.0, left_signal), min(1.0, right_signal)

    def _compute_npc_near_food(self) -> float:
        """
        가장 가까운 NPC-음식 거리 (먹기 직전 신호)

        NPC가 음식에 가까워지면 높은 신호 → 곧 먹을 것 예측

        Returns:
            near_food: 0~1 (0=멀리/없음, 1=매우 가까움)
        """
        if not self.npc_agents or not self.foods:
            return 0.0

        obs_range = self.config.npc_food_observation_range
        best_signal = 0.0

        for npc in self.npc_agents:
            # 플레이어 관찰 범위 내인지 확인
            dx = npc.x - self.agent_x
            dy = npc.y - self.agent_y
            player_dist = math.sqrt(dx * dx + dy * dy)
            if player_dist > obs_range:
                continue

            # NPC와 가장 가까운 음식 거리
            for fx, fy in self.foods:
                food_dist = math.sqrt((npc.x - fx)**2 + (npc.y - fy)**2)
                if food_dist < 30.0:  # 30px 이내면 "가까움"
                    signal = 1.0 - (food_dist / 30.0)
                    # 플레이어와의 거리로 감쇠
                    signal *= (1.0 - player_dist / obs_range)
                    best_signal = max(best_signal, signal)

        return min(1.0, best_signal)

    # === Phase 15c: Theory of Mind 관찰 메서드 ===

    def _compute_npc_intention_food(self) -> float:
        """
        NPC의 음식 추구 의도 강도 (일관성 기반)

        consecutive_food_seeks가 높을수록 "NPC가 음식을 원한다"는 확신 증가.

        Returns:
            intention: 0~1 (0=불확실, 1=확실히 음식 추구 중)
        """
        if not self.npc_agents:
            return 0.0

        obs_range = self.config.npc_intention_observation_range
        threshold = self.config.npc_food_seek_threshold
        best_signal = 0.0

        for npc in self.npc_agents:
            dx = npc.x - self.agent_x
            dy = npc.y - self.agent_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > obs_range or dist < 1.0:
                continue

            # 음식 추구 일관성
            intention = min(1.0, npc.consecutive_food_seeks / threshold)
            # 거리 감쇠
            dist_factor = 1.0 - (dist / obs_range)
            signal = intention * dist_factor

            best_signal = max(best_signal, signal)

        return min(1.0, best_signal)

    def _compute_npc_heading_change(self) -> float:
        """
        NPC 방향 불안정성 (예측 오차 원천)

        NPC가 갑자기 방향을 바꾸면 높은 신호 → 예측 오차.

        Returns:
            heading_change: 0~1 (0=안정, 1=급변)
        """
        if not self.npc_agents:
            return 0.0

        obs_range = self.config.npc_intention_observation_range
        best_signal = 0.0

        for npc in self.npc_agents:
            dx = npc.x - self.agent_x
            dy = npc.y - self.agent_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > obs_range or dist < 1.0:
                continue

            if npc.heading_changed:
                dist_factor = 1.0 - (dist / obs_range)
                best_signal = max(best_signal, dist_factor)

        return min(1.0, best_signal)

    def _compute_npc_competition(self) -> float:
        """
        NPC가 플레이어 근처 음식을 노리고 있는지 감지

        NPC의 target_food가 플레이어 근처 음식과 겹치면 경쟁 신호.

        Returns:
            competition: 0~1 (0=경쟁 없음, 1=강한 경쟁)
        """
        if not self.npc_agents or not self.foods:
            return 0.0

        comp_range = self.config.npc_competition_range
        best_signal = 0.0

        for npc in self.npc_agents:
            if not npc.heading_toward_food or npc.target_food is None:
                continue

            # NPC가 관찰 범위 내인지
            dx = npc.x - self.agent_x
            dy = npc.y - self.agent_y
            npc_dist = math.sqrt(dx * dx + dy * dy)
            if npc_dist > self.config.npc_intention_observation_range:
                continue

            # NPC target_food가 플레이어 근처 음식인지
            tfx, tfy = npc.target_food
            player_to_food = math.sqrt(
                (tfx - self.agent_x)**2 + (tfy - self.agent_y)**2)

            if player_to_food < comp_range:
                # 가까울수록 강한 경쟁 신호
                signal = 1.0 - (player_to_food / comp_range)
                # NPC도 가까울수록 긴급
                npc_to_food = math.sqrt(
                    (npc.x - tfx)**2 + (npc.y - tfy)**2)
                urgency = max(0.0, 1.0 - npc_to_food / comp_range)
                signal *= max(0.3, urgency)

                best_signal = max(best_signal, signal)

        return min(1.0, best_signal)

    def set_brain_info(self, brain_info: dict):
        """뇌 활성화 정보 설정 (시각화용)"""
        self.brain_info = brain_info

    def render(self):
        """Pygame 렌더링"""
        if self.render_mode != "pygame":
            return

        # Lazy init
        if self.screen is None:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((900, 650))  # 환경 400 + 정보 200 + 뇌 300 + 소뇌
            pygame.display.set_caption("ForagerGym - Phase 6a (Cerebellum)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self.brain_info = {}  # 뇌 활성화 정보

        import pygame

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # 배경
        self.screen.fill((40, 40, 40))

        # === 환경 영역 (400x400) ===
        env_surface = pygame.Surface((400, 400))
        env_surface.fill((30, 30, 30))

        # === Phase 2b: Pain Zone (빨간색 가장자리) ===
        if self.config.pain_zone_enabled:
            t = int(self.config.pain_zone_thickness)
            pain_color = (80, 30, 30)  # 어두운 빨강
            # 상단
            pygame.draw.rect(env_surface, pain_color, (0, 0, 400, t))
            # 하단
            pygame.draw.rect(env_surface, pain_color, (0, 400 - t, 400, t))
            # 좌측
            pygame.draw.rect(env_surface, pain_color, (0, 0, t, 400))
            # 우측
            pygame.draw.rect(env_surface, pain_color, (400 - t, 0, t, 400))

        # === Food Patches (반투명 원) ===
        if self.config.food_patch_enabled and self.patches:
            for i, (px, py) in enumerate(self.patches):
                # Patch 영역 표시 (반투명 녹색)
                patch_color = (50, 100, 50, 80)  # 반투명 녹색
                patch_surf = pygame.Surface((int(self.config.patch_radius * 2),
                                            int(self.config.patch_radius * 2)), pygame.SRCALPHA)
                pygame.draw.circle(patch_surf, patch_color,
                                  (int(self.config.patch_radius), int(self.config.patch_radius)),
                                  int(self.config.patch_radius))
                env_surface.blit(patch_surf,
                                (int(px - self.config.patch_radius),
                                 int(py - self.config.patch_radius)))
                # Patch 테두리
                pygame.draw.circle(env_surface, (80, 150, 80),
                                  (int(px), int(py)), int(self.config.patch_radius), 2)

        # Nest 영역 (밝은 색)
        cx, cy = 200, 200  # 400/2
        half = self.config.nest_size // 2
        pygame.draw.rect(env_surface, (50, 60, 50),
                        (cx - half, cy - half, self.config.nest_size, self.config.nest_size))
        pygame.draw.rect(env_surface, (70, 80, 70),
                        (cx - half, cy - half, self.config.nest_size, self.config.nest_size), 2)

        # 음식
        for food_x, food_y in self.foods:
            pygame.draw.circle(env_surface, (255, 220, 50),
                             (int(food_x), int(food_y)), int(self.config.food_radius))

        # Phase 15: NPC 에이전트
        if self.config.social_enabled:
            for npc in self.npc_agents:
                npc_color = (100, 150, 200) if npc.behavior == "forager" else (200, 80, 80)
                pygame.draw.circle(env_surface, npc_color,
                                   (int(npc.x), int(npc.y)), int(npc.radius))
                # NPC 방향 화살표
                npc_arrow_len = npc.radius * 1.2
                npc_ax = npc.x + math.cos(npc.angle) * npc_arrow_len
                npc_ay = npc.y + math.sin(npc.angle) * npc_arrow_len
                pygame.draw.line(env_surface, (180, 180, 180),
                                 (int(npc.x), int(npc.y)),
                                 (int(npc_ax), int(npc_ay)), 1)

        # 에이전트
        agent_color = self._get_agent_color()
        pygame.draw.circle(env_surface, agent_color,
                          (int(self.agent_x), int(self.agent_y)),
                          int(self.config.agent_radius))

        # 에이전트 방향 화살표
        arrow_len = self.config.agent_radius * 1.5
        arrow_x = self.agent_x + np.cos(self.agent_angle) * arrow_len
        arrow_y = self.agent_y + np.sin(self.agent_angle) * arrow_len
        pygame.draw.line(env_surface, (255, 255, 255),
                        (int(self.agent_x), int(self.agent_y)),
                        (int(arrow_x), int(arrow_y)), 2)

        self.screen.blit(env_surface, (0, 0))

        # === 정보 패널 (200x400) ===
        panel_x = 410

        # Energy Bar
        self._draw_bar(panel_x, 20, 180, 25, self.energy / 100,
                      self._get_energy_color(), "Energy", f"{self.energy:.1f}")

        # Homeostasis Zone 표시
        zone_y = 20 + 25 + 5
        zone_x1 = panel_x + int(180 * 0.3)  # 30%
        zone_x2 = panel_x + int(180 * 0.7)  # 70%
        pygame.draw.rect(self.screen, (0, 100, 0), (zone_x1, zone_y, zone_x2 - zone_x1, 3))

        # 정보 텍스트
        info_y = 80
        info_lines = [
            f"Step: {self.steps}",
            f"Food Eaten: {self.total_food_eaten}",
            f"In Nest: {'Yes' if self._in_nest() else 'No'}",
            f"",
            f"Energy Range:",
            f"  Min: {self.min_energy:.1f}",
            f"  Max: {self.max_energy:.1f}",
            f"",
            f"Homeostasis: {self.homeostasis_steps / max(1, self.steps) * 100:.1f}%",
        ]

        # Phase 2b: Pain Zone 정보 추가
        if self.config.pain_zone_enabled:
            info_lines.extend([
                f"",
                f"=== Pain Zone ===",
                f"In Pain: {'YES!' if self._in_pain_zone() else 'No'}",
                f"Visits: {self.pain_zone_visits}",
                f"Time: {self.pain_zone_steps} steps",
                f"Damage: {self.pain_damage_accumulated:.1f}",
            ])

        for i, line in enumerate(info_lines):
            # Pain 관련 라인은 빨간색
            if "YES!" in line or "Pain Zone" in line:
                color = (255, 100, 100)
            else:
                color = (200, 200, 200)
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (panel_x, info_y + i * 18))

        # === 하단 상태 바 ===
        status_y = 420
        pain_str = "PAIN!" if self._in_pain_zone() else "safe"
        status_text = f"[{self.steps:4d}] E={self.energy:5.1f} | Food={self.total_food_eaten:2d} | {pain_str}"
        text_color = (255, 100, 100) if self._in_pain_zone() else (180, 180, 180)
        text = self.font.render(status_text, True, text_color)
        self.screen.blit(text, (10, status_y))

        # Danger 게이지 (Phase 2b)
        if self.config.pain_zone_enabled:
            danger = self._get_danger_signal()
            if danger > 0:
                danger_text = f"DANGER: {danger*100:.0f}%"
                text = self.font.render(danger_text, True, (255, 50, 50))
                self.screen.blit(text, (10, status_y + 25))

        # === 뇌 활성화 패널 (공간 배치형) ===
        self._render_brain_schematic(600, 0)

        pygame.display.flip()
        # 속도 조절: 기본 15 FPS (느리게), --fast 옵션 시 60 FPS
        target_fps = getattr(self, 'render_fps', 15)
        self.clock.tick(target_fps)

    def _render_brain_schematic(self, x: int, y: int):
        """뇌 영역을 공간 배치하고 활성화 시 밝게 표시"""
        import pygame

        if not hasattr(self, 'brain_info') or not self.brain_info:
            return

        info = self.brain_info

        # 패널 배경
        panel_width = 280
        panel_height = 650
        pygame.draw.rect(self.screen, (15, 15, 20), (x, y, panel_width, panel_height))
        pygame.draw.line(self.screen, (60, 60, 70), (x, y), (x, y + panel_height), 2)

        # 타이틀
        title = self.font.render("Brain Schematic", True, (180, 180, 220))
        self.screen.blit(title, (x + 10, y + 5))

        # === 뇌 영역 배치 (위에서 아래로: PFC → BG → Limbic → Brainstem) ===
        region_width = 50
        region_height = 35
        cx = x + panel_width // 2  # 중앙 x

        # 헬퍼: 활성화 기반 색상 계산
        def get_color(base_color, activity):
            activity = min(1.0, max(0.0, activity))
            # 비활성: 어둡게, 활성: 밝게
            dim = 0.2
            brightness = dim + (1.0 - dim) * activity
            return tuple(int(c * brightness) for c in base_color)

        # 헬퍼: 영역 박스 그리기
        def draw_region(cx, cy, w, h, color, label, activity):
            rect = pygame.Rect(cx - w//2, cy - h//2, w, h)
            fill_color = get_color(color, activity)
            pygame.draw.rect(self.screen, fill_color, rect)
            pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)
            # 라벨
            text = self.small_font.render(label, True, (200, 200, 200))
            text_rect = text.get_rect(center=(cx, cy))
            self.screen.blit(text, text_rect)
            # 활성도 표시
            pct = self.small_font.render(f"{activity*100:.0f}%", True, (255, 255, 255))
            self.screen.blit(pct, (cx - w//2 + 2, cy + h//2 - 12))

        # 헬퍼: 연결선 그리기
        def draw_connection(x1, y1, x2, y2, color=(60, 60, 80)):
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 1)

        # === 1. PREFRONTAL CORTEX (최상단) ===
        pfc_y = y + 50
        wm = info.get("working_memory_rate", 0)
        gf = info.get("goal_food_rate", 0)
        gs = info.get("goal_safety_rate", 0)
        inh = info.get("inhibitory_rate", 0)

        draw_region(cx - 55, pfc_y, 45, 30, (180, 150, 255), "WM", wm)
        draw_region(cx, pfc_y, 45, 30, (100, 255, 150), "Goal", max(gf, gs))
        draw_region(cx + 55, pfc_y, 45, 30, (255, 100, 150), "Inhib", inh)

        # PFC 라벨
        lbl = self.small_font.render("PREFRONTAL", True, (150, 120, 200))
        self.screen.blit(lbl, (x + 5, pfc_y - 15))

        # === 2. BASAL GANGLIA ===
        bg_y = y + 110
        striatum = info.get("striatum_rate", 0)
        direct = info.get("direct_rate", 0)
        indirect = info.get("indirect_rate", 0)
        dopamine = info.get("dopamine_level", 0)

        draw_region(cx - 40, bg_y, 50, 30, (255, 180, 50), "Stri", striatum)
        draw_region(cx + 40, bg_y, 50, 30, (255, 255, 50), "DA", dopamine)
        draw_region(cx - 30, bg_y + 40, 40, 25, (100, 255, 100), "Go", direct)
        draw_region(cx + 30, bg_y + 40, 40, 25, (255, 100, 100), "NoGo", indirect)

        # 연결선
        draw_connection(cx - 40, bg_y + 15, cx - 30, bg_y + 27)
        draw_connection(cx - 40, bg_y + 15, cx + 30, bg_y + 27)

        lbl = self.small_font.render("BASAL GANGLIA", True, (200, 150, 80))
        self.screen.blit(lbl, (x + 5, bg_y - 15))

        # === 3. LIMBIC (Hypothalamus + Amygdala) ===
        limbic_y = y + 200
        hunger = info.get("hunger_rate", 0)
        satiety = info.get("satiety_rate", 0)
        fear = info.get("fear_rate", 0)
        la = info.get("la_rate", 0)

        # Hypothalamus (좌)
        draw_region(cx - 55, limbic_y, 45, 30, (255, 150, 50), "Hung", hunger)
        draw_region(cx - 55, limbic_y + 35, 45, 30, (50, 200, 100), "Sati", satiety)

        # Amygdala (우)
        draw_region(cx + 55, limbic_y, 45, 30, (255, 100, 100), "LA", la)
        draw_region(cx + 55, limbic_y + 35, 45, 30, (255, 50, 50), "Fear", fear)

        # 경쟁 표시 (Hunger vs Fear)
        if hunger > 0.3 and fear > 0.3:
            comp_color = (255, 255, 0) if hunger > fear else (255, 100, 100)
            pygame.draw.line(self.screen, comp_color,
                           (cx - 30, limbic_y + 17), (cx + 30, limbic_y + 17), 2)
            vs_text = self.small_font.render("VS", True, comp_color)
            self.screen.blit(vs_text, (cx - 8, limbic_y + 5))

        lbl = self.small_font.render("HYPO", True, (150, 200, 100))
        self.screen.blit(lbl, (x + 5, limbic_y - 10))
        lbl = self.small_font.render("AMYG", True, (200, 100, 100))
        self.screen.blit(lbl, (x + panel_width - 45, limbic_y - 10))

        # === 4. HIPPOCAMPUS (Place Cells 그리드) ===
        hippo_y = y + 290
        place_rate = info.get("place_cell_rate", 0)
        food_mem = info.get("food_memory_rate", 0)

        # Place Cells를 10x10 미니 그리드로 표시 (실제 20x20 요약)
        grid_size = 10
        cell_size = 8
        grid_x = cx - (grid_size * cell_size) // 2
        grid_y = hippo_y

        # Place cell 활성화 패턴 (실제 데이터가 있으면 사용, 없으면 시뮬레이션)
        place_cells_data = info.get("place_cells_grid", None)
        if place_cells_data is None:
            # 에이전트 위치 기반 가우시안 패턴 생성
            agent_x = info.get("agent_grid_x", 5)
            agent_y = info.get("agent_grid_y", 5)
            for i in range(grid_size):
                for j in range(grid_size):
                    # 가우시안 활성화
                    dist = ((i - agent_x)**2 + (j - agent_y)**2) ** 0.5
                    activation = max(0, 1.0 - dist / 3) * place_rate * 5
                    activation = min(1.0, activation)
                    color = (int(50 + 150 * activation), int(50 + 100 * activation), int(150 + 100 * activation))
                    pygame.draw.rect(self.screen, color,
                                   (grid_x + i * cell_size, grid_y + j * cell_size,
                                    cell_size - 1, cell_size - 1))

        # Food Memory 표시
        draw_region(cx + 70, hippo_y + 40, 45, 30, (150, 100, 255), "FMem", food_mem)

        lbl = self.small_font.render("HIPPOCAMPUS", True, (100, 100, 200))
        self.screen.blit(lbl, (x + 5, hippo_y - 15))

        # === 5. CEREBELLUM (소뇌) ===
        cerebellum_y = y + 380
        granule = info.get("granule_rate", 0)
        purkinje = info.get("purkinje_rate", 0)
        deep_nuc = info.get("deep_nuclei_rate", 0)
        error = info.get("error_rate", 0)

        # 소뇌 박스들 (좌우 배치)
        draw_region(cx - 55, cerebellum_y, 40, 25, (200, 255, 150), "Gran", granule)
        draw_region(cx, cerebellum_y, 40, 25, (150, 200, 255), "Purk", purkinje)
        draw_region(cx + 55, cerebellum_y, 40, 25, (255, 200, 150), "Deep", deep_nuc)

        # Error 신호 (오류 시 빨갛게)
        error_color = (255, 50, 50) if error > 0.3 else (100, 50, 50)
        draw_region(cx, cerebellum_y + 30, 50, 20, error_color, "Error", error)

        lbl = self.small_font.render("CEREBELLUM", True, (150, 200, 150))
        self.screen.blit(lbl, (x + 5, cerebellum_y - 15))

        # === 6. THALAMUS (시상 - 감각 게이팅) ===
        thalamus_y = y + 430
        food_relay = info.get("food_relay_rate", 0)
        danger_relay = info.get("danger_relay_rate", 0)
        trn = info.get("trn_rate", 0)
        arousal = info.get("arousal_rate", 0)

        # Thalamus 박스들
        draw_region(cx - 55, thalamus_y, 45, 25, (100, 200, 255), "Food", food_relay)
        draw_region(cx, thalamus_y, 45, 25, (255, 150, 100), "Dang", danger_relay)
        draw_region(cx + 55, thalamus_y, 45, 25, (150, 150, 200), "TRN", trn)

        # Arousal 바 (각성 수준)
        arousal_bar_y = thalamus_y + 30
        bar_width = 100
        pygame.draw.rect(self.screen, (40, 40, 50), (cx - bar_width//2, arousal_bar_y, bar_width, 12))
        arousal_width = int(bar_width * min(1.0, arousal))
        arousal_color = (255, 200, 100) if arousal > 0.5 else (100, 150, 200)
        pygame.draw.rect(self.screen, arousal_color, (cx - bar_width//2, arousal_bar_y, arousal_width, 12))
        arousal_text = self.small_font.render(f"Arousal {arousal*100:.0f}%", True, (200, 200, 200))
        self.screen.blit(arousal_text, (cx - bar_width//2 + 15, arousal_bar_y - 1))

        lbl = self.small_font.render("THALAMUS", True, (150, 180, 220))
        self.screen.blit(lbl, (x + 5, thalamus_y - 15))

        # === 7. MOTOR OUTPUT (하단) ===
        motor_y = y + 510
        motor_l = info.get("motor_left_rate", 0)
        motor_r = info.get("motor_right_rate", 0)

        draw_region(cx - 45, motor_y, 55, 40, (100, 200, 255), "M_L", motor_l)
        draw_region(cx + 45, motor_y, 55, 40, (100, 255, 200), "M_R", motor_r)

        # 턴 방향 화살표
        angle_delta = info.get("angle_delta", 0)
        arrow_x = cx
        arrow_y = motor_y + 50
        if abs(angle_delta) > 0.05:
            arrow_color = (100, 200, 255) if angle_delta < 0 else (100, 255, 200)
            arrow_dir = -1 if angle_delta < 0 else 1
            pygame.draw.polygon(self.screen, arrow_color, [
                (arrow_x, arrow_y),
                (arrow_x + arrow_dir * 30, arrow_y - 10),
                (arrow_x + arrow_dir * 30, arrow_y + 10)
            ])
            turn_text = "LEFT" if angle_delta < 0 else "RIGHT"
        else:
            pygame.draw.circle(self.screen, (150, 150, 150), (arrow_x, arrow_y), 8)
            turn_text = "STRAIGHT"

        text = self.small_font.render(turn_text, True, (200, 200, 200))
        self.screen.blit(text, (arrow_x - 25, arrow_y + 15))

        lbl = self.small_font.render("MOTOR", True, (150, 200, 200))
        self.screen.blit(lbl, (x + 5, motor_y - 15))

        # === 8. 상태 요약 (최하단) ===
        summary_y = y + 590
        energy = info.get("energy_input", 0.5)

        # Energy bar
        pygame.draw.rect(self.screen, (40, 40, 50), (x + 10, summary_y, panel_width - 20, 15))
        energy_width = int((panel_width - 20) * energy)
        energy_color = (50, 200, 50) if energy > 0.5 else (255, 200, 50) if energy > 0.25 else (255, 50, 50)
        pygame.draw.rect(self.screen, energy_color, (x + 10, summary_y, energy_width, 15))
        energy_text = self.small_font.render(f"Energy: {energy*100:.0f}%", True, (255, 255, 255))
        self.screen.blit(energy_text, (x + 15, summary_y + 1))

        # 현재 상태 텍스트
        state_y = summary_y + 25
        if fear > 0.5:
            state = "FEAR MODE"
            state_color = (255, 100, 100)
        elif hunger > 0.5:
            state = "HUNGRY"
            state_color = (255, 200, 100)
        elif satiety > 0.5:
            state = "SATISFIED"
            state_color = (100, 255, 100)
        else:
            state = "EXPLORING"
            state_color = (150, 150, 200)

        state_text = self.font.render(state, True, state_color)
        self.screen.blit(state_text, (x + 10, state_y))

    def _render_brain_panel(self, x: int, y: int):
        """뇌 활성화 상태 시각화 패널"""
        import pygame

        if not hasattr(self, 'brain_info') or not self.brain_info:
            return

        info = self.brain_info
        panel_width = 240
        bar_height = 16
        bar_width = 140
        section_gap = 8

        # 패널 배경
        pygame.draw.rect(self.screen, (25, 25, 35), (x, y, panel_width, 600))
        pygame.draw.line(self.screen, (80, 80, 80), (x, y), (x, y + 600), 2)

        # 타이틀
        title = self.font.render("Brain Activity", True, (200, 200, 255))
        self.screen.blit(title, (x + 10, y + 10))

        current_y = y + 40

        # === 시상하부 (Hypothalamus) ===
        section_title = self.small_font.render("HYPOTHALAMUS", True, (150, 255, 150))
        self.screen.blit(section_title, (x + 10, current_y))
        current_y += 20

        hunger = info.get("hunger_rate", 0)
        satiety = info.get("satiety_rate", 0)
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            hunger, (255, 150, 50), "Hunger")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            satiety, (50, 200, 100), "Satiety")
        current_y += bar_height + section_gap

        # === 편도체 (Amygdala) ===
        section_title = self.small_font.render("AMYGDALA", True, (255, 150, 150))
        self.screen.blit(section_title, (x + 10, current_y))
        current_y += 20

        fear = info.get("fear_rate", 0)
        la = info.get("la_rate", 0)
        cea = info.get("cea_rate", 0)
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            la, (255, 100, 100), "LA")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            cea, (255, 80, 80), "CEA")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            fear, (255, 50, 50), "Fear")
        current_y += bar_height + section_gap

        # === 해마 (Hippocampus) ===
        section_title = self.small_font.render("HIPPOCAMPUS", True, (150, 150, 255))
        self.screen.blit(section_title, (x + 10, current_y))
        current_y += 20

        place = info.get("place_cell_rate", 0)
        food_mem = info.get("food_memory_rate", 0)
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            place, (100, 150, 255), "Place")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            food_mem, (150, 100, 255), "FoodMem")
        current_y += bar_height + section_gap

        # === 기저핵 (Basal Ganglia) ===
        section_title = self.small_font.render("BASAL GANGLIA", True, (255, 200, 100))
        self.screen.blit(section_title, (x + 10, current_y))
        current_y += 20

        striatum = info.get("striatum_rate", 0)
        direct = info.get("direct_rate", 0)
        indirect = info.get("indirect_rate", 0)
        dopamine = info.get("dopamine_level", 0)

        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            striatum, (255, 180, 50), "Striatum")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            direct, (100, 255, 100), "Direct(Go)")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            indirect, (255, 100, 100), "Indirect")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            dopamine, (255, 255, 50), "Dopamine")
        current_y += bar_height + section_gap

        # === 전전두엽 (Prefrontal Cortex) ===
        section_title = self.small_font.render("PREFRONTAL", True, (200, 150, 255))
        self.screen.blit(section_title, (x + 10, current_y))
        current_y += 20

        working_mem = info.get("working_memory_rate", 0)
        goal_food = info.get("goal_food_rate", 0)
        goal_safety = info.get("goal_safety_rate", 0)
        inhibitory = info.get("inhibitory_rate", 0)

        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            working_mem, (180, 150, 255), "WorkMem")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            goal_food, (100, 255, 150), "GoalFood")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            goal_safety, (255, 150, 100), "GoalSafe")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            inhibitory, (255, 100, 150), "Inhibit")
        current_y += bar_height + section_gap

        # === 모터 (Motor) ===
        section_title = self.small_font.render("MOTOR OUTPUT", True, (200, 200, 200))
        self.screen.blit(section_title, (x + 10, current_y))
        current_y += 20

        motor_l = info.get("motor_left_rate", 0)
        motor_r = info.get("motor_right_rate", 0)
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            motor_l, (100, 200, 255), "Motor L")
        current_y += bar_height + 4
        self._draw_brain_bar(x + 15, current_y, bar_width, bar_height,
                            motor_r, (100, 255, 200), "Motor R")
        current_y += bar_height + section_gap

        # === 행동 출력 ===
        angle_delta = info.get("angle_delta", 0)
        turn_text = f"Turn: {'<< LEFT' if angle_delta < -0.1 else '>> RIGHT' if angle_delta > 0.1 else '-- STRAIGHT'}"
        turn_color = (100, 200, 255) if angle_delta < -0.1 else (100, 255, 200) if angle_delta > 0.1 else (180, 180, 180)
        text = self.small_font.render(turn_text, True, turn_color)
        self.screen.blit(text, (x + 15, current_y))

    def _draw_brain_bar(self, x: int, y: int, width: int, height: int,
                        value: float, color: tuple, label: str):
        """뇌 활성화 바 그리기"""
        import pygame

        # 배경
        pygame.draw.rect(self.screen, (40, 40, 50), (x, y, width, height))

        # 채우기
        fill_width = int(width * min(1.0, max(0.0, value)))
        if fill_width > 0:
            # 그라데이션 효과
            dark_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.rect(self.screen, dark_color, (x, y, fill_width, height))
            pygame.draw.rect(self.screen, color, (x, y, fill_width, height // 2))

        # 테두리
        pygame.draw.rect(self.screen, (80, 80, 80), (x, y, width, height), 1)

        # 라벨과 값
        label_text = self.small_font.render(f"{label}", True, (180, 180, 180))
        value_text = self.small_font.render(f"{value*100:.0f}%", True, (255, 255, 255))
        self.screen.blit(label_text, (x + 3, y + 1))
        self.screen.blit(value_text, (x + width - 35, y + 1))

    def _get_agent_color(self) -> Tuple[int, int, int]:
        """에너지 및 Pain 상태에 따른 에이전트 색상"""
        # Phase 2b: Pain Zone에 있으면 보라색 (공포)
        if self.config.pain_zone_enabled and self._in_pain_zone():
            return (200, 50, 200)  # 보라색 (공포/고통)

        # Phase 2a: 에너지 기반 색상
        if self.energy > 70:
            return (50, 200, 50)   # 녹색 (포만)
        elif self.energy > 30:
            return (50, 150, 200)  # 파란색 (정상)
        elif self.energy > 15:
            return (200, 150, 50)  # 노란색 (배고픔)
        else:
            return (200, 50, 50)   # 빨간색 (위험)

    def _get_energy_color(self) -> Tuple[int, int, int]:
        """에너지 바 색상"""
        if self.energy > 70:
            return (50, 200, 50)
        elif self.energy > 30:
            return (50, 150, 200)
        elif self.energy > 15:
            return (200, 150, 50)
        else:
            return (200, 50, 50)

    def _draw_bar(self, x: int, y: int, width: int, height: int,
                  ratio: float, color: Tuple[int, int, int],
                  label: str, value_str: str):
        """상태 바 그리기"""
        import pygame

        # 배경
        pygame.draw.rect(self.screen, (60, 60, 60), (x, y, width, height))

        # 채워진 부분
        fill_width = int(width * min(1.0, max(0.0, ratio)))
        pygame.draw.rect(self.screen, color, (x, y, fill_width, height))

        # 테두리
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, width, height), 1)

        # 라벨
        text = self.small_font.render(f"{label}: {value_str}", True, (255, 255, 255))
        self.screen.blit(text, (x + 5, y + 5))

    def get_episode_summary(self) -> str:
        """에피소드 요약 문자열"""
        reward_freq = (self.total_food_eaten / max(1, self.steps)) * 100
        homeostasis_pct = (self.homeostasis_steps / max(1, self.steps)) * 100
        pain_pct = (self.pain_zone_steps / max(1, self.steps)) * 100 if self.config.pain_zone_enabled else 0

        summary = f"""
{'='*60}
Episode Summary
{'='*60}
  Steps:        {self.steps}
  Final Energy: {self.energy:.1f}
  Food Eaten:   {self.total_food_eaten}
  Reward Freq:  {reward_freq:.2f}%
  Energy Range: {self.min_energy:.1f} ~ {self.max_energy:.1f}
  Homeostasis:  {homeostasis_pct:.1f}% (30-70 range)
"""
        if self.config.pain_zone_enabled:
            summary += f"""
  --- Pain Zone (Phase 2b) ---
  Pain Visits:  {self.pain_zone_visits}
  Pain Time:    {self.pain_zone_steps} steps ({pain_pct:.1f}%)
  Pain Damage:  {self.pain_damage_accumulated:.1f}
"""
        if self.config.food_patch_enabled and self.patches:
            total_patch_visits = sum(self.patch_visits)
            total_patch_food = sum(self.patch_food_eaten)
            summary += f"""
  --- Food Patch (Hebbian Learning) ---
  Patches:      {len(self.patches)}
  Total Visits: {total_patch_visits}
  Patch Food:   {total_patch_food}/{self.total_food_eaten} ({100*total_patch_food/max(1,self.total_food_eaten):.0f}%)
"""
            for i, (px, py) in enumerate(self.patches):
                summary += f"    Patch {i} ({px:.0f},{py:.0f}): {self.patch_visits[i]} visits, {self.patch_food_eaten[i]} food\n"
        summary += f"{'='*60}\n"
        return summary

    def close(self):
        """환경 종료"""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None


# === 테스트 코드 ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ForagerGym Test (Phase 2b)")
    parser.add_argument("--render", choices=["none", "pygame"], default="pygame")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--no-pain", action="store_true", help="Disable Pain Zone (Phase 2a mode)")
    parser.add_argument("--food-patch", action="store_true", help="Enable Food Patch mode (Hebbian learning)")
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    print("=" * 60)
    print("ForagerGym Environment Test - Phase 2b (Fear Conditioning)")
    print("=" * 60)

    config = ForagerConfig()
    if args.no_pain:
        config.pain_zone_enabled = False
        print("  [!] Pain Zone DISABLED (Phase 2a mode)")
    else:
        print(f"  Pain Zone: thickness={config.pain_zone_thickness}, damage={config.pain_damage}")

    if args.food_patch:
        config.food_patch_enabled = True
        print(f"  [!] Food Patch ENABLED: {config.n_patches} patches, radius={config.patch_radius}")
        print(f"      Spawn probability in patch: {config.food_spawn_in_patch_prob*100:.0f}%")

    env = ForagerGym(config, render_mode=args.render)

    for ep in range(args.episodes):
        print(f"\n--- Episode {ep + 1} ---")
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 랜덤 행동
            action = (np.random.uniform(-1, 1),)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # 매 100 스텝마다 로그
            if env.steps % 100 == 0:
                pain_str = "PAIN!" if info.get('in_pain', False) else "safe"
                print(f"[Step {env.steps:4d}] Energy={info['energy']:5.1f} | "
                      f"Food={info['total_food']:2d} | "
                      f"Pain={pain_str}")

            # Pain Zone 진입/탈출 이벤트
            if info.get('in_pain', False) and env.pain_zone_visits == 1 and env.pain_zone_steps == 1:
                print(f"  [!] ENTERED Pain Zone at step {env.steps}!")

        # 에피소드 요약
        print(env.get_episode_summary())
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Death Cause: {info['death_cause']}")

    env.close()
    print("\nTest complete!")
