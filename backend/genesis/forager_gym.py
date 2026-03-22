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

    # 음식 (난이도 조정 - 직선 이동만으론 생존 어려움)
    n_food: int = 15           # 35 → 15 (적당히 희소)
    food_radius: float = 8.0
    food_value: float = 25.0

    # === Food Patch 설정 (Hebbian 학습 검증용) ===
    food_patch_enabled: bool = False           # Food Patch 모드 활성화
    n_patches: int = 2                         # Patch 개수
    patch_radius: float = 50.0                 # Patch 반경
    food_spawn_in_patch_prob: float = 0.8      # Patch 내 음식 생성 확률 (80%)

    # 에너지 (항상성)
    energy_start: float = 50.0  # 기본 시작 에너지
    energy_max: float = 100.0
    energy_decay_field: float = 0.15   # 0.12 → 0.15 (약간 빠른 소모)
    energy_decay_nest: float = 0.05    # 0.04 → 0.05

    # 감각
    n_rays: int = 16  # 음식/벽 감지 레이 수
    view_range: float = 150.0  # 시야 거리 (Phase 7: 120 → 150)

    # 보상
    reward_food: float = 1.0
    reward_starve: float = -10.0
    reward_homeostasis: float = 0.01  # Energy 30-70 유지 시

    # === Phase 2b: Pain Zone (내부 원형 영역) ===
    pain_zone_enabled: bool = True      # Pain Zone 활성화 여부
    pain_zone_count: int = 2            # 내부 원형 Pain Zone 개수
    pain_zone_radius: float = 60.0      # 각 Pain Zone 반경 (px)
    pain_intensity: float = 1.0         # 고통 강도 (0~1)
    pain_damage: float = 0.3            # Pain Zone에서 매 스텝 Energy 감소
    pain_max_damage: float = 200.0      # 누적 Pain 데미지 한계
    danger_range: float = 80.0          # Danger Cue 감지 거리 - 조기 경고

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

    # === Phase 17: NPC Vocalization (발성) ===
    npc_vocalization_enabled: bool = True       # NPC 발성 활성화
    npc_call_range: float = 100.0              # NPC 발성 가청 범위 (pixels)
    npc_call_duration: int = 5                 # 발성 지속 시간 (steps)
    npc_call_food_prob: float = 0.8            # NPC가 먹을 때 food call 확률
    npc_call_danger_prob: float = 0.9          # NPC가 pain zone 근처에서 danger call 확률
    agent_call_range: float = 80.0             # 에이전트 발성 가청 범위
    agent_call_cooldown: int = 10              # 에이전트 발성 쿨다운 (steps)
    npc_call_response_speed: float = 0.15      # NPC가 에이전트 call에 반응하는 강도

    # === Phase L5: Multi-Food Types (지각 학습) ===
    n_food_types: int = 2                  # 음식 종류 수 (1=기존, 2=좋은/나쁜)
    food_type_ratio: float = 0.6           # 좋은 음식 비율 (60%)
    bad_food_energy: float = -5.0          # 나쁜 음식 에너지 (부정적)

    # === Phase L6: Food Cluster Respawn (예측 학습 환경) ===
    food_cluster_respawn: bool = True         # 클러스터 리스폰 활성화
    food_cluster_prob: float = 0.6            # 먹은 위치 근처 리스폰 확률 (60%)
    food_cluster_radius: float = 80.0         # 클러스터 반경 (px)

    # === Phase L12: Danger-Adjacent Food (위험 근접 고보상 음식) ===
    danger_food_enabled: bool = True          # 위험 근접 음식 활성화
    danger_food_ratio: float = 0.3            # 30% 음식이 pain zone 가장자리에 생성
    danger_food_bonus: float = 0.5            # 위험 근접 음식 에너지 +50% 보너스

    # === Predator Config ===
    predator_enabled: bool = True
    n_predators: int = 1                      # 1마리로 시작 (점진적 난이도)
    predator_speed: float = 2.0              # agent 3.0보다 느림 → 탈출 가능
    predator_radius: float = 12.0            # agent 10보다 약간 큼
    predator_turn_rate: float = 0.15         # rad/step (agent 0.3보다 느림)
    predator_chase_range: float = 120.0      # 추적 시작 거리
    predator_catch_radius: float = 15.0      # 접촉 판정 (agent_r + predator_r 근사)
    predator_damage: float = 0.3             # 접촉 시 매 스텝 에너지 감소
    predator_nest_safe: bool = True          # 둥지 = 안전지대
    predator_view_range: float = 120.0       # 에이전트의 포식자 감지 거리
    predator_pain_intensity: float = 0.8     # pain signal 강도 (zone 1.0보다 약간 약함)
    predator_danger_intensity: float = 0.9   # danger signal 강도
    predator_wander_speed: float = 1.5       # 배회 시 속도

    # === Phase L14: Motor Noise + Sensor Jitter ===
    motor_noise_enabled: bool = True
    motor_noise_std: float = 0.05            # σ for angle_delta noise (±~3° at 1σ)
    sensor_jitter_enabled: bool = True
    sensor_jitter_std: float = 0.03          # σ for multiplicative ray noise (±3%)

    # === Environment E1: Obstacles (정적 장애물) ===
    obstacles_enabled: bool = False  # wall_rays 간섭 문제로 비활성화, 별도 obstacle_rays 필요
    n_obstacles: int = 1
    obstacle_min_size: float = 15.0
    obstacle_max_size: float = 30.0

    # 시뮬레이션
    max_steps: int = 3000


class PredatorAgent:
    """포식자 - 에이전트를 추적, 둥지 회피"""

    def __init__(self, x: float, y: float, config: 'ForagerConfig'):
        self.x, self.y = x, y
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.radius = config.predator_radius
        self.state = "wander"  # "wander" | "chase"
        self.speed = config.predator_wander_speed
        self._wander_timer = 0
        self.chase_steps = 0
        self.contact_steps = 0

    def step(self, player_pos, map_w: float, map_h: float, config: 'ForagerConfig',
             obstacles: Optional[List[Tuple[float, float, float, float]]] = None):
        px, py = player_pos
        dist = math.sqrt((self.x - px)**2 + (self.y - py)**2)
        player_in_nest = self._is_in_nest(px, py, config)

        # 이동 전 위치 저장 (장애물 복원용)
        pre_move_x, pre_move_y = self.x, self.y

        # 상태 전환 (히스테리시스: 시작 chase_range, 해제 1.5x)
        if self.state == "wander":
            if dist < config.predator_chase_range and not player_in_nest:
                self.state = "chase"
                self.speed = config.predator_speed
        else:  # chase
            if dist > config.predator_chase_range * 1.5 or player_in_nest:
                self.state = "wander"
                self.speed = config.predator_wander_speed
                self.chase_steps = 0

        if self.state == "chase":
            self._chase(px, py, config)
            self.chase_steps += 1
        else:
            self._wander(map_w, map_h, config)

        # 벽 충돌 — 반사 (구석 박힘 방지)
        old_x, old_y = self.x, self.y
        self.x = np.clip(self.x, self.radius, map_w - self.radius)
        self.y = np.clip(self.y, self.radius, map_h - self.radius)
        if self.x != old_x:  # 좌우 벽
            self.angle = math.pi - self.angle
            self._wander_timer = 0
        if self.y != old_y:  # 상하 벽
            self.angle = -self.angle
            self._wander_timer = 0

        # 장애물 충돌 (AABB-circle)
        if obstacles:
            for ox, oy, ow, oh in obstacles:
                closest_x = max(ox, min(self.x, ox + ow))
                closest_y = max(oy, min(self.y, oy + oh))
                dx_o = self.x - closest_x
                dy_o = self.y - closest_y
                if dx_o * dx_o + dy_o * dy_o < self.radius * self.radius:
                    self.x = pre_move_x
                    self.y = pre_move_y
                    self.angle += np.random.uniform(-0.5, 0.5)
                    self._wander_timer = 0
                    break

    def _chase(self, px: float, py: float, config: 'ForagerConfig'):
        target = math.atan2(py - self.y, px - self.x)
        diff = (target - self.angle + math.pi) % (2 * math.pi) - math.pi
        self.angle += np.clip(diff, -config.predator_turn_rate, config.predator_turn_rate)
        # 둥지 접근 시 회피
        if config.predator_nest_safe:
            nx = self.x + math.cos(self.angle) * self.speed
            ny = self.y + math.sin(self.angle) * self.speed
            if self._is_in_nest(nx, ny, config):
                self.angle += math.pi * 0.5
                return
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

    def _wander(self, map_w: float, map_h: float, config: 'ForagerConfig'):
        self._wander_timer -= 1
        if self._wander_timer <= 0:
            self.angle += np.random.uniform(-0.5, 0.5)
            self._wander_timer = np.random.randint(30, 80)
        # 둥지 회피
        if config.predator_nest_safe:
            nx = self.x + math.cos(self.angle) * self.speed
            ny = self.y + math.sin(self.angle) * self.speed
            if self._is_in_nest(nx, ny, config):
                cx, cy = map_w / 2, map_h / 2
                self.angle = math.atan2(self.y - cy, self.x - cx)
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

    def _is_in_nest(self, x: float, y: float, config: 'ForagerConfig') -> bool:
        cx, cy = config.width / 2, config.height / 2
        half = config.nest_size / 2 + 10  # 10px 마진
        return cx - half <= x <= cx + half and cy - half <= y <= cy + half


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
        # Phase 17: Vocalization
        self.current_call = 0               # 0=none, 1=food, 2=danger
        self.call_timer = 0                 # 발성 잔여 시간
        self.near_pain_zone = False         # pain zone 근접 여부

    def step(self, foods: list, player_pos: Tuple[float, float],
             map_width: float, map_height: float, config: 'ForagerConfig',
             obstacles: Optional[List[Tuple[float, float, float, float]]] = None):
        """NPC 행동 업데이트

        forager: 가장 가까운 음식 방향으로 이동 (70%), 랜덤 탐색 (30%)
        predator: 플레이어 추적
        """
        if self.behavior == "forager":
            self._forager_step(foods, map_width, map_height, config, obstacles)
        elif self.behavior == "predator":
            self._predator_step(player_pos, map_width, map_height, config)

    def _forager_step(self, foods: list, map_w: float, map_h: float,
                      config: 'ForagerConfig',
                      obstacles: Optional[List[Tuple[float, float, float, float]]] = None):
        """음식 탐색 행동"""
        # 가장 가까운 음식 찾기
        best_dist = float('inf')
        best_food = None
        for fx, fy, *_ in foods:
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

        # 장애물 충돌 (AABB-circle)
        if obstacles:
            for ox, oy, ow, oh in obstacles:
                closest_x = max(ox, min(new_x, ox + ow))
                closest_y = max(oy, min(new_y, oy + oh))
                dx = new_x - closest_x
                dy = new_y - closest_y
                if dx * dx + dy * dy < self.radius * self.radius:
                    new_x = self.x
                    new_y = self.y
                    self.angle += np.random.uniform(-0.5, 0.5)
                    break

        self.x = new_x
        self.y = new_y

    def check_food_collision(self, foods: list, config: 'ForagerConfig',
                             current_step: int = 0) -> bool:
        """NPC 음식 충돌 확인 (먹으면 제거 + 새 음식 생성)"""
        if not config.npc_food_eat_enabled:
            return False

        collision_dist = self.radius + config.food_radius
        for i, (fx, fy, *_) in enumerate(foods):
            dist = math.sqrt((self.x - fx)**2 + (self.y - fy)**2)
            if dist < collision_dist:
                foods.pop(i)
                self.food_eaten += 1
                self.last_eat_step = current_step  # Phase 15b
                # Phase 17: 먹을 때 food call 발성
                if config.npc_vocalization_enabled and self.call_timer <= 0:
                    if np.random.random() < config.npc_call_food_prob:
                        self.current_call = 1
                        self.call_timer = config.npc_call_duration
                return True
        return False

    def update_call_state(self, config: 'ForagerConfig', pain_zones: list = None):
        """Phase 17: NPC 발성 상태 업데이트"""
        # 타이머 감소
        if self.call_timer > 0:
            self.call_timer -= 1
            if self.call_timer <= 0:
                self.current_call = 0

        # Pain zone 근접 확인 → danger call (내부 원형 영역)
        near_pain = False
        if pain_zones:
            detect_r = config.pain_zone_radius + config.danger_range
            for cx, cy in pain_zones:
                if math.sqrt((self.x - cx)**2 + (self.y - cy)**2) < detect_r:
                    near_pain = True
                    break
        if near_pain and not self.near_pain_zone and self.call_timer <= 0:
            if np.random.random() < config.npc_call_danger_prob:
                self.current_call = 2
                self.call_timer = config.npc_call_duration
        self.near_pain_zone = near_pain


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
        self.foods: List[Tuple[float, float, int]] = []  # (x, y, food_type) 0=good, 1=bad
        self.steps: int = 0

        # 통계
        self.total_food_eaten: int = 0
        self.min_energy: float = 100
        self.max_energy: float = 0
        self.homeostasis_steps: int = 0  # 30-70 범위 유지 시간
        self.energy_history: List[float] = []
        self.position_history: List[Tuple[float, float, str]] = []  # (x, y, gw_state)

        # === Phase L5: Food Type 통계 ===
        self.good_food_eaten: int = 0
        self.bad_food_eaten: int = 0

        # === Phase L6: Last eaten food position (for cluster respawn) ===
        self._last_eaten_pos: Optional[Tuple[float, float]] = None

        # === Phase 2b: Pain Zone (내부 원형 영역) ===
        self.pain_zones: List[Tuple[float, float]] = []  # [(cx, cy), ...] 원형 pain zone 중심 좌표
        self.pain_damage_accumulated: float = 0.0  # 누적 Pain 데미지
        self.pain_zone_visits: int = 0             # Pain Zone 진입 횟수
        self.pain_zone_steps: int = 0              # Pain Zone 내 체류 시간
        self.was_in_pain: bool = False             # 이전 스텝 Pain Zone 여부 (탈출 감지)
        # === Pain Honest Metrics ===
        self.wall_bounces_total: int = 0           # 전체 벽 반사 횟수
        self.wall_bounces_in_pain: int = 0         # Pain Zone 내 벽 반사 횟수
        self.pain_approach_steps: int = 0          # Pain boundary에 접근 중인 스텝 수
        self.distance_to_pain_sum: float = 0.0     # pain boundary 거리 누적 (평균 계산용)
        self.current_pain_dwell: int = 0           # 현재 연속 pain 체류
        self.max_pain_dwell: int = 0               # 최대 연속 pain 체류
        self.pain_dwell_times: List[int] = []      # 각 방문별 체류 시간

        # === Food Patch 설정 및 통계 ===
        self.patches: List[Tuple[float, float]] = []  # Patch 중심 좌표 (실제 픽셀)
        self.patch_visits: List[int] = []             # 각 Patch 방문 횟수
        self.patch_food_eaten: List[int] = []         # 각 Patch에서 먹은 음식 수
        self.was_in_patch: List[bool] = []            # 이전 스텝 Patch 내 여부
        self._init_patches()

        # === Environment E1: Obstacles ===
        self.obstacles: List[Tuple[float, float, float, float]] = []  # (x, y, w, h)

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

        # Pain Zone 위치 생성 (음식 생성 전에 해야 food가 pain zone을 피함)
        self._generate_pain_zones()

        # 장애물 생성 (음식 생성 전에 해야 food가 장애물을 피함)
        self._generate_obstacles()

        # 음식 생성 (Field에만, Pain Zone 외부)
        self.foods = []
        self._spawn_foods(self.config.n_food)

        # 통계 초기화
        self.steps = 0
        self.total_food_eaten = 0
        self.good_food_eaten = 0
        self.bad_food_eaten = 0
        self.min_energy = self.config.energy_start
        self.max_energy = self.config.energy_start
        self.homeostasis_steps = 0
        self.energy_history = [self.energy]
        self.position_history = [(self.agent_x, self.agent_y, "neutral")]

        # Phase 2b: Pain Zone 통계 초기화
        self.pain_damage_accumulated = 0.0
        self.pain_zone_visits = 0
        self.pain_zone_steps = 0
        self.was_in_pain = False
        # Pain Honest Metrics 초기화
        self.wall_bounces_total = 0
        self.wall_bounces_in_pain = 0
        self.pain_approach_steps = 0
        self.distance_to_pain_sum = 0.0
        self.current_pain_dwell = 0
        self.max_pain_dwell = 0
        self.pain_dwell_times = []

        # Phase L1: Food approach tracking
        self.prev_min_food_dist = self.config.view_range

        # Food Patch 통계 초기화
        self.patch_visits = [0] * self.config.n_patches
        self.patch_food_eaten = [0] * self.config.n_patches
        self.was_in_patch = [False] * self.config.n_patches

        # Phase 17: Agent vocalization state
        self._agent_call_type = 0
        self._agent_call_cooldown = 0

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

        # === Predator Initialization ===
        self.predators = []
        self.predator_damage_total = 0.0
        self.predator_contact_steps = 0
        self._predator_contact = False
        if self.config.predator_enabled:
            for _ in range(self.config.n_predators):
                side = np.random.randint(4)
                margin = 30
                if side == 0:
                    sx, sy = np.random.uniform(margin, self.config.width - margin), float(margin)
                elif side == 1:
                    sx, sy = np.random.uniform(margin, self.config.width - margin), float(self.config.height - margin)
                elif side == 2:
                    sx, sy = float(margin), np.random.uniform(margin, self.config.height - margin)
                else:
                    sx, sy = float(self.config.width - margin), np.random.uniform(margin, self.config.height - margin)
                self.predators.append(PredatorAgent(sx, sy, self.config))

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

        # Phase L14: Motor noise injection
        if self.config.motor_noise_enabled:
            angle_delta += np.random.normal(0, self.config.motor_noise_std)

        # 1. 이동
        self.agent_angle += angle_delta
        self.agent_angle = self.agent_angle % (2 * np.pi)  # 0 ~ 2π 범위 유지

        new_x = self.agent_x + np.cos(self.agent_angle) * self.config.agent_speed
        new_y = self.agent_y + np.sin(self.agent_angle) * self.config.agent_speed

        # 벽 충돌 처리 (부드러운 바운스)
        _bounced = False
        if new_x < self.config.agent_radius:
            new_x = self.config.agent_radius
            self.agent_angle = np.pi - self.agent_angle
            _bounced = True
        elif new_x > self.config.width - self.config.agent_radius:
            new_x = self.config.width - self.config.agent_radius
            self.agent_angle = np.pi - self.agent_angle
            _bounced = True

        if new_y < self.config.agent_radius:
            new_y = self.config.agent_radius
            self.agent_angle = -self.agent_angle
            _bounced = True
        elif new_y > self.config.height - self.config.agent_radius:
            new_y = self.config.height - self.config.agent_radius
            self.agent_angle = -self.agent_angle
            _bounced = True

        if _bounced:
            self.wall_bounces_total += 1

        # === E1: 장애물 충돌 처리 (AABB-circle) ===
        if self.config.obstacles_enabled and self.obstacles:
            for ox, oy, ow, oh in self.obstacles:
                # AABB-circle collision: 가장 가까운 점 찾기
                closest_x = max(ox, min(new_x, ox + ow))
                closest_y = max(oy, min(new_y, oy + oh))
                dx = new_x - closest_x
                dy = new_y - closest_y
                dist_sq = dx * dx + dy * dy
                r = self.config.agent_radius
                if dist_sq < r * r:
                    # 충돌 → 이전 위치로 복원
                    new_x = self.agent_x
                    new_y = self.agent_y
                    _bounced = True
                    break

        self.agent_x = new_x
        self.agent_y = new_y

        # Pain boundary 거리 추적
        _prev_dist = self._distance_to_pain_boundary_xy(
            self.agent_x - np.cos(self.agent_angle) * self.config.agent_speed,
            self.agent_y - np.sin(self.agent_angle) * self.config.agent_speed)
        _cur_dist = self._distance_to_pain_boundary()
        self.distance_to_pain_sum += _cur_dist
        if _cur_dist < _prev_dist:
            self.pain_approach_steps += 1

        # 2. 에너지 감소 (위치에 따라 다름)
        if self._in_nest():
            self.energy -= self.config.energy_decay_nest
        else:
            self.energy -= self.config.energy_decay_field

        self.energy = max(0, self.energy)

        # 3. 음식 섭취
        reward = 0.0
        food_eaten, food_type = self._check_food_collision()
        if food_eaten:
            self.total_food_eaten += 1
            # Phase L12: 위험 근접 보너스 (pain zone 근처에서 먹으면 +50%)
            danger_bonus = 1.0
            if self.config.danger_food_enabled:
                for pz_x, pz_y in self.pain_zones:
                    dist = math.sqrt((self.agent_x - pz_x)**2 + (self.agent_y - pz_y)**2)
                    if dist < self.config.pain_zone_radius + self.config.danger_range:
                        danger_bonus = 1.0 + self.config.danger_food_bonus
                        break
            if food_type == 0:  # 좋은 음식
                self.energy = min(self.config.energy_max,
                                self.energy + self.config.food_value * danger_bonus)
                reward += self.config.reward_food
                self.good_food_eaten += 1
            elif food_type == 1:  # 나쁜 음식
                self.energy = max(0, self.energy + self.config.bad_food_energy)
                self.bad_food_eaten += 1

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
                self.current_pain_dwell = 0
            self.current_pain_dwell += 1
            self.max_pain_dwell = max(self.max_pain_dwell, self.current_pain_dwell)
            # 벽 반사가 pain zone 안에서 발생했는지
            if _bounced:
                self.wall_bounces_in_pain += 1

        # Pain Zone 탈출 보상
        if self.was_in_pain and not in_pain:
            reward += self.config.reward_escape
            self.pain_dwell_times.append(self.current_pain_dwell)
            self.current_pain_dwell = 0

        if not in_pain:
            self.current_pain_dwell = 0

        self.was_in_pain = in_pain

        # === Phase 15: NPC 업데이트 ===
        npc_eating_events = []
        if self.config.social_enabled:
            for npc in self.npc_agents:
                npc.step(self.foods, (self.agent_x, self.agent_y),
                         self.config.width, self.config.height, self.config,
                         obstacles=self.obstacles if self.config.obstacles_enabled else None)
                # NPC 음식 경쟁
                if npc.check_food_collision(self.foods, self.config,
                                            current_step=self.steps):
                    self.npc_food_stolen += 1
                    npc_eating_events.append((npc.x, npc.y, self.steps))
                    self._spawn_foods(1)  # 새 음식 생성 (총 개수 유지)
                # Phase 17: NPC 발성 상태 업데이트
                if self.config.npc_vocalization_enabled:
                    npc.update_call_state(self.config, self.pain_zones)

        # === Phase 17: 에이전트 발성이 NPC에 미치는 영향 ===
        if (self.config.social_enabled and self.config.npc_vocalization_enabled
                and hasattr(self, '_agent_call_type') and self._agent_call_cooldown <= 0):
            if self._agent_call_type in (1, 2):
                for npc in self.npc_agents:
                    dist = math.sqrt((npc.x - self.agent_x)**2 + (npc.y - self.agent_y)**2)
                    if dist < self.config.agent_call_range and dist > 1.0:
                        if self._agent_call_type == 1:
                            # Food call: NPC가 에이전트 쪽으로 회전
                            target_angle = math.atan2(self.agent_y - npc.y, self.agent_x - npc.x)
                        else:
                            # Danger call: NPC가 에이전트 반대쪽으로 회전 (도주)
                            target_angle = math.atan2(npc.y - self.agent_y, npc.x - self.agent_x)
                        angle_diff = (target_angle - npc.angle + math.pi) % (2 * math.pi) - math.pi
                        npc.angle += np.clip(angle_diff, -self.config.npc_call_response_speed,
                                             self.config.npc_call_response_speed)
                self._agent_call_cooldown = self.config.agent_call_cooldown
        if hasattr(self, '_agent_call_cooldown'):
            self._agent_call_cooldown = max(0, self._agent_call_cooldown - 1)

        # === Predator Update ===
        self._predator_contact = False
        if self.config.predator_enabled and self.predators:
            for pred in self.predators:
                pred.step((self.agent_x, self.agent_y),
                          self.config.width, self.config.height, self.config,
                          obstacles=self.obstacles if self.config.obstacles_enabled else None)
                dist = math.sqrt((self.agent_x - pred.x)**2 + (self.agent_y - pred.y)**2)
                if dist < self.config.predator_catch_radius:
                    self._predator_contact = True
                    pred.contact_steps += 1
                    self.predator_contact_steps += 1
                    self.energy -= self.config.predator_damage
                    self.energy = max(0, self.energy)
                    self.predator_damage_total += self.config.predator_damage

        # === Food Patch 방문 추적 ===
        if self.config.food_patch_enabled:
            current_patch = self._get_current_patch()
            for i in range(len(self.patches)):
                in_this_patch = (current_patch == i)
                # 새로 진입한 경우 방문 횟수 증가
                if in_this_patch and not self.was_in_patch[i]:
                    self.patch_visits[i] += 1
                self.was_in_patch[i] = in_this_patch

        # 4b. Food approach signal (Phase L1: dopamine shaping)
        if self.foods:
            min_food_dist = min(
                math.hypot(f[0] - self.agent_x, f[1] - self.agent_y)
                for f in self.foods
            )
        else:
            min_food_dist = self.config.view_range
        food_approach_signal = max(0.0, (self.prev_min_food_dist - min_food_dist) / self.config.view_range)
        food_approach_signal = min(1.0, food_approach_signal)
        self.prev_min_food_dist = min_food_dist

        # 5. 통계 업데이트
        self.min_energy = min(self.min_energy, self.energy)
        self.max_energy = max(self.max_energy, self.energy)
        self.energy_history.append(self.energy)
        gw_state = self.brain_info.get("gw_broadcast", "neutral") if hasattr(self, 'brain_info') and self.brain_info else "neutral"
        self.position_history.append((self.agent_x, self.agent_y, gw_state))

        # 6. 종료 조건
        self.steps += 1
        done = False
        death_cause = None

        if self.energy <= 0:
            done = True
            death_cause = "starve"
            # 포식자 사망 판별: 포식자 피해가 30% 이상이면 predator 사인
            if self.config.predator_enabled and self.predator_damage_total > 0:
                if self.predator_damage_total >= self.config.energy_start * 0.3:
                    death_cause = "predator"
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
            "food_type": food_type if food_eaten else -1,
            "total_food": self.total_food_eaten,
            "good_food_eaten": self.good_food_eaten,
            "bad_food_eaten": self.bad_food_eaten,
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
            "wall_bounces_total": self.wall_bounces_total,
            "wall_bounces_in_pain": self.wall_bounces_in_pain,
            "pain_approach_steps": self.pain_approach_steps,
            "avg_dist_to_pain": self.distance_to_pain_sum / max(1, self.steps),
            "max_pain_dwell": self.max_pain_dwell,
            "avg_pain_dwell": np.mean(self.pain_dwell_times) if self.pain_dwell_times else 0,
            "pain_dwell_times": self.pain_dwell_times.copy(),
            # Food Patch 정보
            "patch_visits": self.patch_visits.copy() if self.config.food_patch_enabled else [],
            "patch_food_eaten": self.patch_food_eaten.copy() if self.config.food_patch_enabled else [],
            "current_patch": self._get_current_patch() if self.config.food_patch_enabled else -1,
            # Phase 15: NPC 정보
            "npc_food_stolen": self.npc_food_stolen,
            "npc_positions": [(npc.x, npc.y) for npc in self.npc_agents] if self.config.social_enabled else [],
            # Phase 15b: NPC 먹기 이벤트
            "npc_eating_events": npc_eating_events,
            # Phase 17: Agent vocalization
            "agent_call_type": getattr(self, '_agent_call_type', 0),
            # Phase L1: Food approach signal (dopamine shaping)
            "food_approach_signal": food_approach_signal,
            # Predator info
            "predator_contact": self._predator_contact if self.config.predator_enabled else False,
            "predator_damage_total": self.predator_damage_total if self.config.predator_enabled else 0,
            "predator_contact_steps": self.predator_contact_steps if self.config.predator_enabled else 0,
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

    def _generate_pain_zones(self):
        """내부 원형 Pain Zone 위치 생성 (매 에피소드 랜덤)"""
        if not self.config.pain_zone_enabled:
            self.pain_zones = []
            return

        r = self.config.pain_zone_radius
        margin = r + 10  # 벽에서 최소 10px 간격
        nest_cx, nest_cy = self.config.width / 2, self.config.height / 2
        nest_half = self.config.nest_size / 2

        zones = []
        for _ in range(self.config.pain_zone_count):
            for _ in range(500):
                cx = np.random.uniform(margin, self.config.width - margin)
                cy = np.random.uniform(margin, self.config.height - margin)

                # 둥지와 겹치지 않기 (circle-rect intersection)
                nearest_x = max(nest_cx - nest_half, min(cx, nest_cx + nest_half))
                nearest_y = max(nest_cy - nest_half, min(cy, nest_cy + nest_half))
                dist_to_nest = math.sqrt((cx - nearest_x)**2 + (cy - nearest_y)**2)
                if dist_to_nest < r + 10:
                    continue

                # 다른 pain zone과 겹치지 않기
                overlap = False
                for oz_x, oz_y in zones:
                    if math.sqrt((cx - oz_x)**2 + (cy - oz_y)**2) < 2 * r + 20:
                        overlap = True
                        break
                if overlap:
                    continue

                zones.append((cx, cy))
                break

        self.pain_zones = zones

    def _generate_obstacles(self):
        """정적 장애물 생성 (매 에피소드 랜덤 배치)"""
        if not self.config.obstacles_enabled:
            self.obstacles = []
            return

        margin = 30.0
        nest_cx, nest_cy = self.config.width / 2, self.config.height / 2
        nest_half = self.config.nest_size / 2
        gap = 20.0  # 장애물 간 최소 간격

        obstacles = []
        for _ in range(self.config.n_obstacles):
            for _ in range(500):
                w = np.random.uniform(self.config.obstacle_min_size,
                                      self.config.obstacle_max_size)
                h = np.random.uniform(self.config.obstacle_min_size,
                                      self.config.obstacle_max_size)
                x = np.random.uniform(margin, self.config.width - margin - w)
                y = np.random.uniform(margin, self.config.height - margin - h)

                # 둥지와 겹치지 않음 (중앙 100x100 + 마진)
                nest_margin = 10
                if (x + w > nest_cx - nest_half - nest_margin and
                        x < nest_cx + nest_half + nest_margin and
                        y + h > nest_cy - nest_half - nest_margin and
                        y < nest_cy + nest_half + nest_margin):
                    continue

                # Pain Zone과 겹치지 않음
                pain_overlap = False
                for pz_x, pz_y in self.pain_zones:
                    # AABB-circle: 장애물 AABB에서 pain zone 중심까지 최소 거리
                    closest_x = max(x, min(pz_x, x + w))
                    closest_y = max(y, min(pz_y, y + h))
                    dist = math.sqrt((pz_x - closest_x)**2 + (pz_y - closest_y)**2)
                    if dist < self.config.pain_zone_radius + 20:
                        pain_overlap = True
                        break
                if pain_overlap:
                    continue

                # 다른 장애물과 겹치지 않음 (gap px 간격)
                obs_overlap = False
                for ox, oy, ow, oh in obstacles:
                    # AABB-AABB 간격 체크
                    if (x - gap < ox + ow and x + w + gap > ox and
                            y - gap < oy + oh and y + h + gap > oy):
                        obs_overlap = True
                        break
                if obs_overlap:
                    continue

                obstacles.append((x, y, w, h))
                break

        self.obstacles = obstacles

    def _point_in_obstacle(self, px: float, py: float, margin: float = 0.0) -> bool:
        """점이 장애물 내부(+마진)에 있는지 확인"""
        for ox, oy, ow, oh in self.obstacles:
            if (ox - margin <= px <= ox + ow + margin and
                    oy - margin <= py <= oy + oh + margin):
                return True
        return False

    def _in_pain_zone(self) -> bool:
        """내부 원형 Pain Zone 내부인지 확인"""
        r_sq = self.config.pain_zone_radius ** 2
        for cx, cy in self.pain_zones:
            if (self.agent_x - cx)**2 + (self.agent_y - cy)**2 < r_sq:
                return True
        return False

    def _distance_to_pain_boundary(self) -> float:
        """현재 위치에서 가장 가까운 pain zone 경계까지의 거리 (외부=양수, 내부=음수)."""
        return self._distance_to_pain_boundary_xy(self.agent_x, self.agent_y)

    def _distance_to_pain_boundary_xy(self, x: float, y: float) -> float:
        """임의 좌표에서 가장 가까운 pain zone 경계까지 거리."""
        if not self.pain_zones:
            return float('inf')
        r = self.config.pain_zone_radius
        min_dist = float('inf')
        for cx, cy in self.pain_zones:
            dist = math.sqrt((x - cx)**2 + (y - cy)**2) - r
            min_dist = min(min_dist, dist)
        return min_dist

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
        return self._distance_to_pain_boundary()

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

    # === Predator Detection Methods ===

    def _cast_predator_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """포식자 방향 레이캐스트 (food ray와 동일 패턴, L/R 분리)"""
        n_half = self.config.n_rays // 2
        rays_l, rays_r = np.zeros(n_half), np.zeros(n_half)
        for i in range(n_half):
            angle_l = self.agent_angle - math.pi / 2 + (i / n_half) * math.pi / 2
            angle_r = self.agent_angle + (i / n_half) * math.pi / 2
            rays_l[i] = self._cast_single_predator_ray(angle_l)
            rays_r[i] = self._cast_single_predator_ray(angle_r)
        return rays_l, rays_r

    def _cast_single_predator_ray(self, angle: float) -> float:
        """단일 레이로 최근접 포식자 감지 (0=없음, 1=매우 근접)"""
        best_dist = self.config.predator_view_range
        for pred in self.predators:
            dx, dy = pred.x - self.agent_x, pred.y - self.agent_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > self.config.predator_view_range:
                continue
            pred_angle = math.atan2(dy, dx)
            diff = abs((pred_angle - angle + math.pi) % (2 * math.pi) - math.pi)
            if diff < 0.26 and dist < best_dist:  # 15° cone
                best_dist = dist
        if best_dist >= self.config.predator_view_range:
            return 0.0
        return 1.0 - (best_dist / self.config.predator_view_range)

    def _get_predator_danger(self) -> float:
        """최근접 포식자 위험 신호 (0=안전, 1=접촉)"""
        min_dist = float('inf')
        for pred in self.predators:
            d = math.sqrt((self.agent_x - pred.x) ** 2 + (self.agent_y - pred.y) ** 2)
            min_dist = min(min_dist, d)
        if min_dist < self.config.predator_catch_radius:
            return 1.0
        if min_dist < self.config.predator_chase_range:
            return 1.0 - (min_dist / self.config.predator_chase_range)
        return 0.0

    def _compute_danger_sound(self) -> Tuple[float, float]:
        """
        Phase 11: Pain Zone에서 발생하는 위험 소리 계산

        Returns:
            (left_sound, right_sound): 좌우 귀에 들리는 소리 강도 (0~1)
        """
        if not self.config.sound_enabled or not self.config.pain_zone_enabled:
            return 0.0, 0.0

        if not self.pain_zones:
            return 0.0, 0.0

        # 가장 가까운 원형 Pain Zone 찾기
        closest_cx, closest_cy = self.pain_zones[0]
        closest_dist = float('inf')
        r = self.config.pain_zone_radius
        for cx, cy in self.pain_zones:
            dist = math.sqrt((self.agent_x - cx)**2 + (self.agent_y - cy)**2) - r
            if dist < closest_dist:
                closest_dist = dist
                closest_cx, closest_cy = cx, cy

        if closest_dist > self.config.danger_sound_range:
            return 0.0, 0.0

        # 거리에 따른 기본 강도 (Pain Zone 내부면 최대)
        if closest_dist <= 0:
            intensity = 1.0
        else:
            intensity = 1.0 - (closest_dist / self.config.danger_sound_range) ** self.config.sound_decay
        intensity = max(0.0, min(1.0, intensity))

        # 가장 가까운 Pain Zone 중심 방향으로의 각도
        danger_angle = math.atan2(closest_cy - self.agent_y, closest_cx - self.agent_x)

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

        for food_x, food_y, *_ in self.foods:
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

        # Phase L5: 타입별 음식 레이캐스트
        if self.config.n_food_types >= 2:
            good_food_rays_l, good_food_rays_r = self._cast_typed_food_rays(0)
            bad_food_rays_l, bad_food_rays_r = self._cast_typed_food_rays(1)
        else:
            n_half = self.config.n_rays // 2
            good_food_rays_l = np.zeros(n_half)
            good_food_rays_r = np.zeros(n_half)
            bad_food_rays_l = np.zeros(n_half)
            bad_food_rays_r = np.zeros(n_half)

        # Phase 2b: Pain 관련 관찰
        if self.config.pain_zone_enabled:
            pain_rays_l, pain_rays_r = self._cast_pain_rays()
            danger_signal = self._get_danger_signal()
        else:
            pain_rays_l = np.zeros(self.config.n_rays // 2)
            pain_rays_r = np.zeros(self.config.n_rays // 2)
            danger_signal = 0.0

        # === Predator sensory injection (이동형 pain source) ===
        predator_rays_l = np.zeros(self.config.n_rays // 2)
        predator_rays_r = np.zeros(self.config.n_rays // 2)
        predator_danger = 0.0
        if self.config.predator_enabled and hasattr(self, 'predators') and self.predators:
            predator_rays_l, predator_rays_r = self._cast_predator_rays()
            predator_danger = self._get_predator_danger()
            # 기존 pain/danger 채널에 주입 (np.maximum: 강한 쪽만)
            pain_rays_l = np.maximum(pain_rays_l, predator_rays_l * self.config.predator_pain_intensity)
            pain_rays_r = np.maximum(pain_rays_r, predator_rays_r * self.config.predator_pain_intensity)
            danger_signal = max(danger_signal, predator_danger * self.config.predator_danger_intensity)

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

        # Phase 17: NPC vocalization observation
        if self.config.npc_vocalization_enabled and self.npc_agents:
            npc_call_food_l, npc_call_food_r = self._compute_npc_call_food()
            npc_call_danger_l, npc_call_danger_r = self._compute_npc_call_danger()
        else:
            npc_call_food_l, npc_call_food_r = 0.0, 0.0
            npc_call_danger_l, npc_call_danger_r = 0.0, 0.0

        # Phase L14: Sensor jitter (multiplicative noise on ray values)
        if self.config.sensor_jitter_enabled:
            jitter_std = self.config.sensor_jitter_std
            n_half = self.config.n_rays // 2
            # pain_rays 제외: Push-Pull(60/-40)은 정밀한 L/R 차이에 의존
            for rays in [food_rays_l, food_rays_r, wall_rays_l, wall_rays_r,
                         good_food_rays_l, good_food_rays_r, bad_food_rays_l, bad_food_rays_r]:
                rays *= (1.0 + np.random.normal(0, jitter_std, rays.shape))
                np.clip(rays, 0.0, 1.0, out=rays)

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

            # Phase 17: NPC vocalization 관찰 채널
            "npc_call_food_left": npc_call_food_l,
            "npc_call_food_right": npc_call_food_r,
            "npc_call_danger_left": npc_call_danger_l,
            "npc_call_danger_right": npc_call_danger_r,

            # Phase L5: 타입별 음식 감각 (지각 학습)
            "good_food_rays_left": good_food_rays_l,
            "good_food_rays_right": good_food_rays_r,
            "bad_food_rays_left": bad_food_rays_l,
            "bad_food_rays_right": bad_food_rays_r,

            # Predator (raw, 미래 뇌 확장용)
            "predator_rays_left": predator_rays_l,
            "predator_rays_right": predator_rays_r,
            "predator_danger": predator_danger,
            "predator_contact": float(getattr(self, '_predator_contact', False)),
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

        for food_x, food_y, *_ in self.foods:
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

    # === Phase L5: 타입별 음식 레이캐스트 ===

    def _cast_typed_food_rays(self, target_type: int) -> Tuple[np.ndarray, np.ndarray]:
        """특정 타입의 음식만 감지하는 레이캐스트 (L/R 분리)

        Args:
            target_type: 감지할 음식 타입 (0=좋은, 1=나쁜)
        """
        n_half = self.config.n_rays // 2
        rays_left = np.zeros(n_half)
        rays_right = np.zeros(n_half)

        for i in range(n_half):
            angle_l = self.agent_angle - np.pi/2 + (i / n_half) * np.pi/2
            rays_left[i] = self._cast_single_typed_food_ray(angle_l, target_type)
            angle_r = self.agent_angle + (i / n_half) * np.pi/2
            rays_right[i] = self._cast_single_typed_food_ray(angle_r, target_type)

        return rays_left, rays_right

    def _cast_single_typed_food_ray(self, angle: float, target_type: int) -> float:
        """단일 레이로 특정 타입의 가장 가까운 음식 감지"""
        best_dist = self.config.view_range

        for food_x, food_y, food_type in self.foods:
            if food_type != target_type:
                continue

            dx = food_x - self.agent_x
            dy = food_y - self.agent_y
            dist = np.sqrt(dx*dx + dy*dy)

            if dist > self.config.view_range:
                continue

            food_angle = np.arctan2(dy, dx)
            angle_diff = abs((food_angle - angle + np.pi) % (2*np.pi) - np.pi)

            if angle_diff < 0.26:
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
        """단일 레이로 벽/장애물까지 거리 (0=멀리, 1=매우가까움)"""
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

        # === E1: 장애물 AABB와 ray intersection ===
        if self.config.obstacles_enabled and self.obstacles:
            ox_pos = self.agent_x
            oy_pos = self.agent_y
            for bx, by, bw, bh in self.obstacles:
                # Ray-AABB intersection (slab method)
                # t_min, t_max for each axis
                if abs(cos_a) > 1e-8:
                    t1 = (bx - ox_pos) / cos_a
                    t2 = (bx + bw - ox_pos) / cos_a
                    if t1 > t2:
                        t1, t2 = t2, t1
                else:
                    if bx <= ox_pos <= bx + bw:
                        t1, t2 = -1e30, 1e30
                    else:
                        continue  # ray parallel and outside

                if abs(sin_a) > 1e-8:
                    t3 = (by - oy_pos) / sin_a
                    t4 = (by + bh - oy_pos) / sin_a
                    if t3 > t4:
                        t3, t4 = t4, t3
                else:
                    if by <= oy_pos <= by + bh:
                        t3, t4 = -1e30, 1e30
                    else:
                        continue

                t_enter = max(t1, t3)
                t_exit = min(t2, t4)

                if t_enter < t_exit and t_exit > 0:
                    hit_t = t_enter if t_enter > 0 else 0.0
                    if hit_t > 0:
                        distances.append(hit_t)

        if not distances:
            return 0.0

        min_dist = min(distances)
        if min_dist >= self.config.view_range:
            return 0.0
        return 1.0 - (min_dist / self.config.view_range)

    def _cast_pain_rays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pain Zone 방향 레이캐스트 (L/R 분리)
        Pain Zone = 내부 원형 영역

        Returns:
            (left_rays, right_rays): 각각 n_rays//2 크기
            0=Pain Zone 멀리, 1=Pain Zone 가까이/내부
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
        단일 레이로 원형 Pain Zone까지 거리 (ray-circle intersection)

        외부: 0=멀리, 1=가까이 (approach gradient)
        내부: exit distance gradient (가까운 edge = 낮은 pain, 먼 edge = 높은 pain)
              → pain_L ≠ pain_R → push-pull이 탈출 방향으로 회전
        """
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        r = self.config.pain_zone_radius
        min_approach = float('inf')  # 외부에서 접근 시 최소 거리
        max_inside_intensity = 0.0   # 내부일 때 최대 강도

        for cx, cy in self.pain_zones:
            dx = self.agent_x - cx
            dy = self.agent_y - cy
            dist_sq = dx * dx + dy * dy

            # Ray-circle intersection: |origin + t*dir - center|² = r²
            # a*t² + b*t + c = 0
            b = 2 * (dx * cos_a + dy * sin_a)
            c = dist_sq - r * r

            discriminant = b * b - 4 * c
            if discriminant < 0:
                continue  # 레이가 원을 빗나감

            sqrt_disc = math.sqrt(discriminant)
            t1 = (-b - sqrt_disc) / 2  # 첫 교점
            t2 = (-b + sqrt_disc) / 2  # 두번째 교점

            if c > 0:
                # 외부: t1 > 0이면 이 방향으로 원에 닿음
                if t1 > 0:
                    min_approach = min(min_approach, t1)
            else:
                # 내부: t2 = exit distance (이 방향으로 원 탈출까지 거리)
                # 가까운 edge → 낮은 pain, 먼 edge → 높은 pain
                if t2 > 0:
                    intensity = min(1.0, t2 / (2 * r))
                    max_inside_intensity = max(max_inside_intensity, intensity)

        # 내부 gradient가 있으면 우선
        if max_inside_intensity > 0:
            return max_inside_intensity

        # 외부 접근 gradient
        if min_approach < self.config.view_range:
            return 1.0 - (min_approach / self.config.view_range)
        return 0.0

    def _check_food_collision(self) -> Tuple[bool, int]:
        """음식 충돌 확인

        Returns:
            (eaten, food_type): eaten=True면 음식 먹음, food_type=0(좋은)/1(나쁜)
        """
        collision_dist = self.config.agent_radius + self.config.food_radius

        for i, (food_x, food_y, food_type) in enumerate(self.foods):
            dist = np.sqrt((self.agent_x - food_x)**2 +
                          (self.agent_y - food_y)**2)
            if dist < collision_dist:
                # Food Patch 통계: 어느 Patch에서 먹었는지 기록
                if self.config.food_patch_enabled:
                    patch_idx = self._in_patch(food_x, food_y)
                    if patch_idx >= 0:
                        self.patch_food_eaten[patch_idx] += 1
                # Phase L6: 먹은 위치 저장 (클러스터 리스폰용)
                self._last_eaten_pos = (food_x, food_y)
                self.foods.pop(i)
                self._spawn_foods(1)  # 새 음식 생성
                return True, food_type
        return False, -1

    def _assign_food_type(self) -> int:
        """Phase L5: 음식 타입 결정 (0=좋은, 1=나쁜)"""
        if self.config.n_food_types <= 1:
            return 0
        return 0 if np.random.random() < self.config.food_type_ratio else 1

    def _spawn_foods(self, n: int):
        """Field에 음식 생성 (Nest 외부, Pain Zone 외부)

        Food Patch 모드 활성화 시:
        - 80%의 음식이 Patch 내에 생성
        - 20%는 랜덤 위치에 생성

        Phase L5: 각 음식에 food_type 부여 (0=좋은, 1=나쁜)
        Phase L12: 30% 음식이 Pain Zone 가장자리에 생성 (위험 근접 고보상)
        """
        cx, cy = self.config.width / 2, self.config.height / 2
        half = self.config.nest_size / 2
        margin = self.config.food_radius * 2
        pain_r = self.config.pain_zone_radius + self.config.food_radius * 2  # pain zone 외부 마진

        for _ in range(n):
            food_type = self._assign_food_type()

            # Phase L12: 위험 근접 음식 (pain zone 바로 바깥 도넛 영역에 생성)
            if (self.config.danger_food_enabled and self.pain_zones
                    and np.random.random() < self.config.danger_food_ratio):
                pz_x, pz_y = self.pain_zones[np.random.randint(len(self.pain_zones))]
                pz_r = self.config.pain_zone_radius
                dr = self.config.danger_range  # danger_range 내 도넛 영역
                for _ in range(50):
                    angle = np.random.uniform(0, 2 * np.pi)
                    # pain zone 경계 + 5px ~ danger_range 사이 도넛
                    r_dist = pz_r + 5 + np.random.uniform(0, dr - 5)
                    x = pz_x + r_dist * np.cos(angle)
                    y = pz_y + r_dist * np.sin(angle)
                    if not (margin <= x <= self.config.width - margin and
                            margin <= y <= self.config.height - margin):
                        continue
                    if (cx - half <= x <= cx + half and cy - half <= y <= cy + half):
                        continue
                    # pain zone 안에 들어가면 안 됨
                    in_zone = False
                    for oz_x, oz_y in self.pain_zones:
                        if (x - oz_x)**2 + (y - oz_y)**2 < (pz_r + margin)**2:
                            in_zone = True
                            break
                    if in_zone:
                        continue
                    # 장애물 내부 제외
                    if self._point_in_obstacle(x, y, self.config.food_radius):
                        continue
                    self.foods.append((x, y, food_type))
                    break
                else:
                    pass  # fall through to other spawn methods
                if len(self.foods) >= self.config.n_food + n:
                    continue

            # Food Patch 모드: 80% Patch 내, 20% 랜덤
            if self.config.food_patch_enabled and self.patches:
                if np.random.random() < self.config.food_spawn_in_patch_prob:
                    # Patch 내에 음식 생성
                    patch = self.patches[np.random.randint(len(self.patches))]
                    for _ in range(100):
                        angle = np.random.uniform(0, 2 * np.pi)
                        r = np.sqrt(np.random.uniform(0, 1)) * self.config.patch_radius
                        x = patch[0] + r * np.cos(angle)
                        y = patch[1] + r * np.sin(angle)

                        # 경계 체크: 맵 내부, Nest 외부, Pain Zone 외부
                        if not (margin <= x <= self.config.width - margin and
                                margin <= y <= self.config.height - margin):
                            continue
                        if (cx - half <= x <= cx + half and cy - half <= y <= cy + half):
                            continue
                        # Pain Zone 내부 체크
                        in_zone = False
                        for pz_x, pz_y in self.pain_zones:
                            if (x - pz_x)**2 + (y - pz_y)**2 < pain_r**2:
                                in_zone = True
                                break
                        if in_zone:
                            continue
                        if self._point_in_obstacle(x, y, self.config.food_radius):
                            continue
                        self.foods.append((x, y, food_type))
                        break
                    continue

            # Phase L6: 클러스터 리스폰 (먹은 위치 근처에 60% 확률로 생성)
            if (self.config.food_cluster_respawn and self._last_eaten_pos is not None
                    and np.random.random() < self.config.food_cluster_prob):
                ex, ey = self._last_eaten_pos
                cr = self.config.food_cluster_radius
                for _ in range(50):
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.sqrt(np.random.uniform(0, 1)) * cr
                    x = ex + r * np.cos(angle)
                    y = ey + r * np.sin(angle)
                    if not (margin <= x <= self.config.width - margin and
                            margin <= y <= self.config.height - margin):
                        continue
                    if (cx - half <= x <= cx + half and cy - half <= y <= cy + half):
                        continue
                    in_zone = False
                    for pz_x, pz_y in self.pain_zones:
                        if (x - pz_x)**2 + (y - pz_y)**2 < pain_r**2:
                            in_zone = True
                            break
                    if in_zone:
                        continue
                    if self._point_in_obstacle(x, y, self.config.food_radius):
                        continue
                    self.foods.append((x, y, food_type))
                    self._last_eaten_pos = None
                    break
                else:
                    self._last_eaten_pos = None  # 클러스터 실패 → 랜덤 폴백
                    # fall through to random spawn below

                if len(self.foods) >= self.config.n_food:
                    continue  # 클러스터로 이미 생성됨

            # 기존 방식: 랜덤 위치 (Patch 모드 OFF 또는 클러스터 실패 시)
            for _ in range(100):
                x = np.random.uniform(margin, self.config.width - margin)
                y = np.random.uniform(margin, self.config.height - margin)

                # Nest 내부면 다시
                if (cx - half <= x <= cx + half and cy - half <= y <= cy + half):
                    continue
                # Pain Zone 내부면 다시
                in_zone = False
                for pz_x, pz_y in self.pain_zones:
                    if (x - pz_x)**2 + (y - pz_y)**2 < pain_r**2:
                        in_zone = True
                        break
                if in_zone:
                    continue
                # 장애물 내부면 다시
                if self._point_in_obstacle(x, y, self.config.food_radius):
                    continue
                self.foods.append((x, y, food_type))
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
            for fx, fy, *_ in self.foods:
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

    # === Phase 17: NPC Vocalization 관찰 메서드 ===

    def _compute_npc_call_food(self) -> Tuple[float, float]:
        """
        NPC food call 감지 (좌우 분리)

        NPC가 current_call==1 (food call)이고 call_timer>0이면,
        거리와 방향에 따라 좌우 신호 발생.

        Returns:
            (call_food_left, call_food_right): 0~1
        """
        left_total = 0.0
        right_total = 0.0

        for npc in self.npc_agents:
            if npc.current_call != 1 or npc.call_timer <= 0:
                continue

            dx = npc.x - self.agent_x
            dy = npc.y - self.agent_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > self.config.npc_call_range or dist < 1.0:
                continue

            # 거리 감쇠
            intensity = 1.0 - (dist / self.config.npc_call_range) ** self.config.sound_decay
            intensity = max(0.0, intensity)
            # 시간 감쇠 (call_timer / duration)
            intensity *= npc.call_timer / self.config.npc_call_duration

            # 좌우 분리
            npc_angle = math.atan2(dy, dx)
            rel_angle = npc_angle - self.agent_angle
            direction = math.sin(rel_angle)

            base = intensity * 0.3
            directional = intensity * 0.7

            if direction < 0:
                left_total += base + directional * abs(direction)
                right_total += base
            else:
                right_total += base + directional * abs(direction)
                left_total += base

        return min(1.0, left_total), min(1.0, right_total)

    def _compute_npc_call_danger(self) -> Tuple[float, float]:
        """
        NPC danger call 감지 (좌우 분리)

        NPC가 current_call==2 (danger call)이고 call_timer>0이면 신호 발생.

        Returns:
            (call_danger_left, call_danger_right): 0~1
        """
        left_total = 0.0
        right_total = 0.0

        for npc in self.npc_agents:
            if npc.current_call != 2 or npc.call_timer <= 0:
                continue

            dx = npc.x - self.agent_x
            dy = npc.y - self.agent_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > self.config.npc_call_range or dist < 1.0:
                continue

            intensity = 1.0 - (dist / self.config.npc_call_range) ** self.config.sound_decay
            intensity = max(0.0, intensity)
            intensity *= npc.call_timer / self.config.npc_call_duration

            npc_angle = math.atan2(dy, dx)
            rel_angle = npc_angle - self.agent_angle
            direction = math.sin(rel_angle)

            base = intensity * 0.3
            directional = intensity * 0.7

            if direction < 0:
                left_total += base + directional * abs(direction)
                right_total += base
            else:
                right_total += base + directional * abs(direction)
                left_total += base

        return min(1.0, left_total), min(1.0, right_total)

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
            # 속도 단계: (렌더 간격, fps캡) - 렌더 간격이 핵심 (GPU 연산이 병목)
            self._render_every = [1, 1, 3, 10, 100]
            self._speed_names = ["SLOW", "x1", "x3", "x10", "MAX"]
            self._speed_idx = 1  # 기본: x1
            self._paused = False

        import pygame

        # 이벤트 처리 (매 스텝 체크 - 키 입력 누락 방지)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT or event.key == pygame.K_UP:
                    self._speed_idx = min(self._speed_idx + 1, len(self._render_every) - 1)
                elif event.key == pygame.K_LEFT or event.key == pygame.K_DOWN:
                    self._speed_idx = max(self._speed_idx - 1, 0)
                elif event.key == pygame.K_SPACE:
                    self._paused = not self._paused

        # 일시정지
        while self._paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self._paused = False
            self.clock.tick(10)

        # 렌더링 건너뛰기 (x3=매 3스텝, x10=매 10스텝, MAX=매 100스텝마다만 그림)
        skip = self._render_every[self._speed_idx]
        if skip > 1 and self.steps % skip != 0:
            return

        # 배경
        self.screen.fill((40, 40, 40))

        # === 환경 영역 (400x400) ===
        env_surface = pygame.Surface((400, 400))
        env_surface.fill((25, 28, 25))

        # Phase L12: 배경 그리드 (미세한 공간 참조)
        for gx in range(0, 401, 50):
            pygame.draw.line(env_surface, (35, 38, 35), (gx, 0), (gx, 400), 1)
        for gy in range(0, 401, 50):
            pygame.draw.line(env_surface, (35, 38, 35), (0, gy), (400, gy), 1)

        # === Phase 2b+L12: Pain Zone (그래디언트 위험 필드) ===
        if self.config.pain_zone_enabled and self.pain_zones:
            r = int(self.config.pain_zone_radius)
            dr = int(self.config.danger_range)
            total_r = r + dr
            for pz_cx, pz_cy in self.pain_zones:
                # 위험 필드 그래디언트 (danger_range 영역 — 바깥→안쪽 fade-in)
                n_rings = 5
                for ring_i in range(n_rings):
                    ring_r = total_r - ring_i * (dr // n_rings)
                    ring_alpha = 15 + ring_i * 12  # 15→75 (바깥→안쪽)
                    ring_surf = pygame.Surface((ring_r * 2, ring_r * 2), pygame.SRCALPHA)
                    pygame.draw.circle(ring_surf, (180, 40, 20, ring_alpha),
                                       (ring_r, ring_r), ring_r)
                    env_surface.blit(ring_surf, (int(pz_cx) - ring_r, int(pz_cy) - ring_r))
                # 실제 Pain Zone (불투명 코어)
                pain_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(pain_surf, (140, 25, 25, 130), (r, r), r)
                env_surface.blit(pain_surf, (int(pz_cx) - r, int(pz_cy) - r))
                # 경계선 (밝은 빨강)
                pygame.draw.circle(env_surface, (220, 60, 60),
                                  (int(pz_cx), int(pz_cy)), r, 2)
                # 위험 필드 경계 (점선 효과 — 대시 원)
                dash_r = total_r
                for deg in range(0, 360, 10):
                    if deg % 20 < 10:  # 10도 그리고 10도 건너뛰기
                        rad = math.radians(deg)
                        dx = int(pz_cx + dash_r * math.cos(rad))
                        dy = int(pz_cy + dash_r * math.sin(rad))
                        dx2 = int(pz_cx + dash_r * math.cos(rad + math.radians(8)))
                        dy2 = int(pz_cy + dash_r * math.sin(rad + math.radians(8)))
                        pygame.draw.line(env_surface, (120, 40, 40), (dx, dy), (dx2, dy2), 1)

        # === E1: Obstacles (정적 장애물) ===
        if self.config.obstacles_enabled and self.obstacles:
            for ox, oy, ow, oh in self.obstacles:
                # 본체 (어두운 회색)
                pygame.draw.rect(env_surface, (60, 65, 60),
                                 (int(ox), int(oy), int(ow), int(oh)))
                # 테두리 (밝은 회색)
                pygame.draw.rect(env_surface, (80, 85, 80),
                                 (int(ox), int(oy), int(ow), int(oh)), 2)

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

        # 음식 (Phase L5+L12: 타입별 색상 + 글로우 + 위험 근접 표시)
        for food_x, food_y, food_type in self.foods:
            fx, fy = int(food_x), int(food_y)
            fr = int(self.config.food_radius)
            if food_type == 1:  # 나쁜 음식
                food_color = (180, 80, 200)  # 보라
            else:  # 좋은 음식 (기본)
                food_color = (100, 220, 80)  # 초록

            # Phase L12: 음식 글로우 (반투명 큰 원)
            glow_r = fr + 4
            glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
            glow_color = (food_color[0], food_color[1], food_color[2], 40)
            pygame.draw.circle(glow_surf, glow_color, (glow_r, glow_r), glow_r)
            env_surface.blit(glow_surf, (fx - glow_r, fy - glow_r))

            # 본체
            pygame.draw.circle(env_surface, food_color, (fx, fy), fr)

            # Phase L12: 위험 근접 음식 표시 (주황 테두리)
            if self.config.danger_food_enabled:
                for pz_x, pz_y in self.pain_zones:
                    dist_sq = (food_x - pz_x)**2 + (food_y - pz_y)**2
                    threshold = self.config.pain_zone_radius + self.config.danger_range
                    if dist_sq < threshold**2:
                        pygame.draw.circle(env_surface, (255, 180, 50), (fx, fy), fr + 2, 2)
                        break

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

        # === Predators ===
        if self.config.predator_enabled and hasattr(self, 'predators'):
            for pred in self.predators:
                color = (220, 40, 40) if pred.state == "chase" else (160, 60, 60)
                pygame.draw.circle(env_surface, color,
                                   (int(pred.x), int(pred.y)), int(pred.radius))
                pygame.draw.circle(env_surface, (255, 80, 80),
                                   (int(pred.x), int(pred.y)), int(pred.radius), 2)
                # 방향 화살표
                arr_len = pred.radius * 1.5
                ax = pred.x + math.cos(pred.angle) * arr_len
                ay = pred.y + math.sin(pred.angle) * arr_len
                pygame.draw.line(env_surface, (255, 150, 150),
                                 (int(pred.x), int(pred.y)), (int(ax), int(ay)), 2)
                # 추적선 (추적 중일 때만)
                if pred.state == "chase":
                    pygame.draw.line(env_surface, (255, 80, 80),
                                     (int(pred.x), int(pred.y)),
                                     (int(self.agent_x), int(self.agent_y)), 1)

        # Phase L12: 궤적 트레일 (GW 상태별 색상 — 선분 + 모드 전환 마커)
        if len(self.position_history) > 2:
            trail_len = min(len(self.position_history), 400)
            trail_start = max(0, len(self.position_history) - trail_len)
            prev_gw = None
            for i in range(trail_start, len(self.position_history) - 1):
                pos = self.position_history[i]
                next_pos = self.position_history[i + 1]
                px, py = int(pos[0]), int(pos[1])
                nx, ny = int(next_pos[0]), int(next_pos[1])
                gw = pos[2] if len(pos) > 2 else "neutral"

                # 알파 (오래될수록 투명)
                age_ratio = (i - trail_start) / trail_len
                alpha = int(40 + 140 * age_ratio)
                thickness = 1 if age_ratio < 0.5 else 2

                # 색상 (GW 상태)
                if gw == "food":
                    color = (50, min(255, 140 + alpha), 50)   # 초록 (음식 탐색)
                elif gw == "safety":
                    color = (min(255, 140 + alpha), 50, 50)   # 빨강 (안전 모드)
                else:
                    color = (80, 80, min(255, 80 + alpha))    # 파랑 (탐색)

                # 선분으로 연결
                pygame.draw.line(env_surface, color, (px, py), (nx, ny), thickness)

                # 모드 전환 마커 (다이아몬드)
                if prev_gw is not None and gw != prev_gw:
                    marker_color = (255, 255, 100)  # 노란 다이아몬드
                    pts = [(px, py - 4), (px + 3, py), (px, py + 4), (px - 3, py)]
                    pygame.draw.polygon(env_surface, marker_color, pts)
                prev_gw = gw

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

        # Phase L12: 목표 상태 헤일로
        if hasattr(self, 'brain_info') and self.brain_info:
            gw = self.brain_info.get("gw_broadcast", "neutral")
            if gw == "food":
                halo_color = (80, 255, 80)    # 초록 헤일로
            elif gw == "safety":
                halo_color = (255, 80, 80)    # 빨강 헤일로
            else:
                halo_color = (120, 120, 200)  # 파란 헤일로
            pygame.draw.circle(env_surface, halo_color,
                               (int(self.agent_x), int(self.agent_y)),
                               int(self.config.agent_radius + 6), 2)

        # Phase L12: GW 모드 오버레이 (환경 좌상단)
        if hasattr(self, 'brain_info') and self.brain_info:
            gw = self.brain_info.get("gw_broadcast", "neutral")
            if gw == "food":
                mode_text = "SEEKING FOOD"
                mode_color = (80, 255, 80)
            elif gw == "safety":
                mode_text = "AVOIDING DANGER"
                mode_color = (255, 80, 80)
            else:
                mode_text = "EXPLORING"
                mode_color = (120, 160, 255)
            # 반투명 배경 박스
            mode_bg = pygame.Surface((140, 20), pygame.SRCALPHA)
            mode_bg.fill((0, 0, 0, 120))
            env_surface.blit(mode_bg, (5, 5))
            mode_surf = self.small_font.render(mode_text, True, mode_color)
            env_surface.blit(mode_surf, (10, 7))

        # Phase L12: 에피소드 진행 바 (환경 하단)
        progress = self.steps / self.config.max_steps
        bar_w = int(396 * progress)
        pygame.draw.rect(env_surface, (40, 45, 40), (2, 394, 396, 4))
        bar_color = (80, 200, 80) if self.energy > 30 else (200, 80, 80)
        pygame.draw.rect(env_surface, bar_color, (2, 394, bar_w, 4))

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

        # === 속도 조절 표시 ===
        speed_name = self._speed_names[self._speed_idx]
        speed_text = f"Speed: {speed_name}  [←/→] speed  [SPACE] pause"
        speed_color = (100, 200, 255) if speed_name == "MAX" else (150, 150, 150)
        text = self.small_font.render(speed_text, True, speed_color)
        self.screen.blit(text, (10, 635))

        # === 뇌 활성화 패널 (공간 배치형) ===
        self._render_brain_schematic(600, 0)

        pygame.display.flip()
        # SLOW 모드만 FPS 캡 (10fps), 나머지는 제한 없음
        if self._speed_idx == 0:
            self.clock.tick(10)

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

        # === 0. GLOBAL WORKSPACE (최상단 — Phase L12) ===
        gw_y = y + 30
        gw_food = info.get("gw_food_l_rate", 0) + info.get("gw_food_r_rate", 0)
        gw_safety = info.get("gw_safety_rate", 0)
        gw_bc = info.get("gw_broadcast", "neutral")

        # 배경 박스
        pygame.draw.rect(self.screen, (20, 25, 20), (x + 5, gw_y - 5, panel_width - 10, 50))
        pygame.draw.rect(self.screen, (80, 80, 100), (x + 5, gw_y - 5, panel_width - 10, 50), 1)

        # 타이틀
        gw_title = self.small_font.render("GLOBAL WORKSPACE", True, (220, 220, 255))
        self.screen.blit(gw_title, (x + 10, gw_y - 3))

        # Food Goal 바
        bar_y = gw_y + 14
        bar_w = panel_width - 80
        food_w = int(bar_w * min(1.0, gw_food))
        pygame.draw.rect(self.screen, (30, 50, 30), (x + 15, bar_y, bar_w, 10))
        pygame.draw.rect(self.screen, (80, 220, 80), (x + 15, bar_y, food_w, 10))
        food_lbl = self.small_font.render(f"Food {gw_food*100:.0f}%", True, (180, 255, 180))
        self.screen.blit(food_lbl, (x + 15 + bar_w + 2, bar_y - 1))

        # Safety Goal 바
        bar_y2 = gw_y + 28
        safety_w = int(bar_w * min(1.0, gw_safety))
        pygame.draw.rect(self.screen, (50, 30, 30), (x + 15, bar_y2, bar_w, 10))
        pygame.draw.rect(self.screen, (220, 80, 80), (x + 15, bar_y2, safety_w, 10))
        safety_lbl = self.small_font.render(f"Safe {gw_safety*100:.0f}%", True, (255, 180, 180))
        self.screen.blit(safety_lbl, (x + 15 + bar_w + 2, bar_y2 - 1))

        # === 1. PREFRONTAL CORTEX ===
        pfc_y = y + 85
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
        bg_y = y + 145
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
        limbic_y = y + 235
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
        hippo_y = y + 325
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
        cerebellum_y = y + 415
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
        thalamus_y = y + 465
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
        motor_y = y + 545
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
        summary_y = y + 625
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
        print(f"  Pain Zone: {config.pain_zone_count}x interior circles, radius={config.pain_zone_radius}, damage={config.pain_damage}")

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
