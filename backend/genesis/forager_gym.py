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

    # 음식 (Phase 2a 테스트용: 희소 음식으로 Hunger 테스트)
    n_food: int = 15          # 30 → 15 (음식 희소화)
    food_radius: float = 6.0  # 8 → 6 (작아서 찾기 어려움)
    food_value: float = 25.0  # 20 → 25 (찾으면 보상 증가)

    # 에너지 (항상성)
    energy_start: float = 50.0  # 기본 시작 에너지
    energy_max: float = 100.0
    energy_decay_field: float = 0.15   # 0.08 → 0.15 (더 빠른 감소)
    energy_decay_nest: float = 0.05    # 0.03 → 0.05

    # 감각
    n_rays: int = 16  # 음식/벽 감지 레이 수
    view_range: float = 120.0  # 시야 거리

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

    # 시뮬레이션
    max_steps: int = 3000


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
                self.foods.pop(i)
                self._spawn_foods(1)  # 새 음식 생성
                return True
        return False

    def _spawn_foods(self, n: int):
        """Field에 음식 생성 (Nest 외부)"""
        cx, cy = self.config.width / 2, self.config.height / 2
        half = self.config.nest_size / 2
        margin = self.config.food_radius * 2

        for _ in range(n):
            # Nest 외부에 생성 시도
            for _ in range(100):  # 최대 100번 시도
                x = np.random.uniform(margin, self.config.width - margin)
                y = np.random.uniform(margin, self.config.height - margin)

                # Nest 내부면 다시
                if not (cx - half <= x <= cx + half and
                       cy - half <= y <= cy + half):
                    self.foods.append((x, y))
                    break

    def render(self):
        """Pygame 렌더링"""
        if self.render_mode != "pygame":
            return

        # Lazy init
        if self.screen is None:
            import pygame
            pygame.init()
            self.screen = pygame.display.set_mode((600, 500))  # 환경 400x400 + 패널
            pygame.display.set_caption("ForagerGym - Phase 2b (Fear Conditioning)")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)

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

        pygame.display.flip()
        self.clock.tick(60)

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
