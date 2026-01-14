"""
Slither.io Gym Environment

A simplified Python environment for training SNN agents before
deploying to the real slither.io game.

Features:
- 800x600 arena
- Agent: snake that grows when eating food
- Food: randomly spawning dots
- Enemies: simple bot snakes (optional)
- Headless or pygame visualization

Phase 1: Scavenger - eat food, grow
Phase 2: Coward - avoid enemies
Phase 3: Parasite - steal kills
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


@dataclass
class SnakeSegment:
    """Single segment of a snake body"""
    x: float
    y: float


@dataclass
class Snake:
    """A snake in the arena"""
    segments: List[SnakeSegment] = field(default_factory=list)
    angle: float = 0.0  # Direction in radians
    speed: float = 3.0  # Base speed
    boosting: bool = False
    alive: bool = True
    color: Tuple[int, int, int] = (0, 255, 0)

    @property
    def head(self) -> SnakeSegment:
        return self.segments[0] if self.segments else None

    @property
    def length(self) -> int:
        return len(self.segments)

    def __post_init__(self):
        if not self.segments:
            # Initialize with 3 segments
            self.segments = [SnakeSegment(400, 300) for _ in range(3)]


@dataclass
class Food:
    """Food item in the arena"""
    x: float
    y: float
    value: int = 1  # How much length it adds
    color: Tuple[int, int, int] = (255, 255, 0)


@dataclass
class SlitherConfig:
    """Environment configuration"""
    width: int = 2000
    height: int = 2000

    # Food
    n_food: int = 500  # More food for bigger map
    food_spawn_rate: float = 0.1  # Per frame

    # Agent
    agent_speed: float = 3.0
    boost_speed: float = 6.0
    boost_cost: float = 0.5  # Segments lost per second

    # Enemies
    n_enemies: int = 0  # Start with 0 for Phase 1
    enemy_speed: float = 2.5

    # Collision
    head_radius: float = 10.0
    food_radius: float = 5.0
    segment_spacing: float = 8.0

    # Rewards
    food_reward: float = 1.0
    death_penalty: float = -10.0
    survival_reward: float = 0.01


class SlitherGym:
    """
    Slither.io training environment

    Observation space: Local view around agent head
    Action space: (angle_delta, boost)
    """

    def __init__(self, config: Optional[SlitherConfig] = None, render_mode: str = "none"):
        self.config = config or SlitherConfig()
        self.render_mode = render_mode  # "none", "pygame", "ascii"

        # State
        self.agent: Snake = None
        self.enemies: List[Snake] = []
        self.foods: List[Food] = []
        self.steps = 0
        self.score = 0

        # Pygame (lazy init)
        self.screen = None
        self.clock = None

        self.reset()

    def reset(self) -> dict:
        """Reset environment to initial state"""
        # Create agent at center
        self.agent = Snake(
            segments=[SnakeSegment(self.config.width/2, self.config.height/2)
                     for _ in range(5)],
            color=(0, 200, 0)
        )

        # Spawn food
        self.foods = []
        for _ in range(self.config.n_food):
            self._spawn_food()

        # Spawn enemies (Phase 2+)
        self.enemies = []
        for i in range(self.config.n_enemies):
            self._spawn_enemy(i)

        self.steps = 0
        self.score = 0

        return self._get_observation()

    def _spawn_food(self):
        """Spawn a food item at random location"""
        x = np.random.uniform(20, self.config.width - 20)
        y = np.random.uniform(20, self.config.height - 20)
        # Random color (bright)
        color = (
            np.random.randint(150, 255),
            np.random.randint(150, 255),
            np.random.randint(50, 150)
        )
        self.foods.append(Food(x, y, value=1, color=color))

    def _spawn_enemy(self, idx: int):
        """Spawn an enemy bot snake"""
        # Spawn away from center
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(150, 300)
        x = self.config.width/2 + dist * np.cos(angle)
        y = self.config.height/2 + dist * np.sin(angle)

        # Clamp to arena
        x = np.clip(x, 50, self.config.width - 50)
        y = np.clip(y, 50, self.config.height - 50)

        enemy = Snake(
            segments=[SnakeSegment(x, y) for _ in range(8)],
            angle=np.random.uniform(0, 2 * np.pi),
            speed=self.config.enemy_speed,
            color=(200, 50, 50)  # Red enemies
        )
        self.enemies.append(enemy)

    def step(self, action: Tuple[float, float, bool]) -> Tuple[dict, float, bool, dict]:
        """
        Take a step in the environment

        Args:
            action: (target_x, target_y, boost) - normalized 0-1 coordinates
                - target_x: Target X position (0-1, maps to screen width)
                - target_y: Target Y position (0-1, maps to screen height)
                - boost: Whether to boost (True/False)

            OR for backwards compatibility:
            action: (angle_delta, boost) - if only 2 elements

        Returns:
            observation, reward, done, info
        """
        # Handle both action formats
        if len(action) == 3:
            target_x, target_y, boost = action
            # Convert normalized coords to screen coords
            screen_x = target_x * self.config.width
            screen_y = target_y * self.config.height
            # Calculate angle to target
            head = self.agent.head
            dx = screen_x - head.x
            dy = screen_y - head.y
            target_angle = math.atan2(dy, dx)
            # Smoothly turn toward target (like real slither.io)
            angle_diff = (target_angle - self.agent.angle + math.pi) % (2 * math.pi) - math.pi
            angle_delta = np.clip(angle_diff * 0.3, -0.3, 0.3)  # Smooth turning
        else:
            angle_delta, boost = action

        # Update agent
        reward = self._update_agent(angle_delta, boost)

        # Update enemies
        for enemy in self.enemies:
            self._update_enemy(enemy)

        # Check collisions
        death_reward = self._check_collisions()
        reward += death_reward

        # Spawn new food
        if np.random.random() < self.config.food_spawn_rate:
            if len(self.foods) < self.config.n_food * 1.5:
                self._spawn_food()

        self.steps += 1
        self.score = self.agent.length

        # Check done
        done = not self.agent.alive

        # Survival reward
        if not done:
            reward += self.config.survival_reward

        obs = self._get_observation()
        info = {
            'length': self.agent.length,
            'steps': self.steps,
            'foods_eaten': self.score - 5,  # Started with 5
        }

        return obs, reward, done, info

    def _update_agent(self, angle_delta: float, boost: bool) -> float:
        """Update agent position and return reward"""
        reward = 0.0

        # Clamp angle delta
        angle_delta = np.clip(angle_delta, -0.3, 0.3)  # Max turn rate
        self.agent.angle += angle_delta

        # Determine speed
        speed = self.config.boost_speed if boost else self.config.agent_speed
        self.agent.boosting = boost

        # Boost cost (lose segments)
        if boost and self.agent.length > 3:
            # Lose segment occasionally when boosting
            if np.random.random() < self.config.boost_cost * 0.1:
                self.agent.segments.pop()

        # Move head
        head = self.agent.head
        new_x = head.x + speed * np.cos(self.agent.angle)
        new_y = head.y + speed * np.sin(self.agent.angle)

        # No wrap - wall collision handled in _check_collisions
        # Just clamp to prevent going too far out
        new_x = np.clip(new_x, 0, self.config.width)
        new_y = np.clip(new_y, 0, self.config.height)

        # Add new head position
        self.agent.segments.insert(0, SnakeSegment(new_x, new_y))

        # Check food collision
        head = self.agent.head
        foods_to_remove = []
        for food in self.foods:
            dist = math.sqrt((head.x - food.x)**2 + (head.y - food.y)**2)
            if dist < self.config.head_radius + self.config.food_radius:
                foods_to_remove.append(food)
                reward += self.config.food_reward * food.value
                # Grow by not removing tail

        # Remove eaten food
        for food in foods_to_remove:
            self.foods.remove(food)

        # Remove tail (if didn't eat)
        if not foods_to_remove:
            self.agent.segments.pop()

        return reward

    def _update_enemy(self, enemy: Snake):
        """Update enemy bot with simple AI"""
        if not enemy.alive:
            return

        # Simple behavior: wander + occasionally turn toward food
        if np.random.random() < 0.02:
            # Random turn
            enemy.angle += np.random.uniform(-0.5, 0.5)

        # Move toward nearest food occasionally
        if np.random.random() < 0.05 and self.foods:
            nearest = min(self.foods,
                         key=lambda f: (f.x - enemy.head.x)**2 + (f.y - enemy.head.y)**2)
            target_angle = math.atan2(nearest.y - enemy.head.y, nearest.x - enemy.head.x)
            enemy.angle += 0.1 * (target_angle - enemy.angle)

        # Move
        head = enemy.head
        new_x = head.x + enemy.speed * np.cos(enemy.angle)
        new_y = head.y + enemy.speed * np.sin(enemy.angle)

        # Bounce off walls
        margin = 30
        if new_x < margin or new_x > self.config.width - margin:
            enemy.angle = math.pi - enemy.angle  # Reflect horizontally
            new_x = np.clip(new_x, margin, self.config.width - margin)
        if new_y < margin or new_y > self.config.height - margin:
            enemy.angle = -enemy.angle  # Reflect vertically
            new_y = np.clip(new_y, margin, self.config.height - margin)

        enemy.segments.insert(0, SnakeSegment(new_x, new_y))

        # Check food
        foods_to_remove = []
        for food in self.foods:
            dist = math.sqrt((head.x - food.x)**2 + (head.y - food.y)**2)
            if dist < self.config.head_radius + self.config.food_radius:
                foods_to_remove.append(food)

        for food in foods_to_remove:
            self.foods.remove(food)

        if not foods_to_remove:
            enemy.segments.pop()

    def _check_collisions(self) -> float:
        """Check for fatal collisions, return penalty if dead"""
        head = self.agent.head

        # NO self-collision in real slither.io!
        # (Removed self-body collision check)

        # Collision with map boundary (wall death)
        margin = 10
        if (head.x < margin or head.x > self.config.width - margin or
            head.y < margin or head.y > self.config.height - margin):
            self.agent.alive = False
            return self.config.death_penalty

        # Collision with enemy bodies (head hits enemy body = death)
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            for seg in enemy.segments:
                dist = math.sqrt((head.x - seg.x)**2 + (head.y - seg.y)**2)
                if dist < self.config.head_radius + 5:
                    self.agent.alive = False
                    # In real slither.io, dead snake becomes food
                    self._spawn_death_food(self.agent)
                    return self.config.death_penalty

        return 0.0

    def _spawn_death_food(self, dead_snake: Snake):
        """Spawn food where snake died (like real slither.io)"""
        for seg in dead_snake.segments[::3]:  # Every 3rd segment
            self.foods.append(Food(
                seg.x + np.random.uniform(-5, 5),
                seg.y + np.random.uniform(-5, 5),
                value=2,  # Death food is more valuable
                color=(255, 100, 100)
            ))

    def _get_observation(self) -> dict:
        """
        Get observation for agent

        Returns local view around head:
        - Food positions (relative to head)
        - Enemy positions (relative to head)
        - Own body positions
        - Current heading
        """
        if not self.agent.alive:
            return {'food': [], 'enemies': [], 'body': [], 'heading': 0, 'length': 0}

        head = self.agent.head
        view_range = 150  # Pixels

        # Nearby food (relative coords)
        nearby_food = []
        for food in self.foods:
            dx = food.x - head.x
            dy = food.y - head.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < view_range:
                angle = math.atan2(dy, dx) - self.agent.angle  # Relative to heading
                nearby_food.append({
                    'dist': dist / view_range,  # Normalized
                    'angle': angle,
                    'value': food.value
                })

        # Nearby enemies
        nearby_enemies = []
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            for seg in enemy.segments:
                dx = seg.x - head.x
                dy = seg.y - head.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < view_range:
                    angle = math.atan2(dy, dx) - self.agent.angle
                    nearby_enemies.append({
                        'dist': dist / view_range,
                        'angle': angle
                    })

        # Own body (for self-collision avoidance)
        own_body = []
        for seg in self.agent.segments[5:]:  # Skip first segments
            dx = seg.x - head.x
            dy = seg.y - head.y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < view_range:
                angle = math.atan2(dy, dx) - self.agent.angle
                own_body.append({
                    'dist': dist / view_range,
                    'angle': angle
                })

        return {
            'food': nearby_food[:20],  # Limit for efficiency
            'enemies': nearby_enemies[:30],
            'body': own_body[:10],
            'heading': self.agent.angle,
            'length': self.agent.length,
            'position': (head.x / self.config.width, head.y / self.config.height)
        }

    def get_sensor_input(self, n_rays: int = 16) -> np.ndarray:
        """
        Get ray-cast sensor input for SNN

        Casts rays in all directions and returns:
        - Distance to nearest food (per ray)
        - Distance to nearest enemy (per ray)
        - Distance to nearest body segment (per ray)

        Returns: (3, n_rays) array normalized 0-1
        """
        if not self.agent.alive:
            return np.zeros((3, n_rays))

        head = self.agent.head
        view_range = 150.0

        food_rays = np.ones(n_rays)  # 1 = nothing, 0 = close
        enemy_rays = np.ones(n_rays)
        body_rays = np.ones(n_rays)

        for i in range(n_rays):
            ray_angle = self.agent.angle + (2 * np.pi * i / n_rays) - np.pi

            # Check food
            for food in self.foods:
                dx = food.x - head.x
                dy = food.y - head.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < view_range:
                    food_angle = math.atan2(dy, dx)
                    angle_diff = abs((food_angle - ray_angle + np.pi) % (2*np.pi) - np.pi)
                    if angle_diff < np.pi / n_rays:  # Within ray cone
                        food_rays[i] = min(food_rays[i], dist / view_range)

            # Check enemies
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                for seg in enemy.segments:
                    dx = seg.x - head.x
                    dy = seg.y - head.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < view_range:
                        seg_angle = math.atan2(dy, dx)
                        angle_diff = abs((seg_angle - ray_angle + np.pi) % (2*np.pi) - np.pi)
                        if angle_diff < np.pi / n_rays:
                            enemy_rays[i] = min(enemy_rays[i], dist / view_range)

            # Check own body
            for seg in self.agent.segments[10:]:
                dx = seg.x - head.x
                dy = seg.y - head.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < view_range:
                    seg_angle = math.atan2(dy, dx)
                    angle_diff = abs((seg_angle - ray_angle + np.pi) % (2*np.pi) - np.pi)
                    if angle_diff < np.pi / n_rays:
                        body_rays[i] = min(body_rays[i], dist / view_range)

        # Invert so closer = higher signal
        return np.stack([1 - food_rays, 1 - enemy_rays, 1 - body_rays])

    def render(self):
        """Render the environment"""
        if self.render_mode == "none":
            return

        if self.render_mode == "pygame":
            self._render_pygame()
        elif self.render_mode == "ascii":
            self._render_ascii()

    def _render_pygame(self):
        """Render using pygame"""
        try:
            import pygame
        except ImportError:
            print("pygame not installed, falling back to ascii")
            self.render_mode = "ascii"
            return self._render_ascii()

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.config.width, self.config.height))
            pygame.display.set_caption("Slither Gym")
            self.clock = pygame.time.Clock()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Clear screen
        self.screen.fill((20, 20, 30))  # Dark background

        # Draw food
        for food in self.foods:
            pygame.draw.circle(self.screen, food.color, (int(food.x), int(food.y)),
                             int(self.config.food_radius))

        # Draw enemies
        for enemy in self.enemies:
            if enemy.alive:
                for seg in enemy.segments:
                    pygame.draw.circle(self.screen, enemy.color,
                                      (int(seg.x), int(seg.y)), 8)

        # Draw agent
        if self.agent.alive:
            for i, seg in enumerate(self.agent.segments):
                # Gradient from head to tail
                brightness = max(100, 200 - i * 5)
                color = (0, brightness, 0)
                radius = max(5, 10 - i * 0.3)
                pygame.draw.circle(self.screen, color,
                                  (int(seg.x), int(seg.y)), int(radius))

            # Draw direction indicator
            head = self.agent.head
            end_x = head.x + 20 * np.cos(self.agent.angle)
            end_y = head.y + 20 * np.sin(self.agent.angle)
            pygame.draw.line(self.screen, (255, 255, 255),
                           (int(head.x), int(head.y)), (int(end_x), int(end_y)), 2)

        # Draw HUD
        font = pygame.font.Font(None, 36)
        text = font.render(f"Length: {self.agent.length}  Steps: {self.steps}",
                          True, (255, 255, 255))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def _render_ascii(self):
        """Simple ASCII rendering"""
        if self.steps % 10 != 0:
            return
        print(f"\rStep: {self.steps:5d} | Length: {self.agent.length:3d} | "
              f"Foods: {len(self.foods):3d} | Alive: {self.agent.alive}", end="")

    def close(self):
        """Clean up"""
        if self.screen is not None:
            import pygame
            pygame.quit()


# Demo / Test
if __name__ == "__main__":
    print("Slither Gym - Demo Mode")
    print("Control with MOUSE (like real slither.io)")
    print("Hold SPACE or CLICK for boost")
    print()

    # Create environment with pygame
    env = SlitherGym(render_mode="pygame")
    obs = env.reset()

    try:
        import pygame
        pygame.init()

        # Viewport settings (camera follows player)
        VIEWPORT_W, VIEWPORT_H = 800, 600
        screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        pygame.display.set_caption("Slither Gym - Mouse Control")
        clock = pygame.time.Clock()
        env.screen = screen  # Override env screen

        done = False
        total_reward = 0

        while not done:
            # Get mouse position relative to viewport center
            mouse_x, mouse_y = pygame.mouse.get_pos()

            # Convert to world coordinates (camera centered on player)
            head = env.agent.head
            cam_x = head.x - VIEWPORT_W / 2
            cam_y = head.y - VIEWPORT_H / 2

            world_mouse_x = cam_x + mouse_x
            world_mouse_y = cam_y + mouse_y

            # Normalize to 0-1 for action
            target_x = world_mouse_x / env.config.width
            target_y = world_mouse_y / env.config.height

            # Boost on mouse click or space
            mouse_buttons = pygame.mouse.get_pressed()
            keys = pygame.key.get_pressed()
            boost = mouse_buttons[0] or keys[pygame.K_SPACE]

            # Step with mouse target
            obs, reward, done, info = env.step((target_x, target_y, boost))
            total_reward += reward

            # Render with camera offset
            screen.fill((20, 20, 30))

            # Draw food (offset by camera)
            for food in env.foods:
                fx = food.x - cam_x
                fy = food.y - cam_y
                if -10 < fx < VIEWPORT_W + 10 and -10 < fy < VIEWPORT_H + 10:
                    pygame.draw.circle(screen, food.color, (int(fx), int(fy)), 5)

            # Draw enemies
            for enemy in env.enemies:
                if enemy.alive:
                    for seg in enemy.segments:
                        sx = seg.x - cam_x
                        sy = seg.y - cam_y
                        if -10 < sx < VIEWPORT_W + 10 and -10 < sy < VIEWPORT_H + 10:
                            pygame.draw.circle(screen, enemy.color, (int(sx), int(sy)), 8)

            # Draw agent
            if env.agent.alive:
                for i, seg in enumerate(env.agent.segments):
                    sx = seg.x - cam_x
                    sy = seg.y - cam_y
                    brightness = max(100, 200 - i * 3)
                    color = (0, brightness, 0)
                    radius = max(5, 10 - i * 0.2)
                    pygame.draw.circle(screen, color, (int(sx), int(sy)), int(radius))

                # Direction line to mouse
                pygame.draw.line(screen, (100, 100, 100),
                               (VIEWPORT_W // 2, VIEWPORT_H // 2),
                               (mouse_x, mouse_y), 1)

            # Draw wall boundary indicator
            # Left wall
            if cam_x < 50:
                pygame.draw.rect(screen, (255, 0, 0), (0, 0, 5, VIEWPORT_H))
            # Right wall
            if cam_x + VIEWPORT_W > env.config.width - 50:
                pygame.draw.rect(screen, (255, 0, 0), (VIEWPORT_W - 5, 0, 5, VIEWPORT_H))
            # Top wall
            if cam_y < 50:
                pygame.draw.rect(screen, (255, 0, 0), (0, 0, VIEWPORT_W, 5))
            # Bottom wall
            if cam_y + VIEWPORT_H > env.config.height - 50:
                pygame.draw.rect(screen, (255, 0, 0), (0, VIEWPORT_H - 5, VIEWPORT_W, 5))

            # HUD
            font = pygame.font.Font(None, 36)
            text = font.render(f"Length: {env.agent.length}  Steps: {env.steps}", True, (255, 255, 255))
            screen.blit(text, (10, 10))

            boost_text = "BOOST!" if boost else ""
            if boost_text:
                bt = font.render(boost_text, True, (255, 255, 0))
                screen.blit(bt, (10, 50))

            pygame.display.flip()
            clock.tick(60)

            # Check quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    done = True

        print(f"\n\nGame Over!")
        print(f"Final Length: {info['length']}")
        print(f"Steps Survived: {info['steps']}")
        print(f"Total Reward: {total_reward:.1f}")

    except ImportError:
        print("pygame not installed. Running headless demo...")

        for _ in range(1000):
            # Random target
            target_x = np.random.uniform(0.3, 0.7)
            target_y = np.random.uniform(0.3, 0.7)
            boost = np.random.random() < 0.1

            obs, reward, done, info = env.step((target_x, target_y, boost))
            env.render()

            if done:
                break

        print(f"\n\nDemo finished. Length: {info['length']}, Steps: {info['steps']}")

    env.close()
