"""
Game Agent - SNN 뇌로 게임 규칙 자율 학습
==========================================

FEP 없음. 생물학적 메커니즘만 사용:
- STDP (시냅스 가소성)
- 도파민 시스템 (보상/novelty 기반)
- 습관화 (반복 자극 감쇠)

게임 규칙을 직접 알려주지 않음 - 뇌가 스스로 발견해야 함
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from enum import IntEnum

from snn_brain_biological import BiologicalBrain, BiologicalConfig, DEVICE


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class SnakeGame:
    """
    간단한 뱀 게임

    규칙 (에이전트는 모름):
    - 뱀이 음식을 먹으면 길어짐
    - 벽이나 자기 몸에 부딪히면 게임 오버
    - 목표: 최대한 오래 생존하며 음식 먹기
    """

    def __init__(self, grid_size: int = 10):
        self.grid_size = grid_size
        self.reset()

    def reset(self) -> np.ndarray:
        """게임 리셋"""
        # 뱀 초기 위치 (중앙)
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = Direction.RIGHT

        # 음식 생성
        self._spawn_food()

        # 상태
        self.score = 0
        self.steps = 0
        self.done = False

        return self._get_observation()

    def _spawn_food(self):
        """빈 공간에 음식 생성"""
        empty = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in self.snake:
                    empty.append((x, y))

        if empty:
            self.food = empty[np.random.randint(len(empty))]
        else:
            self.food = None  # 꽉 참

    def _get_observation(self) -> np.ndarray:
        """
        시각적 관측 (grid_size x grid_size)

        값:
        - 0.0: 빈 공간
        - 0.3: 뱀 몸통
        - 0.6: 뱀 머리
        - 1.0: 음식
        """
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # 뱀 몸통
        for segment in self.snake[1:]:
            if 0 <= segment[0] < self.grid_size and 0 <= segment[1] < self.grid_size:
                obs[segment[1], segment[0]] = 0.3

        # 뱀 머리
        head = self.snake[0]
        if 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size:
            obs[head[1], head[0]] = 0.6

        # 음식
        if self.food:
            obs[self.food[1], self.food[0]] = 1.0

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, dict]:
        """
        행동 실행

        action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT

        Returns:
            obs: 새 관측
            info: {
                'ate_food': bool,
                'died': bool,
                'score': int,
                'steps': int
            }
        """
        if self.done:
            return self._get_observation(), {
                'ate_food': False,
                'died': True,
                'score': self.score,
                'steps': self.steps
            }

        self.steps += 1

        # 방향 변경 (180도 회전 방지)
        opposite = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }

        if action != opposite.get(self.direction, -1):
            self.direction = Direction(action)

        # 새 머리 위치 계산
        head = self.snake[0]
        if self.direction == Direction.UP:
            new_head = (head[0], head[1] - 1)
        elif self.direction == Direction.RIGHT:
            new_head = (head[0] + 1, head[1])
        elif self.direction == Direction.DOWN:
            new_head = (head[0], head[1] + 1)
        else:  # LEFT
            new_head = (head[0] - 1, head[1])

        # 충돌 체크
        ate_food = False
        died = False

        # 벽 충돌
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            died = True
            self.done = True
        # 자기 몸 충돌
        elif new_head in self.snake:
            died = True
            self.done = True
        else:
            # 이동
            self.snake.insert(0, new_head)

            # 음식 먹기
            if new_head == self.food:
                ate_food = True
                self.score += 1
                self._spawn_food()
            else:
                # 꼬리 제거 (음식 안 먹으면)
                self.snake.pop()

        info = {
            'ate_food': ate_food,
            'died': died,
            'score': self.score,
            'steps': self.steps,
            'snake_length': len(self.snake)
        }

        return self._get_observation(), info


class PongGame:
    """
    간단한 퐁 게임 (1인용)

    규칙 (에이전트는 모름):
    - 패들로 공을 튕겨야 함
    - 공을 놓치면 게임 오버
    """

    def __init__(self, width: int = 20, height: int = 15):
        self.width = width
        self.height = height
        self.paddle_size = 3
        self.reset()

    def reset(self) -> np.ndarray:
        # 패들 (하단 중앙)
        self.paddle_x = self.width // 2

        # 공
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = np.random.choice([-1, 1])
        self.ball_dy = -1  # 위로 시작

        self.score = 0
        self.steps = 0
        self.done = False

        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((self.height, self.width), dtype=np.float32)

        # 패들
        for i in range(self.paddle_size):
            px = self.paddle_x - self.paddle_size // 2 + i
            if 0 <= px < self.width:
                obs[self.height - 1, px] = 0.5

        # 공
        if 0 <= self.ball_x < self.width and 0 <= self.ball_y < self.height:
            obs[int(self.ball_y), int(self.ball_x)] = 1.0

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, dict]:
        """
        action: 0=LEFT, 1=STAY, 2=RIGHT
        """
        if self.done:
            return self._get_observation(), {
                'hit': False, 'missed': True, 'score': self.score, 'steps': self.steps
            }

        self.steps += 1

        # 패들 이동
        if action == 0:  # LEFT
            self.paddle_x = max(self.paddle_size // 2, self.paddle_x - 1)
        elif action == 2:  # RIGHT
            self.paddle_x = min(self.width - 1 - self.paddle_size // 2, self.paddle_x + 1)

        # 공 이동
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        hit = False
        missed = False

        # 좌우 벽 반사
        if self.ball_x <= 0 or self.ball_x >= self.width - 1:
            self.ball_dx *= -1
            self.ball_x = np.clip(self.ball_x, 0, self.width - 1)

        # 상단 벽 반사
        if self.ball_y <= 0:
            self.ball_dy = 1
            self.ball_y = 0

        # 하단 - 패들 체크
        if self.ball_y >= self.height - 1:
            paddle_left = self.paddle_x - self.paddle_size // 2
            paddle_right = self.paddle_x + self.paddle_size // 2

            if paddle_left <= self.ball_x <= paddle_right:
                # 패들에 맞음
                self.ball_dy = -1
                self.ball_y = self.height - 2
                self.score += 1
                hit = True
            else:
                # 놓침
                missed = True
                self.done = True

        info = {
            'hit': hit,
            'missed': missed,
            'score': self.score,
            'steps': self.steps
        }

        return self._get_observation(), info


class GameDopamineSystem:
    """
    게임용 도파민 시스템

    Browser 버전과 유사하지만 게임 이벤트에 최적화:
    - 음식/히트: 큰 도파민 분비
    - 죽음: 도파민 급감 + 학습 신호
    - 가까워짐: 약한 도파민
    """

    def __init__(self):
        self.dopamine_level = 0.0
        self.dopamine_history = deque(maxlen=100)

        # 예측 오차 기반 학습
        self.prediction_error = 0.0

        # 상태 추적
        self.last_distance_to_food = None
        self.consecutive_closer = 0
        self.consecutive_farther = 0

    def process_game_event(self, info: dict, obs: np.ndarray) -> float:
        """
        게임 이벤트 처리 → 도파민 분비

        Returns:
            dopamine_level: 현재 도파민 수준
        """
        dopamine_burst = 0.0

        # === Snake 이벤트 ===
        if 'ate_food' in info:
            if info['ate_food']:
                # 음식 획득 - 큰 보상
                dopamine_burst = 2.0
                self.consecutive_closer = 0
            elif info.get('died', False):
                # 죽음 - 도파민 급감
                dopamine_burst = -1.5
                self.prediction_error = 1.0  # 높은 학습 신호
            else:
                # 일반 이동 - 음식과의 거리 변화로 판단
                dopamine_burst = self._compute_approach_reward(obs)

        # === Pong 이벤트 ===
        if 'hit' in info:
            if info['hit']:
                dopamine_burst = 1.5
            elif info.get('missed', False):
                dopamine_burst = -1.5
                self.prediction_error = 1.0

        # 도파민 업데이트 (감쇠 포함)
        self.dopamine_level = self.dopamine_level * 0.8 + dopamine_burst
        self.dopamine_level = np.clip(self.dopamine_level, -2.0, 2.0)

        self.dopamine_history.append(self.dopamine_level)

        # 예측 오차 감쇠
        self.prediction_error *= 0.9

        return self.dopamine_level

    def _compute_approach_reward(self, obs: np.ndarray) -> float:
        """음식에 가까워지면 약한 도파민"""
        # 음식과 머리 위치 찾기
        food_pos = np.where(obs == 1.0)
        head_pos = np.where(obs == 0.6)

        if len(food_pos[0]) == 0 or len(head_pos[0]) == 0:
            return 0.0

        food_y, food_x = food_pos[0][0], food_pos[1][0]
        head_y, head_x = head_pos[0][0], head_pos[1][0]

        distance = abs(food_x - head_x) + abs(food_y - head_y)

        reward = 0.0
        if self.last_distance_to_food is not None:
            if distance < self.last_distance_to_food:
                self.consecutive_closer += 1
                self.consecutive_farther = 0
                reward = 0.1 * min(self.consecutive_closer, 3)
            elif distance > self.last_distance_to_food:
                self.consecutive_farther += 1
                self.consecutive_closer = 0
                reward = -0.05 * min(self.consecutive_farther, 3)

        self.last_distance_to_food = distance
        return reward

    def get_learning_rate_modifier(self) -> float:
        """도파민에 따른 학습률 조절"""
        # 높은 도파민 또는 높은 예측 오차 → 더 많이 학습
        base = 1.0
        dopamine_mod = 1.0 + self.dopamine_level * 0.3
        error_mod = 1.0 + self.prediction_error * 0.5

        return base * dopamine_mod * error_mod

    def get_exploration_pressure(self) -> float:
        """도파민 낮으면 탐색 압력 증가"""
        if len(self.dopamine_history) < 5:
            return 0.0

        recent = list(self.dopamine_history)[-5:]
        avg = np.mean(recent)

        if avg < 0.0:
            return min(1.0, abs(avg))
        return 0.0


class GameSNNAgent:
    """
    SNN 뇌로 게임 학습

    FEP 없음 - STDP + 도파민만 사용
    """

    def __init__(self, obs_shape: Tuple[int, int], n_actions: int):
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        # 관측 크기에 맞게 뇌 구성
        obs_size = obs_shape[0] * obs_shape[1]

        config = BiologicalConfig(
            visual_v1=max(2000, obs_size * 10),
            visual_v2=1000,
            auditory_a1=500,
            temporal=1000,
            parietal=500,
            prefrontal=1000,
            hippocampus=500,
            motor=n_actions * 100,
            intra_region_sparsity=0.02,
            inter_region_sparsity=0.01,
        )

        self.brain = BiologicalBrain(config)
        self.dopamine = GameDopamineSystem()

        # 행동 기록
        self.action_history = deque(maxlen=20)
        self.recent_actions_per_state = {}  # 상태별 행동 기록

    def _resize_observation(self, obs: np.ndarray, target_size: int = 64) -> np.ndarray:
        """관측을 target_size x target_size로 리사이즈"""
        h, w = obs.shape
        result = np.zeros((target_size, target_size), dtype=np.float32)

        # 간단한 nearest-neighbor 업스케일
        scale_h = target_size / h
        scale_w = target_size / w

        for i in range(target_size):
            for j in range(target_size):
                src_i = min(int(i / scale_h), h - 1)
                src_j = min(int(j / scale_w), w - 1)
                result[i, j] = obs[src_i, src_j]

        return result

    def act(self, obs: np.ndarray, info: dict, n_steps: int = 5) -> int:
        """
        관측 → 행동 선택
        """
        # 도파민 처리
        dopamine = self.dopamine.process_game_event(info, obs)
        exploration = self.dopamine.get_exploration_pressure()

        # 관측을 64x64로 리사이즈 (BiologicalBrain 요구사항)
        obs_resized = self._resize_observation(obs, target_size=64)
        visual_tensor = torch.tensor(obs_resized, dtype=torch.float32)

        # 뇌 시뮬레이션 (도파민 전달)
        motor_accumulator = torch.zeros(self.brain.config.motor, device=DEVICE)

        for _ in range(n_steps):
            # 도파민을 뇌의 STDP 학습에 전달 (3-factor rule)
            spikes = self.brain.step(visual_input=visual_tensor, learn=True, dopamine=dopamine)
            motor_accumulator += spikes['motor']

        # 행동 선택
        action = self._decode_action(motor_accumulator, dopamine, exploration)

        self.action_history.append(action)

        return action

    def _decode_action(self, motor_spikes: torch.Tensor,
                       dopamine: float, exploration: float) -> int:
        """운동 피질 → 행동"""
        motor_np = motor_spikes.cpu().numpy()
        group_size = len(motor_np) // self.n_actions

        # 각 행동의 활성화
        activities = []
        for i in range(self.n_actions):
            start = i * group_size
            end = start + group_size
            activities.append(motor_np[start:end].sum())

        activities = np.array(activities, dtype=np.float64)

        # 도파민 낮을 때 반복 행동 억제
        if dopamine < 0.0 and len(self.action_history) >= 3:
            recent = list(self.action_history)[-3:]
            for act in set(recent):
                count = recent.count(act)
                if count >= 2 and act < len(activities):
                    activities[act] *= 0.3 ** count

        # 탐색 압력
        if exploration > 0.3:
            recent_set = set(list(self.action_history)[-5:]) if len(self.action_history) >= 5 else set()
            for i in range(self.n_actions):
                if i not in recent_set:
                    activities[i] += exploration * 20

        # 확률적 선택
        temperature = 1.0 - dopamine * 0.3
        temperature = max(0.3, min(1.5, temperature))

        if activities.sum() > 0:
            activities = activities - activities.max()
            probs = np.exp(activities / (temperature * 10 + 1e-6))
            probs = probs / probs.sum()
            action = np.random.choice(self.n_actions, p=probs)
        else:
            action = np.random.randint(self.n_actions)

        return int(action)

    def reset(self):
        """에피소드 리셋"""
        self.brain.reset()
        self.dopamine = GameDopamineSystem()
        self.action_history.clear()


def run_snake_experiment(n_episodes: int = 10, max_steps: int = 200):
    """Snake 게임 실험"""
    print("=" * 60)
    print("Snake Game - SNN Brain Learning")
    print("=" * 60)
    print("\nNo FEP, No explicit reward function")
    print("Learning: STDP + Dopamine only")
    print("Goal: Discover game rules autonomously")

    game = SnakeGame(grid_size=10)
    agent = GameSNNAgent(obs_shape=(10, 10), n_actions=4)

    scores = []

    for ep in range(n_episodes):
        obs = game.reset()
        agent.reset()

        info = {'ate_food': False, 'died': False, 'score': 0, 'steps': 0}

        for step in range(max_steps):
            action = agent.act(obs, info)
            obs, info = game.step(action)

            if info['died']:
                break

        scores.append(info['score'])

        print(f"Episode {ep+1:3d}: Score={info['score']:2d}, "
              f"Steps={info['steps']:3d}, Length={info.get('snake_length', 1)}")

    print(f"\n{'='*40}")
    print(f"Results:")
    print(f"  Average Score: {np.mean(scores):.2f}")
    print(f"  Max Score: {max(scores)}")
    print(f"  Score Trend: {np.mean(scores[:3]):.2f} → {np.mean(scores[-3:]):.2f}")

    return scores


def run_pong_experiment(n_episodes: int = 10, max_steps: int = 500):
    """Pong 게임 실험"""
    print("\n" + "=" * 60)
    print("Pong Game - SNN Brain Learning")
    print("=" * 60)

    game = PongGame(width=20, height=15)
    agent = GameSNNAgent(obs_shape=(15, 20), n_actions=3)

    scores = []

    for ep in range(n_episodes):
        obs = game.reset()
        agent.reset()

        info = {'hit': False, 'missed': False, 'score': 0, 'steps': 0}

        for step in range(max_steps):
            action = agent.act(obs, info)
            obs, info = game.step(action)

            if info['missed']:
                break

        scores.append(info['score'])

        print(f"Episode {ep+1:3d}: Score={info['score']:2d}, Steps={info['steps']:3d}")

    print(f"\n{'='*40}")
    print(f"Results:")
    print(f"  Average Score: {np.mean(scores):.2f}")
    print(f"  Max Score: {max(scores)}")
    print(f"  Score Trend: {np.mean(scores[:3]):.2f} → {np.mean(scores[-3:]):.2f}")

    return scores


if __name__ == "__main__":
    snake_scores = run_snake_experiment(n_episodes=100, max_steps=200)
    # pong_scores = run_pong_experiment(n_episodes=100, max_steps=500)
    pong_scores = []  # Skip for now

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("\nKey Question: Did the brain discover game rules?")
    print(f"  Snake improvement: {np.mean(snake_scores[-10:]) - np.mean(snake_scores[:10]):+.2f}")

    # 10구간별 평균 출력
    print("\nSnake Score by Segment:")
    for i in range(0, len(snake_scores), 10):
        segment = snake_scores[i:i+10]
        if segment:
            print(f"  Episodes {i+1:3d}-{i+len(segment):3d}: avg={np.mean(segment):.2f}, max={max(segment)}")
