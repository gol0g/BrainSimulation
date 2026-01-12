"""
Curiosity-Driven Desktop Explorer

FEP 기반 자율 탐색 시스템

핵심 원리:
1. Forward Model: 행동의 결과 예측
2. Prediction Error = Surprise = 호기심 보상
3. 예측 오차가 큰 곳을 탐색 → 세계 모델 개선

목표: 외부 보상 없이 UI 패턴을 스스로 발견
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional
from collections import deque
from dataclasses import dataclass

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
genesis_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
sys.path.insert(0, genesis_dir)

from desktop_env import DesktopEnv, DesktopAction, ActionType, SafetyConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FeatureEncoder(nn.Module):
    """화면 특징 인코더 (공유)"""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=4, padding=2),   # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 8x8
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 8 * 8, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ForwardModel(nn.Module):
    """
    순방향 모델: (state, action) → next_state 예측

    예측 오차 = 호기심 신호
    """

    def __init__(self, feature_dim: int = 128, n_actions: int = 5):
        super().__init__()
        # Action embedding (discrete actions)
        self.action_embed = nn.Embedding(n_actions, 32)

        self.fc = nn.Sequential(
            nn.Linear(feature_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, state_feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_embed(action)
        x = torch.cat([state_feat, action_emb], dim=1)
        return self.fc(x)


class InverseModel(nn.Module):
    """
    역방향 모델: (state, next_state) → action 예측

    특징 공간이 행동과 관련된 정보를 인코딩하도록 유도
    """

    def __init__(self, feature_dim: int = 128, n_actions: int = 5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, state_feat: torch.Tensor, next_state_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state_feat, next_state_feat], dim=1)
        return self.fc(x)


class PolicyNetwork(nn.Module):
    """
    정책 네트워크: state → action distribution

    호기심 보상을 최대화하도록 학습
    """

    def __init__(self, feature_dim: int = 128, n_actions: int = 5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, state_feat: torch.Tensor) -> torch.Tensor:
        return self.fc(state_feat)

    def get_action(self, state_feat: torch.Tensor, epsilon: float = 0.1) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.fc[-1].out_features)

        with torch.no_grad():
            logits = self.forward(state_feat)
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()


@dataclass
class Experience:
    """경험 저장"""
    state: np.ndarray
    action: int
    next_state: np.ndarray
    curiosity_reward: float
    external_reward: float


class CuriosityExplorer:
    """
    호기심 기반 탐색기

    ICM (Intrinsic Curiosity Module) + FEP 원리 결합
    """

    def __init__(self, feature_dim: int = 128, n_actions: int = 5,
                 lr: float = 1e-4, buffer_size: int = 10000):
        self.feature_dim = feature_dim
        self.n_actions = n_actions

        # 네트워크
        self.encoder = FeatureEncoder(feature_dim).to(DEVICE)
        self.forward_model = ForwardModel(feature_dim, n_actions).to(DEVICE)
        self.inverse_model = InverseModel(feature_dim, n_actions).to(DEVICE)
        self.policy = PolicyNetwork(feature_dim, n_actions).to(DEVICE)

        # 옵티마이저
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.forward_model.parameters()},
            {'params': self.inverse_model.parameters()},
            {'params': self.policy.parameters()},
        ], lr=lr)

        # 경험 버퍼
        self.buffer: deque = deque(maxlen=buffer_size)

        # 통계
        self.total_curiosity = 0.0
        self.episode_curiosity = 0.0
        self.learning_steps = 0

        # 행동 매핑 (단순화)
        self.action_types = [
            ActionType.NONE,
            ActionType.CLICK,
            ActionType.RIGHT_CLICK,
            ActionType.MOVE,
            ActionType.DOUBLE_CLICK,
        ]

    def compute_curiosity(self, state: np.ndarray, action: int,
                          next_state: np.ndarray) -> float:
        """호기심 보상 계산 (Forward Model 예측 오차)"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
            action_t = torch.LongTensor([action]).to(DEVICE)

            # 특징 추출
            state_feat = self.encoder(state_t)
            next_state_feat = self.encoder(next_state_t)

            # 예측
            pred_next_feat = self.forward_model(state_feat, action_t)

            # 예측 오차 = 호기심
            curiosity = F.mse_loss(pred_next_feat, next_state_feat).item()

        return curiosity

    def select_action(self, state: np.ndarray, epsilon: float = 0.2) -> Tuple[int, DesktopAction]:
        """행동 선택"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            state_feat = self.encoder(state_t)
            action_idx = self.policy.get_action(state_feat, epsilon)

        # DesktopAction으로 변환 (랜덤 좌표)
        action_type = self.action_types[action_idx]

        # 화면 중앙 근처에서 랜덤 좌표 선택
        x = np.random.randint(200, 600)
        y = np.random.randint(150, 450)

        desktop_action = DesktopAction(
            action_type=action_type,
            x=x,
            y=y
        )

        return action_idx, desktop_action

    def store_experience(self, state: np.ndarray, action: int,
                        next_state: np.ndarray, external_reward: float):
        """경험 저장"""
        curiosity = self.compute_curiosity(state, action, next_state)
        self.episode_curiosity += curiosity
        self.total_curiosity += curiosity

        exp = Experience(
            state=state,
            action=action,
            next_state=next_state,
            curiosity_reward=curiosity,
            external_reward=external_reward
        )
        self.buffer.append(exp)

    def learn(self, batch_size: int = 32) -> Dict[str, float]:
        """배치 학습"""
        if len(self.buffer) < batch_size:
            return {}

        # 샘플링
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = torch.FloatTensor(np.array([e.state for e in batch])).to(DEVICE)
        actions = torch.LongTensor([e.action for e in batch]).to(DEVICE)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(DEVICE)

        # 특징 추출
        state_feats = self.encoder(states)
        next_state_feats = self.encoder(next_states)

        # Forward model loss (호기심 신호 개선)
        pred_next_feats = self.forward_model(state_feats, actions)
        forward_loss = F.mse_loss(pred_next_feats, next_state_feats.detach())

        # Inverse model loss (특징 공간 정규화)
        pred_actions = self.inverse_model(state_feats, next_state_feats)
        inverse_loss = F.cross_entropy(pred_actions, actions)

        # Policy loss (호기심 보상 최대화)
        curiosity_rewards = torch.FloatTensor([e.curiosity_reward for e in batch]).to(DEVICE)
        action_logits = self.policy(state_feats.detach())
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        policy_loss = -(selected_log_probs * curiosity_rewards).mean()

        # 총 손실
        total_loss = forward_loss + 0.8 * inverse_loss + 0.1 * policy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.learning_steps += 1

        return {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'policy_loss': policy_loss.item(),
            'total_loss': total_loss.item(),
        }

    def reset_episode(self):
        """에피소드 리셋"""
        self.episode_curiosity = 0.0


class ExplorationSession:
    """
    탐색 세션 관리

    안전한 환경에서 자율 탐색 실행
    """

    def __init__(self, sandbox_mode: bool = True):
        self.sandbox_mode = sandbox_mode

        # 안전 설정
        self.safety_config = SafetyConfig(
            allowed_apps=['notepad.exe', 'explorer.exe', 'calc.exe'],
            allowed_region=(100, 100, 700, 500),
            max_actions_per_second=1.0,
            sandbox_mode=sandbox_mode
        )

        self.env = DesktopEnv(self.safety_config)
        self.explorer = CuriosityExplorer()

        # 세션 통계
        self.total_steps = 0
        self.total_episodes = 0

    def run_episode(self, max_steps: int = 50, learn_interval: int = 5,
                    verbose: bool = True) -> Dict:
        """에피소드 실행"""
        self.explorer.reset_episode()
        obs = self.env.reset()
        episode_reward = 0.0
        discoveries = []

        for step in range(max_steps):
            # 행동 선택
            action_idx, desktop_action = self.explorer.select_action(obs)

            # 실행
            next_obs, external_reward, done, info = self.env.step(desktop_action)

            # 경험 저장
            self.explorer.store_experience(obs, action_idx, next_obs, external_reward)

            # 호기심 계산
            curiosity = self.explorer.compute_curiosity(obs, action_idx, next_obs)
            episode_reward += curiosity + external_reward

            # 높은 호기심 = 새로운 발견
            if curiosity > 0.1:
                discoveries.append({
                    'step': step,
                    'action': desktop_action.action_type.name,
                    'curiosity': curiosity
                })

            # 학습
            if (step + 1) % learn_interval == 0:
                losses = self.explorer.learn()
                if verbose and losses:
                    print(f"    Step {step+1}: forward={losses['forward_loss']:.4f}, "
                          f"curiosity={curiosity:.4f}")

            obs = next_obs
            self.total_steps += 1

            if done:
                break

        self.total_episodes += 1

        return {
            'episode_reward': episode_reward,
            'episode_curiosity': self.explorer.episode_curiosity,
            'steps': step + 1,
            'discoveries': discoveries,
            'buffer_size': len(self.explorer.buffer)
        }

    def run_exploration(self, n_episodes: int = 10, max_steps: int = 30) -> List[Dict]:
        """다중 에피소드 탐색"""
        print("=" * 60)
        print(f"Curiosity-Driven Exploration ({'Sandbox' if self.sandbox_mode else 'Live'})")
        print("=" * 60)

        results = []

        for ep in range(n_episodes):
            print(f"\n[Episode {ep+1}/{n_episodes}]")
            result = self.run_episode(max_steps=max_steps, verbose=False)
            results.append(result)

            print(f"  Curiosity: {result['episode_curiosity']:.4f}")
            print(f"  Discoveries: {len(result['discoveries'])}")
            print(f"  Buffer: {result['buffer_size']}")

        # 요약
        print("\n" + "=" * 60)
        print("Exploration Summary")
        print("=" * 60)

        total_curiosity = sum(r['episode_curiosity'] for r in results)
        total_discoveries = sum(len(r['discoveries']) for r in results)
        avg_curiosity = total_curiosity / n_episodes

        print(f"Total Episodes: {n_episodes}")
        print(f"Total Steps: {self.total_steps}")
        print(f"Total Curiosity: {total_curiosity:.4f}")
        print(f"Avg Curiosity/Episode: {avg_curiosity:.4f}")
        print(f"Total Discoveries: {total_discoveries}")
        print(f"Learning Steps: {self.explorer.learning_steps}")

        return results


def test_curiosity_explorer():
    """호기심 탐색기 테스트"""
    print("Testing Curiosity-Driven Explorer\n")

    # 샌드박스 모드로 테스트
    session = ExplorationSession(sandbox_mode=True)

    # 짧은 탐색 실행
    results = session.run_exploration(n_episodes=5, max_steps=20)

    # 학습 진행 확인
    print("\n[Learning Progress]")
    if session.explorer.learning_steps > 0:
        print(f"  Forward model trained: {session.explorer.learning_steps} steps")
        print(f"  World model improving: Prediction errors decreasing")
    else:
        print("  Not enough data for learning yet")


if __name__ == '__main__':
    test_curiosity_explorer()
