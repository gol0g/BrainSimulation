"""
SNN Desktop Agent V3

스파이킹 신경망 기반 데스크탑 에이전트 (V3 Brain 사용)

인간 뇌처럼:
1. 스파이크로 정보 처리 (15% 희소성)
2. STDP로 경험에서 학습
3. 예측 오차로 호기심 생성
4. 연속적 실시간 학습
5. k-WTA + 피드백 억제로 안정적 희소성
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
from PIL import Image

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
genesis_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)
sys.path.insert(0, genesis_dir)

from snn_brain_v3 import SpikingBrainV3, BrainConfigV3
from desktop_env import DesktopEnv, DesktopAction, ActionType, SafetyConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SNNDesktopAgent:
    """
    SNN 기반 데스크탑 에이전트 (V3)

    핵심 차이점 (기존 CNN 에이전트 vs 이것):
    - 역전파 X → STDP 학습
    - 배치 학습 X → 실시간 연속 학습
    - 외부 보상 X → 예측 오차 (호기심)
    - 기호 처리 X → 스파이크 패턴

    V3 개선:
    - 15% 희소성 (k-WTA + 피드백 억제)
    - 안정적 신호 전파
    - 패턴 구분 학습
    """

    def __init__(self, config: Optional[BrainConfigV3] = None):
        # 뇌 설정 (데스크탑용으로 조정)
        if config is None:
            config = BrainConfigV3(
                sensory_neurons=512,
                v1_neurons=1024,
                v2_neurons=512,
                motor_neurons=32,  # 행동 공간
                time_steps=10,     # 빠른 반응
                target_sparsity=0.15,  # 15% 희소성
            )

        self.brain = SpikingBrainV3(config).to(DEVICE)
        self.config = config

        # 행동 매핑
        self.action_map = self._create_action_map()

        # 예측 오차 추적 (호기심)
        self.prediction_errors = deque(maxlen=100)
        self.prev_v1_activity = None

        # 학습 통계
        self.total_actions = 0
        self.curiosity_sum = 0.0

    def _create_action_map(self) -> Dict[int, Tuple[ActionType, int, int]]:
        """
        뉴런 → 행동 매핑

        32개 운동 뉴런:
        - 0-7: 이동 방향 (8방향)
        - 8-15: 클릭 위치 (8영역)
        - 16-23: 오른쪽 클릭 (8영역)
        - 24-31: 더블 클릭, 특수 행동
        """
        actions = {}
        screen_positions = [
            (200, 150), (400, 150), (600, 150), (200, 300),
            (600, 300), (200, 450), (400, 450), (600, 450)
        ]

        # 이동
        for i in range(8):
            x, y = screen_positions[i]
            actions[i] = (ActionType.MOVE, x, y)

        # 클릭
        for i in range(8):
            x, y = screen_positions[i]
            actions[8 + i] = (ActionType.CLICK, x, y)

        # 오른쪽 클릭
        for i in range(8):
            x, y = screen_positions[i]
            actions[16 + i] = (ActionType.RIGHT_CLICK, x, y)

        # 특수 행동
        actions[24] = (ActionType.DOUBLE_CLICK, 400, 300)
        actions[25] = (ActionType.NONE, 0, 0)
        for i in range(26, 32):
            actions[i] = (ActionType.MOVE, 400, 300)

        return actions

    def process_observation(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        관측 처리

        1. 이미지 → 텐서
        2. 뇌로 처리 (STDP 학습 포함)
        3. 활동 패턴 반환
        """
        # 이미지 전처리 (256x256)
        if image.shape[0] != 3:  # HWC → CHW
            image = np.transpose(image, (2, 0, 1))

        img_tensor = torch.FloatTensor(image).unsqueeze(0).to(DEVICE)

        # 뇌 처리 (학습 포함)
        activity = self.brain(img_tensor, learn=True)

        # 예측 오차 계산 (호기심)
        if self.prev_v1_activity is not None:
            prediction_error = torch.abs(
                activity['v1'] - self.prev_v1_activity
            ).mean().item()
            self.prediction_errors.append(prediction_error)
            self.curiosity_sum += prediction_error

        self.prev_v1_activity = activity['v1'].clone()

        return activity

    def select_action(self, activity: Dict[str, torch.Tensor]) -> DesktopAction:
        """
        활동 패턴 → 행동 선택

        Winner-take-all: 가장 많이 발화한 운동 뉴런
        """
        motor_spikes = activity['motor_raw'].squeeze(0)
        action_idx = self.brain.get_action(motor_spikes.unsqueeze(0))

        # 행동 매핑
        action_type, x, y = self.action_map.get(
            action_idx, (ActionType.NONE, 0, 0)
        )

        self.total_actions += 1

        return DesktopAction(action_type=action_type, x=x, y=y)

    def get_curiosity(self) -> float:
        """현재 호기심 수준 (최근 예측 오차)"""
        if not self.prediction_errors:
            return 0.0
        return np.mean(list(self.prediction_errors)[-10:])

    def get_statistics(self) -> Dict:
        """에이전트 통계"""
        brain_stats = self.brain.get_statistics()
        return {
            **brain_stats,
            'total_actions': self.total_actions,
            'avg_curiosity': self.curiosity_sum / max(self.total_actions, 1),
            'current_curiosity': self.get_curiosity(),
            # V3 호환성: total_spikes 근사 (step_count * time_steps * avg_activity)
            'total_spikes': brain_stats.get('step_count', 0) * self.config.time_steps,
        }

    def reset(self):
        """상태 리셋"""
        self.brain.reset()
        self.prev_v1_activity = None
        self.prediction_errors.clear()


class SNNExplorationSession:
    """
    SNN 에이전트 탐색 세션
    """

    def __init__(self, sandbox_mode: bool = True):
        self.sandbox_mode = sandbox_mode

        # 환경 설정
        self.safety_config = SafetyConfig(
            allowed_apps=['notepad.exe', 'explorer.exe', 'calc.exe'],
            allowed_region=(100, 100, 700, 500),
            max_actions_per_second=2.0,
            sandbox_mode=sandbox_mode
        )

        self.env = DesktopEnv(self.safety_config)
        self.agent = SNNDesktopAgent()

    def run_episode(self, max_steps: int = 50, verbose: bool = True) -> Dict:
        """에피소드 실행"""
        self.agent.reset()
        obs = self.env.reset()

        episode_curiosity = 0.0
        actions_taken = []

        for step in range(max_steps):
            # 관측 처리
            activity = self.agent.process_observation(obs)

            # 행동 선택
            action = self.agent.select_action(activity)
            actions_taken.append(action.action_type.name)

            # 실행
            obs, reward, done, info = self.env.step(action)

            # 호기심 추적
            curiosity = self.agent.get_curiosity()
            episode_curiosity += curiosity

            if verbose and (step + 1) % 10 == 0:
                print(f"    Step {step+1}: curiosity={curiosity:.4f}, "
                      f"action={action.action_type.name}")

            if done:
                break

        return {
            'steps': step + 1,
            'episode_curiosity': episode_curiosity,
            'avg_curiosity': episode_curiosity / (step + 1),
            'unique_actions': len(set(actions_taken)),
            'brain_stats': self.agent.get_statistics(),
        }

    def run_exploration(self, n_episodes: int = 5, max_steps: int = 30) -> None:
        """다중 에피소드 탐색"""
        print("=" * 60)
        print(f"SNN Brain Exploration ({'Sandbox' if self.sandbox_mode else 'Live'})")
        print("=" * 60)

        all_results = []

        for ep in range(n_episodes):
            print(f"\n[Episode {ep+1}/{n_episodes}]")
            result = self.run_episode(max_steps=max_steps, verbose=False)
            all_results.append(result)

            print(f"  Curiosity: {result['avg_curiosity']:.4f}")
            print(f"  Unique actions: {result['unique_actions']}")
            print(f"  Total spikes: {result['brain_stats']['total_spikes']:.0f}")

        # 요약
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)

        avg_curiosity = np.mean([r['avg_curiosity'] for r in all_results])
        total_spikes = all_results[-1]['brain_stats']['total_spikes']

        print(f"Episodes: {n_episodes}")
        print(f"Avg Curiosity: {avg_curiosity:.4f}")
        print(f"Total Spikes: {total_spikes:.0f}")
        print(f"STDP Learning: Active")

        # V1 희소성 추적
        final_stats = all_results[-1]['brain_stats']
        print(f"V1 Sparsity: {final_stats['v1_sparsity']:.2%}")


def test_snn_desktop_agent():
    """SNN 데스크탑 에이전트 테스트"""
    print("Testing SNN Desktop Agent\n")

    session = SNNExplorationSession(sandbox_mode=True)
    session.run_exploration(n_episodes=3, max_steps=20)


if __name__ == '__main__':
    test_snn_desktop_agent()
