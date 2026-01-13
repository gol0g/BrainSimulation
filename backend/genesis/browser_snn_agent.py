"""
Browser SNN Agent - Playwright + 50만 뉴런 곤충 뇌

실제 브라우저와 SNN을 연결하는 통합 에이전트:
1. Playwright로 브라우저 화면 캡처
2. Visual Encoder로 스파이크 변환
3. Insect Brain (50만 뉴런) 처리
4. Motor Decoder로 행동 변환
5. Playwright로 마우스/클릭 실행

학습 목표: "빨간 점을 클릭하라"
"""

import asyncio
import numpy as np
import torch
import pickle
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Playwright
try:
    from playwright.async_api import async_playwright, Page, Browser
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("WARNING: Playwright not installed. Run: pip install playwright && playwright install")

# Import Insect Brain
from insect_brain import InsectBrain, InsectBrainConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class AgentConfig:
    """에이전트 설정"""
    # 뇌 크기
    brain_neurons: int = 50000  # 테스트용 (500K는 프로덕션용)

    # 시각
    visual_width: int = 128
    visual_height: int = 128

    # 행동 스케일
    move_scale: float = 8.0  # 이동 감도 (5.0 → 8.0 증가, 512px 화면용)
    click_threshold: float = 0.3  # 클릭 임계값

    # 학습
    learning_rate: float = 0.01
    reward_decay: float = 0.95

    # 파일
    brain_file: str = "insect_brain_weights.pkl"


class BrowserSNNAgent:
    """
    Playwright + SNN 통합 에이전트

    "50만 뉴런 곤충 뇌로 브라우저를 탐색한다"
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

        # 뇌 생성
        brain_config = InsectBrainConfig(
            total_neurons=self.config.brain_neurons,
            visual_width=self.config.visual_width,
            visual_height=self.config.visual_height,
        )
        self.brain = InsectBrain(brain_config)

        # 가중치 로드 시도
        self._load_brain()

        # Playwright 상태
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        # 학습 통계
        self.total_steps = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        self.hits = 0

        # 이전 상태 (학습용)
        self.prev_reward = 0.0

    def _load_brain(self):
        """장기 기억 로드"""
        brain_path = Path(__file__).parent / self.config.brain_file
        if brain_path.exists():
            try:
                with open(brain_path, 'rb') as f:
                    weights = pickle.load(f)

                # 시냅스 가중치 복원
                self._restore_weights(weights)
                print(f"Long-term memory loaded from {brain_path}")
            except Exception as e:
                print(f"Failed to load brain: {e}")
        else:
            print("New brain born - no previous memory")

    def _restore_weights(self, weights: Dict):
        """가중치 복원"""
        # Optic Lobe
        if 'optic_l1_l2' in weights:
            self.brain.optic_lobe.syn_l1_l2.weights = weights['optic_l1_l2'].to(DEVICE)
            self.brain.optic_lobe.syn_l1_l2._rebuild_sparse()
        if 'optic_l2_l3' in weights:
            self.brain.optic_lobe.syn_l2_l3.weights = weights['optic_l2_l3'].to(DEVICE)
            self.brain.optic_lobe.syn_l2_l3._rebuild_sparse()

        # Mushroom Body
        if 'mushroom_input_kc' in weights:
            self.brain.mushroom_body.syn_input_kc.weights = weights['mushroom_input_kc'].to(DEVICE)
            self.brain.mushroom_body.syn_input_kc._rebuild_sparse()
        if 'mushroom_kc_output' in weights:
            self.brain.mushroom_body.syn_kc_output.weights = weights['mushroom_kc_output'].to(DEVICE)
            self.brain.mushroom_body.syn_kc_output._rebuild_sparse()

        # Central Complex
        if 'central_compass_motor' in weights:
            self.brain.central_complex.syn_compass_motor.weights = weights['central_compass_motor'].to(DEVICE)
            self.brain.central_complex.syn_compass_motor._rebuild_sparse()
        if 'central_motor_output' in weights:
            self.brain.central_complex.syn_motor_output.weights = weights['central_motor_output'].to(DEVICE)
            self.brain.central_complex.syn_motor_output._rebuild_sparse()

    def save_brain(self):
        """장기 기억 저장"""
        weights = {
            'optic_l1_l2': self.brain.optic_lobe.syn_l1_l2.weights.cpu(),
            'optic_l2_l3': self.brain.optic_lobe.syn_l2_l3.weights.cpu(),
            'mushroom_input_kc': self.brain.mushroom_body.syn_input_kc.weights.cpu(),
            'mushroom_kc_output': self.brain.mushroom_body.syn_kc_output.weights.cpu(),
            'central_compass_motor': self.brain.central_complex.syn_compass_motor.weights.cpu(),
            'central_motor_output': self.brain.central_complex.syn_motor_output.weights.cpu(),
        }

        brain_path = Path(__file__).parent / self.config.brain_file
        with open(brain_path, 'wb') as f:
            pickle.dump(weights, f)

        print(f"Brain saved to {brain_path}")

    async def connect_browser(self, headless: bool = False):
        """브라우저 연결"""
        if not HAS_PLAYWRIGHT:
            raise RuntimeError("Playwright not installed")

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=headless,
            args=[
                '--disable-gpu',
                '--disable-dev-shm-usage',
                '--disable-animations',
                '--force-device-scale-factor=1',  # 확대/축소 방지
            ]
        )
        self.page = await self.browser.new_page(
            viewport={"width": 512, "height": 512},
            device_scale_factor=1,  # 고정 스케일
        )

        print("Browser connected")

    async def load_arena(self):
        """테스트 환경 로드"""
        arena_path = Path(__file__).parent / "test_arena.html"
        await self.page.goto(f"file://{arena_path}")
        await asyncio.sleep(0.5)  # 로딩 대기
        print("Arena loaded")

    async def capture_screen(self) -> np.ndarray:
        """화면 캡처 → numpy 배열"""
        screenshot = await self.page.screenshot(
            animations="disabled",  # 애니메이션 중지 후 캡처
            type="jpeg",  # PNG보다 빠름
            quality=80,
        )

        # JPEG 바이트 → numpy
        import io
        from PIL import Image

        img = Image.open(io.BytesIO(screenshot))
        img = img.convert('RGB')
        img = img.resize((self.config.visual_width, self.config.visual_height))

        return np.array(img)

    async def execute_action(self, action: Dict) -> float:
        """행동 실행 → 보상 반환"""
        dx = action['dx'] * self.config.move_scale
        dy = action['dy'] * self.config.move_scale
        click = action['click']

        # 이동
        reward = await self.page.evaluate(f"window.moveCursor({dx}, {dy})")

        # 클릭
        if click:
            click_reward = await self.page.evaluate("window.clickAction()")
            reward += click_reward
            if click_reward > 5:  # Hit!
                self.hits += 1

        return float(reward)

    async def get_state(self) -> Dict:
        """게임 상태 조회"""
        return await self.page.evaluate("window.getState()")

    async def reset_game(self):
        """게임 리셋"""
        await self.page.evaluate("window.resetGame()")
        self.brain.reset()

    async def step(self) -> Tuple[Dict, float]:
        """
        한 스텝 실행

        Returns:
            (brain_output, reward)
        """
        # 1. 화면 캡처
        frame = await self.capture_screen()

        # 2. 뇌 처리
        result = self.brain.forward(frame, external_reward=self.prev_reward)
        action = result['action']

        # 3. 행동 실행
        reward = await self.execute_action(action)

        # 4. 학습 업데이트
        self.prev_reward = reward
        self.total_reward += reward
        self.total_steps += 1

        # 5. 상태 업데이트
        dopamine = result['dopamine']
        status_text = f"Step: {self.total_steps} | Reward: {self.total_reward:.1f} | DA: {dopamine:.3f}"
        await self.page.evaluate(f"window.setStatus('{status_text}')")

        return result, reward

    async def run_episode(self, max_steps: int = 500, render_delay: float = 0.03) -> float:
        """
        에피소드 실행

        Args:
            max_steps: 최대 스텝 수
            render_delay: 렌더링 딜레이 (시각화용)

        Returns:
            에피소드 총 보상
        """
        await self.reset_game()
        episode_reward = 0.0

        for step in range(max_steps):
            result, reward = await self.step()
            episode_reward += reward

            # 진행 상황 출력 (매 100 스텝)
            if step % 100 == 0:
                state = await self.get_state()
                print(f"  Step {step}: reward={episode_reward:.1f}, hits={state['hits']}, "
                      f"dx={result['action']['dx']:.1f}, dy={result['action']['dy']:.1f}")

            # 렌더링 딜레이
            if render_delay > 0:
                await asyncio.sleep(render_delay)

        self.episode_rewards.append(episode_reward)
        return episode_reward

    async def train(self, n_episodes: int = 10, steps_per_episode: int = 500):
        """
        학습 실행

        Args:
            n_episodes: 에피소드 수
            steps_per_episode: 에피소드당 스텝 수
        """
        print("=" * 60)
        print(f"Training: {n_episodes} episodes, {steps_per_episode} steps each")
        print(f"Brain: {self.config.brain_neurons:,} neurons")
        print("=" * 60)

        for episode in range(n_episodes):
            print(f"\n[Episode {episode + 1}/{n_episodes}]")

            reward = await self.run_episode(steps_per_episode, render_delay=0.02)
            state = await self.get_state()

            print(f"  Total reward: {reward:.1f}")
            print(f"  Hits: {state['hits']}")
            print(f"  Score: {state['score']}")

            # 매 에피소드 후 뇌 저장
            self.save_brain()

        # 최종 통계
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total steps: {self.total_steps:,}")
        print(f"Total reward: {self.total_reward:.1f}")
        print(f"Total hits: {self.hits}")
        print(f"Average reward per episode: {np.mean(self.episode_rewards):.1f}")

    async def close(self):
        """브라우저 종료"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        print("Browser closed")


async def main():
    """메인 실행"""
    print("=" * 60)
    print("Browser SNN Agent - 50만 뉴런 곤충 뇌")
    print("Mission: Click the Red Dot!")
    print("=" * 60)

    # 에이전트 생성 (테스트용 5만 뉴런)
    config = AgentConfig(
        brain_neurons=50000,  # 테스트용
        visual_width=128,
        visual_height=128,
        move_scale=3.0,
    )

    agent = BrowserSNNAgent(config)

    try:
        # 브라우저 연결
        await agent.connect_browser(headless=False)

        # 테스트 환경 로드
        await agent.load_arena()

        # 학습 실행
        await agent.train(n_episodes=5, steps_per_episode=300)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 뇌 저장 및 종료
        agent.save_brain()
        await agent.close()


def test_without_browser():
    """브라우저 없이 테스트 (SNN만)"""
    print("=" * 60)
    print("SNN-only Test (no browser)")
    print("=" * 60)

    config = AgentConfig(brain_neurons=50000)

    # 뇌만 생성
    brain_config = InsectBrainConfig(
        total_neurons=config.brain_neurons,
        visual_width=config.visual_width,
        visual_height=config.visual_height,
    )
    brain = InsectBrain(brain_config)

    # 가상 프레임으로 테스트
    print("\n[Test: Random frames]")
    for i in range(10):
        # 랜덤 프레임 (빨간 점 시뮬레이션)
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[30:70, 30:70, :] = 100  # 회색 배경
        frame[45:55, 45:55, 0] = 255  # 빨간 점

        result = brain.forward(frame, external_reward=0.5 if i > 5 else 0)

        print(f"  Frame {i}: dx={result['action']['dx']:.1f}, "
              f"dy={result['action']['dy']:.1f}, "
              f"click={result['action']['click']}, "
              f"DA={result['dopamine']:.3f}")

    print("\n[Test: Moving target simulation]")

    # 목표 이동 시뮬레이션
    cursor_x, cursor_y = 64, 64
    target_x, target_y = 100, 30

    for step in range(50):
        # 프레임 생성
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[20:108, 20:108, :] = 80  # 배경

        # 빨간 목표
        tx, ty = int(target_x), int(target_y)
        frame[max(0,ty-5):min(128,ty+5), max(0,tx-5):min(128,tx+5), 0] = 255

        # 녹색 커서
        cx, cy = int(cursor_x), int(cursor_y)
        frame[max(0,cy-3):min(128,cy+3), max(0,cx-3):min(128,cx+3), 1] = 255

        # 거리 기반 보상
        dist = np.sqrt((cursor_x - target_x)**2 + (cursor_y - target_y)**2)
        if dist < 10:
            reward = 1.0
        elif dist < 30:
            reward = 0.3
        else:
            reward = 0.0

        # 뇌 처리
        result = brain.forward(frame, external_reward=reward)

        # 커서 이동
        cursor_x = np.clip(cursor_x + result['action']['dx'] * 0.5, 0, 127)
        cursor_y = np.clip(cursor_y + result['action']['dy'] * 0.5, 0, 127)

        if step % 10 == 0:
            print(f"  Step {step}: cursor=({cursor_x:.0f},{cursor_y:.0f}), "
                  f"target=({target_x},{target_y}), dist={dist:.1f}, "
                  f"reward={reward:.1f}")

    final_dist = np.sqrt((cursor_x - target_x)**2 + (cursor_y - target_y)**2)
    print(f"\n  Final distance: {final_dist:.1f} px")
    print(f"  Initial distance: {np.sqrt((64-100)**2 + (64-30)**2):.1f} px")


if __name__ == '__main__':
    import sys

    if '--no-browser' in sys.argv or not HAS_PLAYWRIGHT:
        test_without_browser()
    else:
        asyncio.run(main())
