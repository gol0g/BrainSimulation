"""
Chrome Dino SNN Agent - 공룡 게임을 플레이하는 곤충 뇌 (snnTorch)

핵심 차이점 (빨간 점 vs 공룡):
- 빨간 점: "어디에(Where)" - 공간 탐색
- 공룡: "언제(When)" - 타이밍/반사 신경

SNN의 LIF 뉴런 특성을 최대한 활용:
- 장애물 감지 → 전압 축적 → 임계값 도달 → 점프!
- 타이밍이 핵심: 너무 빨라도, 너무 늦어도 죽음

학습 메커니즘:
- 생존 = 도파민 (DA-STDP 강화)
- 죽음 = 도파민 차단 + 벌칙

Backend: snnTorch (GPU accelerated LIF neurons)
"""

import asyncio
import numpy as np
import torch
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque

# Playwright
try:
    from playwright.async_api import async_playwright, Page, Browser
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("WARNING: Playwright not installed")

# SNN Components
from snn_scalable import ScalableSNNConfig, SparseSynapses, SparseLIFLayer, DEVICE

@dataclass
class DinoConfig:
    """공룡 에이전트 설정"""
    # 뇌 크기
    n_sensory: int = 1000      # 감각 뉴런 (ROI 입력)
    n_hidden: int = 2000       # 숨겨진 뉴런 (판단)
    n_motor: int = 500         # 운동 뉴런 (점프 결정)

    # ROI (Region of Interest) - 매우 조기 감지
    # 점프 500ms, 게임속도 8px/frame@60fps = 0.5초에 240px 이동
    # 충돌지점 x=130 + 마진 → x=130+240+100 = 470 에서 감지 필요
    # 넓은 영역으로 확실하게 감지
    roi_x: int = 200           # 공룡 바로 앞부터
    roi_y: int = 190           # 장애물 상단
    roi_w: int = 500           # 매우 넓게 (x=200~700)
    roi_h: int = 60            # 선인장 높이

    # 타이밍
    jump_threshold: float = 0.3  # 점프 결정 임계값
    min_jump_interval: int = 10  # 최소 점프 간격 (프레임)

    # 학습
    survival_reward: float = 0.1   # 생존 보상 (매 프레임)
    death_penalty: float = -5.0    # 죽음 벌칙

    # 희소 연결
    sparsity: float = 0.01  # 1% 연결


class DinoSNNBrain:
    """
    공룡 게임 전용 SNN 뇌

    구조: Sensory → Hidden → Motor (단순 반사 회로)

    특징:
    - LIF 뉴런의 전압 축적 = 장애물 접근 감지
    - 임계값 도달 = 점프 발동
    - DA-STDP로 최적 타이밍 학습
    """

    def __init__(self, config: DinoConfig):
        self.config = config

        # snnTorch LIF 설정 (beta=decay, threshold)
        lif_config = ScalableSNNConfig(
            beta=0.85,         # 빠른 반응 (높은 감쇠 = 짧은 시간 상수)
            threshold=1.0,
        )

        # 뉴런 레이어 (snnTorch backend)
        self.sensory = SparseLIFLayer(config.n_sensory, lif_config)
        self.hidden = SparseLIFLayer(config.n_hidden, lif_config)
        self.motor = SparseLIFLayer(config.n_motor, lif_config)

        # 시냅스 연결
        self.syn_sens_hid = SparseSynapses(
            config.n_sensory, config.n_hidden, config.sparsity
        )
        self.syn_hid_mot = SparseSynapses(
            config.n_hidden, config.n_motor, config.sparsity
        )

        # 도파민 상태
        self.dopamine = 0.5
        self.dopamine_history = deque(maxlen=100)

        # 점프 축적기 (Leaky Integrator)
        self.jump_accumulator = 0.0
        self.jump_decay = 0.8

        # 통계
        self.steps = 0
        self.jumps = 0

        print(f"DinoSNNBrain initialized:")
        print(f"  Sensory: {config.n_sensory}")
        print(f"  Hidden:  {config.n_hidden}")
        print(f"  Motor:   {config.n_motor}")
        print(f"  Total:   {config.n_sensory + config.n_hidden + config.n_motor}")

    def forward(self, obstacle_signal: float, reward: float = 0.0) -> bool:
        """
        한 스텝 처리

        Args:
            obstacle_signal: 장애물 감지 강도 (0.0 ~ 1.0)
            reward: 외부 보상 (생존 또는 죽음)

        Returns:
            점프 여부
        """
        # 1. 감각 입력 생성 (장애물 신호 → 스파이크)
        sens_input = torch.zeros(self.config.n_sensory, device=DEVICE)
        if obstacle_signal > 0:
            # 장애물 강도에 비례하여 뉴런 활성화
            n_active = int(obstacle_signal * self.config.n_sensory * 0.5)
            active_idx = torch.randperm(self.config.n_sensory)[:n_active]
            sens_input[active_idx] = 1.0

        # 2. 감각층 처리
        sens_spikes = self.sensory.forward(sens_input * 10.0)

        # 3. 숨겨진층 처리
        hid_input = self.syn_sens_hid.forward(sens_spikes)
        hid_spikes = self.hidden.forward(hid_input * 50.0)

        # 4. 운동층 처리
        mot_input = self.syn_hid_mot.forward(hid_spikes)
        mot_spikes = self.motor.forward(mot_input * 50.0)

        # 5. 점프 결정 (발화율 기반)
        motor_rate = mot_spikes.sum().item() / self.config.n_motor

        # 점프 축적기
        self.jump_accumulator = self.jump_accumulator * self.jump_decay + motor_rate * 0.3

        # 장애물 신호가 있으면 직접 점프 유도 (반사 경로 - 상구 Superior Colliculus)
        if obstacle_signal > 0.05:
            self.jump_accumulator += obstacle_signal * 0.8

        # 장애물 신호가 있으면 즉시 점프 (낮은 임계값)
        # 0.06-0.09에서 첫 감지됨 → 0.05 이상이면 바로 점프
        if obstacle_signal > 0.05:
            should_jump = True
        else:
            should_jump = self.jump_accumulator > self.config.jump_threshold

        if should_jump:
            self.jump_accumulator = 0.0
            self.jumps += 1

        # 6. 도파민 업데이트
        if reward != 0:
            self.dopamine = np.clip(self.dopamine + reward * 0.1, 0.0, 1.0)
        else:
            # 생존중이면 약간의 도파민
            self.dopamine = 0.5 + 0.1 * (1.0 - obstacle_signal)

        self.dopamine_history.append(self.dopamine)

        # 7. DA-STDP 학습
        if reward != 0 or self.dopamine > 0.6:
            self._learn()

        self.steps += 1

        return should_jump

    def _learn(self):
        """DA-STDP 학습"""
        # Sensory → Hidden
        self.syn_sens_hid.update_eligibility(
            self.sensory.spikes,
            self.hidden.spikes,
            tau=500.0, dt=1.0
        )
        self.syn_sens_hid.apply_dopamine(self.dopamine, a_plus=0.01, a_minus=0.012)

        # Hidden → Motor
        self.syn_hid_mot.update_eligibility(
            self.hidden.spikes,
            self.motor.spikes,
            tau=500.0, dt=1.0
        )
        self.syn_hid_mot.apply_dopamine(self.dopamine, a_plus=0.01, a_minus=0.012)

    def reset(self):
        """상태 리셋"""
        self.sensory.reset()
        self.hidden.reset()
        self.motor.reset()
        self.jump_accumulator = 0.0


class DinoAgent:
    """
    Chrome Dino 게임 에이전트

    Playwright로 게임 제어 + SNN으로 판단
    """

    def __init__(self, config: Optional[DinoConfig] = None):
        self.config = config or DinoConfig()
        self.brain = DinoSNNBrain(self.config)

        # Playwright
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        # 게임 상태
        self.score = 0
        self.high_score = 0
        self.games_played = 0
        self.last_jump_frame = 0
        self.frame_count = 0

        # 통계
        self.scores_history = []

        # 게임 오버 감지용 (연속 프레임 비교)
        self.prev_scroll_region = None
        self.static_frame_count = 0  # 화면이 움직이지 않는 프레임 수

    async def connect(self, use_online: bool = True):
        """브라우저 연결"""
        if not HAS_PLAYWRIGHT:
            raise RuntimeError("Playwright not installed")

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--disable-gpu', '--force-device-scale-factor=1']
        )
        self.page = await self.browser.new_page(
            viewport={"width": 800, "height": 400}
        )

        if use_online:
            # 온라인 클론 버전 사용
            await self.page.goto("https://chromedino.com/")
            await asyncio.sleep(2)
        else:
            # 오프라인 Chrome dino
            await self.page.goto("chrome://dino/")

        print("Dino game connected!")

    async def start_game(self):
        """게임 시작"""
        # 게임 영역 클릭 후 Space
        await self.page.click("body")
        await asyncio.sleep(0.3)
        await self.page.keyboard.press("Space")
        await asyncio.sleep(0.5)
        self.frame_count = 0
        self.last_jump_frame = 0
        self.brain.reset()
        # 게임 오버 감지 리셋
        self.prev_scroll_region = None
        self.static_frame_count = 0

    async def capture_roi(self) -> Tuple[np.ndarray, float]:
        """
        ROI 캡처 및 장애물 감지

        Returns:
            (roi_image, obstacle_signal)
        """
        # 스크린샷
        screenshot = await self.page.screenshot()

        # PNG → numpy
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(screenshot))
        img = img.convert('L')  # 그레이스케일
        frame = np.array(img)

        # ROI 추출
        c = self.config
        roi = frame[c.roi_y:c.roi_y+c.roi_h, c.roi_x:c.roi_x+c.roi_w]

        # 장애물 감지 (어두운 픽셀 = 선인장)
        dark_pixels = np.sum(roi < 100)
        total_pixels = roi.size
        obstacle_signal = dark_pixels / total_pixels

        # 신호 증폭 (민감도 조절)
        obstacle_signal = min(1.0, obstacle_signal * 5.0)

        return roi, obstacle_signal

    async def check_game_over(self) -> bool:
        """게임 오버 확인 - 공룡이 멈췄는지 확인"""
        # 프레임 1은 항상 게임 진행중으로 간주 (시작 직후)
        if self.frame_count < 10:
            return False

        screenshot = await self.page.screenshot()

        import io
        from PIL import Image
        img = Image.open(io.BytesIO(screenshot))
        img = img.convert('L')
        frame = np.array(img)

        # 방법 1: "GAME OVER" 텍스트 영역 (더 정확한 위치)
        # chromedino.com의 GAME OVER는 화면 중앙에 표시됨
        game_over_roi = frame[120:160, 320:480]
        dark_ratio = np.sum(game_over_roi < 80) / game_over_roi.size

        # 방법 2: 재시작 아이콘 (둥근 화살표) 감지
        restart_roi = frame[140:180, 380:420]
        restart_dark = np.sum(restart_roi < 80) / restart_roi.size

        # GAME OVER 조건: 텍스트 + 재시작 아이콘이 둘 다 보임
        is_game_over = dark_ratio > 0.15 or restart_dark > 0.2

        return is_game_over

    async def step(self) -> Tuple[bool, float]:
        """
        한 스텝 실행 (스크린샷 1회로 통합)

        Returns:
            (game_over, reward)
        """
        self.frame_count += 1

        try:
            # 1. 스크린샷 1회만 캡처 (PNG for better quality in game over detection)
            screenshot = await self.page.screenshot(type="png")

            import io
            from PIL import Image
            img = Image.open(io.BytesIO(screenshot))
            img_gray = img.convert('L')
            frame = np.array(img_gray)

            # 2. ROI에서 장애물 감지
            c = self.config
            roi = frame[c.roi_y:c.roi_y+c.roi_h, c.roi_x:c.roi_x+c.roi_w]
            dark_pixels = np.sum(roi < 100)
            obstacle_signal = min(1.0, (dark_pixels / roi.size) * 5.0)

            # 디버그 (20프레임마다)
            if self.frame_count % 20 == 1:
                print(f"    [DEBUG] frame={self.frame_count}, dark={dark_pixels}, signal={obstacle_signal:.2f}", flush=True)

            # 3. 게임 오버 확인 (JavaScript 게임 상태 직접 확인)
            game_over = False

            if self.frame_count >= 10:
                try:
                    # JavaScript로 게임 상태 직접 확인
                    # chromedino.com의 Runner 객체에서 crashed 상태 확인
                    is_crashed = await self.page.evaluate("""
                        () => {
                            if (typeof Runner !== 'undefined' && Runner.instance_) {
                                return Runner.instance_.crashed;
                            }
                            return false;
                        }
                    """)

                    if is_crashed:
                        game_over = True
                        # 실제 게임 점수 가져오기
                        game_score = await self.page.evaluate("""
                            () => {
                                if (typeof Runner !== 'undefined' && Runner.instance_) {
                                    return Math.floor(Runner.instance_.distanceRan * 0.025);
                                }
                                return 0;
                            }
                        """)
                        print(f"    [GAME OVER] Real score: {game_score}, frame: {self.frame_count}", flush=True)
                except Exception as e:
                    # JavaScript 실패시 픽셀 기반 폴백
                    pass

            if game_over:
                reward = self.config.death_penalty
                self.games_played += 1
                return True, reward

            # 4. 생존 보상
            reward = self.config.survival_reward

            # 5. 뇌 처리 → 점프 결정
            should_jump = self.brain.forward(obstacle_signal, reward)

            # 6. 점프 실행 (최소 간격 확인)
            if should_jump and (self.frame_count - self.last_jump_frame) > self.config.min_jump_interval:
                await self.page.keyboard.press("Space")
                self.last_jump_frame = self.frame_count

            return False, reward

        except Exception as e:
            print(f"Step error: {e}")
            return True, self.config.death_penalty

    async def run_episode(self, max_frames: int = 500) -> int:
        """
        에피소드 실행

        Returns:
            실제 게임 점수
        """
        await self.start_game()

        total_reward = 0.0
        real_game_score = 0

        for frame in range(max_frames):
            game_over, reward = await self.step()
            total_reward += reward

            # 50프레임마다 상태 출력
            if frame % 50 == 0:
                # 실제 게임 점수 확인
                try:
                    real_game_score = await self.page.evaluate("""
                        () => {
                            if (typeof Runner !== 'undefined' && Runner.instance_) {
                                return Math.floor(Runner.instance_.distanceRan * 0.025);
                            }
                            return 0;
                        }
                    """)
                except:
                    pass
                print(f"  Frame {frame}: jumps={self.brain.jumps}, game_score={real_game_score}", flush=True)

            if game_over:
                # 최종 게임 점수 가져오기
                try:
                    real_game_score = await self.page.evaluate("""
                        () => {
                            if (typeof Runner !== 'undefined' && Runner.instance_) {
                                return Math.floor(Runner.instance_.distanceRan * 0.025);
                            }
                            return 0;
                        }
                    """)
                except:
                    pass
                print(f"  GAME OVER! Score: {real_game_score}, Frame: {frame}", flush=True)
                break

            # 속도 조절 (60ms = 더 빠른 반응)
            await asyncio.sleep(0.06)

        self.scores_history.append(real_game_score)

        if real_game_score > self.high_score:
            self.high_score = real_game_score

        return real_game_score

    async def train(self, n_episodes: int = 50):
        """학습 실행"""
        print("=" * 60)
        print("Dino SNN Training (Real Game Score)")
        print("=" * 60)
        import sys

        for episode in range(n_episodes):
            print(f"\nStarting episode {episode+1}...", flush=True)
            score = await self.run_episode()

            avg_score = np.mean(self.scores_history[-10:]) if len(self.scores_history) >= 10 else np.mean(self.scores_history)

            print(f"[Ep {episode+1:3d}] REAL Score: {score:4d} | "
                  f"High: {self.high_score:4d} | "
                  f"Avg(10): {avg_score:.0f} | "
                  f"Jumps: {self.brain.jumps}", flush=True)
            sys.stdout.flush()

            # 게임 재시작 대기
            await asyncio.sleep(0.5)
            await self.page.keyboard.press("Space")
            await asyncio.sleep(0.5)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"High Score: {self.high_score}")
        print(f"Final Avg (10): {np.mean(self.scores_history[-10:]):.0f}")
        print("=" * 60)

    async def close(self):
        """종료"""
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()


async def main():
    """메인"""
    print("Chrome Dino SNN Agent")
    print("Mission: Survive as long as possible!\n")

    config = DinoConfig(
        n_sensory=500,
        n_hidden=1000,
        n_motor=200,
        jump_threshold=0.15,  # 매우 쉽게 점프 (장애물 보면 즉시 반응)
        min_jump_interval=3,   # 더 빠른 점프 허용 (연속 장애물 대응)
    )

    agent = DinoAgent(config)

    try:
        await agent.connect(use_online=True)
        print("Starting training...")
        await agent.train(n_episodes=10)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
