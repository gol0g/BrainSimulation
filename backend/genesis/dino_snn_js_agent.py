"""
Chrome Dino SNN Agent - JS API + snnTorch Brain Hybrid

Uses JavaScript API for precise obstacle detection,
snnTorch-based SNN brain for learning and decision making.

Backend: snnTorch (GPU accelerated LIF neurons)
"""

import asyncio
import numpy as np
import torch
from typing import Optional
from dataclasses import dataclass
from collections import deque
from playwright.async_api import async_playwright, Page, Browser

# SNN Components
from snn_scalable import ScalableSNNConfig, SparseSynapses, SparseLIFLayer, DEVICE


@dataclass
class DinoConfig:
    """Agent configuration"""
    # Brain size
    n_sensory: int = 500
    n_hidden: int = 1000
    n_motor: int = 200

    # Timing (discovered from JS API testing)
    jump_gap: int = 100  # Jump when obstacle is this far - jump later to be at peak when passing
    min_jump_interval: int = 300  # ms between jumps

    # Learning
    survival_reward: float = 0.1
    death_penalty: float = -5.0

    # SNN
    sparsity: float = 0.01


class DinoSNNBrain:
    """SNN brain that learns optimal jump timing (snnTorch backend)"""

    def __init__(self, config: DinoConfig):
        self.config = config

        # snnTorch uses beta (decay) and threshold
        lif_config = ScalableSNNConfig(beta=0.9, threshold=1.0)

        self.sensory = SparseLIFLayer(config.n_sensory, lif_config)
        self.hidden = SparseLIFLayer(config.n_hidden, lif_config)
        self.motor = SparseLIFLayer(config.n_motor, lif_config)

        self.syn_sens_hid = SparseSynapses(config.n_sensory, config.n_hidden, config.sparsity)
        self.syn_hid_mot = SparseSynapses(config.n_hidden, config.n_motor, config.sparsity)

        self.dopamine = 0.5
        self.jump_accumulator = 0.0
        self.steps = 0
        self.jumps = 0

        print(f"DinoSNNBrain initialized:")
        print(f"  Sensory: {config.n_sensory}")
        print(f"  Hidden:  {config.n_hidden}")
        print(f"  Motor:   {config.n_motor}")
        print(f"  Total:   {config.n_sensory + config.n_hidden + config.n_motor}")

    def process(self, gap: float, reward: float = 0.0) -> bool:
        """Process gap signal and decide whether to jump"""
        # Normalize gap to 0-1 signal (closer = stronger signal)
        if gap > 0:
            signal = max(0, 1.0 - gap / 500.0)  # Stronger as obstacle approaches
        else:
            signal = 0.0

        # Sensory input
        sens_input = torch.zeros(self.config.n_sensory, device=DEVICE)
        if signal > 0:
            n_active = int(signal * self.config.n_sensory * 0.5)
            active_idx = torch.randperm(self.config.n_sensory)[:n_active]
            sens_input[active_idx] = 1.0

        # Forward pass
        sens_spikes = self.sensory.forward(sens_input * 10.0)
        hid_input = self.syn_sens_hid.forward(sens_spikes)
        hid_spikes = self.hidden.forward(hid_input * 50.0)
        mot_input = self.syn_hid_mot.forward(hid_spikes)
        mot_spikes = self.motor.forward(mot_input * 50.0)

        # Jump accumulator
        motor_rate = mot_spikes.sum().item() / self.config.n_motor
        self.jump_accumulator = self.jump_accumulator * 0.8 + motor_rate * 0.3

        # Direct reflex pathway (Superior Colliculus)
        if signal > 0.5:  # Strong signal = obstacle close
            self.jump_accumulator += signal * 0.5

        # Decision
        should_jump = self.jump_accumulator > 0.3

        if should_jump:
            self.jump_accumulator = 0.0
            self.jumps += 1

        # Dopamine and learning
        if reward != 0:
            self.dopamine = np.clip(self.dopamine + reward * 0.1, 0.0, 1.0)
            self._learn()
        else:
            self.dopamine = 0.5 + 0.1 * (1.0 - signal)

        self.steps += 1
        return should_jump

    def _learn(self):
        """DA-STDP learning"""
        self.syn_sens_hid.update_eligibility(self.sensory.spikes, self.hidden.spikes, tau=500.0, dt=1.0)
        self.syn_sens_hid.apply_dopamine(self.dopamine, a_plus=0.01, a_minus=0.012)
        self.syn_hid_mot.update_eligibility(self.hidden.spikes, self.motor.spikes, tau=500.0, dt=1.0)
        self.syn_hid_mot.apply_dopamine(self.dopamine, a_plus=0.01, a_minus=0.012)

    def reset(self):
        """Reset state for new episode"""
        self.sensory.reset()
        self.hidden.reset()
        self.motor.reset()
        self.jump_accumulator = 0.0


class DinoJSAgent:
    """Agent using JS API for detection + SNN for learning"""

    def __init__(self, config: Optional[DinoConfig] = None):
        self.config = config or DinoConfig()
        self.brain = DinoSNNBrain(self.config)

        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.scores = []

    async def connect(self):
        """Connect to browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=['--disable-gpu', '--force-device-scale-factor=1']
        )
        self.page = await self.browser.new_page(viewport={"width": 800, "height": 400})
        await self.page.goto("https://chromedino.com/")
        await asyncio.sleep(2)
        print("Dino game connected!")

    async def get_game_state(self):
        """Get game state via JS API"""
        return await self.page.evaluate("""() => {
            const r = Runner.instance_;
            if (!r || !r.horizon) return null;
            const obs = r.horizon.obstacles;
            return {
                firstObs: obs.length > 0 ? {
                    x: obs[0].xPos,
                    w: obs[0].width,
                    h: obs[0].typeConfig ? obs[0].typeConfig.height : 0,
                    type: obs[0].typeConfig ? obs[0].typeConfig.type : 'unknown'
                } : null,
                tRexX: r.tRex.xPos,
                tRexY: r.tRex.yPos,
                jumping: r.tRex.jumping,
                ducking: r.tRex.ducking,
                crashed: r.crashed,
                distance: Math.floor(r.distanceRan * 0.025)
            };
        }""")

    async def run_episode(self, max_frames: int = 2000) -> int:
        """Run one episode"""
        # Start game
        await self.page.click('body')
        await self.page.keyboard.press('Space')
        await asyncio.sleep(0.3)

        self.brain.reset()
        last_jump_time = 0
        frame = 0

        while frame < max_frames:
            state = await self.get_game_state()

            if not state:
                frame += 1
                await asyncio.sleep(0.016)
                continue

            # Check game over
            if state['crashed']:
                self.brain.process(0, self.config.death_penalty)
                return state['distance']

            # Calculate gap
            gap = 0
            if state['firstObs']:
                gap = state['firstObs']['x'] - state['tRexX']

            # Debug: log when obstacle is close
            if gap > 0 and gap < 500 and frame % 5 == 0:
                jmp = "AIR" if state['jumping'] else "GND"
                obs_type = state['firstObs'].get('type', '?') if state['firstObs'] else '?'
                obs_h = state['firstObs'].get('h', 0) if state['firstObs'] else 0
                trex_y = state.get('tRexY', 0)
                print(f"  [{frame:3d}] Gap={gap:4.0f} {jmp} Y={trex_y:.0f} Obs:{obs_type}(h={obs_h})")

            # SNN processes the gap signal
            current_time = frame * 16  # ms
            should_jump_snn = self.brain.process(gap, self.config.survival_reward)

            # Check if jump is allowed
            can_jump = (
                not state['jumping'] and
                current_time - last_jump_time > self.config.min_jump_interval
            )

            # Rule-based jump trigger (primary)
            should_jump_rule = can_jump and gap > 0 and gap < self.config.jump_gap

            # SNN can also trigger jump (use same gap threshold)
            should_jump_snn_allowed = can_jump and should_jump_snn and gap > 0 and gap < self.config.jump_gap

            # Combined decision
            if should_jump_rule or should_jump_snn_allowed:
                await self.page.keyboard.press('Space')
                last_jump_time = current_time
                print(f"  >>> JUMP at Gap={gap:.0f}")

            frame += 1
            await asyncio.sleep(0.016)

        return 0

    async def train(self, n_episodes: int = 10):
        """Train the agent"""
        print("\n" + "="*60)
        print("Dino SNN Training (JS API)")
        print("="*60)

        for ep in range(n_episodes):
            print(f"Starting episode {ep+1}...")
            score = await self.run_episode()
            self.scores.append(score)

            high = max(self.scores)
            avg = sum(self.scores[-10:]) / min(len(self.scores), 10)
            print(f"[Ep {ep+1:3d}] Score: {score} | High: {high} | Avg(10): {avg:.0f} | Jumps: {self.brain.jumps}")

            await asyncio.sleep(1)  # Wait before next episode

        print("\n" + "="*60)
        print(f"Training Complete! High: {max(self.scores)}, Final Avg: {sum(self.scores)/len(self.scores):.1f}")
        print("="*60)

    async def close(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()


async def main():
    print("Chrome Dino SNN Agent (JS API)")
    print("Mission: Survive as long as possible!")
    print()

    agent = DinoJSAgent()
    await agent.connect()

    try:
        await agent.train(n_episodes=20)
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
