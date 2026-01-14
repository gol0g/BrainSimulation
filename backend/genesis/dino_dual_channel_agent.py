"""
Chrome Dino Dual-Channel Vision Agent

Ground Eye + Sky Eye architecture with inhibitory circuits.
- Ground Eye: Detects low obstacles (cacti) → triggers JUMP
- Sky Eye: Detects high obstacles (birds) → triggers DUCK + INHIBITS JUMP

This solves the "Wall of Despair" at 600+ points where birds appear.
"""

import asyncio
import numpy as np
import torch
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from playwright.async_api import async_playwright, Page, Browser

from snn_scalable import ScalableSNNConfig, SparseSynapses, SparseLIFLayer, DEVICE

# Checkpoint directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "dino_dual"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DualChannelConfig:
    """Dual-channel agent configuration"""
    # Brain architecture - two sensory channels
    n_ground_eye: int = 500      # Ground sensor (cacti detection)
    n_sky_eye: int = 500         # Sky sensor (bird detection)
    n_hidden: int = 2000         # Processing layer
    n_motor_jump: int = 300      # Jump motor neurons
    n_motor_duck: int = 300      # Duck motor neurons

    # Timing parameters
    jump_gap: int = 100          # Jump when obstacle is this far
    duck_gap: int = 180          # Duck when bird is this far (earlier than jump!)
    min_action_interval: int = 250  # ms between actions

    # Height threshold to distinguish birds vs cacti
    bird_height_threshold: int = 50  # Y position above this = bird territory

    # Learning
    survival_reward: float = 0.1
    death_penalty: float = -5.0

    # Inhibition strength (Sky Eye → Jump inhibition)
    inhibition_strength: float = 0.8

    # SNN
    sparsity: float = 0.01


class DualChannelBrain:
    """
    Dual-Channel SNN Brain with ISOLATED Pathways + Inhibitory Circuits

    Architecture (Two Parallel Pathways):

    ┌─────────────┐                    ┌─────────────┐
    │  Ground Eye │                    │   Sky Eye   │
    │  (Cacti)    │                    │  (Birds)    │
    └──────┬──────┘                    └──────┬──────┘
           │                                  │
           ▼                                  ▼
    ┌──────────────┐                   ┌──────────────┐
    │ Ground Hidden│                   │  Sky Hidden  │
    └──────┬───────┘                   └──────┬───────┘
           │                                  │
           ▼                                  ▼
    ┌──────────────┐    INHIBIT        ┌──────────────┐
    │  Jump Motor  │◄──────────────────│  Duck Motor  │
    └──────────────┘                   └──────────────┘

    Key: Sky pathway can INHIBIT Jump pathway (bird = don't jump)
    """

    def __init__(self, config: DualChannelConfig):
        self.config = config

        lif_config = ScalableSNNConfig(beta=0.9, threshold=1.0)

        # === GROUND PATHWAY (Cacti → Jump) ===
        self.ground_eye = SparseLIFLayer(config.n_ground_eye, lif_config)
        self.ground_hidden = SparseLIFLayer(config.n_hidden // 2, lif_config)  # Dedicated hidden
        self.motor_jump = SparseLIFLayer(config.n_motor_jump, lif_config)

        # === SKY PATHWAY (Birds → Duck) ===
        self.sky_eye = SparseLIFLayer(config.n_sky_eye, lif_config)
        self.sky_hidden = SparseLIFLayer(config.n_hidden // 2, lif_config)  # Dedicated hidden
        self.motor_duck = SparseLIFLayer(config.n_motor_duck, lif_config)

        # === GROUND PATHWAY SYNAPSES ===
        self.syn_ground_ghid = SparseSynapses(
            config.n_ground_eye, config.n_hidden // 2, config.sparsity
        )
        self.syn_ghid_jump = SparseSynapses(
            config.n_hidden // 2, config.n_motor_jump, config.sparsity
        )

        # === SKY PATHWAY SYNAPSES ===
        self.syn_sky_shid = SparseSynapses(
            config.n_sky_eye, config.n_hidden // 2, config.sparsity
        )
        self.syn_shid_duck = SparseSynapses(
            config.n_hidden // 2, config.n_motor_duck, config.sparsity
        )

        # === INHIBITORY CROSS-CONNECTION ===
        # Sky Eye → Jump Motor (direct inhibition) - bird detection suppresses jumping
        self.syn_sky_jump_inhib = SparseSynapses(
            config.n_sky_eye, config.n_motor_jump, config.sparsity * 2
        )

        # State tracking
        self.dopamine = 0.5
        self.jump_accumulator = 0.0
        self.duck_accumulator = 0.0
        self.sky_inhibition = 0.0  # Current inhibition level
        self.steps = 0
        self.jumps = 0
        self.ducks = 0

        # Statistics
        self.sky_activations = 0
        self.ground_activations = 0
        self.inhibitions_applied = 0

        total_neurons = (config.n_ground_eye + config.n_sky_eye +
                        config.n_hidden + config.n_motor_jump + config.n_motor_duck)
        print(f"DualChannelBrain initialized (ISOLATED PATHWAYS):")
        print(f"  Ground Pathway: Eye({config.n_ground_eye}) → Hidden({config.n_hidden//2}) → Jump({config.n_motor_jump})")
        print(f"  Sky Pathway:    Eye({config.n_sky_eye}) → Hidden({config.n_hidden//2}) → Duck({config.n_motor_duck})")
        print(f"  Total neurons:  {total_neurons}")
        print(f"  Cross-inhibit:  Sky Eye --| Jump Motor (strength={config.inhibition_strength})")

    def process(self, ground_signal: float, sky_signal: float, reward: float = 0.0) -> tuple[bool, bool]:
        """
        Process dual-channel input through ISOLATED pathways and decide action

        Args:
            ground_signal: 0-1 signal from ground (lower obstacles - cacti)
            sky_signal: 0-1 signal from sky (upper obstacles - birds)
            reward: Learning signal

        Returns:
            (should_jump, should_duck)
        """
        # === GROUND EYE INPUT ===
        ground_input = torch.zeros(self.config.n_ground_eye, device=DEVICE)
        if ground_signal > 0:
            n_active = int(ground_signal * self.config.n_ground_eye * 0.5)
            active_idx = torch.randperm(self.config.n_ground_eye)[:n_active]
            ground_input[active_idx] = 1.0
            self.ground_activations += 1

        # === SKY EYE INPUT ===
        sky_input = torch.zeros(self.config.n_sky_eye, device=DEVICE)
        if sky_signal > 0:
            n_active = int(sky_signal * self.config.n_sky_eye * 0.6)
            active_idx = torch.randperm(self.config.n_sky_eye)[:n_active]
            sky_input[active_idx] = 1.0
            self.sky_activations += 1

        # ========== GROUND PATHWAY (Cacti → Jump) ==========
        ground_spikes = self.ground_eye.forward(ground_input * 10.0)
        ghid_input = self.syn_ground_ghid.forward(ground_spikes)
        ghid_spikes = self.ground_hidden.forward(ghid_input * 50.0)
        jump_input = self.syn_ghid_jump.forward(ghid_spikes)

        # ========== SKY PATHWAY (Birds → Duck) ==========
        sky_spikes = self.sky_eye.forward(sky_input * 12.0)
        shid_input = self.syn_sky_shid.forward(sky_spikes)
        shid_spikes = self.sky_hidden.forward(shid_input * 50.0)
        duck_input = self.syn_shid_duck.forward(shid_spikes)

        # ========== CROSS-INHIBITION: Sky Eye --| Jump Motor ==========
        sky_inhibition_input = self.syn_sky_jump_inhib.forward(sky_spikes)
        inhibition_magnitude = sky_inhibition_input.sum().item() / self.config.n_motor_jump
        self.sky_inhibition = inhibition_magnitude * self.config.inhibition_strength

        if self.sky_inhibition > 0.1:
            self.inhibitions_applied += 1

        # Apply inhibition to jump pathway
        jump_input_final = jump_input * 50.0 - sky_inhibition_input * self.config.inhibition_strength * 100.0
        jump_input_final = torch.clamp(jump_input_final, min=0)

        # ========== MOTOR OUTPUT ==========
        jump_spikes = self.motor_jump.forward(jump_input_final)
        duck_spikes = self.motor_duck.forward(duck_input * 50.0)

        # === ACCUMULATORS ===
        jump_rate = jump_spikes.sum().item() / self.config.n_motor_jump
        duck_rate = duck_spikes.sum().item() / self.config.n_motor_duck

        self.jump_accumulator = self.jump_accumulator * 0.7 + jump_rate * 0.4
        self.duck_accumulator = self.duck_accumulator * 0.7 + duck_rate * 0.4

        # === REFLEX PATHWAYS (Superior Colliculus) ===
        # Ground reflex → Jump (only if no sky inhibition)
        if ground_signal > 0.5 and self.sky_inhibition < 0.3:
            self.jump_accumulator += ground_signal * 0.5

        # Sky reflex → Duck (and boost inhibition)
        if sky_signal > 0.4:
            self.duck_accumulator += sky_signal * 0.6
            self.jump_accumulator *= (1.0 - sky_signal * 0.7)  # Strong jump suppression

        # === DECISIONS ===
        should_jump = self.jump_accumulator > 0.35 and self.sky_inhibition < 0.4
        should_duck = self.duck_accumulator > 0.35  # Raised threshold (was 0.3)

        # Duck takes priority (mutual exclusion)
        if should_duck:
            should_jump = False

        # Reset accumulators on action
        if should_jump:
            self.jump_accumulator = 0.0
            self.jumps += 1
        if should_duck:
            self.duck_accumulator = 0.0
            self.ducks += 1

        # === LEARNING ===
        if reward != 0:
            self.dopamine = np.clip(self.dopamine + reward * 0.1, 0.0, 1.0)
            self._learn()
        else:
            self.dopamine = 0.5 + 0.1 * (1.0 - max(ground_signal, sky_signal))

        self.steps += 1
        return should_jump, should_duck

    def _learn(self):
        """DA-STDP learning for all synapses in both pathways"""
        # === GROUND PATHWAY ===
        # Ground Eye → Ground Hidden
        self.syn_ground_ghid.update_eligibility(
            self.ground_eye.spikes, self.ground_hidden.spikes, tau=500.0, dt=1.0
        )
        self.syn_ground_ghid.apply_dopamine(self.dopamine, a_plus=0.01, a_minus=0.012)

        # Ground Hidden → Jump Motor
        self.syn_ghid_jump.update_eligibility(
            self.ground_hidden.spikes, self.motor_jump.spikes, tau=500.0, dt=1.0
        )
        self.syn_ghid_jump.apply_dopamine(self.dopamine, a_plus=0.01, a_minus=0.012)

        # === SKY PATHWAY ===
        # Sky Eye → Sky Hidden
        self.syn_sky_shid.update_eligibility(
            self.sky_eye.spikes, self.sky_hidden.spikes, tau=500.0, dt=1.0
        )
        self.syn_sky_shid.apply_dopamine(self.dopamine, a_plus=0.01, a_minus=0.012)

        # Sky Hidden → Duck Motor
        self.syn_shid_duck.update_eligibility(
            self.sky_hidden.spikes, self.motor_duck.spikes, tau=500.0, dt=1.0
        )
        self.syn_shid_duck.apply_dopamine(self.dopamine, a_plus=0.01, a_minus=0.012)

        # === CROSS-INHIBITION ===
        # Sky Eye → Jump Motor (Inhibitory)
        self.syn_sky_jump_inhib.update_eligibility(
            self.sky_eye.spikes, self.motor_jump.spikes, tau=500.0, dt=1.0
        )
        # Invert dopamine for inhibitory synapse (strengthen when surviving bird encounter)
        inhib_da = 1.0 - self.dopamine if self.dopamine < 0.5 else self.dopamine
        self.syn_sky_jump_inhib.apply_dopamine(inhib_da, a_plus=0.008, a_minus=0.01)

    def reset(self):
        """Reset state for new episode"""
        # Ground pathway
        self.ground_eye.reset()
        self.ground_hidden.reset()
        self.motor_jump.reset()
        # Sky pathway
        self.sky_eye.reset()
        self.sky_hidden.reset()
        self.motor_duck.reset()
        # Accumulators
        self.jump_accumulator = 0.0
        self.duck_accumulator = 0.0
        self.sky_inhibition = 0.0

    def get_stats(self) -> dict:
        """Get brain statistics"""
        return {
            'ground_activations': self.ground_activations,
            'sky_activations': self.sky_activations,
            'inhibitions_applied': self.inhibitions_applied,
            'jumps': self.jumps,
            'ducks': self.ducks,
            'steps': self.steps
        }

    def save(self, path: Path):
        """Save all synapse weights"""
        state = {
            'syn_ground_ghid': self.syn_ground_ghid.weights.cpu(),
            'syn_ghid_jump': self.syn_ghid_jump.weights.cpu(),
            'syn_sky_shid': self.syn_sky_shid.weights.cpu(),
            'syn_shid_duck': self.syn_shid_duck.weights.cpu(),
            'syn_sky_jump_inhib': self.syn_sky_jump_inhib.weights.cpu(),
            'stats': self.get_stats(),
        }
        torch.save(state, path)
        print(f"  Model saved: {path}")

    def load(self, path: Path):
        """Load synapse weights"""
        if not path.exists():
            print(f"  No checkpoint found: {path}")
            return False
        state = torch.load(path, map_location=DEVICE)
        self.syn_ground_ghid.weights = state['syn_ground_ghid'].to(DEVICE)
        self.syn_ghid_jump.weights = state['syn_ghid_jump'].to(DEVICE)
        self.syn_sky_shid.weights = state['syn_sky_shid'].to(DEVICE)
        self.syn_shid_duck.weights = state['syn_shid_duck'].to(DEVICE)
        self.syn_sky_jump_inhib.weights = state['syn_sky_jump_inhib'].to(DEVICE)
        print(f"  Model loaded: {path}")
        if 'stats' in state:
            print(f"  Stats: {state['stats']}")
        return True


class DinoDualChannelAgent:
    """Chrome Dino Agent with Dual-Channel Vision"""

    def __init__(self, config: Optional[DualChannelConfig] = None):
        self.config = config or DualChannelConfig()
        self.brain = DualChannelBrain(self.config)

        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.scores = []
        self.best_score = 0

    def save_model(self, name: str = "model"):
        """Save model to checkpoint"""
        path = CHECKPOINT_DIR / f"{name}.pt"
        self.brain.save(path)

    def load_model(self, name: str = "model") -> bool:
        """Load model from checkpoint"""
        path = CHECKPOINT_DIR / f"{name}.pt"
        return self.brain.load(path)

    def save_best(self, score: int):
        """Save if this is the best score"""
        if score > self.best_score:
            self.best_score = score
            path = CHECKPOINT_DIR / f"best_{score}.pt"
            self.brain.save(path)
            # Also save as 'best.pt' for easy loading
            self.brain.save(CHECKPOINT_DIR / "best.pt")
            return True
        return False

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
        """Get game state with obstacle height information"""
        return await self.page.evaluate("""() => {
            const r = Runner.instance_;
            if (!r || !r.horizon) return null;
            const obs = r.horizon.obstacles;
            return {
                firstObs: obs.length > 0 ? {
                    x: obs[0].xPos,
                    y: obs[0].yPos,  // Y position for height detection
                    w: obs[0].width,
                    h: obs[0].typeConfig ? obs[0].typeConfig.height : 0,
                    type: obs[0].typeConfig ? obs[0].typeConfig.type : 'unknown'
                } : null,
                tRexX: r.tRex.xPos,
                tRexY: r.tRex.yPos,
                jumping: r.tRex.jumping,
                ducking: r.tRex.ducking,
                crashed: r.crashed,
                distance: Math.floor(r.distanceRan * 0.025),
                speed: r.currentSpeed
            };
        }""")

    async def run_episode(self, max_frames: int = 5000) -> int:
        """Run one episode"""
        # Start game
        await self.page.click('body')
        await self.page.keyboard.press('Space')
        await asyncio.sleep(0.3)

        self.brain.reset()
        last_action_time = 0
        frame = 0
        is_ducking = False

        while frame < max_frames:
            state = await self.get_game_state()

            if not state:
                frame += 1
                await asyncio.sleep(0.016)
                continue

            # Check game over
            if state['crashed']:
                self.brain.process(0, 0, self.config.death_penalty)
                return state['distance']

            # Speed adaptation
            distance = state['distance']
            speed_scale = 1.0 + distance / 1000.0
            speed_scale = min(speed_scale, 2.0)

            # Calculate signals for both channels
            ground_signal = 0.0
            sky_signal = 0.0

            if state['firstObs']:
                obs = state['firstObs']
                gap = obs['x'] - state['tRexX']
                obs_y = obs.get('y', 100)  # Y position (lower = higher on screen)
                obs_type = str(obs.get('type', '')).upper()

                # Determine if obstacle is in sky (bird) or ground (cactus)
                is_bird = 'PTERO' in obs_type

                # Debug output
                if gap > 0 and gap < 400 and frame % 15 == 0:
                    status = "AIR" if state['jumping'] else ("DCK" if state.get('ducking') else "GND")
                    channel = "SKY" if is_bird else "GND"
                    print(f"  [{frame:3d}] Gap={gap:4.0f} Y={obs_y:3.0f} {obs_type[:6]:6s} CH={channel} {status} spd={speed_scale:.2f}")

                if gap > 0:
                    # Calculate signal strength (closer = stronger)
                    base_signal = max(0, 1.0 - gap / 500.0)

                    if is_bird:
                        # Bird detected → Sky Eye
                        sky_signal = base_signal * 1.2  # Boost for urgency
                        # Still give some ground signal for mixed scenarios
                        ground_signal = base_signal * 0.2
                    else:
                        # Cactus detected → Ground Eye
                        ground_signal = base_signal
                        sky_signal = 0.0

            # Process through brain
            current_time = frame * 16  # ms
            should_jump, should_duck = self.brain.process(
                ground_signal, sky_signal, self.config.survival_reward
            )

            # Adjust timing based on speed
            adjusted_jump_gap = int(self.config.jump_gap * speed_scale)
            adjusted_duck_gap = int(self.config.duck_gap * speed_scale)

            # Action conditions
            can_act = current_time - last_action_time > self.config.min_action_interval

            # Release duck if ducking
            if is_ducking and (should_jump or not should_duck):
                await self.page.keyboard.up('ArrowDown')
                is_ducking = False

            # Execute actions
            if state['firstObs']:
                gap = state['firstObs']['x'] - state['tRexX']
                obs_type = str(state['firstObs'].get('type', '')).upper()
                is_bird = 'PTERO' in obs_type

                # DUCK for birds (priority)
                if is_bird and should_duck and can_act and not state['jumping']:
                    if gap > 0 and gap < adjusted_duck_gap:
                        await self.page.keyboard.down('ArrowDown')
                        is_ducking = True
                        last_action_time = current_time
                        print(f"  >>> DUCK at Gap={gap:.0f} (SKY EYE triggered)")

                # JUMP for cacti (only if not inhibited by sky)
                elif not is_bird and should_jump and can_act:
                    if not state['jumping'] and not state.get('ducking', False):
                        if gap > 0 and gap < adjusted_jump_gap:
                            await self.page.keyboard.press('Space')
                            last_action_time = current_time
                            if frame % 5 == 0:
                                inhib = self.brain.sky_inhibition
                                print(f"  >>> JUMP at Gap={gap:.0f} (GND EYE, inhib={inhib:.2f})")

            frame += 1
            await asyncio.sleep(0.016)

        return 0

    async def train(self, n_episodes: int = 100, resume: bool = False):
        """Train the agent"""
        print("\n" + "="*60)
        print("Dino DUAL-CHANNEL Training (Ground Eye + Sky Eye)")
        print("="*60)

        # Resume from best model if requested
        if resume:
            if self.load_model("best"):
                print("  Resumed from best model!")

        for ep in range(n_episodes):
            print(f"\nStarting episode {ep+1}...")
            score = await self.run_episode()
            self.scores.append(score)

            # Auto-save best model
            if self.save_best(score):
                print(f"  ★ NEW BEST! Saved model (score={score})")

            high = max(self.scores)
            avg = sum(self.scores[-10:]) / min(len(self.scores), 10)
            stats = self.brain.get_stats()

            print(f"[Ep {ep+1:3d}] Score: {score} | High: {high} | Avg(10): {avg:.0f}")
            print(f"         GND={stats['ground_activations']} SKY={stats['sky_activations']} INH={stats['inhibitions_applied']} J={stats['jumps']} D={stats['ducks']}")

            await asyncio.sleep(1)

        # Final save
        self.save_model("final")

        print("\n" + "="*60)
        print(f"Training Complete!")
        print(f"  High Score: {max(self.scores)}")
        print(f"  Final Avg:  {sum(self.scores)/len(self.scores):.1f}")
        stats = self.brain.get_stats()
        print(f"  Total Jumps: {stats['jumps']}, Ducks: {stats['ducks']}")
        print(f"  Sky Inhibitions: {stats['inhibitions_applied']}")
        print(f"  Models saved to: {CHECKPOINT_DIR}")
        print("="*60)

    async def close(self):
        """Close browser"""
        if self.browser:
            await self.browser.close()


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from best model')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode (load best, no training)')
    args = parser.parse_args()

    print("Chrome Dino DUAL-CHANNEL Agent")
    print("Architecture: Ground Eye (cacti) + Sky Eye (birds)")
    print("Key Feature: Sky Eye --| Jump Motor (Inhibitory)")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print()

    agent = DinoDualChannelAgent()

    if args.eval:
        # Evaluation mode
        agent.load_model("best")
        await agent.connect()
        try:
            score = await agent.run_episode()
            print(f"\nEvaluation Score: {score}")
        finally:
            await agent.close()
    else:
        # Training mode
        await agent.connect()
        try:
            await agent.train(n_episodes=args.episodes, resume=args.resume)
        finally:
            await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
