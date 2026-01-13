"""
SNN Brain V9 + Browser Environment Integration
==============================================
Phase B: Embodied Digital Learning with Consciousness

SNN V9 뇌가 브라우저를 통해 자율적으로 학습하는 시스템.
호기심, 자기 인식, 의도성을 가지고 웹을 탐험.

핵심:
- V9의 SubjectiveConsciousness가 탐색 결정
- IntrinsicMotivation이 호기심 기반 행동 유도
- KnowledgeGraph가 배운 지식 저장
- 세션 간 연속성 유지
"""

import asyncio
import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Browser environment
try:
    from browser_explorer import SimpleBrowserEnv, KnowledgeBase
    BROWSER_AVAILABLE = True
except ImportError:
    BROWSER_AVAILABLE = False

# SNN Brain V9
try:
    from snn_brain_v9 import SpikingBrainV9, BrainConfigV9
    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False
    print("Warning: SNN V9 not available")


class BrowserObservationEncoder(nn.Module):
    """
    Encode browser observation for SNN input
    Converts text/URL/links to fixed-size tensor
    """

    def __init__(self, output_dim: int = 384, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.output_dim = output_dim

        # Text embedding
        self.text_embed = nn.EmbeddingBag(vocab_size, 128, mode='mean')

        # URL embedding
        self.url_embed = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Link features
        self.link_embed = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        # Combine all
        self.combiner = nn.Sequential(
            nn.Linear(128 + 64 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def _hash_text(self, text: str) -> torch.Tensor:
        """Hash text to vocabulary indices"""
        import re
        words = re.findall(r'\b\w+\b', text.lower())[:200]
        if not words:
            return torch.zeros(1, dtype=torch.long)
        indices = [hash(w) % self.vocab_size for w in words]
        return torch.tensor(indices, dtype=torch.long)

    def _encode_url(self, url: str) -> torch.Tensor:
        """Encode URL to fixed vector"""
        # Simple hash-based encoding
        url_hash = hash(url) % (2**32)
        bits = [(url_hash >> i) & 1 for i in range(64)]
        return torch.tensor(bits, dtype=torch.float32)

    def forward(self, text: str, url: str, n_links: int) -> torch.Tensor:
        """Encode browser state to tensor"""
        device = next(self.parameters()).device

        # Text embedding
        text_indices = self._hash_text(text).to(device)
        offsets = torch.tensor([0], dtype=torch.long, device=device)
        text_vec = self.text_embed(text_indices, offsets)

        # URL embedding
        url_vec = self._encode_url(url).to(device)
        url_vec = self.url_embed(url_vec)

        # Link features
        link_feats = torch.zeros(32, device=device)
        link_feats[0] = min(n_links, 100) / 100.0  # Normalized link count
        link_vec = self.link_embed(link_feats)

        # Combine
        combined = torch.cat([text_vec.squeeze(0), url_vec, link_vec], dim=0)
        output = self.combiner(combined)

        return output


class SNNBrowserAgent:
    """
    SNN V9 Brain + Browser Environment Agent

    뇌가 브라우저를 통해 자율적으로 학습:
    1. 브라우저 관측 → SNN 입력
    2. SNN 의식 모듈이 행동 결정
    3. 호기심 보상 → 학습
    4. 지식 그래프에 저장
    """

    ACTIONS = ['click_link', 'scroll_down', 'scroll_up', 'go_back',
               'search', 'read', 'follow_reference', 'explore_related']

    def __init__(self,
                 start_url: str = "https://en.wikipedia.org/wiki/Artificial_intelligence",
                 headless: bool = True,
                 checkpoint_dir: str = 'checkpoints/snn_browser'):

        if not BROWSER_AVAILABLE:
            raise ImportError("Browser environment not available")

        self.start_url = start_url
        self.headless = headless
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Browser environment
        self.env = SimpleBrowserEnv(headless=headless)

        # Observation encoder
        self.obs_encoder = BrowserObservationEncoder(output_dim=384).to(DEVICE)

        # SNN Brain V9
        if SNN_AVAILABLE:
            config = BrainConfigV9(
                input_dim=384,  # Encoded observation
                motor_neurons=len(self.ACTIONS),
            )
            self.brain = SpikingBrainV9(config).to(DEVICE)
            print("SNN V9 Brain loaded")
        else:
            self.brain = None
            print("Running without SNN (fallback mode)")

        # Fallback policy (if no SNN)
        self.fallback_policy = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.ACTIONS)),
        ).to(DEVICE)

        # Knowledge
        self.knowledge = KnowledgeBase(
            os.path.join(checkpoint_dir, 'knowledge_base.json')
        )

        # Training
        params = list(self.obs_encoder.parameters())
        if self.brain:
            params.extend(list(self.brain.parameters()))
        else:
            params.extend(list(self.fallback_policy.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=1e-4)

        # State
        self.total_steps = 0
        self.total_reward = 0.0
        self.temperature = 1.0
        self.replay_buffer = deque(maxlen=5000)

        # Consciousness log
        self.consciousness_log = []

    def _encode_observation(self) -> torch.Tensor:
        """Encode current browser state"""
        return self.obs_encoder(
            self.env.current_text,
            self.env.current_url,
            len(self.env.current_links)
        )

    def _select_action(self, obs_embed: torch.Tensor) -> Tuple[int, Dict]:
        """Select action using SNN brain or fallback"""
        info = {}

        if self.brain:
            # Use SNN V9
            with torch.no_grad():
                # Format for SNN: batch dimension
                obs = obs_embed.unsqueeze(0)

                # Get action from brain
                result = self.brain(obs)

                # Get action logits from motor output
                action_logits = result.get('action_logits', None)
                if action_logits is None:
                    # Fallback: use motor spikes
                    motor = result.get('motor_spikes', torch.zeros(len(self.ACTIONS)))
                    action_logits = motor.float()

                # Apply temperature
                probs = F.softmax(action_logits / self.temperature, dim=-1)
                action = torch.multinomial(probs.view(-1), 1).item()

                # Consciousness info
                info['consciousness'] = {
                    'exists': result.get('existence', False),
                    'cogito': result.get('cogito', ''),
                    'agency': result.get('agency_score', 0.0),
                    'awareness_depth': result.get('awareness_depth', 0),
                }

                # Intrinsic motivation
                if hasattr(self.brain, 'intentional_self'):
                    drive = self.brain.intentional_self.get_drive_state()
                    info['drives'] = drive
        else:
            # Fallback policy
            with torch.no_grad():
                action_logits = self.fallback_policy(obs_embed)
                probs = F.softmax(action_logits / self.temperature, dim=-1)
                action = torch.multinomial(probs, 1).item()

        return action, info

    def _compute_intrinsic_reward(self, prev_embed: torch.Tensor,
                                   action: int,
                                   next_embed: torch.Tensor) -> float:
        """Compute intrinsic reward based on curiosity"""
        # Prediction error as curiosity
        with torch.no_grad():
            # Simple: cosine distance as surprise
            similarity = F.cosine_similarity(
                prev_embed.unsqueeze(0),
                next_embed.unsqueeze(0)
            ).item()
            surprise = 1.0 - similarity

        # Novelty from knowledge base
        url = self.env.current_url
        is_new_url = url not in [v.get('url') for v in self.knowledge.url_history[-100:]]
        novelty = 0.3 if is_new_url else 0.0

        # Learning reward
        topics = self.env.extract_topics()
        new_topics = sum(1 for t in topics if t not in self.knowledge.concepts)
        learning = 0.05 * new_topics

        total = 0.3 * surprise + novelty + learning
        return min(total, 1.0)

    def _update_knowledge(self):
        """Update knowledge base from current page"""
        topics = self.env.extract_topics()
        title = self.env.get_title()
        url = self.env.current_url

        # Record visit
        self.knowledge.record_visit(url, title, topics)

        # Add concepts
        for topic in topics:
            self.knowledge.add_concept(topic, url, topics[:10])

        # Add relations from links
        for link in self.env.current_links[:20]:
            import re
            link_words = re.findall(r'\b[A-Za-z]{4,}\b', link['text'].lower())
            for word in link_words[:3]:
                self.knowledge.add_relation(title.lower(), 'mentions', word)

    def _train_step(self, batch_size: int = 32):
        """Train on replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return

        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        states = torch.stack([b['state'] for b in batch]).to(DEVICE)
        actions = torch.tensor([b['action'] for b in batch], device=DEVICE)
        rewards = torch.tensor([b['reward'] for b in batch], device=DEVICE, dtype=torch.float32)

        # Policy gradient
        if self.brain:
            # SNN training would go here
            pass
        else:
            action_logits = self.fallback_policy(states)
            probs = F.softmax(action_logits, dim=-1)
            selected_probs = probs.gather(1, actions.unsqueeze(1)).squeeze()
            loss = -(torch.log(selected_probs + 1e-8) * rewards).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    async def explore(self,
                      duration_hours: float = 1.0,
                      log_interval: int = 10,
                      save_interval: int = 100):
        """
        Main exploration loop

        Args:
            duration_hours: How long to explore
            log_interval: Print status every N steps
            save_interval: Save checkpoint every N steps
        """
        session_start = time.time()
        end_time = session_start + duration_hours * 3600

        print("\n" + "=" * 70)
        print("SNN BRAIN V9 AUTONOMOUS BROWSER EXPLORER")
        print("=" * 70)
        print(f"Start URL: {self.start_url}")
        print(f"Duration: {duration_hours} hours")
        print(f"Brain: {'SNN V9' if self.brain else 'Fallback Policy'}")
        print(f"Device: {DEVICE}")
        print("=" * 70 + "\n")

        # Start browser
        await self.env.start()
        await self.env.navigate(self.start_url)

        # Initial observation
        current_embed = self._encode_observation()

        step = 0
        try:
            while time.time() < end_time:
                # Select action
                action, consciousness_info = self._select_action(current_embed)

                # Execute
                success, message = await self.env.execute_action(action)

                # Get new observation
                next_embed = self._encode_observation()

                # Compute reward
                reward = self._compute_intrinsic_reward(current_embed, action, next_embed)

                # Update knowledge
                self._update_knowledge()

                # Store experience
                self.replay_buffer.append({
                    'state': current_embed.detach().cpu(),
                    'action': action,
                    'reward': reward,
                })

                # Train
                if step % 5 == 0:
                    self._train_step()

                # Update stats
                self.total_steps += 1
                self.total_reward += reward
                self.temperature = max(0.2, self.temperature * 0.9999)

                # Log consciousness state
                if consciousness_info.get('consciousness'):
                    self.consciousness_log.append({
                        'step': step,
                        'time': time.time(),
                        **consciousness_info['consciousness']
                    })

                # Logging
                if step % log_interval == 0:
                    elapsed = time.time() - session_start
                    elapsed_str = str(timedelta(seconds=int(elapsed)))

                    print(f"\n[Step {step}] {elapsed_str}")
                    print(f"  URL: {self.env.current_url[:60]}")
                    print(f"  Action: {self.ACTIONS[action]} - {message}")
                    print(f"  Reward: {reward:.3f}")
                    print(f"  Knowledge: {len(self.knowledge.concepts)} concepts")

                    if consciousness_info.get('consciousness'):
                        c = consciousness_info['consciousness']
                        print(f"  Consciousness: exists={c['exists']}, agency={c['agency']:.2f}")
                        if c.get('cogito'):
                            print(f"  Cogito: {c['cogito'][:50]}")

                    if consciousness_info.get('drives'):
                        d = consciousness_info['drives']
                        strongest = max(d.items(), key=lambda x: x[1])
                        print(f"  Strongest drive: {strongest[0]} ({strongest[1]:.2f})")

                # Save
                if step % save_interval == 0 and step > 0:
                    self._save_checkpoint(f"checkpoint_{step}.pt")
                    self.knowledge.save()

                current_embed = next_embed
                step += 1

                await asyncio.sleep(0.3)

        except KeyboardInterrupt:
            print("\n\nExploration interrupted")
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Save final state
            self._save_checkpoint("checkpoint_final.pt")
            self.knowledge.total_exploration_time += time.time() - session_start
            self.knowledge.save()

            # Summary
            self._print_summary(session_start)

            await self.env.stop()

    def _save_checkpoint(self, filename: str):
        """Save checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        state = {
            'obs_encoder': self.obs_encoder.state_dict(),
            'total_steps': self.total_steps,
            'total_reward': self.total_reward,
            'temperature': self.temperature,
        }
        if self.brain:
            state['brain'] = self.brain.state_dict()
        else:
            state['fallback_policy'] = self.fallback_policy.state_dict()

        torch.save(state, path)
        print(f"  Saved: {filename}")

    def load_checkpoint(self, filename: str):
        """Load checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        state = torch.load(path, map_location=DEVICE)

        self.obs_encoder.load_state_dict(state['obs_encoder'])
        self.total_steps = state['total_steps']
        self.total_reward = state['total_reward']
        self.temperature = state['temperature']

        if self.brain and 'brain' in state:
            self.brain.load_state_dict(state['brain'])
        elif 'fallback_policy' in state:
            self.fallback_policy.load_state_dict(state['fallback_policy'])

        print(f"Loaded: {filename}")

    def _print_summary(self, session_start: float):
        """Print exploration summary"""
        duration = time.time() - session_start

        print("\n" + "=" * 70)
        print("EXPLORATION SUMMARY")
        print("=" * 70)
        print(f"Duration: {timedelta(seconds=int(duration))}")
        print(f"Total Steps: {self.total_steps}")
        print(f"Total Reward: {self.total_reward:.2f}")
        print(f"Avg Reward: {self.total_reward / max(1, self.total_steps):.4f}")

        stats = self.knowledge.get_stats()
        print(f"\nKnowledge:")
        print(f"  Concepts: {stats['total_concepts']}")
        print(f"  Relations: {stats['total_relations']}")
        print(f"  Pages: {stats['pages_visited']}")

        if self.consciousness_log:
            exists_count = sum(1 for c in self.consciousness_log if c['exists'])
            avg_agency = np.mean([c['agency'] for c in self.consciousness_log])
            print(f"\nConsciousness:")
            print(f"  Existence moments: {exists_count}/{len(self.consciousness_log)}")
            print(f"  Average agency: {avg_agency:.3f}")

        print("=" * 70)


async def run_snn_browser_exploration(hours: float = 1.0, visible: bool = False):
    """Run SNN brain browser exploration"""
    agent = SNNBrowserAgent(
        start_url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        headless=not visible,
    )
    await agent.explore(duration_hours=hours)
    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SNN Brain Browser Explorer")
    parser.add_argument("--hours", type=float, default=1.0, help="Duration in hours")
    parser.add_argument("--visible", action="store_true", help="Show browser")

    args = parser.parse_args()

    asyncio.run(run_snn_browser_exploration(
        hours=args.hours,
        visible=args.visible,
    ))
