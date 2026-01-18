"""
MiniGrid Place Cell Agent - 공간 기억을 가진 SNN (snnTorch)

생물학적 영감:
- Place Cells (해마 CA1): 특정 위치에서만 발화하는 뉴런
- Head Direction Cells (전정핵): 머리 방향을 인코딩
- Grid Cells (내후각피질): 공간의 주기적 패턴 (추후 구현)

아키텍처:
┌─────────────────┐     ┌─────────────────┐
│   Visual Input  │     │ Head Direction  │
│   (7x7 view)    │     │   (4 neurons)   │
│    500 LIF      │     │    100 LIF      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│              Place Cells                │
│         (19x19 = 361 neurons)           │
│      각 뉴런 = 특정 (x,y) 위치          │
└────────────────────┬────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │  Left   │ │ Forward │ │  Right  │
    │ 100 LIF │ │ 100 LIF │ │ 100 LIF │
    └─────────┘ └─────────┘ └─────────┘

핵심 메커니즘:
1. Path Integration: 이동할 때마다 place cell 활성화 업데이트
2. Landmark Correction: 시각 입력으로 위치 오차 보정
3. Goal Memory: 목표를 보면 해당 place cell 강화
4. STDP: place cell → motor 연결 학습
"""

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
import gymnasium as gym
import minigrid  # MiniGrid 환경 등록
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import time
import argparse


# Checkpoint directory
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints" / "minigrid_place_cell"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PlaceCellConfig:
    """Place Cell 에이전트 설정"""
    # Grid dimensions (환경에 맞게 조정)
    grid_width: int = 8
    grid_height: int = 8

    # Population sizes
    n_visual: int = 250          # 시각 입력 뉴런 (7x7 view에 충분)
    n_place_cells: int = 64      # 8x8 place cells
    n_head_direction: int = 40   # 4방향 x 10 뉴런
    n_motor: int = 50            # 각 행동당

    # LIF parameters
    beta: float = 0.9            # membrane decay
    threshold: float = 1.0       # spike threshold

    # STDP parameters
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    a_plus: float = 0.005
    a_minus: float = 0.006
    w_max: float = 1.0
    w_min: float = 0.0

    # WTA (Winner-Take-All) for motors
    wta_inhibition: float = 0.8

    # Place cell parameters
    place_field_sigma: float = 1.5   # place field 크기 (그리드 단위)

    # Dopamine
    dopamine_baseline: float = 0.0
    dopamine_reward: float = 1.0
    dopamine_decay: float = 0.95

    # Simulation
    steps_per_action: int = 10   # 25 → 10 (속도 향상)
    exploration_rate: float = 0.2  # 탐색률 증가

    # Sparse connectivity
    sparsity: float = 0.1


class SparseSynapses:
    """희소 시냅스 연결 (STDP 지원)"""

    def __init__(self, n_pre: int, n_post: int, sparsity: float = 0.1,
                 w_init: float = 0.3, device: str = "cuda"):
        self.n_pre = n_pre
        self.n_post = n_post
        self.device = device

        # 희소 연결 생성
        mask = torch.rand(n_pre, n_post, device=device) < sparsity
        self.indices = mask.nonzero()  # (N, 2) tensor of connections

        # 가중치 초기화
        n_connections = self.indices.shape[0]
        self.weights = torch.ones(n_connections, device=device) * w_init

        # Eligibility trace for STDP
        self.eligibility = torch.zeros(n_connections, device=device)

        # 희소 행렬 캐시
        self._sparse_matrix = None
        self._needs_rebuild = True

    def forward(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """희소 행렬-벡터 곱셈"""
        if self._needs_rebuild:
            self._build_sparse_matrix()

        # pre_spikes: (n_pre,) -> (n_post,)
        return torch.sparse.mm(self._sparse_matrix, pre_spikes.unsqueeze(1)).squeeze(1)

    def _build_sparse_matrix(self):
        """희소 행렬 재구성"""
        indices = self.indices.T  # (2, N)
        self._sparse_matrix = torch.sparse_coo_tensor(
            indices, self.weights,
            size=(self.n_post, self.n_pre),
            device=self.device
        ).coalesce()
        self._needs_rebuild = False

    def update_eligibility(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor,
                          tau: float = 20.0, dt: float = 1.0):
        """STDP eligibility trace 업데이트"""
        decay = np.exp(-dt / tau)
        self.eligibility *= decay

        # 각 연결에 대해 pre/post 스파이크 확인
        pre_idx = self.indices[:, 0]
        post_idx = self.indices[:, 1]

        pre_fired = pre_spikes[pre_idx]
        post_fired = post_spikes[post_idx]

        # LTP: post 발화 시 pre가 발화했으면 양의 적격
        # LTD: pre 발화 시 post가 발화 안 했으면 음의 적격
        self.eligibility += pre_fired * post_fired * 0.1
        self.eligibility -= pre_fired * (1 - post_fired) * 0.05

    def apply_dopamine(self, dopamine: float, a_plus: float = 0.01,
                       a_minus: float = 0.012, w_max: float = 1.0, w_min: float = 0.0):
        """도파민 조절 가소성 적용"""
        delta = dopamine * self.eligibility

        # LTP (양의 변화)
        ltp_mask = delta > 0
        self.weights[ltp_mask] += a_plus * delta[ltp_mask] * (w_max - self.weights[ltp_mask])

        # LTD (음의 변화)
        ltd_mask = delta < 0
        self.weights[ltd_mask] += a_minus * delta[ltd_mask] * (self.weights[ltd_mask] - w_min)

        # 클램핑
        self.weights.clamp_(w_min, w_max)
        self._needs_rebuild = True


class PlaceCellLayer(nn.Module):
    """Place Cell 레이어 - 위치 기반 활성화"""

    def __init__(self, n_cells: int, grid_width: int, grid_height: int,
                 sigma: float = 1.5, beta: float = 0.9, device: str = "cuda"):
        super().__init__()
        self.n_cells = n_cells
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.sigma = sigma
        self.device = device

        # LIF 뉴런
        self.lif = snn.Leaky(beta=beta, threshold=1.0, reset_mechanism="zero")

        # 각 place cell의 선호 위치 (x, y)
        self.preferred_pos = torch.zeros(n_cells, 2, device=device)
        for i in range(n_cells):
            self.preferred_pos[i, 0] = i % grid_width
            self.preferred_pos[i, 1] = i // grid_width

        # 막전위
        self.mem = torch.zeros(n_cells, device=device)

    def get_place_activation(self, pos: torch.Tensor) -> torch.Tensor:
        """위치 기반 Gaussian 활성화 계산"""
        # pos: (2,) tensor [x, y]
        diff = self.preferred_pos - pos.unsqueeze(0)  # (n_cells, 2)
        dist_sq = (diff ** 2).sum(dim=1)  # (n_cells,)
        activation = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        return activation

    def forward(self, external_current: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            external_current: 외부 입력 (visual + head direction)
            pos: 현재 위치 (x, y)
        Returns:
            spikes, membrane potential
        """
        # Place field 활성화를 전류로 변환
        place_current = self.get_place_activation(pos) * 2.0

        # 총 입력 전류
        total_current = external_current + place_current

        # LIF 업데이트
        spk, self.mem = self.lif(total_current, self.mem)
        return spk, self.mem

    def reset(self):
        """막전위 리셋"""
        self.mem = torch.zeros(self.n_cells, device=self.device)


class PlaceCellAgent:
    """해마 Place Cell 기반 공간 탐색 에이전트 (snnTorch)"""

    def __init__(self, config: PlaceCellConfig = None):
        self.config = config or PlaceCellConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        # 도파민 레벨
        self.dopamine = self.config.dopamine_baseline

        # 현재 위치 (에이전트)
        self.current_pos = torch.tensor([9.0, 9.0], device=self.device)
        self.current_dir = 0

        # Goal memory
        self.goal_memory = torch.zeros(self.config.n_place_cells, device=self.device)

        # Stats
        self.episode = 0
        self.total_steps = 0
        self.best_reward = 0

        self._build_network()

    def _build_network(self):
        """snnTorch 네트워크 빌드"""
        print("Building Place Cell SNN (snnTorch)...")

        cfg = self.config

        # === LIF 뉴런 레이어 ===
        # Visual input layer
        self.visual_lif = snn.Leaky(beta=cfg.beta, threshold=cfg.threshold,
                                     reset_mechanism="zero")
        self.visual_mem = torch.zeros(cfg.n_visual, device=self.device)

        # Head direction layer
        self.hd_lif = snn.Leaky(beta=cfg.beta, threshold=cfg.threshold,
                                 reset_mechanism="zero")
        self.hd_mem = torch.zeros(cfg.n_head_direction, device=self.device)

        # Place cell layer (custom)
        self.place_layer = PlaceCellLayer(
            cfg.n_place_cells, cfg.grid_width, cfg.grid_height,
            sigma=cfg.place_field_sigma, beta=cfg.beta, device=self.device
        )

        # Motor layers
        self.motor_lif = {
            "left": snn.Leaky(beta=cfg.beta, threshold=cfg.threshold, reset_mechanism="zero"),
            "forward": snn.Leaky(beta=cfg.beta, threshold=cfg.threshold, reset_mechanism="zero"),
            "right": snn.Leaky(beta=cfg.beta, threshold=cfg.threshold, reset_mechanism="zero")
        }
        self.motor_mem = {
            "left": torch.zeros(cfg.n_motor, device=self.device),
            "forward": torch.zeros(cfg.n_motor, device=self.device),
            "right": torch.zeros(cfg.n_motor, device=self.device)
        }

        # === Synapses ===
        # Visual → Place Cells
        self.syn_visual_place = SparseSynapses(
            cfg.n_visual, cfg.n_place_cells,
            sparsity=cfg.sparsity, w_init=0.3, device=self.device
        )

        # Head Direction → Place Cells
        self.syn_hd_place = SparseSynapses(
            cfg.n_head_direction, cfg.n_place_cells,
            sparsity=cfg.sparsity, w_init=0.3, device=self.device
        )

        # Place Cells → Motors
        self.syn_place_motor = {
            "left": SparseSynapses(cfg.n_place_cells, cfg.n_motor,
                                   sparsity=cfg.sparsity, w_init=0.3, device=self.device),
            "forward": SparseSynapses(cfg.n_place_cells, cfg.n_motor,
                                      sparsity=cfg.sparsity, w_init=0.3, device=self.device),
            "right": SparseSynapses(cfg.n_place_cells, cfg.n_motor,
                                    sparsity=cfg.sparsity, w_init=0.3, device=self.device)
        }

        print(f"  Visual: {cfg.n_visual}")
        print(f"  Head Direction: {cfg.n_head_direction}")
        print(f"  Place Cells: {cfg.n_place_cells}")
        print(f"  Motor (per action): {cfg.n_motor}")
        print(f"  Total neurons: {self._count_neurons()}")
        print("Network built successfully!")

    def _count_neurons(self) -> int:
        """총 뉴런 수"""
        cfg = self.config
        return cfg.n_visual + cfg.n_head_direction + cfg.n_place_cells + cfg.n_motor * 3

    def _encode_visual(self, obs: np.ndarray) -> torch.Tensor:
        """7x7 시각 입력을 스파이크 확률로 인코딩"""
        # obs shape: (7, 7, 3)
        visual_input = obs[:, :, 0].flatten()  # 7x7 = 49 values

        # 뉴런 수에 맞게 확장
        rates = torch.zeros(self.config.n_visual, device=self.device)
        n_repeats = self.config.n_visual // 49

        for i in range(49):
            val = visual_input[i]
            # 목표(8)를 보면 높은 활성화
            if val == 8:  # goal
                rate = 1.0
            elif val == 2:  # wall
                rate = 0.3
            elif val == 1:  # empty
                rate = 0.1
            else:
                rate = 0.0

            for j in range(n_repeats):
                if i * n_repeats + j < self.config.n_visual:
                    rates[i * n_repeats + j] = rate

        return rates

    def _encode_direction(self, direction: int) -> torch.Tensor:
        """방향을 head direction cell 활성화로 인코딩"""
        rates = torch.zeros(self.config.n_head_direction, device=self.device)
        neurons_per_dir = self.config.n_head_direction // 4

        # 현재 방향에 해당하는 뉴런들 활성화
        start = direction * neurons_per_dir
        end = start + neurons_per_dir
        rates[start:end] = 1.0

        return rates

    def act(self, obs: Dict) -> int:
        """관찰에서 행동 선택"""
        cfg = self.config

        # 입력 인코딩
        visual_rates = self._encode_visual(obs["image"])
        hd_rates = self._encode_direction(obs["direction"])

        # 스파이크 카운트 초기화
        motor_spikes = {"left": 0, "forward": 0, "right": 0}

        # 시뮬레이션 스텝 실행
        for step in range(cfg.steps_per_action):
            # 입력 스파이크 생성 (Poisson)
            visual_spk = (torch.rand(cfg.n_visual, device=self.device) < visual_rates * 0.3).float()
            hd_spk = (torch.rand(cfg.n_head_direction, device=self.device) < hd_rates * 0.5).float()

            # Visual LIF
            visual_current = visual_spk * 2.0
            visual_out, self.visual_mem = self.visual_lif(visual_current, self.visual_mem)

            # Head Direction LIF
            hd_current = hd_spk * 2.0
            hd_out, self.hd_mem = self.hd_lif(hd_current, self.hd_mem)

            # Place Cells
            visual_to_place = self.syn_visual_place.forward(visual_out)
            hd_to_place = self.syn_hd_place.forward(hd_out)
            place_input = visual_to_place + hd_to_place
            place_spk, _ = self.place_layer(place_input, self.current_pos)

            # Motor neurons
            for motor_name in ["left", "forward", "right"]:
                motor_input = self.syn_place_motor[motor_name].forward(place_spk)

                # WTA 억제 (다른 모터의 활성화에 따라)
                for other_name in ["left", "forward", "right"]:
                    if other_name != motor_name:
                        other_activity = self.motor_mem[other_name].mean()
                        motor_input -= cfg.wta_inhibition * other_activity

                motor_spk, self.motor_mem[motor_name] = self.motor_lif[motor_name](
                    motor_input, self.motor_mem[motor_name]
                )
                motor_spikes[motor_name] += motor_spk.sum().item()

            # STDP eligibility 업데이트
            for motor_name in ["left", "forward", "right"]:
                motor_activity = self.motor_mem[motor_name] > 0.5
                self.syn_place_motor[motor_name].update_eligibility(
                    place_spk, motor_activity.float(), tau=cfg.tau_plus
                )

        # WTA: 가장 많이 발화한 행동 선택
        max_action = max(motor_spikes, key=motor_spikes.get)

        # 행동 매핑: left=0, right=1, forward=2
        action_map = {"left": 0, "right": 1, "forward": 2}
        action = action_map[max_action]

        # Exploration
        if np.random.random() < cfg.exploration_rate:
            action = np.random.choice([0, 1, 2])

        self.total_steps += 1
        return action

    def update_position(self, new_pos: np.ndarray, new_dir: int):
        """위치 업데이트"""
        self.current_pos = torch.tensor(new_pos, dtype=torch.float32, device=self.device)
        self.current_dir = new_dir

    def reward(self, r: float):
        """보상 처리 (DA-STDP)"""
        if r > 0:
            self.dopamine = self.config.dopamine_reward

            # Goal memory 업데이트
            place_activation = self.place_layer.get_place_activation(self.current_pos)
            self.goal_memory += place_activation * r
            self.goal_memory.clamp_(0, 1)

            # DA-STDP 적용
            for motor_name in ["left", "forward", "right"]:
                self.syn_place_motor[motor_name].apply_dopamine(
                    self.dopamine,
                    a_plus=self.config.a_plus,
                    a_minus=self.config.a_minus,
                    w_max=self.config.w_max,
                    w_min=self.config.w_min
                )
        else:
            self.dopamine *= self.config.dopamine_decay

    def reset(self):
        """에피소드 리셋"""
        self.current_pos = torch.tensor([9.0, 9.0], device=self.device)
        self.current_dir = 0
        self.dopamine = self.config.dopamine_baseline

        # 막전위 리셋
        self.visual_mem.zero_()
        self.hd_mem.zero_()
        self.place_layer.reset()
        for mem in self.motor_mem.values():
            mem.zero_()

        self.episode += 1


def train(episodes: int = 500, render: bool = False, env_name: str = "MiniGrid-FourRooms-v0"):
    """Place Cell 에이전트 훈련"""

    print("=" * 60)
    print("MiniGrid Place Cell Agent Training (snnTorch)")
    print("=" * 60)

    # 환경 생성
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)

    # 에이전트 생성
    agent = PlaceCellAgent()

    # 통계
    episode_rewards = []
    episode_lengths = []
    best_reward = 0
    successes = 0

    print(f"\nEnvironment: {env_name}")
    print(f"Starting training for {episodes} episodes...")
    print("-" * 60)

    start_time = time.time()

    for ep in range(episodes):
        obs, info = env.reset()
        agent.reset()

        total_reward = 0
        steps = 0
        done = False
        truncated = False

        # 초기 위치 설정
        actual_pos = np.array(env.unwrapped.agent_pos)
        actual_dir = env.unwrapped.agent_dir
        agent.update_position(actual_pos, actual_dir)

        while not done and not truncated:
            # 행동 선택
            action = agent.act(obs)

            # 환경 스텝
            obs, reward, done, truncated, info = env.step(action)

            # 위치 업데이트
            actual_pos = np.array(env.unwrapped.agent_pos)
            actual_dir = env.unwrapped.agent_dir
            agent.update_position(actual_pos, actual_dir)

            # 보상 처리
            agent.reward(reward)
            total_reward += reward
            steps += 1

            if render:
                time.sleep(0.02)

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        if total_reward > 0:
            successes += 1

        if total_reward > best_reward:
            best_reward = total_reward

        # 10 에피소드마다 출력
        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_steps = np.mean(episode_lengths[-10:])
            success_rate = successes / (ep + 1) * 100
            elapsed = time.time() - start_time

            print(f"[Ep {ep+1:4d}] Reward: {total_reward:.3f} | "
                  f"Avg(10): {avg_reward:.3f} | Steps: {steps:3d} | "
                  f"Success: {success_rate:.1f}% | Time: {elapsed:.1f}s")

    env.close()

    # 최종 결과
    elapsed = time.time() - start_time
    final_success_rate = successes / episodes * 100

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Episodes: {episodes}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/episodes:.2f}s/ep)")
    print(f"  Best Reward: {best_reward:.3f}")
    print(f"  Success Rate: {final_success_rate:.1f}%")
    print(f"  Final Avg(10): {np.mean(episode_rewards[-10:]):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGrid Place Cell Agent")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-8x8-v0",
                        help="MiniGrid environment name")
    args = parser.parse_args()

    train(episodes=args.episodes, render=args.render, env_name=args.env)
