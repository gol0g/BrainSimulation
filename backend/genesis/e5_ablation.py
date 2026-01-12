"""
E5 Ablation Framework

E5a: Reward Ablation - 보상 구조 변경해도 ratio 유지되는지
E5b: Encoding Ablation - 관측 표현 변경해도 ratio 유지되는지

PASS 기준:
- memory_ratio > 1.15
- hierarchy_ratio > 1.10
- FULL이 BASE를 이기는 비율 유지
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum


class EncodingVariant(Enum):
    """E5b: 관측 인코딩 변형"""
    ORIGINAL = "original"      # 기본 (dx, dy)
    POLAR = "polar"            # (angle, distance)
    ONEHOT_DIST = "onehot"     # direction one-hot + distance
    RANDPROJ = "randproj"      # random linear projection
    PERMFLIP = "permflip"      # permutation + sign flip


class RewardVariant(Enum):
    """E5a: 보상 구조 변형"""
    ORIGINAL = "original"
    NO_DOOR = "no_door"        # Task A: door 보상 제거
    GOAL_ONLY = "goal_only"    # Task A: goal 도달만 보상
    REDUCED_PENALTY = "reduced_penalty"  # Task B: penalty -1.0 → -0.5
    RESET_STYLE = "reset_style"  # Task B: wrong → progress reset


@dataclass
class AblationConfig:
    """Ablation 설정"""
    encoding: EncodingVariant = EncodingVariant.ORIGINAL
    reward: RewardVariant = RewardVariant.ORIGINAL

    # Random projection matrix (RANDPROJ용)
    proj_matrix: Optional[np.ndarray] = None

    # Permutation indices (PERMFLIP용)
    perm_indices: Optional[np.ndarray] = None
    sign_flips: Optional[np.ndarray] = None


class EncodingTransform:
    """
    관측 인코딩 변환기

    E5b' 핵심: semantic-preserving transforms만 사용
    - polar, onehot: 역변환 가능 (의미론 보존)
    - randproj, permflip: 정보 파괴 → E5b'에서 제외
    """

    def __init__(self, config: AblationConfig, obs_dim: int, seed: int = 42):
        self.config = config
        self.obs_dim = obs_dim
        self.rng = np.random.RandomState(seed)

        # Initialize random matrices if needed (for backward compat, but excluded from E5b')
        if config.encoding == EncodingVariant.RANDPROJ and config.proj_matrix is None:
            config.proj_matrix = self.rng.randn(obs_dim, obs_dim).astype(np.float32)
            # Normalize rows
            config.proj_matrix /= np.linalg.norm(config.proj_matrix, axis=1, keepdims=True)

        if config.encoding == EncodingVariant.PERMFLIP:
            if config.perm_indices is None:
                config.perm_indices = self.rng.permutation(obs_dim)
            if config.sign_flips is None:
                config.sign_flips = self.rng.choice([-1, 1], size=obs_dim).astype(np.float32)

    def transform_dxdy(self, dx: float, dy: float) -> Tuple[float, float]:
        """dx, dy 쌍을 변환"""
        if self.config.encoding == EncodingVariant.ORIGINAL:
            return dx, dy

        elif self.config.encoding == EncodingVariant.POLAR:
            # Cartesian → Polar (angle normalized to [-1,1], distance normalized)
            angle = np.arctan2(dy, dx) / np.pi  # [-1, 1]
            distance = np.sqrt(dx**2 + dy**2)
            return angle, distance

        elif self.config.encoding == EncodingVariant.ONEHOT_DIST:
            # Direction as primary axis + distance
            # Return (primary_direction_code, distance)
            # Direction code: 0=N, 0.25=E, 0.5=S, 0.75=W
            if abs(dx) > abs(dy):
                direction = 0.5 if dx > 0 else 0.0  # S or N
            else:
                direction = 0.25 if dy > 0 else 0.75  # E or W
            distance = np.sqrt(dx**2 + dy**2)
            return direction, distance

        else:
            return dx, dy

    def transform_obs(self, obs: np.ndarray) -> np.ndarray:
        """전체 관측 벡터 변환"""
        if self.config.encoding == EncodingVariant.ORIGINAL:
            return obs

        elif self.config.encoding in [EncodingVariant.POLAR, EncodingVariant.ONEHOT_DIST]:
            # Transform each (visible, dx, dy) triplet
            transformed = obs.copy()
            # Indices for dx, dy pairs (assuming structure: [vis, dx, dy, vis, dx, dy, ...])
            dxdy_pairs = [(1, 2), (4, 5), (7, 8), (10, 11)]  # cp_a, cp_b, cp_c, goal
            for dx_idx, dy_idx in dxdy_pairs:
                if dx_idx < len(obs) and dy_idx < len(obs):
                    new_dx, new_dy = self.transform_dxdy(obs[dx_idx], obs[dy_idx])
                    transformed[dx_idx] = new_dx
                    transformed[dy_idx] = new_dy
            return transformed

        elif self.config.encoding == EncodingVariant.RANDPROJ:
            # Random linear projection
            if self.config.proj_matrix is not None:
                # Only transform the first obs_dim elements
                n = min(len(obs), self.config.proj_matrix.shape[0])
                transformed = obs.copy()
                transformed[:n] = self.config.proj_matrix[:n, :n] @ obs[:n]
                return transformed
            return obs

        elif self.config.encoding == EncodingVariant.PERMFLIP:
            # Permutation + sign flip
            transformed = obs.copy()
            if self.config.perm_indices is not None and self.config.sign_flips is not None:
                n = min(len(obs), len(self.config.perm_indices))
                permuted = obs[:n][self.config.perm_indices[:n]]
                transformed[:n] = permuted * self.config.sign_flips[:n]
            return transformed

        return obs

    # ========================================================================
    # E5b' Canonicalization: Inverse transforms (semantic-preserving only)
    # ========================================================================

    def inverse_dxdy(self, v1: float, v2: float) -> Tuple[float, float]:
        """역변환: 변환된 (v1, v2)를 원래 (dx, dy)로 복원"""
        if self.config.encoding == EncodingVariant.ORIGINAL:
            return v1, v2

        elif self.config.encoding == EncodingVariant.POLAR:
            # Polar → Cartesian: (angle, distance) → (dx, dy)
            angle = v1 * np.pi  # angle was normalized to [-1, 1]
            distance = v2
            dx = distance * np.cos(angle)
            dy = distance * np.sin(angle)
            return dx, dy

        elif self.config.encoding == EncodingVariant.ONEHOT_DIST:
            # Direction code + distance → approximate (dx, dy)
            # Direction: 0=N(-1,0), 0.25=E(0,1), 0.5=S(1,0), 0.75=W(0,-1)
            direction = v1
            distance = v2
            if direction < 0.125:  # N
                dx, dy = -distance, 0
            elif direction < 0.375:  # E
                dx, dy = 0, distance
            elif direction < 0.625:  # S
                dx, dy = distance, 0
            else:  # W
                dx, dy = 0, -distance
            return dx, dy

        # randproj, permflip: 역변환 불가 (정보 파괴)
        return v1, v2

    def canonicalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Canonicalization: 어떤 인코딩이든 원래 형식으로 복원

        E5b' 핵심: decode(encode(x)) == x_canon
        """
        if self.config.encoding == EncodingVariant.ORIGINAL:
            return obs

        elif self.config.encoding in [EncodingVariant.POLAR, EncodingVariant.ONEHOT_DIST]:
            # 역변환 가능한 경우만 처리
            canonical = obs.copy()
            # dx, dy pairs 위치 (Task A: indices 1,2,4,5 / Task B: varies)
            dxdy_pairs = [(1, 2), (4, 5), (7, 8), (10, 11)]
            for dx_idx, dy_idx in dxdy_pairs:
                if dx_idx < len(obs) and dy_idx < len(obs):
                    orig_dx, orig_dy = self.inverse_dxdy(obs[dx_idx], obs[dy_idx])
                    canonical[dx_idx] = orig_dx
                    canonical[dy_idx] = orig_dy
            return canonical

        # randproj, permflip: 역변환 불가
        return obs

    def verify_roundtrip(self, original_obs: np.ndarray) -> bool:
        """
        Roundtrip 검증: encode → decode가 원본과 일치하는지

        Returns True if semantic-preserving (polar, onehot)
        Returns False if information-destroying (randproj, permflip)
        """
        if self.config.encoding in [EncodingVariant.RANDPROJ, EncodingVariant.PERMFLIP]:
            return False  # 정보 파괴 인코딩은 roundtrip 불가

        transformed = self.transform_obs(original_obs)
        recovered = self.canonicalize_obs(transformed)

        # 허용 오차 내에서 일치 확인 (부동소수점 오차)
        return np.allclose(original_obs, recovered, atol=1e-5)


class RewardModifier:
    """보상 구조 변환기"""

    def __init__(self, config: AblationConfig):
        self.config = config

    def modify_task_a_config(self, env_config) -> None:
        """Task A (Key-Door) 보상 설정 수정"""
        if self.config.reward == RewardVariant.NO_DOOR:
            # Door 통과 보상 제거 (door_reward가 있다면)
            if hasattr(env_config, 'door_reward'):
                env_config.door_reward = 0.0

        elif self.config.reward == RewardVariant.GOAL_ONLY:
            # Goal 도달만 보상
            if hasattr(env_config, 'key_reward'):
                env_config.key_reward = 0.0
            if hasattr(env_config, 'door_reward'):
                env_config.door_reward = 0.0

    def modify_task_b_config(self, env_config) -> None:
        """Task B (Checkpoint) 보상 설정 수정"""
        if self.config.reward == RewardVariant.REDUCED_PENALTY:
            # Wrong penalty 축소
            if hasattr(env_config, 'wrong_checkpoint_penalty'):
                env_config.wrong_checkpoint_penalty = -0.5

        elif self.config.reward == RewardVariant.RESET_STYLE:
            # Reset style: penalty 대신 보상만 축소
            if hasattr(env_config, 'wrong_checkpoint_penalty'):
                env_config.wrong_checkpoint_penalty = -0.1  # 거의 무시
            if hasattr(env_config, 'correct_checkpoint_reward'):
                env_config.correct_checkpoint_reward = 5.0  # 더 강조


@dataclass
class AblationGateResult:
    """Ablation Gate 결과"""
    passed: bool
    reason: str

    # E5a/b combined metrics
    encoding_variant: str
    reward_variant: str

    # Task A metrics
    task_a_memory_ratio: float
    task_a_full_success: float
    task_a_passed: bool

    # Task B metrics
    task_b_hierarchy_ratio: float
    task_b_full_success: float
    task_b_passed: bool

    # Stability
    wrong_confidence_violations: int


class AblationGate:
    """E5 Ablation Gate"""

    # Thresholds (original보다 약간 낮게 - ablation이니까)
    MIN_MEMORY_RATIO = 1.15
    MIN_HIERARCHY_RATIO = 1.10
    MIN_SUCCESS_RATE = 0.60  # Ablation에서는 60%면 OK

    def evaluate(
        self,
        encoding: EncodingVariant,
        reward: RewardVariant,
        task_a_base_reward: float,
        task_a_mem_reward: float,
        task_a_full_success: float,
        task_b_base_reward: float,
        task_b_full_reward: float,  # Changed from hier_reward - use FULL for combined benefit
        task_b_full_success: float,
        wrong_confidence_violations: int = 0,
    ) -> AblationGateResult:
        """Gate 평가"""

        # Task A: Memory ratio (handle negative rewards)
        # For negative rewards, improvement = (mem - base) / |base| + 1
        # This converts "less negative = better" into ratio form
        if task_a_base_reward > 0:
            memory_ratio = task_a_mem_reward / task_a_base_reward
        elif task_a_base_reward < 0:
            # Improvement-based ratio for negative rewards
            improvement = (task_a_mem_reward - task_a_base_reward) / abs(task_a_base_reward)
            memory_ratio = 1.0 + improvement
        else:
            memory_ratio = 1.0
        task_a_passed = memory_ratio >= self.MIN_MEMORY_RATIO

        # Task B: Hierarchy ratio (FULL vs BASE - combined memory+hierarchy benefit)
        # +HIE alone can't work because it doesn't remember the sequence
        if task_b_base_reward > 0:
            hierarchy_ratio = task_b_full_reward / task_b_base_reward
        elif task_b_base_reward < 0:
            improvement = (task_b_full_reward - task_b_base_reward) / abs(task_b_base_reward)
            hierarchy_ratio = 1.0 + improvement
        else:
            hierarchy_ratio = 1.0
        task_b_passed = hierarchy_ratio >= self.MIN_HIERARCHY_RATIO

        # Overall
        passed = task_a_passed and task_b_passed and wrong_confidence_violations == 0

        if passed:
            reason = "PASS"
        else:
            reasons = []
            if not task_a_passed:
                reasons.append(f"memory_ratio={memory_ratio:.2f}<{self.MIN_MEMORY_RATIO}")
            if not task_b_passed:
                reasons.append(f"hierarchy_ratio={hierarchy_ratio:.2f}<{self.MIN_HIERARCHY_RATIO}")
            if wrong_confidence_violations > 0:
                reasons.append(f"wrong_confidence={wrong_confidence_violations}")
            reason = ", ".join(reasons)

        return AblationGateResult(
            passed=passed,
            reason=reason,
            encoding_variant=encoding.value,
            reward_variant=reward.value,
            task_a_memory_ratio=memory_ratio,
            task_a_full_success=task_a_full_success,
            task_a_passed=task_a_passed,
            task_b_hierarchy_ratio=hierarchy_ratio,
            task_b_full_success=task_b_full_success,
            task_b_passed=task_b_passed,
            wrong_confidence_violations=wrong_confidence_violations,
        )


# Ablation matrix for batch running
ENCODING_VARIANTS = [
    EncodingVariant.ORIGINAL,
    EncodingVariant.POLAR,
    EncodingVariant.ONEHOT_DIST,
    EncodingVariant.RANDPROJ,
    EncodingVariant.PERMFLIP,
]

# E5b' Semantic-preserving encodings only (역변환 가능)
# randproj, permflip은 정보 파괴 → 제외
E5B_PRIME_ENCODINGS = [
    EncodingVariant.ORIGINAL,
    EncodingVariant.POLAR,
    EncodingVariant.ONEHOT_DIST,
]

REWARD_VARIANTS_TASK_A = [
    RewardVariant.ORIGINAL,
    RewardVariant.NO_DOOR,
    RewardVariant.GOAL_ONLY,
]

REWARD_VARIANTS_TASK_B = [
    RewardVariant.ORIGINAL,
    RewardVariant.REDUCED_PENALTY,
    RewardVariant.RESET_STYLE,
]
