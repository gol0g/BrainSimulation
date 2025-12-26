"""
v3.7: Reproducibility System - Seed Management & Testing

재현성 보장을 위한 시드 관리:
1. np.random과 random 모듈 동시 고정
2. 시뮬레이션 상태 해시로 재현성 검증
3. 재현성 테스트 유틸리티

용도:
- 동일 seed → 동일 결과 보장
- 디버깅: 특정 상황 재현
- 벤치마크: 공정한 비교
"""

import numpy as np
import random
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any
import json


@dataclass
class SimulationFingerprint:
    """시뮬레이션 상태 지문 - 재현성 검증용"""
    seed: int
    step_count: int
    total_food: int
    total_deaths: int
    final_energy: float
    agent_pos: tuple
    avg_G: float
    action_counts: Dict[int, int]

    def to_hash(self) -> str:
        """상태를 해시로 변환 (비교용)"""
        # 부동소수점은 소수점 4자리까지만 비교
        data = {
            'seed': self.seed,
            'step_count': self.step_count,
            'total_food': self.total_food,
            'total_deaths': self.total_deaths,
            'final_energy': round(self.final_energy, 4),
            'agent_pos': self.agent_pos,
            'avg_G': round(self.avg_G, 4),
            'action_counts': self.action_counts
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()[:12]


class SeedManager:
    """
    중앙 시드 관리자

    사용법:
        seed_mgr = SeedManager()
        seed_mgr.set_seed(42)  # 모든 랜덤 소스 고정

        # 나중에 같은 seed로 복원
        seed_mgr.set_seed(42)  # 동일한 시퀀스 재시작
    """

    def __init__(self):
        self.current_seed: Optional[int] = None
        self._initial_seed: Optional[int] = None

    def set_seed(self, seed: int):
        """모든 랜덤 소스의 시드 고정"""
        self.current_seed = seed
        self._initial_seed = seed

        # NumPy 시드 고정
        np.random.seed(seed)

        # Python random 시드 고정
        random.seed(seed)

    def get_seed(self) -> Optional[int]:
        """현재 시드 반환"""
        return self.current_seed

    def reset_to_initial(self):
        """초기 시드로 리셋 (동일 시퀀스 재시작)"""
        if self._initial_seed is not None:
            self.set_seed(self._initial_seed)

    def get_status(self) -> Dict:
        """상태 조회"""
        return {
            'current_seed': self.current_seed,
            'initial_seed': self._initial_seed,
            'is_set': self.current_seed is not None
        }


# 글로벌 시드 매니저 인스턴스
_global_seed_manager = SeedManager()


def set_global_seed(seed: int):
    """글로벌 시드 설정 (모든 랜덤 소스)"""
    _global_seed_manager.set_seed(seed)


def get_global_seed() -> Optional[int]:
    """현재 글로벌 시드 반환"""
    return _global_seed_manager.get_seed()


def get_seed_manager() -> SeedManager:
    """글로벌 시드 매니저 인스턴스 반환"""
    return _global_seed_manager


@dataclass
class ReproducibilityTestResult:
    """재현성 테스트 결과"""
    passed: bool
    seed: int
    n_runs: int
    fingerprints: List[str]
    details: Dict[str, Any]
    message: str


def run_reproducibility_test(
    agent,
    world,
    action_selector,
    seed: int = 42,
    n_steps: int = 100,
    n_runs: int = 3
) -> ReproducibilityTestResult:
    """
    재현성 테스트 실행

    동일 seed로 n_runs번 시뮬레이션을 돌려서
    모든 결과가 동일한지 검증

    Args:
        agent: GenesisAgent 인스턴스
        world: World 인스턴스
        action_selector: ActionSelector 인스턴스
        seed: 테스트용 시드
        n_steps: 각 실행당 스텝 수
        n_runs: 반복 실행 횟수

    Returns:
        ReproducibilityTestResult
    """
    fingerprints = []
    all_results = []

    for run_idx in range(n_runs):
        # 시드 리셋
        set_global_seed(seed)

        # 월드 & 에이전트 리셋
        world.reset()
        world.reset_statistics()
        agent.reset()

        # 시뮬레이션 실행
        action_counts = {}
        G_values = []
        last_action = 0

        for step in range(n_steps):
            obs = world.get_observation()

            if step == 0:
                state = agent.step(obs)
            else:
                state = agent.step_with_action(obs, last_action)

            action = int(state.action)
            action_counts[action] = action_counts.get(action, 0) + 1

            # G 값 기록
            if hasattr(state, 'G_decomposition') and action in state.G_decomposition:
                G_values.append(state.G_decomposition[action].G)

            # 행동 실행
            outcome = world.execute_action(action)

            # 학습
            obs_after = world.get_observation()
            action_selector.update_transition_model(action, obs, obs_after)

            last_action = action

        # 지문 생성
        fingerprint = SimulationFingerprint(
            seed=seed,
            step_count=world.step_count,
            total_food=world.total_food,
            total_deaths=world.total_deaths,
            final_energy=world.energy,
            agent_pos=tuple(world.agent_pos),
            avg_G=np.mean(G_values) if G_values else 0.0,
            action_counts=action_counts
        )

        fingerprints.append(fingerprint.to_hash())
        all_results.append(asdict(fingerprint))

    # 모든 지문이 동일한지 확인
    unique_fingerprints = set(fingerprints)
    passed = len(unique_fingerprints) == 1

    if passed:
        message = f"✓ 재현성 테스트 통과: {n_runs}회 실행 모두 동일 (hash: {fingerprints[0]})"
    else:
        message = f"✗ 재현성 테스트 실패: {len(unique_fingerprints)}개의 서로 다른 결과"

    return ReproducibilityTestResult(
        passed=passed,
        seed=seed,
        n_runs=n_runs,
        fingerprints=fingerprints,
        details={
            'unique_count': len(unique_fingerprints),
            'runs': all_results
        },
        message=message
    )


def compare_fingerprints(fp1: SimulationFingerprint, fp2: SimulationFingerprint) -> Dict[str, Any]:
    """두 지문 비교 (디버깅용)"""
    differences = {}

    if fp1.step_count != fp2.step_count:
        differences['step_count'] = (fp1.step_count, fp2.step_count)
    if fp1.total_food != fp2.total_food:
        differences['total_food'] = (fp1.total_food, fp2.total_food)
    if fp1.total_deaths != fp2.total_deaths:
        differences['total_deaths'] = (fp1.total_deaths, fp2.total_deaths)
    if abs(fp1.final_energy - fp2.final_energy) > 0.0001:
        differences['final_energy'] = (fp1.final_energy, fp2.final_energy)
    if fp1.agent_pos != fp2.agent_pos:
        differences['agent_pos'] = (fp1.agent_pos, fp2.agent_pos)
    if abs(fp1.avg_G - fp2.avg_G) > 0.0001:
        differences['avg_G'] = (fp1.avg_G, fp2.avg_G)
    if fp1.action_counts != fp2.action_counts:
        differences['action_counts'] = (fp1.action_counts, fp2.action_counts)

    return {
        'identical': len(differences) == 0,
        'differences': differences
    }
