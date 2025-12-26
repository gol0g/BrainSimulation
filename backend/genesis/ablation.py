"""
Ablation Study Framework - v4.4

G1 Gate에서 각 메커니즘의 기여도를 측정하기 위한 ablation 테스트

테스트 대상:
- Memory (LTM): 장기 기억이 적응에 도움/방해?
- Sleep (Consolidation): 수면이 일반화에 도움/방해?
- THINK: 메타인지가 급변 상황에서 유리?
- Hierarchy: 계층적 컨텍스트가 drift 감지에 도움?
- Regret (v4.4): 반사실적 추론이 학습에 도움?

사용법:
    runner = AblationRunner(agent, world, action_selector)
    results = runner.run_ablation_suite(drift_type='reverse', duration=100)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import time


@dataclass
class AblationConfig:
    """단일 ablation 테스트 설정"""
    name: str
    memory_enabled: bool = False
    sleep_enabled: bool = False
    think_enabled: bool = False
    hierarchy_enabled: bool = False
    regret_enabled: bool = False  # v4.4

    def __str__(self):
        flags = []
        if self.memory_enabled: flags.append("MEM")
        if self.sleep_enabled: flags.append("SLP")
        if self.think_enabled: flags.append("THK")
        if self.hierarchy_enabled: flags.append("HIE")
        if self.regret_enabled: flags.append("REG")
        return f"{self.name}[{','.join(flags) if flags else 'BASE'}]"


@dataclass
class AblationResult:
    """단일 ablation 테스트 결과"""
    config: AblationConfig
    drift_type: str
    duration: int
    seed: int

    # G1 Gate metrics
    pre_food: int = 0
    post_food: int = 0
    peak_std_ratio: float = 1.0
    volatility_auc: float = 0.0
    time_to_recovery: int = -1
    food_rate_pre: float = 0.0
    food_rate_shock: float = 0.0
    food_rate_adapt: float = 0.0

    # 추가 메트릭
    total_steps: int = 0
    passed: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class AblationSuiteResult:
    """전체 ablation suite 결과"""
    drift_type: str
    duration: int
    seed: int
    configs_tested: int
    results: List[AblationResult] = field(default_factory=list)

    # 비교 분석
    best_config: Optional[str] = None
    worst_config: Optional[str] = None
    ranking: List[str] = field(default_factory=list)  # 성능 순 config 이름


# 표준 ablation 설정들
STANDARD_ABLATIONS = [
    AblationConfig("BASE", memory_enabled=False, sleep_enabled=False, think_enabled=False, hierarchy_enabled=False, regret_enabled=False),
    AblationConfig("+MEM", memory_enabled=True, sleep_enabled=False, think_enabled=False, hierarchy_enabled=False, regret_enabled=False),
    AblationConfig("+SLP", memory_enabled=False, sleep_enabled=True, think_enabled=False, hierarchy_enabled=False, regret_enabled=False),
    AblationConfig("+THK", memory_enabled=False, sleep_enabled=False, think_enabled=True, hierarchy_enabled=False, regret_enabled=False),
    AblationConfig("+HIE", memory_enabled=False, sleep_enabled=False, think_enabled=False, hierarchy_enabled=True, regret_enabled=False),
    AblationConfig("+REG", memory_enabled=False, sleep_enabled=False, think_enabled=False, hierarchy_enabled=False, regret_enabled=True),  # v4.4
    AblationConfig("FULL", memory_enabled=True, sleep_enabled=True, think_enabled=True, hierarchy_enabled=True, regret_enabled=True),  # v4.4: regret 추가
    AblationConfig("MEM+SLP", memory_enabled=True, sleep_enabled=True, think_enabled=False, hierarchy_enabled=False, regret_enabled=False),
    AblationConfig("MEM+HIE", memory_enabled=True, sleep_enabled=False, think_enabled=False, hierarchy_enabled=True, regret_enabled=False),
    AblationConfig("MEM+REG", memory_enabled=True, sleep_enabled=False, think_enabled=False, hierarchy_enabled=False, regret_enabled=True),  # v4.4
]


class AblationRunner:
    """
    Ablation 테스트 실행기

    각 config에 대해:
    1. 에이전트 리셋 + config 적용
    2. DRIFT 시나리오 실행
    3. G1 Gate 결과 수집
    4. 비교 분석
    """

    def __init__(self, agent, world, action_selector, scenario_manager):
        self.agent = agent
        self.world = world
        self.action_selector = action_selector
        self.scenario_manager = scenario_manager

    def apply_config(self, config: AblationConfig):
        """에이전트에 ablation config 적용"""
        # Memory
        if config.memory_enabled:
            self.action_selector.enable_memory(store_threshold=0.5)
        else:
            self.action_selector.disable_memory()

        # Sleep (Consolidation)
        if config.sleep_enabled:
            self.action_selector.enable_consolidation(auto_trigger=True)
        else:
            self.action_selector.disable_consolidation()

        # THINK
        if config.think_enabled:
            self.action_selector.enable_think()
        else:
            self.action_selector.disable_think()

        # Hierarchy
        if config.hierarchy_enabled:
            self.action_selector.enable_hierarchy(K=4, update_interval=10)
        else:
            self.action_selector.disable_hierarchy()

        # Regret (v4.4)
        if config.regret_enabled:
            self.action_selector.enable_regret()
        else:
            self.action_selector.disable_regret()

    def run_single_ablation(
        self,
        config: AblationConfig,
        drift_type: str = 'reverse',
        duration: int = 100,
        drift_after: int = 50,
        seed: int = 42
    ) -> AblationResult:
        """
        단일 ablation 테스트 실행

        Returns:
            AblationResult with G1 Gate metrics
        """
        from genesis.reproducibility import set_global_seed

        # 1. 리셋 및 시드 설정
        set_global_seed(seed)
        self.world.reset()
        self.agent.reset()

        # 2. Config 적용
        self.apply_config(config)

        # 3. DRIFT 시나리오 시작
        from genesis.scenarios import ScenarioType
        self.scenario_manager.start_scenario(
            ScenarioType.DRIFT,
            duration=duration,
            drift_after=drift_after,
            drift_type=drift_type
        )

        # 4. 시뮬레이션 실행
        for step in range(duration):
            # Step 실행 (simplified - actual step logic in main_genesis.py)
            self.scenario_manager._drift_step_count += 1
            self.scenario_manager.check_and_activate_drift()

        # 5. 결과 수집
        g1_result = self.scenario_manager.get_g1_gate_result()

        if g1_result is None:
            return AblationResult(
                config=config,
                drift_type=drift_type,
                duration=duration,
                seed=seed,
                total_steps=duration,
                passed=False,
                reasons=["G1 Gate 결과 없음"]
            )

        return AblationResult(
            config=config,
            drift_type=drift_type,
            duration=duration,
            seed=seed,
            pre_food=g1_result.pre_drift_food,
            post_food=g1_result.post_drift_food,
            peak_std_ratio=g1_result.peak_std_ratio,
            volatility_auc=g1_result.volatility_auc,
            time_to_recovery=g1_result.time_to_recovery,
            food_rate_pre=g1_result.food_rate_pre,
            food_rate_shock=g1_result.food_rate_shock,
            food_rate_adapt=g1_result.food_rate_adapt,
            total_steps=duration,
            passed=g1_result.passed,
            reasons=g1_result.reasons
        )

    def run_ablation_suite(
        self,
        drift_type: str = 'reverse',
        duration: int = 100,
        drift_after: int = 50,
        seed: int = 42,
        configs: Optional[List[AblationConfig]] = None
    ) -> AblationSuiteResult:
        """
        전체 ablation suite 실행

        Args:
            drift_type: drift 종류 (rotate, flip_x, flip_y, reverse, partial, delayed, probabilistic)
            duration: 총 스텝 수
            drift_after: drift 시작 스텝
            seed: 재현성용 시드
            configs: 테스트할 config 목록 (None이면 STANDARD_ABLATIONS 사용)

        Returns:
            AblationSuiteResult with all results and ranking
        """
        if configs is None:
            configs = STANDARD_ABLATIONS

        suite_result = AblationSuiteResult(
            drift_type=drift_type,
            duration=duration,
            seed=seed,
            configs_tested=len(configs)
        )

        for config in configs:
            print(f"Running ablation: {config}")
            result = self.run_single_ablation(
                config=config,
                drift_type=drift_type,
                duration=duration,
                drift_after=drift_after,
                seed=seed
            )
            suite_result.results.append(result)

        # 랭킹 계산 (food_rate_adapt 기준)
        sorted_results = sorted(
            suite_result.results,
            key=lambda r: r.food_rate_adapt,
            reverse=True
        )
        suite_result.ranking = [r.config.name for r in sorted_results]
        suite_result.best_config = suite_result.ranking[0] if suite_result.ranking else None
        suite_result.worst_config = suite_result.ranking[-1] if suite_result.ranking else None

        return suite_result


def compute_ablation_contribution(suite_result: AblationSuiteResult) -> Dict[str, float]:
    """
    각 메커니즘의 기여도 계산

    방법: BASE 대비 해당 메커니즘 ON일 때의 성능 변화

    Returns:
        dict: {mechanism_name: contribution_score}
    """
    # BASE 결과 찾기
    base_result = next((r for r in suite_result.results if r.config.name == "BASE"), None)
    if base_result is None:
        return {}

    base_adapt = base_result.food_rate_adapt

    contributions = {}

    # 각 메커니즘별 기여도
    for result in suite_result.results:
        if result.config.name.startswith("+"):
            mechanism = result.config.name[1:]  # "+MEM" -> "MEM"
            improvement = result.food_rate_adapt - base_adapt
            contributions[mechanism] = round(improvement, 4)

    return contributions
