"""
v5.9: PC-Z Bridge - Neural Predictive Coding ↔ Z-State Integration

핵심 철학:
- Neural PC의 prediction error가 regime_score로 흘러들어감
- Z-state가 λ_prior (prior precision)를 직접 조절
- Action Circuit의 deliberation budget도 Z-state에 연동

연결 구조:
    [Neural PC] ──error──→ [Regime Score] ──score──→ [Self Model]
         ↑                                                  │
         │                                                  ↓
         └─────────────── λ_prior ←──────────────── [Z-state]
                                                           │
                                                           ↓
    [Action Circuit] ←───── act/learn coupling ←── [Interaction Gating]

z별 λ_prior 조절:
- z=0 (stable): λ=1.0 (normal prior influence)
- z=1 (exploring): λ=0.4 (데이터 의존, prior 약화)
- z=2 (reflecting): λ=1.5 (prior/memory 강화)
- z=3 (fatigued): λ=0.7 (보수적, 약간 prior 의존)

z별 deliberation 조절:
- z=0 (stable): budget=1.0 (normal)
- z=1 (exploring): budget=1.5 (더 많이 숙고)
- z=2 (reflecting): budget=2.0 (최대 숙고)
- z=3 (fatigued): budget=0.5 (빠른 결정)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from collections import deque

from .neural_pc import NeuralPCLayer, PCState
from .action_circuit import ActionCompetitionCircuit
from .self_model import SelfModel, SelfState
from .regime_score import RegimeChangeScore, RegimeScoreState
from .interaction_gating import InteractionGating, GatingModifiers


@dataclass
class PCZBridgeConfig:
    """PC-Z Bridge 설정"""
    # Z → λ_prior 매핑
    lambda_by_z: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,   # stable: normal
        1: 0.4,   # exploring: data-driven
        2: 1.5,   # reflecting: prior-heavy
        3: 0.7,   # fatigued: conservative
    })

    # Z → deliberation budget 매핑
    budget_by_z: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,   # stable: normal
        1: 1.5,   # exploring: think more
        2: 2.0,   # reflecting: max deliberation
        3: 0.5,   # fatigued: quick decisions
    })

    # PC error → regime_score 연결 강도
    error_to_score_weight: float = 0.6

    # Smoothing
    error_ema_alpha: float = 0.3

    # 추가 modulation 범위
    lambda_min: float = 0.2
    lambda_max: float = 2.0


@dataclass
class PCZBridgeState:
    """Bridge 상태"""
    # 현재 modulation 값들
    current_lambda_prior: float = 1.0
    current_budget_modifier: float = 1.0

    # PC error 추적
    error_ema: float = 0.0
    error_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # Z-state 추적
    z_history: deque = field(default_factory=lambda: deque(maxlen=100))

    # 연결 통계
    total_steps: int = 0
    z_changes: int = 0
    last_z: int = 0


class PCZBridge:
    """
    Neural PC ↔ Z-State 연결 모듈

    역할:
    1. PC prediction error → regime_score 입력
    2. Z-state → λ_prior 조절
    3. Z-state → action deliberation budget 조절
    """

    def __init__(
        self,
        config: Optional[PCZBridgeConfig] = None,
        pc_layer: Optional[NeuralPCLayer] = None,
        action_circuit: Optional[ActionCompetitionCircuit] = None,
        self_model: Optional[SelfModel] = None,
        regime_score: Optional[RegimeChangeScore] = None,
        gating: Optional[InteractionGating] = None,
    ):
        self.config = config or PCZBridgeConfig()
        self.state = PCZBridgeState()

        # 연결할 컴포넌트들 (나중에 set 가능)
        self.pc_layer = pc_layer
        self.action_circuit = action_circuit
        self.self_model = self_model
        self.regime_score = regime_score
        self.gating = gating

    def set_components(
        self,
        pc_layer: Optional[NeuralPCLayer] = None,
        action_circuit: Optional[ActionCompetitionCircuit] = None,
        self_model: Optional[SelfModel] = None,
        regime_score: Optional[RegimeChangeScore] = None,
        gating: Optional[InteractionGating] = None,
    ):
        """컴포넌트 연결 (lazy initialization 지원)"""
        if pc_layer:
            self.pc_layer = pc_layer
        if action_circuit:
            self.action_circuit = action_circuit
        if self_model:
            self.self_model = self_model
        if regime_score:
            self.regime_score = regime_score
        if gating:
            self.gating = gating

    def update_from_pc_error(self, pc_state: PCState) -> float:
        """
        PC prediction error를 받아서 regime_score에 전달

        Returns:
            normalized error (0~1)
        """
        # Error norm을 0~1로 정규화
        error_norm = pc_state.error_norm
        normalized_error = np.clip(error_norm / 2.0, 0, 1)  # 2.0 = expected max

        # EMA smoothing
        alpha = self.config.error_ema_alpha
        self.state.error_ema = alpha * normalized_error + (1 - alpha) * self.state.error_ema

        # 히스토리 저장
        self.state.error_history.append(normalized_error)

        # regime_score에 전달 (연결되어 있으면)
        if self.regime_score:
            # regime_score.update()에 error 값 전달
            # 이 값은 regime_score의 error 입력으로 사용됨
            pass  # regime_score는 별도 update 메서드로 호출

        return self.state.error_ema

    def get_lambda_prior_from_z(self, z: int) -> float:
        """
        Z-state에 따른 λ_prior 값 반환

        Args:
            z: current z-state (0~3)

        Returns:
            λ_prior value
        """
        base_lambda = self.config.lambda_by_z.get(z, 1.0)

        # 추가 modulation: error가 높으면 λ 낮춤 (더 데이터 의존)
        error_adjustment = 1.0 - 0.3 * self.state.error_ema

        final_lambda = base_lambda * error_adjustment
        final_lambda = np.clip(
            final_lambda,
            self.config.lambda_min,
            self.config.lambda_max
        )

        self.state.current_lambda_prior = final_lambda
        return final_lambda

    def get_budget_modifier_from_z(self, z: int) -> float:
        """
        Z-state에 따른 deliberation budget modifier 반환

        Args:
            z: current z-state (0~3)

        Returns:
            budget modifier (0.5 ~ 2.0)
        """
        modifier = self.config.budget_by_z.get(z, 1.0)
        self.state.current_budget_modifier = modifier
        return modifier

    def step(
        self,
        pc_state: Optional[PCState] = None,
        z: Optional[int] = None,
        volatility: float = 0.0,
        regret_rate: float = 0.0,
    ) -> Dict:
        """
        한 스텝 업데이트: PC error → regime_score, Z → modulation

        Args:
            pc_state: Neural PC의 현재 상태
            z: 현재 z-state (None이면 self_model에서 가져옴)
            volatility: transition volatility (regime_score용)
            regret_rate: regret spike rate (regime_score용)

        Returns:
            Dict with lambda_prior, budget_modifier, regime_score 등
        """
        self.state.total_steps += 1

        # 1. PC error 처리
        error_normalized = 0.0
        if pc_state:
            error_normalized = self.update_from_pc_error(pc_state)

        # 2. Z-state 가져오기
        if z is None and self.self_model:
            z = self.self_model.state.z
        z = z if z is not None else 0

        # Z 변화 추적
        if z != self.state.last_z:
            self.state.z_changes += 1
        self.state.last_z = z
        self.state.z_history.append(z)

        # 3. Regime score 업데이트 (연결되어 있으면)
        regime_score_value = 0.0
        if self.regime_score:
            # PC error를 regime_score의 prediction_error로 전달
            score_result = self.regime_score.update(
                prediction_error=error_normalized * self.config.error_to_score_weight,
                volatility=volatility,
                regret_spike_rate=regret_rate,
            )
            regime_score_value = score_result.get('score', 0.0)

        # 4. Z → λ_prior
        lambda_prior = self.get_lambda_prior_from_z(z)

        # 5. Z → budget modifier
        budget_modifier = self.get_budget_modifier_from_z(z)

        # 6. Gating modifiers 가져오기 (연결되어 있으면)
        gating_mods = None
        if self.gating:
            gating_mods = self.gating.get_current_modifiers()

        return {
            'lambda_prior': lambda_prior,
            'budget_modifier': budget_modifier,
            'error_normalized': error_normalized,
            'regime_score': regime_score_value,
            'z': z,
            'gating': gating_mods.to_dict() if gating_mods else None,
        }

    def apply_to_pc(self, lambda_prior: float):
        """
        λ_prior를 Neural PC에 적용

        PC의 다음 infer() 호출 시 이 값이 사용되도록 설정
        """
        if self.pc_layer:
            # PC layer의 default lambda_prior 업데이트
            self.pc_layer.config.default_lambda_prior = lambda_prior

    def apply_to_action_circuit(self, budget_modifier: float):
        """
        Budget modifier를 Action Circuit에 적용
        """
        if self.action_circuit:
            # Action circuit의 base budget 조절
            self.action_circuit.config.base_budget = budget_modifier

    def get_diagnostics(self) -> Dict:
        """진단 정보 반환"""
        z_dist = {}
        if self.state.z_history:
            for z in self.state.z_history:
                z_dist[z] = z_dist.get(z, 0) + 1
            total = len(self.state.z_history)
            z_dist = {k: v/total for k, v in z_dist.items()}

        return {
            'total_steps': self.state.total_steps,
            'z_changes': self.state.z_changes,
            'z_change_rate': self.state.z_changes / max(1, self.state.total_steps),
            'z_distribution': z_dist,
            'current_lambda': self.state.current_lambda_prior,
            'current_budget': self.state.current_budget_modifier,
            'error_ema': self.state.error_ema,
            'avg_error': np.mean(list(self.state.error_history)) if self.state.error_history else 0,
        }

    def reset(self):
        """상태 초기화"""
        self.state = PCZBridgeState()


# =============================================================================
# Standalone Test Functions
# =============================================================================

def test_lambda_modulation():
    """Z-state에 따른 λ_prior 변화 테스트"""
    bridge = PCZBridge()

    print("Z-state → λ_prior mapping:")
    for z in range(4):
        lam = bridge.get_lambda_prior_from_z(z)
        z_names = ['stable', 'exploring', 'reflecting', 'fatigued']
        print(f"  z={z} ({z_names[z]}): λ={lam:.2f}")

    # Error 영향 테스트
    print("\nError → λ adjustment:")
    bridge.state.error_ema = 0.0
    print(f"  error=0.0, z=1: λ={bridge.get_lambda_prior_from_z(1):.2f}")

    bridge.state.error_ema = 0.5
    print(f"  error=0.5, z=1: λ={bridge.get_lambda_prior_from_z(1):.2f}")

    bridge.state.error_ema = 1.0
    print(f"  error=1.0, z=1: λ={bridge.get_lambda_prior_from_z(1):.2f}")


def test_budget_modulation():
    """Z-state에 따른 budget 변화 테스트"""
    bridge = PCZBridge()

    print("Z-state → budget modifier mapping:")
    for z in range(4):
        budget = bridge.get_budget_modifier_from_z(z)
        z_names = ['stable', 'exploring', 'reflecting', 'fatigued']
        print(f"  z={z} ({z_names[z]}): budget={budget:.1f}x")


def test_step_integration():
    """통합 step 테스트"""
    from .neural_pc import PCState

    bridge = PCZBridge()

    print("\nStep integration test:")

    # Mock PC state (using correct PCState fields)
    mock_pc_state = PCState(
        mu=np.zeros(16),
        epsilon=np.ones(8) * 0.5,
        o_hat=np.ones(8) * 0.3,
        d_mu=np.zeros(16),
        converged=True,
        iterations=20,
        error_norm=0.8,
        initial_error=1.0,
    )

    for step in range(5):
        z = step % 4  # Cycle through z states
        result = bridge.step(
            pc_state=mock_pc_state,
            z=z,
            volatility=0.2,
            regret_rate=0.1,
        )
        print(f"  Step {step}: z={z}, λ={result['lambda_prior']:.2f}, "
              f"budget={result['budget_modifier']:.1f}x, "
              f"error={result['error_normalized']:.2f}")

    print("\nDiagnostics:")
    diag = bridge.get_diagnostics()
    for k, v in diag.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    print("=" * 60)
    print("  PC-Z Bridge Test Suite")
    print("=" * 60)

    test_lambda_modulation()
    print()
    test_budget_modulation()
    print()
    test_step_integration()
