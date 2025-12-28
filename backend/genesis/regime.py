"""
v4.7 Regime-tagged Memory - Regime Tracker

Core concept: Regime = "which world rules are active"
- Q(r) is a belief distribution over regimes (soft, not hard switch)
- Transition error pattern + persistence determines regime changes
- Hysteresis prevents noise-triggered regime flips

MVP: K=2 regimes (pre-drift / post-drift)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class RegimeState:
    """Current regime belief and detection state"""
    # Q(r) - belief distribution over K regimes
    Q: np.ndarray  # shape (K,)

    # Current dominant regime
    current_regime: int = 0

    # Hysteresis: minimum steps before allowing regime switch
    steps_in_current: int = 0
    min_steps_before_switch: int = 10  # hysteresis threshold

    # Detection signals (running statistics)
    transition_error_ema: float = 0.0
    transition_error_baseline: float = 0.1  # learned baseline
    volatility_ema: float = 0.0

    # Regime switch history
    switch_count: int = 0
    last_switch_step: int = 0

    # Post-switch grace period (extra suppression)
    grace_period_remaining: int = 0
    grace_period_length: int = 15  # steps of extra caution after switch


@dataclass
class RegimeConfig:
    """Configuration for regime tracker"""
    K: int = 2  # number of regimes (MVP: 2 = pre/post drift)

    # Detection thresholds
    spike_threshold: float = 2.0  # error > baseline * threshold = regime change signal
    persistence_required: int = 5  # consecutive spikes needed to trigger switch

    # Hysteresis
    min_steps_before_switch: int = 10
    grace_period_length: int = 15

    # EMA parameters
    error_ema_alpha: float = 0.3  # fast tracking for current error
    baseline_ema_alpha: float = 0.02  # slow tracking for baseline

    # Q(r) update parameters
    belief_update_rate: float = 0.2  # how fast Q(r) shifts
    belief_decay_rate: float = 0.98  # slow decay toward uniform


class RegimeTracker:
    """
    Tracks which "regime" (world rule set) is currently active.

    Uses transition error patterns to detect regime changes:
    - Sudden spike in transition error = world rules changed
    - Persistent high error = confirmed regime change
    - Hysteresis prevents noise-triggered flips

    Q(r) is a soft belief, allowing graceful handling of ambiguous cases.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.state = self._create_initial_state()
        self._spike_counter = 0  # consecutive spike count
        self._step_count = 0

    def _create_initial_state(self) -> RegimeState:
        """Create initial regime state"""
        Q = np.zeros(self.config.K)
        Q[0] = 1.0  # start in regime 0 with full confidence

        return RegimeState(
            Q=Q,
            current_regime=0,
            min_steps_before_switch=self.config.min_steps_before_switch,
            grace_period_length=self.config.grace_period_length
        )

    def update(self, transition_error: float, volatility: float = 0.0) -> Dict:
        """
        Update regime belief based on transition error signal.

        Args:
            transition_error: Mean transition prediction error this step
            volatility: Optional volatility signal

        Returns:
            Dict with update info including any regime switch
        """
        self._step_count += 1
        state = self.state
        cfg = self.config

        # Update EMAs
        state.transition_error_ema = (
            cfg.error_ema_alpha * transition_error +
            (1 - cfg.error_ema_alpha) * state.transition_error_ema
        )
        state.volatility_ema = (
            cfg.error_ema_alpha * volatility +
            (1 - cfg.error_ema_alpha) * state.volatility_ema
        )

        # Update baseline (slow) - only when not in spike
        spike_ratio = state.transition_error_ema / (state.transition_error_baseline + 1e-6)
        is_spike = spike_ratio > cfg.spike_threshold

        if not is_spike:
            state.transition_error_baseline = (
                cfg.baseline_ema_alpha * transition_error +
                (1 - cfg.baseline_ema_alpha) * state.transition_error_baseline
            )
            self._spike_counter = 0
        else:
            self._spike_counter += 1

        # Check for regime switch
        switched = False
        old_regime = state.current_regime

        state.steps_in_current += 1
        if state.grace_period_remaining > 0:
            state.grace_period_remaining -= 1

        # Regime switch conditions:
        # 1. Spike detected AND persistent (N consecutive spikes)
        # 2. Hysteresis satisfied (been in current regime long enough)
        if (self._spike_counter >= cfg.persistence_required and
            state.steps_in_current >= cfg.min_steps_before_switch):

            # Switch to next regime (MVP: toggle between 0 and 1)
            new_regime = (state.current_regime + 1) % cfg.K

            # Update Q(r) - shift belief toward new regime
            state.Q *= (1 - cfg.belief_update_rate)
            state.Q[new_regime] += cfg.belief_update_rate
            state.Q = state.Q / state.Q.sum()  # normalize

            # If belief is strong enough, commit to regime switch
            if state.Q[new_regime] > 0.6:
                state.current_regime = new_regime
                state.steps_in_current = 0
                state.switch_count += 1
                state.last_switch_step = self._step_count
                state.grace_period_remaining = cfg.grace_period_length

                # Reset baseline for new regime
                state.transition_error_baseline = state.transition_error_ema
                self._spike_counter = 0
                switched = True
        else:
            # No switch - slowly reinforce current regime belief
            state.Q[state.current_regime] = min(
                0.95,
                state.Q[state.current_regime] + 0.01
            )
            # Decay others
            for r in range(cfg.K):
                if r != state.current_regime:
                    state.Q[r] *= cfg.belief_decay_rate
            state.Q = state.Q / state.Q.sum()

        return {
            'switched': switched,
            'old_regime': old_regime if switched else None,
            'new_regime': state.current_regime if switched else None,
            'is_spike': is_spike,
            'spike_ratio': spike_ratio,
            'spike_counter': self._spike_counter,
            'Q': state.Q.tolist(),
            'current_regime': state.current_regime,
            'grace_period_remaining': state.grace_period_remaining,
            'in_grace_period': state.grace_period_remaining > 0
        }

    def get_current_regime(self) -> int:
        """Get current dominant regime (argmax Q)"""
        return self.state.current_regime

    def get_regime_belief(self) -> np.ndarray:
        """Get full Q(r) distribution"""
        return self.state.Q.copy()

    def get_recall_weight_modifier(self) -> float:
        """
        Get recall weight modifier based on regime state.

        Returns lower weight during:
        - Grace period after switch (most dangerous time)
        - Low confidence in current regime
        """
        state = self.state

        # Base modifier from regime confidence
        confidence = state.Q[state.current_regime]
        base_modifier = 0.5 + 0.5 * confidence  # 0.5 to 1.0

        # Extra reduction during grace period
        if state.grace_period_remaining > 0:
            grace_factor = state.grace_period_remaining / self.config.grace_period_length
            grace_modifier = 0.3 + 0.7 * (1 - grace_factor)  # 0.3 to 1.0
            return base_modifier * grace_modifier

        return base_modifier

    def should_store_memory(self) -> Tuple[bool, int]:
        """
        Determine if memory should be stored and to which regime bank.

        During grace period, memories are still stored but marked
        with the new regime.

        Returns:
            (should_store, regime_id)
        """
        # Don't store during first few steps of grace period
        # (too uncertain which regime we're really in)
        if self.state.grace_period_remaining > self.config.grace_period_length - 3:
            return False, self.state.current_regime

        return True, self.state.current_regime

    def reset(self):
        """Reset regime tracker to initial state"""
        self.state = self._create_initial_state()
        self._spike_counter = 0
        self._step_count = 0

    def get_status(self) -> Dict:
        """Get full status for API/debugging"""
        state = self.state
        return {
            'enabled': True,
            'K': self.config.K,
            'current_regime': state.current_regime,
            'Q': state.Q.tolist(),
            'confidence': float(state.Q[state.current_regime]),
            'steps_in_current': state.steps_in_current,
            'switch_count': state.switch_count,
            'last_switch_step': state.last_switch_step,
            'grace_period_remaining': state.grace_period_remaining,
            'in_grace_period': state.grace_period_remaining > 0,
            'transition_error_ema': state.transition_error_ema,
            'transition_error_baseline': state.transition_error_baseline,
            'spike_counter': self._spike_counter,
            'recall_weight_modifier': self.get_recall_weight_modifier(),
            'config': {
                'spike_threshold': self.config.spike_threshold,
                'persistence_required': self.config.persistence_required,
                'min_steps_before_switch': self.config.min_steps_before_switch,
                'grace_period_length': self.config.grace_period_length
            }
        }
