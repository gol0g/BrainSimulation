"""
Core Brain Architecture - Unified Principles

This module implements the fundamental operating principles
that replace the fragmented module approach.

Philosophy:
- ONE cost function explains all behavior
- Emotions EMERGE from prediction errors and viability state
- Fast (alarm) and slow (deliberation) pathways work together
- Death is not a game rule but an absorbing state (loss of agency)

Components:
- viability.py: Replaces HP with viability kernel concept
- world_model.py: Prediction and error computation
- unified_value.py: Single cost function for all decisions
- alarm.py: Fast pathway for immediate responses
- policy.py: Action selection integrating all components
"""

from .viability import ViabilitySystem, ViabilityState, ViabilityParams
from .world_model import WorldModel, Prediction, PredictionError
from .unified_value import UnifiedValue, CostBreakdown
from .alarm import AlarmSystem, AlarmState, PolicyModulation
from .policy import UnifiedPolicy, ActionDecision

__all__ = [
    # Viability
    'ViabilitySystem', 'ViabilityState', 'ViabilityParams',
    # World Model
    'WorldModel', 'Prediction', 'PredictionError',
    # Unified Value
    'UnifiedValue', 'CostBreakdown',
    # Alarm
    'AlarmSystem', 'AlarmState', 'PolicyModulation',
    # Policy
    'UnifiedPolicy', 'ActionDecision',
]
