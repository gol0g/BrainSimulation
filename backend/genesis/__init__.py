"""
Genesis Brain - Free Energy Principle Implementation

This is not a simulation of behaviors.
This is an implementation of the fundamental principle from which all behaviors emerge.

Core equation:
    F = D_KL[Q(s) || P(s|o)] - log P(o)

Simplified:
    F = prediction_error + complexity

Everything else - perception, action, emotion, goal, learning -
is derived from minimizing this single quantity.

NO explicit emotion labels.
NO hardcoded goals.
NO reward signals.

Just F minimization. Everything else emerges.
"""

from .free_energy import FreeEnergyEngine, FreeEnergyState
from .generative_model import GenerativeModel, ModelState
from .inference import InferenceEngine, InferenceResult
from .action_selection import ActionSelector, ActionResult
from .agent import GenesisAgent, AgentState

__all__ = [
    'FreeEnergyEngine',
    'FreeEnergyState',
    'GenerativeModel',
    'ModelState',
    'InferenceEngine',
    'InferenceResult',
    'ActionSelector',
    'ActionResult',
    'GenesisAgent',
    'AgentState'
]
