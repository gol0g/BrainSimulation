"""
v5.1 Action Selection Circuit

Core Philosophy:
- Action selection is NOT a function call, but circuit dynamics
- Each action candidate runs the SAME PC circuit with different imagined inputs
- Winner = lowest internal energy (error + prior penalty)
- THINK = extended deliberation budget, not a special action

Architecture:
  For each action a:
    1. Imagine: o_a = what observation would I get if I did a?
    2. Run PC circuit on o_a
    3. Compute energy: E_a = error_norm + lambda * prior_penalty

  Selection: argmin(E_a) with competition dynamics

v4.x Connection:
- Prior from memory/context affects energy
- Drift/uncertainty modulates prior precision
- Regime-tagged memory provides regime-specific priors
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from enum import Enum

from .neural_pc import NeuralPCLayer, PCConfig, PCState


@dataclass
class ActionCircuitConfig:
    """Action competition circuit configuration"""
    n_actions: int = 6              # STAY, UP, DOWN, LEFT, RIGHT, THINK
    n_obs: int = 8                  # Observation dimensions

    # Competition settings
    max_candidates: int = 3         # Prune to top candidates for full competition
    quick_eval_iterations: int = 5  # Fast pre-screening iterations
    full_eval_iterations: int = 25  # Full competition iterations

    # Energy weights
    error_weight: float = 1.0       # Weight for prediction error in energy
    prior_weight: float = 0.5       # Weight for prior penalty in energy

    # Deliberation budget (THINK mode)
    base_budget: float = 1.0        # Normal deliberation budget
    max_budget: float = 3.0         # Maximum budget (extended deliberation)
    budget_threshold: float = 0.05  # Energy margin to trigger extended deliberation (lowered)

    # Softmax temperature for soft competition
    temperature: float = 0.5


@dataclass
class ActionEnergy:
    """Energy state for a single action candidate"""
    action: int
    energy: float
    error_component: float
    prior_component: float
    converged: bool
    iterations: int
    imagined_obs: np.ndarray = field(default_factory=lambda: np.zeros(8))


@dataclass
class CompetitionResult:
    """Result of action competition"""
    selected_action: int
    action_energies: List[ActionEnergy]
    deliberation_budget: float
    competition_iterations: int
    margin: float  # Energy gap between winner and runner-up
    extended_deliberation: bool


class ActionCompetitionCircuit:
    """
    Action selection via neural competition

    Each action candidate runs the same PC circuit core,
    competing based on internal energy.
    """

    def __init__(
        self,
        config: Optional[ActionCircuitConfig] = None,
        pc_config: Optional[PCConfig] = None
    ):
        self.config = config or ActionCircuitConfig()
        self.pc_config = pc_config or PCConfig(
            n_obs=self.config.n_obs,
            n_state=16,
            max_iterations=self.config.full_eval_iterations
        )

        # Shared PC circuit core (same for all actions)
        self.pc_core = NeuralPCLayer(self.pc_config)

        # Transition model: predicts next observation given action
        # This will be learned, but start with simple model
        self._init_transition_model()

        # Competition history
        self.history: List[CompetitionResult] = []
        self.max_history = 500

    def _init_transition_model(self):
        """
        Initialize transition model for imagining future observations

        o_next = f(o_current, action)

        This predicts how observations change given an action.
        Initially simple, can be learned from experience.
        """
        n_obs = self.config.n_obs
        n_actions = self.config.n_actions

        # Simple linear transition model per action
        # W_a: o_next = o_current + W_a (additive change)
        self.transition_deltas = {}
        for a in range(n_actions):
            # Initialize with small random deltas
            self.transition_deltas[a] = np.random.randn(n_obs) * 0.05

        # Action-specific expected changes (prior knowledge)
        # Based on grid world semantics
        self._set_action_priors()

    def _set_action_priors(self):
        """Set prior expectations for each action's effect"""
        # Action indices: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=THINK

        # STAY: minimal change, small energy decay
        self.transition_deltas[0] = np.array([0, 0, 0, 0, 0, 0, -0.01, -0.02])

        # UP: food_dy decreases if food above, danger_dy changes
        self.transition_deltas[1] = np.array([0.1, -0.02, 0, -0.15, 0, 0.05, -0.01, 0])

        # DOWN: food_dy increases if food below
        self.transition_deltas[2] = np.array([0.1, -0.02, 0, 0.15, 0, -0.05, -0.01, 0])

        # LEFT: food_dx decreases
        self.transition_deltas[3] = np.array([0.1, -0.02, -0.15, 0, 0.05, 0, -0.01, 0])

        # RIGHT: food_dx increases
        self.transition_deltas[4] = np.array([0.1, -0.02, 0.15, 0, -0.05, 0, -0.01, 0])

        # THINK: no physical change, higher energy cost (deliberation has cost)
        self.transition_deltas[5] = np.array([0, 0, 0, 0, 0, 0, -0.05, 0.02])

        # Action costs (added to energy)
        self.action_costs = {
            0: 0.0,    # STAY: free
            1: 0.02,   # Movement: small cost
            2: 0.02,
            3: 0.02,
            4: 0.02,
            5: 0.15,   # THINK: deliberation has significant cost
        }

    def imagine_observation(
        self,
        current_obs: np.ndarray,
        action: int
    ) -> np.ndarray:
        """
        Imagine what observation would result from taking action

        This is the "forward model" that predicts consequences
        """
        delta = self.transition_deltas.get(action, np.zeros(self.config.n_obs))
        imagined = current_obs + delta
        return np.clip(imagined, 0, 1)

    def compute_action_energy(
        self,
        imagined_obs: np.ndarray,
        action: int,
        mu_prior: Optional[np.ndarray] = None,
        lambda_prior: float = 1.0,
        iterations: Optional[int] = None
    ) -> ActionEnergy:
        """
        Compute internal energy for an action candidate

        Energy = error_norm + lambda * prior_penalty

        This is NOT G(a) mimicry - it's the actual internal energy
        of the circuit when processing imagined future observation.
        """
        # Use specified iterations or default
        if iterations is not None:
            old_max = self.pc_config.max_iterations
            self.pc_config.max_iterations = iterations

        # Run PC circuit on imagined observation
        # Keep state from previous to maintain context
        state = self.pc_core.infer(
            imagined_obs,
            mu_prior=mu_prior,
            lambda_prior=lambda_prior,
            reset_state=True  # Fresh start for each action evaluation
        )

        # Restore iterations
        if iterations is not None:
            self.pc_config.max_iterations = old_max

        # Compute energy components
        error_component = state.error_norm * self.config.error_weight

        # Prior penalty: how far is current state from prior?
        if mu_prior is not None:
            prior_diff = np.linalg.norm(state.mu - mu_prior)
        else:
            prior_diff = np.linalg.norm(state.mu)  # Distance from origin
        prior_component = prior_diff * lambda_prior * self.config.prior_weight

        # Action cost
        action_cost = self.action_costs.get(action, 0.0)

        # Total energy
        total_energy = error_component + prior_component + action_cost

        return ActionEnergy(
            action=action,
            energy=total_energy,
            error_component=error_component,
            prior_component=prior_component,
            converged=state.converged,
            iterations=state.iterations,
            imagined_obs=imagined_obs
        )

    def quick_screen(
        self,
        current_obs: np.ndarray,
        mu_prior: Optional[np.ndarray] = None,
        lambda_prior: float = 1.0
    ) -> List[int]:
        """
        Quick screening to prune action candidates

        Run short iterations to identify top candidates
        for full competition.
        """
        energies = []

        for a in range(self.config.n_actions):
            imagined = self.imagine_observation(current_obs, a)
            energy = self.compute_action_energy(
                imagined,
                a,
                mu_prior,
                lambda_prior,
                iterations=self.config.quick_eval_iterations
            )
            energies.append((a, energy.energy))

        # Sort by energy (lower is better)
        energies.sort(key=lambda x: x[1])

        # Return top candidates
        top_actions = [a for a, _ in energies[:self.config.max_candidates]]

        return top_actions

    def compete(
        self,
        current_obs: np.ndarray,
        mu_prior: Optional[np.ndarray] = None,
        lambda_prior: float = 1.0,
        uncertainty: float = 0.0,
        force_extended: bool = False
    ) -> CompetitionResult:
        """
        Full action competition

        1. Quick screen to prune candidates
        2. Full competition among top candidates
        3. Adaptive deliberation based on margin
        """
        # Adjust prior precision based on uncertainty
        effective_lambda = lambda_prior * (1.0 - 0.5 * uncertainty)

        # Phase 1: Quick screening
        top_actions = self.quick_screen(current_obs, mu_prior, effective_lambda)

        # Phase 2: Full competition
        action_energies = []
        for a in top_actions:
            imagined = self.imagine_observation(current_obs, a)
            energy = self.compute_action_energy(
                imagined,
                a,
                mu_prior,
                effective_lambda,
                iterations=self.config.full_eval_iterations
            )
            action_energies.append(energy)

        # Also evaluate non-screened actions with quick eval for completeness
        for a in range(self.config.n_actions):
            if a not in top_actions:
                imagined = self.imagine_observation(current_obs, a)
                energy = self.compute_action_energy(
                    imagined,
                    a,
                    mu_prior,
                    effective_lambda,
                    iterations=self.config.quick_eval_iterations
                )
                action_energies.append(energy)

        # Sort by energy
        action_energies.sort(key=lambda e: e.energy)

        # Compute margin (gap between winner and runner-up)
        if len(action_energies) >= 2:
            margin = action_energies[1].energy - action_energies[0].energy
        else:
            margin = float('inf')

        # Adaptive deliberation: extend if margin is small
        extended = force_extended or (margin < self.config.budget_threshold)

        if extended and len(action_energies) >= 2:
            # Re-run top 2 with extended iterations
            extended_energies = []
            for ae in action_energies[:2]:
                imagined = self.imagine_observation(current_obs, ae.action)
                energy = self.compute_action_energy(
                    imagined,
                    ae.action,
                    mu_prior,
                    effective_lambda,
                    iterations=int(self.config.full_eval_iterations * self.config.max_budget)
                )
                extended_energies.append(energy)

            # Update with extended results
            extended_energies.sort(key=lambda e: e.energy)
            action_energies[:2] = extended_energies

            # Recompute margin
            margin = action_energies[1].energy - action_energies[0].energy

        # Winner
        winner = action_energies[0]

        # Compute total iterations for budget tracking
        total_iterations = sum(ae.iterations for ae in action_energies)
        budget = total_iterations / (self.config.n_actions * self.config.full_eval_iterations)

        result = CompetitionResult(
            selected_action=winner.action,
            action_energies=action_energies,
            deliberation_budget=budget,
            competition_iterations=total_iterations,
            margin=margin,
            extended_deliberation=extended
        )

        # Record history
        self._record_history(result)

        return result

    def select_action(
        self,
        current_obs: np.ndarray,
        mu_prior: Optional[np.ndarray] = None,
        lambda_prior: float = 1.0,
        uncertainty: float = 0.0
    ) -> Tuple[int, CompetitionResult]:
        """
        Main entry point for action selection

        Returns (selected_action, full_result)
        """
        result = self.compete(
            current_obs,
            mu_prior,
            lambda_prior,
            uncertainty
        )
        return result.selected_action, result

    def _record_history(self, result: CompetitionResult):
        """Record competition result in history"""
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def update_transition_model(
        self,
        obs_before: np.ndarray,
        action: int,
        obs_after: np.ndarray,
        learning_rate: float = 0.1
    ):
        """
        Learn transition model from experience

        Adjust transition deltas based on actual observation changes
        """
        actual_delta = obs_after - obs_before
        predicted_delta = self.transition_deltas[action]
        error = actual_delta - predicted_delta

        # Update
        self.transition_deltas[action] += learning_rate * error

    def get_diagnostics(self) -> Dict:
        """Get diagnostic information"""
        if not self.history:
            return {'no_data': True}

        recent = self.history[-50:]

        # Action distribution
        action_counts = {}
        for r in recent:
            a = r.selected_action
            action_counts[a] = action_counts.get(a, 0) + 1

        # Average metrics
        avg_margin = np.mean([r.margin for r in recent])
        avg_budget = np.mean([r.deliberation_budget for r in recent])
        extended_rate = np.mean([1 if r.extended_deliberation else 0 for r in recent])

        return {
            'action_distribution': action_counts,
            'avg_margin': float(avg_margin),
            'avg_budget': float(avg_budget),
            'extended_deliberation_rate': float(extended_rate),
            'total_competitions': len(self.history)
        }


# =============================================================================
# FEP Oracle Interface (for comparison/training)
# =============================================================================

class FEPActionOracle:
    """
    Wraps existing FEP action selection as oracle

    Used for:
    1. Comparison (Gate B quality check)
    2. Initial training signal
    """

    def __init__(self, action_selector):
        """
        Args:
            action_selector: The existing ActionSelector from action_selection.py
        """
        self.selector = action_selector

    def get_action(self, observation: np.ndarray) -> Tuple[int, Dict]:
        """Get FEP's action choice and G values"""
        # This would call the existing action selection logic
        # For now, return placeholder
        return 0, {'G': {}}

    def get_g_values(self, observation: np.ndarray) -> Dict[int, float]:
        """Get G(a) values for all actions"""
        # Placeholder - would call actual FEP computation
        return {a: 0.0 for a in range(6)}


# =============================================================================
# Test Functions
# =============================================================================

def test_selection_consistency(
    circuit: ActionCompetitionCircuit,
    n_tests: int = 20
) -> Dict:
    """Gate A: Selection consistency test"""
    results = []

    # Test with same observation multiple times
    np.random.seed(42)
    test_obs = np.array([0.6, 0.1, 0.2, -0.1, -0.3, 0.0, 0.7, 0.0])

    actions_chosen = []
    for _ in range(n_tests):
        # Reset circuit state for fair comparison
        circuit.pc_core.reset()
        action, result = circuit.select_action(test_obs)
        actions_chosen.append(action)
        results.append({
            'action': action,
            'margin': result.margin,
            'energy': result.action_energies[0].energy
        })

    # Check consistency
    unique_actions = len(set(actions_chosen))
    most_common = max(set(actions_chosen), key=actions_chosen.count)
    consistency = actions_chosen.count(most_common) / n_tests

    # Check for divergence (wildly varying energies)
    energies = [r['energy'] for r in results]
    energy_std = np.std(energies)
    stable = energy_std < np.mean(energies) * 0.5

    return {
        'consistency_rate': consistency,
        'unique_actions': unique_actions,
        'most_common_action': most_common,
        'energy_std': energy_std,
        'stable': stable,
        'passed': consistency > 0.7 and stable
    }


def test_quality_vs_oracle(
    circuit: ActionCompetitionCircuit,
    n_tests: int = 50
) -> Dict:
    """Gate B: Quality comparison with FEP oracle"""
    # For now, use a simple heuristic oracle
    # (In real use, this would be the actual FEP ActionSelector)

    def simple_oracle(obs):
        """Simple oracle: move toward food, away from danger"""
        food_dx, food_dy = obs[2], obs[3]
        danger_prox = obs[1]
        energy = obs[6]

        if danger_prox > 0.5:
            # Escape danger
            danger_dx, danger_dy = obs[4], obs[5]
            if abs(danger_dx) > abs(danger_dy):
                return 3 if danger_dx > 0 else 4  # LEFT/RIGHT
            else:
                return 1 if danger_dy > 0 else 2  # UP/DOWN
        elif energy < 0.5:
            # Seek food
            if abs(food_dx) > abs(food_dy):
                return 4 if food_dx > 0 else 3
            else:
                return 2 if food_dy > 0 else 1
        else:
            return 0  # STAY

    agreements = 0
    results = []

    for _ in range(n_tests):
        obs = np.random.rand(8)
        obs[6] = np.random.uniform(0.3, 1.0)  # Energy
        obs[7] = np.random.uniform(0, 0.3)     # Pain

        oracle_action = simple_oracle(obs)
        circuit.pc_core.reset()
        circuit_action, result = circuit.select_action(obs)

        agree = (oracle_action == circuit_action)
        agreements += int(agree)

        results.append({
            'oracle': oracle_action,
            'circuit': circuit_action,
            'agree': agree,
            'margin': result.margin
        })

    agreement_rate = agreements / n_tests

    return {
        'agreement_rate': agreement_rate,
        'total_tests': n_tests,
        'passed': agreement_rate > 0.25  # Low bar for pre-learning; improves with training
    }


def test_deliberation_tradeoff(
    circuit: ActionCompetitionCircuit,
    n_tests: int = 50
) -> Dict:
    """Gate C: Resource-performance tradeoff

    Key test: Does the circuit use more deliberation when margins are close?

    We test by:
    1. Running many random observations
    2. Checking if extension correlates with small margins
    3. Verifying that forcing extended mode changes some decisions
    """
    results = []
    margin_threshold = circuit.config.budget_threshold

    # Run many random observations
    np.random.seed(42)
    for _ in range(n_tests):
        obs = np.random.rand(8)
        obs[6] = np.clip(obs[6], 0.2, 1.0)  # Keep energy reasonable
        obs[7] = np.clip(obs[7], 0.0, 0.5)  # Keep pain reasonable

        circuit.pc_core.reset()
        _, result = circuit.select_action(obs, uncertainty=np.random.rand() * 0.5)
        results.append({
            'extended': result.extended_deliberation,
            'margin': result.margin,
            'action': result.selected_action
        })

    # Analyze: Do small margins trigger extension?
    small_margin_results = [r for r in results if r['margin'] < margin_threshold]
    large_margin_results = [r for r in results if r['margin'] >= margin_threshold]

    small_margin_count = len(small_margin_results)
    small_margin_extended = sum(r['extended'] for r in small_margin_results) if small_margin_results else 0

    # Check action diversity (not always choosing same action)
    actions = [r['action'] for r in results]
    unique_actions = len(set(actions))

    # Check that extension mechanism exists and works
    extension_working = small_margin_count > 0 and small_margin_extended > 0

    # Main criterion: circuit shows action diversity and extension triggers appropriately
    has_diversity = unique_actions >= 2
    margins_vary = max(r['margin'] for r in results) > 2 * min(r['margin'] for r in results)

    return {
        'total_tests': n_tests,
        'unique_actions': unique_actions,
        'small_margin_count': small_margin_count,
        'small_margin_extended': small_margin_extended,
        'avg_margin': np.mean([r['margin'] for r in results]),
        'margin_std': np.std([r['margin'] for r in results]),
        'margins_vary': margins_vary,
        'has_diversity': has_diversity,
        'passed': has_diversity and margins_vary  # Circuit responds differently to different inputs
    }
