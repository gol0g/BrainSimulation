"""
Unified Policy - Final Action Selection

Core Concept:
Policy is the bridge between evaluation and action.
It integrates:
- Fast pathway (alarm): Immediate bias toward survival
- Slow pathway (deliberation): Careful cost evaluation
- Learning: SNN weights encode past experience

Two modes of operation:
1. REACTIVE: Alarm high → alarm modulation dominates
2. DELIBERATIVE: Alarm low → cost evaluation dominates

Why not just use cost function?
- Cost function tells us what's good/bad
- Policy tells us HOW to select actions
- The "how" includes:
  - Exploration vs exploitation
  - Attention allocation
  - When to think more vs act fast

Output:
- Selected action
- Action source (reactive/deliberative/exploratory)
- Explanation (which cost component drove decision)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import math

from .alarm import AlarmSystem, PolicyModulation
from .unified_value import UnifiedValue, CostBreakdown
from .world_model import WorldModel
from .viability import ViabilitySystem


@dataclass
class ActionDecision:
    """Complete action decision with explanation."""
    action: str              # 'up', 'down', 'left', 'right', 'stay'
    action_code: int         # 0-4 for compatibility
    source: str              # 'reactive', 'deliberative', 'exploratory'
    confidence: float        # How confident in this decision
    explanation: str         # Korean explanation
    scores: Dict[str, float] # Per-action scores

    # Internal details
    alarm_influence: float   # How much alarm affected decision
    cost_breakdown: Optional[CostBreakdown] = None


class UnifiedPolicy:
    """
    Makes final action decisions integrating all systems.

    This is the "executive" that produces behavior.
    But unlike a separate goal system, it operates on
    the unified cost function.
    """

    def __init__(self,
                 viability: ViabilitySystem,
                 world_model: WorldModel,
                 alarm: AlarmSystem,
                 value: UnifiedValue):
        self.viability = viability
        self.world_model = world_model
        self.alarm = alarm
        self.value = value

        # === POLICY PARAMETERS ===
        self.base_exploration = 0.15  # Base exploration rate
        self.temperature = 1.0        # Softmax temperature

        # Action codes
        self.actions = ['stay', 'up', 'down', 'left', 'right']
        self.action_codes = {a: i for i, a in enumerate(self.actions)}

        # === DIRECTION MAPPINGS ===
        self.opposite = {
            'up': 'down', 'down': 'up',
            'left': 'right', 'right': 'left'
        }

    def select_action(self,
                      sensory: Dict[str, float],
                      predator_info: Optional[Dict] = None,
                      snn_preferences: Optional[Dict[str, float]] = None) -> ActionDecision:
        """
        Select an action based on current state.

        Integrates:
        1. Alarm (fast pathway) - immediate modulation
        2. Cost evaluation (slow pathway) - deliberate analysis
        3. SNN preferences - learned reflexes
        4. Exploration - information gathering
        """
        # === GET CURRENT STATE ===
        viability_metrics = self.viability.get_visualization_data()
        prediction_errors = self.world_model.get_prediction_error_summary()
        emergent_state = self.world_model.get_emergent_state(viability_metrics)

        # === FAST PATHWAY: ALARM ===
        predator_distance = None
        if predator_info and predator_info.get('position'):
            px, py = predator_info['position']
            ax, ay = predator_info.get('agent_pos', (0, 0))
            predator_distance = abs(px - ax) + abs(py - ay)

        hazard = self.alarm.assess_hazard(
            sensory,
            viability_metrics,
            took_damage=False,
            predator_distance=predator_distance
        )
        alarm_mod = self.alarm.update(hazard)

        # === CHECK FOR FREEZE ===
        if random.random() < alarm_mod.freeze_probability:
            return ActionDecision(
                action='stay',
                action_code=0,
                source='reactive',
                confidence=0.9,
                explanation='공포로 얼어붙음',
                scores={},
                alarm_influence=1.0
            )

        # === COMPUTE ACTION SCORES ===
        scores = {}
        explanations = {}

        for action in ['up', 'down', 'left', 'right']:
            score, explanation = self._evaluate_action(
                action,
                sensory,
                viability_metrics,
                prediction_errors,
                emergent_state,
                predator_info,
                alarm_mod,
                snn_preferences
            )
            scores[action] = score
            explanations[action] = explanation

        # === DETERMINE ACTION SOURCE ===
        alarm_influence = alarm_mod.avoidance_strength / 5.0

        if alarm_influence > 0.7:
            source = 'reactive'
        elif self._should_explore(emergent_state, alarm_mod):
            source = 'exploratory'
        else:
            source = 'deliberative'

        # === SELECT ACTION ===
        if source == 'exploratory':
            # Pure exploration
            action = random.choice(['up', 'down', 'left', 'right'])
            confidence = 0.3
            explanation = '탐험 중'
        else:
            # Select based on scores (lower cost = better)
            action, confidence = self._softmax_select(scores, alarm_mod)
            explanation = explanations[action]

        # Get cost breakdown for explanation
        cost_breakdown = self.value.compute_cost(
            viability_metrics, prediction_errors, emergent_state
        )

        return ActionDecision(
            action=action,
            action_code=self.action_codes[action],
            source=source,
            confidence=confidence,
            explanation=explanation,
            scores={a: -s for a, s in scores.items()},  # Negate for display (higher = better)
            alarm_influence=alarm_influence,
            cost_breakdown=cost_breakdown
        )

    def _evaluate_action(self,
                         action: str,
                         sensory: Dict,
                         viability_metrics: Dict,
                         prediction_errors: Dict,
                         emergent_state: Dict,
                         predator_info: Optional[Dict],
                         alarm_mod: PolicyModulation,
                         snn_preferences: Optional[Dict]) -> Tuple[float, str]:
        """
        Evaluate expected cost of an action.

        Returns (cost, explanation).
        """
        cost = 0.0
        reasons = []

        # === BASE: LEARNED VALUE FROM WORLD MODEL ===
        dir_vals = self.world_model.direction_values.get(action, {})
        food_value = dir_vals.get('food', 0)
        threat_value = dir_vals.get('threat', 0)
        wall_value = dir_vals.get('wall', 0)

        # Food reduces cost
        cost -= food_value * 2.0
        if food_value > 0.3:
            reasons.append('음식 기대')

        # Threat increases cost
        cost += threat_value * 3.0
        if threat_value > 0.3:
            reasons.append('위험 감지')

        # Wall increases cost
        cost += wall_value * 2.0

        # === VIABILITY-BASED EVALUATION ===
        urgencies = viability_metrics.get('urgencies', {})

        # If hungry, food direction is more valuable
        if urgencies.get('energy', 0) > 0.3:
            if sensory.get(f'food_{action}', 0) > 0:
                cost -= urgencies['energy'] * 3.0
                reasons.append('배고픔 해결')

        # === ALARM-BASED AVOIDANCE ===
        if predator_info and predator_info.get('position'):
            threat_dir = self._get_threat_direction(
                predator_info['position'],
                predator_info.get('agent_pos', (0, 0))
            )
            if action == threat_dir:
                # Moving toward threat - high cost
                cost += alarm_mod.avoidance_strength * 2.0
                reasons.append('위협 방향!')
            elif action == self.opposite.get(threat_dir, ''):
                # Moving away from threat - reduced cost
                cost -= alarm_mod.avoidance_strength * 0.5
                reasons.append('안전 방향')

        # === SNN PREFERENCES ===
        if snn_preferences:
            snn_score = snn_preferences.get(action, 0)
            cost -= snn_score * 0.5  # SNN has learned preferences

        # === PREDICTION ERROR REDUCTION ===
        # If this direction has high uncertainty, there's information value
        uncertainty = prediction_errors.get('uncertainty', 0)
        if emergent_state.get('curiosity_like', 0) > 0.3:
            # Uncertain directions are more interesting when curious
            cost -= uncertainty * 0.5
            if uncertainty > 0.5:
                reasons.append('불확실성 탐색')

        # Default explanation
        if not reasons:
            reasons.append('비용 최소화')

        return cost, ' + '.join(reasons)

    def _get_threat_direction(self,
                              predator_pos: Tuple,
                              agent_pos: Tuple) -> str:
        """Determine direction of threat."""
        dx = predator_pos[0] - agent_pos[0]
        dy = predator_pos[1] - agent_pos[1]

        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'

    def _should_explore(self,
                        emergent_state: Dict,
                        alarm_mod: PolicyModulation) -> bool:
        """Decide if we should explore."""
        # Base exploration rate, modulated by state
        exploration_rate = self.base_exploration

        # Curiosity increases exploration
        exploration_rate += emergent_state.get('curiosity_like', 0) * 0.2

        # Alarm suppresses exploration
        exploration_rate *= (1.0 - alarm_mod.exploration_suppression)

        # Fear suppresses exploration
        exploration_rate *= (1.0 - emergent_state.get('fear_like', 0) * 0.8)

        return random.random() < exploration_rate

    def _softmax_select(self,
                        scores: Dict[str, float],
                        alarm_mod: PolicyModulation) -> Tuple[str, float]:
        """
        Select action using softmax over negative costs.

        Returns (action, confidence).
        """
        # Temperature: lower = more deterministic
        # Alarm increases determinism (less random)
        temp = self.temperature * (1.0 - alarm_mod.avoidance_strength / 10.0)
        temp = max(0.1, temp)

        # Convert costs to probabilities (lower cost = higher prob)
        actions = list(scores.keys())
        costs = [scores[a] for a in actions]

        # Negate costs (lower cost = better = higher value)
        values = [-c for c in costs]

        # Softmax
        max_val = max(values)
        exp_values = [math.exp((v - max_val) / temp) for v in values]
        sum_exp = sum(exp_values)
        probs = [e / sum_exp for e in exp_values]

        # Sample
        r = random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                return actions[i], probs[i]

        return actions[-1], probs[-1]

    def get_behavior_explanation(self) -> Dict:
        """
        Generate human-readable explanation of current behavior.

        This is NOT the internal mechanism - it's a summary for observers.
        """
        viability_metrics = self.viability.get_visualization_data()
        prediction_errors = self.world_model.get_prediction_error_summary()
        emergent_state = self.world_model.get_emergent_state(viability_metrics)

        # Get dominant concern
        concern = self.value.get_dominant_concern(
            viability_metrics, prediction_errors, emergent_state
        )

        # Get trajectory
        trajectory = self.value.get_cost_trajectory()

        explanations = {
            'survival': '생존 위협에 반응 중',
            'understanding': '세상을 이해하려는 중',
            'control': '통제력 회복 중',
            'exploration': '새로운 정보 탐색 중'
        }

        trend_explanations = {
            'improving': '상황이 나아지고 있음',
            'worsening': '상황이 악화되고 있음',
            'stable': '안정적'
        }

        return {
            'dominant_concern': concern,
            'concern_korean': explanations.get(concern, ''),
            'trend': trajectory['trend'],
            'trend_korean': trend_explanations.get(trajectory['trend'], ''),
            'current_cost': trajectory['current'],
            'is_improving': trajectory['is_improving'],
            'emergent_state': emergent_state
        }

    def reset(self):
        """Reset policy state."""
        pass  # Stateless mostly

    def get_visualization_data(self,
                               last_decision: Optional[ActionDecision] = None) -> Dict:
        """Data for frontend."""
        explanation = self.get_behavior_explanation()

        data = {
            'explanation': explanation,
            'exploration_rate': self.base_exploration,
            'temperature': self.temperature
        }

        if last_decision:
            data['last_action'] = {
                'action': last_decision.action,
                'source': last_decision.source,
                'confidence': last_decision.confidence,
                'explanation': last_decision.explanation,
                'scores': last_decision.scores,
                'alarm_influence': last_decision.alarm_influence
            }

        return data
