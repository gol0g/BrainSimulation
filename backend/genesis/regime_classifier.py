"""
Regime Classifier (M5) - Auto-recommend defense policy based on environment characteristics.

Based on E6-E8 findings:
- TTC: High threat predictability, justified opportunity cost
- TTC*: Medium predictability, needs approach-gating
- RF: Low predictability OR high defense cost (random walk threats)

Key insight from E8 phase diagram:
- p_chase=0.00: TTC wins (random threat doesn't pursue, safer to be proactive)
- p_chase=0.05: RF wins (weak signal causes TTC overreaction)
- p_chase>=0.10: TTC wins (predictable tracking, preemption justified)

Usage:
    classifier = RegimeClassifier()

    # During observation period, feed each step's data
    for step in observation_steps:
        classifier.observe(danger_dist, prev_danger_dist, in_defense, got_food, energy)

    # Get recommendation
    result = classifier.recommend()
    print(result['policy'])  # 'TTC', 'TTC*', or 'RF'

Calibration:
    # Use calibrate() with known environment data to tune thresholds
    classifier.calibrate(reference_logs, known_optimal_policies)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional


@dataclass
class RegimeMetrics:
    """Computed metrics for regime classification."""
    # Threat predictability indicators
    approach_streak_rate: float = 0.0      # % of steps where danger is closing
    mean_approach_streak: float = 0.0      # Average consecutive approach steps
    max_approach_streak: int = 0           # Longest approach streak
    closing_speed_std: float = 0.0         # Variability of approach speed
    long_streak_ratio: float = 0.0         # % of streaks that are 3+ steps (intentional chase)

    # Defense cost indicators
    defense_time_ratio: float = 0.0        # % of time in defense mode
    food_loss_rate: float = 0.0            # Food opportunities missed during defense
    energy_drain_rate: float = 0.0         # Energy lost per step during defense

    # Near-miss indicators (for TTC calibration)
    near_miss_count: int = 0               # Times danger got very close
    near_miss_ttc_p90: float = 0.0         # 90th percentile TTC at near-misses

    # Derived scores
    predictability_score: float = 0.0      # 0-1, higher = more predictable threat
    cost_score: float = 0.0                # 0-1, higher = more expensive defense

    # Sample size
    n_observations: int = 0


@dataclass
class ClassifierConfig:
    """Tunable thresholds for regime classification."""
    # Decision thresholds
    ttc_pred_threshold: float = 0.5        # Predictability >= this for TTC
    ttc_cost_threshold: float = 0.5        # Cost <= this for TTC
    ttc_star_pred_threshold: float = 0.25  # Predictability >= this for TTC*
    ttc_star_cost_threshold: float = 0.25  # Cost <= this for TTC*

    # Metric computation
    long_streak_min: int = 3               # Minimum steps for "intentional chase"
    near_miss_threshold: float = 2.0       # Distance threshold for near-miss

    # Score weights (predictability)
    pred_rate_weight: float = 0.3
    pred_streak_weight: float = 0.3
    pred_long_streak_weight: float = 0.3
    pred_consistency_weight: float = 0.1

    # Score weights (cost)
    cost_time_weight: float = 0.3
    cost_food_weight: float = 0.4
    cost_energy_weight: float = 0.3


@dataclass
class RegimeClassifier:
    """
    Collects environment observations and recommends optimal defense policy.

    Decision logic based on E8 phase diagram:
    - High predictability (long streaks, consistent) AND low cost → TTC
    - Medium predictability OR very low cost → TTC*
    - Low predictability AND high cost → RF only

    Key E8 insights:
    - Random walk (no tracking): TTC can win if cost is low
    - Weak tracking (p_chase~0.05): "Dip" zone where RF wins (false triggers)
    - Strong tracking: TTC clearly wins
    """

    # Observation buffers
    danger_distances: List[float] = field(default_factory=list)
    closing_speeds: List[float] = field(default_factory=list)
    approach_flags: List[bool] = field(default_factory=list)
    defense_flags: List[bool] = field(default_factory=list)
    food_events: List[bool] = field(default_factory=list)
    energy_values: List[float] = field(default_factory=list)
    near_miss_ttcs: List[float] = field(default_factory=list)

    # Streak tracking
    current_approach_streak: int = 0
    approach_streaks: List[int] = field(default_factory=list)

    # Config
    min_observations: int = 100
    config: ClassifierConfig = field(default_factory=ClassifierConfig)

    def observe(self,
                danger_dist: float,
                prev_danger_dist: Optional[float],
                in_defense: bool,
                got_food: bool,
                energy: float) -> None:
        """
        Record one step of environment observation.

        Args:
            danger_dist: Current distance to nearest danger
            prev_danger_dist: Previous distance (None for first step)
            in_defense: Whether agent is in defense mode this step
            got_food: Whether agent got food this step
            energy: Current energy level
        """
        self.danger_distances.append(danger_dist)
        self.defense_flags.append(in_defense)
        self.food_events.append(got_food)
        self.energy_values.append(energy)

        if prev_danger_dist is not None:
            # Closing speed (positive = approaching)
            closing = prev_danger_dist - danger_dist
            self.closing_speeds.append(closing)

            # Approach flag
            is_approaching = closing > 0.05  # Small threshold to ignore noise
            self.approach_flags.append(is_approaching)

            # Track streaks
            if is_approaching:
                self.current_approach_streak += 1
            else:
                if self.current_approach_streak > 0:
                    self.approach_streaks.append(self.current_approach_streak)
                self.current_approach_streak = 0

            # Track near-misses with TTC estimate
            if danger_dist < self.config.near_miss_threshold and closing > 0:
                ttc_estimate = danger_dist / max(closing, 0.01)
                self.near_miss_ttcs.append(ttc_estimate)

    def compute_metrics(self) -> RegimeMetrics:
        """Compute all regime classification metrics from observations."""
        n = len(self.danger_distances)

        if n < self.min_observations:
            return RegimeMetrics(n_observations=n)

        metrics = RegimeMetrics(n_observations=n)

        # === Threat predictability ===

        # Approach streak rate
        if self.approach_flags:
            metrics.approach_streak_rate = sum(self.approach_flags) / len(self.approach_flags)

        # Streak statistics
        # Include current streak if ongoing
        all_streaks = self.approach_streaks.copy()
        if self.current_approach_streak > 0:
            all_streaks.append(self.current_approach_streak)

        if all_streaks:
            metrics.mean_approach_streak = np.mean(all_streaks)
            metrics.max_approach_streak = max(all_streaks)
            # Long streak ratio: intentional chase indicator
            long_streaks = [s for s in all_streaks if s >= self.config.long_streak_min]
            metrics.long_streak_ratio = len(long_streaks) / len(all_streaks)

        # Closing speed variability (lower = more predictable)
        if self.closing_speeds:
            positive_speeds = [s for s in self.closing_speeds if s > 0]
            if positive_speeds:
                metrics.closing_speed_std = np.std(positive_speeds)

        # Near-miss statistics
        if self.near_miss_ttcs:
            metrics.near_miss_count = len(self.near_miss_ttcs)
            metrics.near_miss_ttc_p90 = np.percentile(self.near_miss_ttcs, 90)

        # === Defense cost ===

        # Defense time ratio
        if self.defense_flags:
            metrics.defense_time_ratio = sum(self.defense_flags) / len(self.defense_flags)

        # Food loss during defense
        defense_steps = sum(self.defense_flags)
        if defense_steps > 0:
            food_during_defense = sum(f for f, d in zip(self.food_events, self.defense_flags) if d)
            # Compare to baseline food rate
            total_food = sum(self.food_events)
            non_defense_steps = len(self.defense_flags) - defense_steps
            if non_defense_steps > 0:
                baseline_food_rate = (total_food - food_during_defense) / non_defense_steps
                defense_food_rate = food_during_defense / defense_steps
                # Food loss = how much worse is defense mode
                metrics.food_loss_rate = max(0, baseline_food_rate - defense_food_rate)

        # Energy drain during defense
        if len(self.energy_values) > 1:
            energy_diffs = np.diff(self.energy_values)
            defense_diffs = [d for d, f in zip(energy_diffs, self.defense_flags[1:]) if f]
            if defense_diffs:
                metrics.energy_drain_rate = -np.mean(defense_diffs)  # Positive = losing energy

        # === Derived scores ===
        cfg = self.config

        # Predictability: approach rate + streak length + long streak ratio + consistency
        rate_score = metrics.approach_streak_rate
        streak_score = min(1.0, metrics.mean_approach_streak / 5.0)  # 5+ steps = max
        long_streak_score = metrics.long_streak_ratio  # Key differentiator for tracking
        consistency_score = max(0, 1.0 - metrics.closing_speed_std)  # Lower std = higher score

        metrics.predictability_score = (
            cfg.pred_rate_weight * rate_score +
            cfg.pred_streak_weight * streak_score +
            cfg.pred_long_streak_weight * long_streak_score +
            cfg.pred_consistency_weight * consistency_score
        )

        # Cost: defense time + food loss + energy drain
        time_cost = metrics.defense_time_ratio
        food_cost = min(1.0, metrics.food_loss_rate * 10)  # Scale to 0-1
        energy_cost = min(1.0, metrics.energy_drain_rate * 5)  # Scale to 0-1

        metrics.cost_score = (
            cfg.cost_time_weight * time_cost +
            cfg.cost_food_weight * food_cost +
            cfg.cost_energy_weight * energy_cost
        )

        return metrics

    def recommend(self) -> Dict:
        """
        Get policy recommendation based on collected observations.

        Returns:
            Dict with:
                - policy: 'TTC', 'TTC*', or 'RF'
                - confidence: 0-1 confidence in recommendation
                - metrics: RegimeMetrics object
                - reasoning: Human-readable explanation
        """
        metrics = self.compute_metrics()

        if metrics.n_observations < self.min_observations:
            return {
                'policy': 'RF',
                'confidence': 0.0,
                'metrics': metrics,
                'reasoning': f'Insufficient observations ({metrics.n_observations}/{self.min_observations}). Defaulting to RF (safest).'
            }

        p = metrics.predictability_score
        c = metrics.cost_score
        cfg = self.config

        # Decision thresholds (from E8 phase diagram)
        # TTC: High predictability, acceptable cost
        if p >= cfg.ttc_pred_threshold and c <= cfg.ttc_cost_threshold:
            policy = 'TTC'
            confidence = min(p, 1.0 - c)
            reasoning = f'High threat predictability ({p:.2f}) with acceptable cost ({c:.2f}). TTC preemptive defense justified.'

        # TTC*: Medium predictability OR very low cost
        elif p >= cfg.ttc_star_pred_threshold or c <= cfg.ttc_star_cost_threshold:
            policy = 'TTC*'
            confidence = 0.5 + 0.25 * p - 0.25 * c
            reasoning = f'Medium predictability ({p:.2f}) or low cost ({c:.2f}). TTC* with approach-gating recommended.'

        # RF: Low predictability AND high cost
        else:
            policy = 'RF'
            confidence = 1.0 - p + c * 0.5
            reasoning = f'Low predictability ({p:.2f}) with high cost ({c:.2f}). RF reactive-only is optimal.'

        confidence = np.clip(confidence, 0.0, 1.0)

        return {
            'policy': policy,
            'confidence': confidence,
            'metrics': metrics,
            'reasoning': reasoning
        }

    def reset(self) -> None:
        """Clear all observations."""
        self.danger_distances.clear()
        self.closing_speeds.clear()
        self.approach_flags.clear()
        self.defense_flags.clear()
        self.food_events.clear()
        self.energy_values.clear()
        self.near_miss_ttcs.clear()
        self.current_approach_streak = 0
        self.approach_streaks.clear()

    def summary(self) -> str:
        """Get human-readable summary of metrics and recommendation."""
        result = self.recommend()
        m = result['metrics']

        lines = [
            "=" * 50,
            "REGIME CLASSIFIER REPORT",
            "=" * 50,
            "",
            f"Observations: {m.n_observations}",
            "",
            "--- Threat Predictability ---",
            f"  Approach streak rate:  {m.approach_streak_rate:.1%}",
            f"  Mean approach streak:  {m.mean_approach_streak:.1f} steps",
            f"  Max approach streak:   {m.max_approach_streak} steps",
            f"  Long streak ratio:     {m.long_streak_ratio:.1%}",
            f"  Closing speed std:     {m.closing_speed_std:.3f}",
            f"  >> Predictability score: {m.predictability_score:.2f}",
            "",
            "--- Defense Cost ---",
            f"  Defense time ratio:    {m.defense_time_ratio:.1%}",
            f"  Food loss rate:        {m.food_loss_rate:.3f}",
            f"  Energy drain rate:     {m.energy_drain_rate:.3f}",
            f"  >> Cost score: {m.cost_score:.2f}",
            "",
            "--- Near-Miss Analysis ---",
            f"  Near-miss count:       {m.near_miss_count}",
            f"  TTC (p90):             {m.near_miss_ttc_p90:.2f} steps",
            "",
            "--- Recommendation ---",
            f"  Policy: {result['policy']}",
            f"  Confidence: {result['confidence']:.1%}",
            f"  Reasoning: {result['reasoning']}",
            "=" * 50,
        ]

        return "\n".join(lines)

    def get_ttc_calibration(self) -> Dict:
        """
        Get recommended TTC thresholds based on near-miss analysis.

        Returns:
            Dict with tau_on, tau_off, and m (approach streak min) recommendations.
        """
        metrics = self.compute_metrics()

        if metrics.near_miss_count < 10:
            return {
                'tau_on': 10.0,  # Default
                'tau_off': 15.0,
                'm': 2,
                'calibrated': False,
                'reason': 'Insufficient near-miss data'
            }

        # tau_on: Use 80th percentile of near-miss TTC
        tau_on = np.percentile(self.near_miss_ttcs, 80)

        # tau_off: 1.5x tau_on for hysteresis
        tau_off = tau_on * 1.5

        # m: Based on long streak patterns
        # If long streaks are common, lower m (more aggressive TTC)
        # If rare, higher m (more conservative)
        if metrics.long_streak_ratio > 0.3:
            m = 2  # Common long streaks = intentional tracking
        elif metrics.long_streak_ratio > 0.1:
            m = 3  # Some long streaks = mixed regime
        else:
            m = 4  # Rare long streaks = mostly random

        return {
            'tau_on': tau_on,
            'tau_off': tau_off,
            'm': m,
            'calibrated': True,
            'reason': f'Based on {metrics.near_miss_count} near-miss events'
        }


# Convenience function for quick classification
def classify_regime(observations: List[Dict]) -> Dict:
    """
    Quick regime classification from a list of observation dicts.

    Each observation dict should have:
        - danger_dist: float
        - in_defense: bool
        - got_food: bool
        - energy: float

    Returns policy recommendation dict.
    """
    classifier = RegimeClassifier()

    prev_dist = None
    for obs in observations:
        classifier.observe(
            danger_dist=obs['danger_dist'],
            prev_danger_dist=prev_dist,
            in_defense=obs.get('in_defense', False),
            got_food=obs.get('got_food', False),
            energy=obs.get('energy', 0.5)
        )
        prev_dist = obs['danger_dist']

    return classifier.recommend()
