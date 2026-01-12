"""
v5.13 Operations Monitor

Shadow deployment 1주 운영용 도구:
1. 4칸 대시보드 출력
2. 주간 리포트 생성
3. v5.14 트리거 판정
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class WeeklyMetrics:
    """1주간 수집된 메트릭"""
    early_recovery_events: List[Dict] = field(default_factory=list)
    bad_phase_patterns: List[Dict] = field(default_factory=list)
    zone_distributions: List[Dict] = field(default_factory=list)
    lag_metrics: List[float] = field(default_factory=list)
    premature_impacts: List[Dict] = field(default_factory=list)

    # Zone-tagged impact (stable/transition/shock)
    zone_tagged_impacts: List[Dict] = field(default_factory=list)

    # 시나리오별 분류
    by_scenario: Dict[str, List[Dict]] = field(default_factory=dict)

    # Fingerprint별 seed 추적 (reproducibility용)
    by_fingerprint: Dict[str, Dict[str, List[Dict]]] = field(default_factory=dict)


class OpsMonitor:
    """v5.13 운영 모니터"""

    def __init__(self, ops_spec_path: Optional[str] = None):
        if ops_spec_path is None:
            ops_spec_path = os.path.join(os.path.dirname(__file__), 'v513_ops_spec.json')

        with open(ops_spec_path, 'r', encoding='utf-8') as f:
            self.spec = json.load(f)

        self.weekly_metrics = WeeklyMetrics()
        self.alerts: List[Dict] = []

    def ingest_monitoring_report(self, report: Dict, scenario: str = "unknown"):
        """모니터링 리포트 수집"""
        timestamp = datetime.now().isoformat()

        # Zone distribution
        if 'zone_distribution' in report:
            self.weekly_metrics.zone_distributions.append({
                'timestamp': timestamp,
                'scenario': scenario,
                'data': report['zone_distribution']
            })

        # Early recovery
        if 'early_recovery' in report:
            early = report['early_recovery']
            if early.get('count', 0) > 0:
                for event in early.get('events', []):
                    self.weekly_metrics.early_recovery_events.append({
                        'timestamp': timestamp,
                        'scenario': scenario,
                        **event
                    })

        # Phase analysis (bad patterns)
        if 'phase_analysis' in report:
            phase = report['phase_analysis']
            if phase.get('bad_pattern_rate', 0) > 0:
                self.weekly_metrics.bad_phase_patterns.append({
                    'timestamp': timestamp,
                    'scenario': scenario,
                    'rate': phase['bad_pattern_rate'],
                    'count': phase.get('bad_count', 0)
                })

        # Lag metric
        if 'lag_metric' in report:
            lag = report['lag_metric']
            if lag.get('lag_steps') is not None:
                self.weekly_metrics.lag_metrics.append(lag['lag_steps'])

        # Premature impact
        if 'premature_impact' in report:
            impact = report['premature_impact']
            if impact.get('cost_detected', False):
                self.weekly_metrics.premature_impacts.append({
                    'timestamp': timestamp,
                    'scenario': scenario,
                    **impact
                })

        # 시나리오별 분류
        if scenario not in self.weekly_metrics.by_scenario:
            self.weekly_metrics.by_scenario[scenario] = []
        self.weekly_metrics.by_scenario[scenario].append({
            'timestamp': timestamp,
            'report': report
        })

    def check_thresholds(self, report: Dict, scenario: str = "unknown", seed: int = 0) -> List[Dict]:
        """
        경고 임계값 확인 (3-stage: WARNING, CANDIDATE, CONFIRMED)

        WARNING = OR (early_rate > 10% OR bad_pattern > 5%)
        UPGRADE_CANDIDATE = AND (early_rate > 10% AND bad_pattern > 5%)
        """
        alerts = []
        thresholds = self.spec['alert_thresholds']
        warning_th = thresholds.get('warning', {})
        candidate_th = thresholds.get('upgrade_candidate', {})

        early_rate = 0
        bad_pattern_rate = 0

        if 'early_recovery' in report:
            early_rate = report['early_recovery'].get('rate', 0)

        if 'phase_analysis' in report:
            bad_pattern_rate = report['phase_analysis'].get('bad_pattern_rate', 0)

        # Check thresholds
        early_warning = early_rate > warning_th.get('early_recovery_rate', 0.10)
        bad_warning = bad_pattern_rate > warning_th.get('bad_phase_pattern_rate', 0.05)

        # WARNING = OR
        if early_warning or bad_warning:
            alerts.append({
                'level': 'WARNING',
                'early_rate': early_rate,
                'bad_pattern_rate': bad_pattern_rate,
                'triggered_by': 'early' if early_warning else 'bad_pattern',
                'scenario': scenario,
                'seed': seed,
            })

        # UPGRADE_CANDIDATE = AND
        if early_warning and bad_warning:
            alerts.append({
                'level': 'UPGRADE_CANDIDATE',
                'early_rate': early_rate,
                'bad_pattern_rate': bad_pattern_rate,
                'scenario': scenario,
                'seed': seed,
            })

        # Zone-tagged impact tracking
        if 'zone_distribution' in report:
            for zone in ['stable', 'transition', 'shock']:
                zone_data = report['zone_distribution'].get(zone, {})
                if zone_data.get('count', 0) > 0:
                    self.weekly_metrics.zone_tagged_impacts.append({
                        'zone': zone,
                        'weight_mean': zone_data.get('weight_mean', 0),
                        'scenario': scenario,
                        'seed': seed,
                        'has_cost': early_warning or bad_warning,
                    })

        # Track by fingerprint for reproducibility
        fingerprint = scenario  # fingerprint = scenario name for now
        if fingerprint not in self.weekly_metrics.by_fingerprint:
            self.weekly_metrics.by_fingerprint[fingerprint] = {}
        seed_key = str(seed)
        if seed_key not in self.weekly_metrics.by_fingerprint[fingerprint]:
            self.weekly_metrics.by_fingerprint[fingerprint][seed_key] = []
        self.weekly_metrics.by_fingerprint[fingerprint][seed_key].extend(alerts)

        self.alerts.extend(alerts)
        return alerts

    def evaluate_v514_trigger(self) -> Tuple[bool, str, Dict]:
        """
        v5.14 트리거 조건 평가 (3-stage)

        Stage 1: WARNING (OR) - log and observe
        Stage 2: UPGRADE_CANDIDATE (AND + cost) - prepare branch
        Stage 3: UPGRADE_CONFIRMED (candidate + reproducibility) - deploy

        Returns:
            (should_trigger, reason, evidence)
        """
        evidence = {
            'stage': 'HEALTHY',
            'warning_detected': False,
            'candidate_detected': False,
            'cost_verified': False,
            'reproducibility': False,
            'zone_impact': {},
            'details': {}
        }

        # 1. Check for WARNING (OR pattern)
        warning_alerts = [a for a in self.alerts if a['level'] == 'WARNING']
        if warning_alerts:
            evidence['warning_detected'] = True
            evidence['stage'] = 'WARNING'
            evidence['details']['warning_count'] = len(warning_alerts)

        # 2. Check for UPGRADE_CANDIDATE (AND pattern)
        candidate_alerts = [a for a in self.alerts if a['level'] == 'UPGRADE_CANDIDATE']
        if candidate_alerts:
            evidence['candidate_detected'] = True
            evidence['stage'] = 'UPGRADE_CANDIDATE'
            evidence['details']['candidate_alerts'] = candidate_alerts

        # 3. Cost verification with zone weighting
        zone_weights = self.spec.get('cost_thresholds', {}).get('zone_weighting', {
            'stable': 1.5, 'transition': 1.0, 'shock': 0.5
        })

        # Calculate zone-tagged impact
        zone_costs = {'stable': 0, 'transition': 0, 'shock': 0}
        for impact in self.weekly_metrics.zone_tagged_impacts:
            if impact.get('has_cost', False):
                zone = impact.get('zone', 'shock')
                zone_costs[zone] = zone_costs.get(zone, 0) + 1

        evidence['zone_impact'] = zone_costs
        evidence['details']['zone_costs'] = zone_costs

        # Cost is verified if stable zone has significant cost
        # (stable zone cost is most critical)
        if zone_costs.get('stable', 0) > 0:
            evidence['cost_verified'] = True
            evidence['details']['cost_zone'] = 'stable'
        elif self.weekly_metrics.premature_impacts:
            cost_events = [i for i in self.weekly_metrics.premature_impacts
                          if i.get('cost_detected', False)]
            if cost_events:
                evidence['cost_verified'] = True
                evidence['details']['cost_events'] = len(cost_events)

        # 4. Reproducibility check (2/3 seeds or 30% of 10 seeds)
        repro_spec = self.spec.get('reproducibility', {})
        min_seeds = repro_spec.get('min_seeds', 3)
        min_rate = repro_spec.get('min_trigger_rate', 0.67)

        for fingerprint, seeds_data in self.weekly_metrics.by_fingerprint.items():
            total_seeds = len(seeds_data)
            triggered_seeds = sum(
                1 for seed_alerts in seeds_data.values()
                if any(a['level'] == 'UPGRADE_CANDIDATE' for a in seed_alerts)
            )

            if total_seeds >= min_seeds:
                trigger_rate = triggered_seeds / total_seeds
                if trigger_rate >= min_rate:
                    evidence['reproducibility'] = True
                    evidence['details']['reproducibility'] = {
                        'fingerprint': fingerprint,
                        'triggered_seeds': triggered_seeds,
                        'total_seeds': total_seeds,
                        'rate': trigger_rate,
                    }
                    break

        # Final stage determination
        if evidence['candidate_detected'] and evidence['cost_verified'] and evidence['reproducibility']:
            evidence['stage'] = 'UPGRADE_CONFIRMED'
        elif evidence['candidate_detected'] and evidence['cost_verified']:
            evidence['stage'] = 'UPGRADE_CANDIDATE'
        elif evidence['warning_detected']:
            evidence['stage'] = 'WARNING'

        should_trigger = evidence['stage'] == 'UPGRADE_CONFIRMED'

        # Reason message
        stage_messages = {
            'HEALTHY': "v5.13 healthy - no patterns detected",
            'WARNING': f"WARNING: pattern detected (OR), observing... ({len(warning_alerts)} alerts)",
            'UPGRADE_CANDIDATE': f"CANDIDATE: pattern (AND) + cost verified, awaiting reproducibility",
            'UPGRADE_CONFIRMED': f"CONFIRMED: all conditions met - deploy v5.14",
        }
        reason = stage_messages.get(evidence['stage'], "unknown state")

        return should_trigger, reason, evidence

    def print_dashboard(self):
        """5칸 대시보드 출력 (zone-tagged impact 포함)"""
        print("\n" + "=" * 70)
        print("  v5.13 OPERATIONS DASHBOARD")
        print("=" * 70)

        # 계산
        n_early = len(self.weekly_metrics.early_recovery_events)
        n_total = len(self.weekly_metrics.zone_distributions)
        early_rate = n_early / max(n_total, 1)

        bad_patterns = self.weekly_metrics.bad_phase_patterns
        bad_rate = sum(p['rate'] for p in bad_patterns) / max(len(bad_patterns), 1) if bad_patterns else 0

        lag_values = self.weekly_metrics.lag_metrics
        lag_mean = sum(lag_values) / max(len(lag_values), 1) if lag_values else 0

        n_impact = len([i for i in self.weekly_metrics.premature_impacts if i.get('cost_detected')])
        impact_rate = n_impact / max(n_total, 1)

        # Zone-tagged impact 계산
        zone_costs = {'stable': 0, 'transition': 0, 'shock': 0}
        for impact in self.weekly_metrics.zone_tagged_impacts:
            if impact.get('has_cost', False):
                zone = impact.get('zone', 'shock')
                zone_costs[zone] = zone_costs.get(zone, 0) + 1

        # 상태 판정 (새 spec 구조)
        warning_th = self.spec['alert_thresholds'].get('warning', {})

        def status_or(early, bad):
            """OR 기준 WARNING"""
            e_warn = early > warning_th.get('early_recovery_rate', 0.10)
            b_warn = bad > warning_th.get('bad_phase_pattern_rate', 0.05)
            if e_warn and b_warn:
                return "UPGRADE"
            elif e_warn or b_warn:
                return "WARNING"
            return "OK"

        combined_status = status_or(early_rate, bad_rate)

        # 출력
        print(f"""
+---------------------------+---------------------------+
|  EARLY RECOVERY RATE      |  BAD PHASE PATTERN RATE   |
|  {early_rate:>6.1%}                    |  {bad_rate:>6.1%}                     |
+---------------------------+---------------------------+
|  LAG METRIC (mean)        |  PREMATURE IMPACT RATE    |
|  {lag_mean:>6.1f} steps              |  {impact_rate:>6.1%}                     |
+---------------------------+---------------------------+
|  ZONE-TAGGED IMPACT (cost occurred in which zone)     |
|  stable: {zone_costs['stable']:>3}  transition: {zone_costs['transition']:>3}  shock: {zone_costs['shock']:>3}          |
|  * stable zone cost = v5.14 highly valuable           |
+-------------------------------------------------------+
|  PATTERN STATUS: [{combined_status:^8}]                           |
+-------------------------------------------------------+
""")

        # v5.14 3-stage 판정
        should_trigger, reason, evidence = self.evaluate_v514_trigger()
        stage = evidence.get('stage', 'HEALTHY')

        stage_icons = {
            'HEALTHY': '[  OK  ]',
            'WARNING': '[ WARN ]',
            'UPGRADE_CANDIDATE': '[CANDID]',
            'UPGRADE_CONFIRMED': '[DEPLOY]',
        }

        print(f"  Stage: {stage_icons.get(stage, '[????]')} {stage}")
        print(f"  {reason}")
        print("=" * 70 + "\n")

    def generate_weekly_report(self) -> str:
        """주간 1페이지 리포트 생성"""
        import numpy as np

        lines = []
        lines.append("=" * 70)
        lines.append(f"  v5.13 WEEKLY REPORT - {datetime.now().strftime('%Y-%m-%d')}")
        lines.append("=" * 70)

        # (1) Zone distribution / Lag metric 추이
        lines.append("\n[1] ZONE DISTRIBUTION & LAG METRIC TREND")
        lines.append("-" * 50)

        for scenario, entries in self.weekly_metrics.by_scenario.items():
            if not entries:
                continue
            lines.append(f"\n  Scenario: {scenario}")
            lines.append(f"    Samples: {len(entries)}")

            # Zone weights 평균
            zone_weights = {'shock': [], 'transition': [], 'stable': []}
            for e in entries:
                zd = e['report'].get('zone_distribution', {})
                for zone in zone_weights:
                    if zone in zd:
                        zone_weights[zone].append(zd[zone].get('weight_mean', 0))

            for zone, weights in zone_weights.items():
                if weights:
                    lines.append(f"    {zone}: mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")

        if self.weekly_metrics.lag_metrics:
            lag_arr = np.array(self.weekly_metrics.lag_metrics)
            lines.append(f"\n  Lag Metric: mean={np.mean(lag_arr):.2f}, std={np.std(lag_arr):.2f}")

        # (2) Early recovery top 5
        lines.append("\n[2] EARLY RECOVERY TOP 5 EVENTS")
        lines.append("-" * 50)

        events = sorted(
            self.weekly_metrics.early_recovery_events,
            key=lambda x: x.get('weight_delta', 0),
            reverse=True
        )[:5]

        if events:
            for i, e in enumerate(events, 1):
                lines.append(f"  #{i}: step={e.get('step', '?')}, "
                           f"score={e.get('score', 0):.2f}, "
                           f"residual={e.get('residual', 0):.3f}, "
                           f"weight_delta={e.get('weight_delta', 0):.4f}")
        else:
            lines.append("  (No early recovery events)")

        # (3) Bad phase pattern rate / Premature impact + Zone-tagged
        lines.append("\n[3] PATTERN & IMPACT SUMMARY")
        lines.append("-" * 50)

        if self.weekly_metrics.bad_phase_patterns:
            rates = [p['rate'] for p in self.weekly_metrics.bad_phase_patterns]
            lines.append(f"  Bad Phase Pattern Rate: mean={np.mean(rates):.2%}, max={np.max(rates):.2%}")
        else:
            lines.append("  Bad Phase Pattern Rate: 0%")

        if self.weekly_metrics.premature_impacts:
            cost_count = len([i for i in self.weekly_metrics.premature_impacts if i.get('cost_detected')])
            lines.append(f"  Premature Impact Events: {cost_count} with cost detected")
        else:
            lines.append("  Premature Impact Events: 0")

        # Zone-tagged impact
        zone_costs = {'stable': 0, 'transition': 0, 'shock': 0}
        for impact in self.weekly_metrics.zone_tagged_impacts:
            if impact.get('has_cost', False):
                zone = impact.get('zone', 'shock')
                zone_costs[zone] = zone_costs.get(zone, 0) + 1

        lines.append(f"\n  Zone-Tagged Impact:")
        lines.append(f"    stable: {zone_costs['stable']} (critical)")
        lines.append(f"    transition: {zone_costs['transition']}")
        lines.append(f"    shock: {zone_costs['shock']} (expected)")

        # (4) 결론 (3-stage)
        lines.append("\n[4] CONCLUSION (3-Stage Trigger)")
        lines.append("-" * 50)

        should_trigger, reason, evidence = self.evaluate_v514_trigger()
        stage = evidence.get('stage', 'HEALTHY')

        lines.append(f"\n  5-Point Checklist:")
        lines.append(f"    [{'X' if evidence.get('warning_detected') else ' '}] 1. WARNING (OR): early>10% OR bad>5%")
        lines.append(f"    [{'X' if evidence.get('candidate_detected') else ' '}] 2. CANDIDATE (AND): early>10% AND bad>5%")
        lines.append(f"    [{'X' if zone_costs.get('stable', 0) > 0 else ' '}] 3. Cost in STABLE zone")
        lines.append(f"    [{'X' if evidence.get('cost_verified') else ' '}] 4. Cost verified (residual+20% or efficiency-10%)")
        lines.append(f"    [{'X' if evidence.get('reproducibility') else ' '}] 5. Reproducibility (2/3 seeds)")

        lines.append(f"\n  Current Stage: {stage}")
        lines.append(f"  Decision: {'>>> DEPLOY v5.14 <<<' if should_trigger else 'MAINTAIN v5.13'}")
        lines.append(f"  Reason: {reason}")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)


def run_dashboard_demo():
    """대시보드 데모 (3-stage trigger 시연)"""
    import numpy as np

    print("=" * 70)
    print("  DEMO 1: CANDIDATE stage (reproducibility 미달)")
    print("=" * 70)

    monitor = OpsMonitor()
    scenarios = ['stable_only', 'single_shock', 'oscillating']

    # 시나리오별 여러 seed 데이터 주입
    for scenario in scenarios:
        for seed in range(5):
            # oscillating에서 3/5 seeds에 문제 (60% < 67% 기준)
            has_problem = (scenario == 'oscillating' and seed in [0, 1, 2])

            report = {
                'zone_distribution': {
                    'shock': {'weight_mean': 0.02, 'count': 5},
                    'transition': {'weight_mean': 0.04, 'count': 3},
                    'stable': {'weight_mean': 0.15, 'count': 20},
                },
                'early_recovery': {
                    'count': 3 if has_problem else 0,
                    'rate': 0.15 if has_problem else 0.02,
                    'events': []
                },
                'phase_analysis': {
                    'bad_pattern_rate': 0.08 if has_problem else 0.02,
                    'bad_count': 5 if has_problem else 1
                },
                'lag_metric': {'lag_steps': 5 + np.random.randn()},
                'premature_impact': {'cost_detected': has_problem}
            }

            monitor.ingest_monitoring_report(report, scenario)
            monitor.check_thresholds(report, scenario=scenario, seed=seed)

    monitor.print_dashboard()

    # DEMO 2: CONFIRMED stage
    print("\n" + "=" * 70)
    print("  DEMO 2: CONFIRMED stage (reproducibility 충족 - 4/5 seeds)")
    print("=" * 70)

    monitor2 = OpsMonitor()

    for scenario in scenarios:
        for seed in range(5):
            # oscillating에서 4/5 seeds에 문제 (80% > 67% 기준)
            has_problem = (scenario == 'oscillating' and seed in [0, 1, 2, 3])

            report = {
                'zone_distribution': {
                    'shock': {'weight_mean': 0.02, 'count': 5},
                    'transition': {'weight_mean': 0.04, 'count': 3},
                    'stable': {'weight_mean': 0.15, 'count': 20},
                },
                'early_recovery': {
                    'count': 3 if has_problem else 0,
                    'rate': 0.15 if has_problem else 0.02,
                    'events': []
                },
                'phase_analysis': {
                    'bad_pattern_rate': 0.08 if has_problem else 0.02,
                    'bad_count': 5 if has_problem else 1
                },
                'lag_metric': {'lag_steps': 5},
                'premature_impact': {'cost_detected': has_problem}
            }

            monitor2.ingest_monitoring_report(report, scenario)
            monitor2.check_thresholds(report, scenario=scenario, seed=seed)

    monitor2.print_dashboard()
    print(monitor2.generate_weekly_report())


if __name__ == "__main__":
    run_dashboard_demo()
