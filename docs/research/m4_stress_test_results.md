# M4 Environment Stress Tests

> Date: 2026-04-12

## Results Summary

| Test | Conditions | Survival | vs Baseline |
|------|-----------|----------|------------|
| Baseline | 250px, 1 pred, no reversal | 70% | — |
| Contingency reversal | good↔bad every 2500 steps | 65% | -5pp |
| Partial observability | view 150px | 60% | -10pp |
| Partial observability | view 100px | 65% | -5pp |
| **EXTREME** | **100px + reversal 2000 + switch 1500 + 2 pred** | **75%** | **+5pp** |

## Interpretation

The brain handles all quantitative stressors without breaking:
- Reduced vision → hippocampal spatial memory compensates
- Rule flipping → D1/D2 R-STDP + Garcia effect relearns online
- Combined extreme → still 75% (noise range, but NOT degraded)

## What This Means

Current 28K brain is **robust to quantitative difficulty increases**.
The circuits (Push-Pull, R-STDP, ACh gate) are sufficient for:
- Reactive foraging with reduced sensors
- Online rule relearning
- Multi-threat environments

## What's Needed to Break It

**Qualitative challenges** that require capabilities the brain doesn't have:
1. **Hidden-state aliasing**: same sensory input, different correct action
   depending on an earlier cue (requires WM integration with action)
2. **Conditional rules**: "green food is good in zone A, bad in zone B"
   (requires context-dependent valuation)
3. **Sequential dependencies**: "must visit beacon X before food Y is edible"
   (requires planning/sequencing)

These test whether Phase 12-20 (WM, PFC, Metacognition) are actually
contributing to decision-making, not just providing background context.
