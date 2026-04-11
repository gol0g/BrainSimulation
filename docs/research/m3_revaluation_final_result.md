# M3 Revaluation SWR — Final Validation Result

> Date: 2026-04-12

## Result: FAIL — Replay is maladaptive after contingency change

### 5-Seed 3-Way Ablation (reverse backup WORKING, preds=3700+)

| Condition | New zone | Old zone | Food(100) |
|-----------|----------|----------|-----------|
| **No replay** | **26.8%** | 14.6% | **4.2** |
| Revaluation | 0.0% | 28.8% | 3.0 |
| Consolidation | 0.0% | 30.2% | 2.8 |

Per-seed delta (reval vs no_replay): 0/5 positive. Mean: -26.8pp.

## Root Cause

NOT a threshold bug (that was fixed — reverse backup confirmed working).
The problem is **fundamental**: W_pp transition graph was learned from
the OLD environment. Reverse backup propagates value along OLD paths.

Both replay types (revaluation AND consolidation) trap the agent in
the old zone because they reinforce stale spatial associations.

## Scientific Interpretation

**"SWR replay after a latent contingency change is maladaptive when
the learned transition structure no longer reflects the current environment."**

This is actually a meaningful finding:
- Replay helps in STABLE environments (consolidation)
- Replay HURTS after environment restructuring
- This matches biological literature: replay is prioritized offline
  UPDATING, not direct online planning

## What Would Fix It

1. **W_pp needs to be updated BEFORE replay** — agent must explore
   new environment enough to rewrite transition graph
2. **Or**: surprise/ACh gate should SUPPRESS replay when uncertainty
   is high (don't replay when you know the world changed)
3. **Or**: separate "old map" from "new map" (context-dependent replay)

## Impact on Project Direction

GPT's Option E (paperize) becomes more relevant:
- "Replay helps consolidation but hurts after contingency change" IS
  a publishable finding if properly ablated
- The ACh uncertainty gate SHOULD suppress replay when surprised —
  test whether surprise-gated replay suppression improves adaptation
