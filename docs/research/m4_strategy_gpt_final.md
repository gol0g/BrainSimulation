# M4 Strategy — GPT Final Review

> Date: 2026-04-11
> Source: ChatGPT (M3 Review and Next Steps)

## Revised Priority

**D/E → A or B (conditional on failure mode) → C (last)**

NOT D→B→A. Key change: Option E = "paperize M3 before adding architecture."

## Immediate Next Steps

### 1. Validate M3 (8-10 seeds + ablation)
- Freeze M3 architecture
- Run detour test with 8-10 seeds
- 3-way ablation:
  - No replay (baseline)
  - Replay WITHOUT reverse backup (consolidation only)
  - Replay WITHOUT ACh priority (no surprise modulation)
- Report: median effect, confidence intervals, per-seed paired deltas

### 2. Build Revaluation Benchmark Suite (4 tasks)
1. **Delayed-cue detour**: cue at start, ambiguity in middle, decision later
2. **Multi-branch reroute**: 2-3 alternative paths after barrier shift
3. **Reward vs transition split**: reward moved vs path changed (separate tests)
4. **Combinatorial remix**: unseen barrier-goal-pain combinations

### 3. Decision Rule After Benchmark
- Failure = hidden state / branch rollout → **A (active dendrites)**
- Failure = interference across accumulated rules → **B (continual learning)**
- Both pass → **E: write the replay/revaluation paper**

## Publishable Package (minimum viable)

**Claim**: "Surprise-prioritized reverse replay over a learned transition graph
supports transition revaluation and detour replanning in a spiking
hippocampal-style agent."

**Required**:
1. Behavioral suite: reward reval + transition reval + partial observability
2. Mechanistic ablations: reverse backup / replay priority / transition learning
3. Neural-style signatures:
   - Replay increases after surprise/context change
   - Replay content shifts toward changed contingencies
   - Replay content predicts later behavioral improvement

**Metrics**: success rate, first-hit latency, food in first 100 steps,
path inefficiency, replay-event statistics. Effect sizes + CIs.

## Why NOT jump to A yet
- M3's transition revaluation is already a serious model class
- Active dendrites are legitimate but not yet a proven bottleneck
- Better to prove WHERE point neurons fail, THEN upgrade
- "Justified by evidence rather than ambition"

## Why C (dreaming) is last
- Literature: awake replay has marginal support as direct online planning
- Stronger support for replay as prioritized offline updating
- Forward imagination needs world model → requires A first anyway
