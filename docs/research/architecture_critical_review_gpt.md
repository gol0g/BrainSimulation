# Architecture Critical Review — GPT Consultation

> Date: 2026-04-07
> Source: ChatGPT (SNN Project Review)
> Status: Received, action items to be implemented

## Blunt Verdict

"Your concern is valid. The current learning stack is under-constrained enough that adding more neurons will mostly add scaffolding, saturation, and compute cost unless you first fix competition, stabilization, and causal evaluation."

"The architecture may contain a real sensorimotor core, but a lot of the present 'brain-like' mass is probably not carrying proportional functional load."

## Issue 1: R-STDP Fast Saturation — DESIGN MISMATCH

**Diagnosis**: NOT a fundamental R-STDP flaw. It's a design mismatch:
- Global reinforcement rule applied where competition + prediction-specific credit assignment is needed
- Many place cells co-active during reward, broad modulatory signal, small target → uniform saturation expected
- No postsynaptic competition, normalization, or heterosynaptic counterforce

**Solution**: 
- **Move place→prediction AWAY from R-STDP** — use predictive/self-supervised sequence learning instead
- Keep R-STDP ONLY for value/action pathways (pred→BG, KC→D1)
- This split matches biology: state transition learning ≠ reward-modulated action selection

**Missing stabilizers** (need at least 1-2):
- Postsynaptic competition (k-WTA/lateral inhibition among prediction neurons)
- Per-neuron incoming-weight normalization
- Heterosynaptic plasticity
- Sliding thresholds / metaplasticity
- Decorrelated upstream place codes

**Key insight**: Weight decay alone doesn't create selectivity. Higher w_max only delays saturation.

## Issue 2: Curiosity Gate — LIKELY DECORATIVE

**Diagnosis**: Low rate (0.005-0.020) not inherently wrong — novelty signals ARE transient/sparse biologically. But:
- Signal definition is "semantically muddy" — mixing novelty + uncertainty + conflict into one weak additive channel
- Strong safety veto overwhelms weak curiosity inputs
- "Circuit is presently more decorative than functional"

**Validation required** (4 conditions):
1. Intact
2. Curiosity-off (--no-curiosity)
3. Curiosity-amplified (increase weights 2x)
4. Curiosity-shuffled (randomize spike timing)

**Metrics that matter**:
- Unique-state visitation increase
- First-food latency after environment shifts
- Map coverage / path entropy
- Performance in sparse-reward or changed environments

**Fix**: Separate novelty, surprise/conflict, and uncertainty instead of mixing into one scalar.

## Issue 3: ~15K Neuron Scaffolding — DEAD WEIGHT

**Diagnosis**: "Very likely a dead-weight problem right now."
- "Anatomy gets added faster than causal pathways"
- "Labels become richer while computation stays roughly the same"
- "Classic trap in biologically inspired projects"

**Effective system is much smaller than 28K neurons.**
- Likely functional core: sensory salience + pain avoidance + hippocampal context + BG action selection + some WM/uncertainty loops
- Rest: "spectators, relays, or concept placeholders"

**Module admission criteria** (each population must pass 3 tests):
1. Carries decodable task information
2. Lesioning changes behavior or internal prediction quality
3. Benefit exceeds compute cost

**Ablation battery needed**: full lesion, spike-train shuffling, output-weight randomization, readout decodability. Test combinations, not just single lesions.

## Issue 4: Saturation = Limitation, Not Victory

**Diagnosis**: "Widespread saturation across many pathways means the rule has lost dynamic range."
- Garcia 5/5, Forward Model 10/10, Body→Narr 14/14 = NOT reassuring
- Apparent learning capacity is inflated on paper
- Increasing w_max is usually the wrong first response

**Better responses**:
- Maintain soft bounds (not hard clip)
- Normalize incoming weight totals
- Add heterosynaptic or synaptic-scaling terms
- Introduce metaplastic thresholds (BCM-style)
- Split synapses into fast/slow components

## Architecture Verdict: "Partly yes, mostly not yet for scaling"

**Core idea is coherent**: embodied SNNs + hippocampal context + uncertainty + safety + BG action selection.

**Scaling path is currently wrong**: "Adding named cortical-style modules faster than establishing clear causal roles, proper competition/homeostasis, and task pressure."

**Risk**: "100K neurons will likely produce a system that is larger, slower, and more fragile, but not proportionally more cognitive."

**"The project scales anatomical decoration faster than functional mechanism."**

## GPT's Recommended Priority Order (before 100K neurons)

### 1. Stop adding modules. Fix the learning substrate.
- Replace place→prediction R-STDP with predictive/self-supervised rule + competition + heterosynaptic stabilization
- Keep reward modulation for value/action pathways only

### 2. Make "module admission" strict.
- Every population must pass: decodable info, lesion impact, benefit > cost

### 3. Track capacity explicitly.
- Fraction of synapses at bounds
- Entropy of weight distributions
- Effective rank / diversity of readouts
- Recovery after task reversal
- Performance after environment shift

### 4. Separate novelty, surprise, uncertainty.
- Don't mix into one curiosity scalar
- Test whether each actually buys exploration

## Bottom Line

**"Do not scale this architecture yet. Fix competition, stabilization, and causal auditing first."**
