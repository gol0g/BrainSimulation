# Context-Dependent Value Learning — GPT Design

> Date: 2026-04-12

## Core Fix

**Duplicate D1/D2 by context + place-driven context gating.**
NOT active dendrites, NOT DA modulation alone.

## Why It Fails Now

Same KC→D1 synapse stores opposite values for two contexts.
DA is global — bad outcome in Zone B depresses ALL connections for that food pattern.

## Minimum Circuit: +40 neurons

### Context Encoder (8 neurons)
- CtxA: 4 LIF — active when in left half (Zone A)
- CtxB: 4 LIF — active when in right half (Zone B)
- Input: place_cells → CtxA/CtxB (biased init or competitive Hebbian)
- WTA: CtxA ↔ CtxB mutual inhibition

### Context-Specific D1/D2 (32 neurons)
- D1_A_L, D1_A_R, D2_A_L, D2_A_R (4 each = 16)
- D1_B_L, D1_B_R, D2_B_L, D2_B_R (4 each = 16)
- Initialize from pretrained global D1/D2 weights (preserve 70%)

### Gating
- CtxA → D1_A/D2_A excitatory, → D1_B/D2_B inhibitory
- CtxB → D1_B/D2_B excitatory, → D1_A/D2_A inhibitory
- Only active context's SPNs near threshold → only they learn

### Learning Rule (4-factor)
```
e_ij^z(t+1) = λ * e_ij^z(t) + pre_i * post_j^z * ctx_z * act_j
Δw_ij^{D1,z} = η_D1 * δ_DA * e_ij^z
```
4 factors: pre(KC) × post(SPN) × context × dopamine

### Motor Output
- D1_A_L + D1_B_L → motor_left (both contribute, but only active context fires)
- Same push-pull as before

## Initialization Trick
Copy trained KC→D1/D2 weights into BOTH A and B channels.
Both start identical → policy unchanged → survival preserved.
Then A and B diverge where outcomes disagree.

## Success Metrics
1. Context purity: CtxA active on left >0.9, CtxB on right >0.9
2. Weight divergence: KC→D1_A ≠ KC→D1_B for same food cue
3. Zone-conditional selectivity: >0.6 (vs current 0.51)
