# C4 Experience-Based Prediction — GPT Design Consultation

> Date: 2026-04-03
> Source: ChatGPT (Phase C4 Design SNN conversation)
> Status: Received, pending critical analysis

## GPT Recommendation Summary

### 1. Minimal Circuit Addition
- Do NOT add a full "prediction cortex"
- Add small `Pred_FoodSoon` readout: 8-16 excitatory neurons + WTA inhibitory pool
- Reuse existing Assoc_Binding + WM for context (location + sound + rich-zone trace)

### 2. Mechanism Choice
- **Core**: Successor-representation-like hippocampal prediction
  - "food will appear soon given current context" = future occupancy prediction
  - Hippocampal predictive map fits best
- **Auxiliary**: Existing PE circuits for mismatch detection
  - Expected food not arrived → negative PE
  - Unexpected food → positive PE
- NOT full predictive coding (too complex, not needed)

### 3. Concrete Synapses
1. **CA3→CA1 predictive weights**: Keep existing Hebbian/STDP + replay
2. **Context → Pred_FoodSoon**: Plastic from sound-category + WM rich-trace + coarse place
   - Rule: Hebbian + eligibility trace, gated by outcome
3. **CA1_future_readout → Pred_FoodSoon**: Key addition
   - Rule: R-STDP / 3-factor: `Δw = η × e_ij × δ_pred`
   - e_ij: eligibility trace, decay ~0.98–0.995
   - δ_pred: +1 if food appears within horizon H, -1 if deadline passes
4. **Pred_FoodSoon → BG**: Excitatory bias to Go/approach, weak exploration suppression

### 4. Behavioral Measurement (Prediction vs Reaction)
- **Delay task**: context cue now → food not yet visible → food appears after H steps
- **Metrics**:
  - A. Anticipatory heading bias (turn toward food zone BEFORE food visible)
  - B. Pre-spawn occupancy (already in correct zone during pre-food window)
  - C. Omission PE (PE spikes when predicted food fails to arrive)
  - D. Reversal speed (replay speeds relearning after contingency change)

### 5. Realistic Scope
**Achievable:**
- Short-horizon "food soon" prediction
- Context-conditioned anticipatory turning
- Omission-triggered PE
- Replay-assisted faster re-learning after reversal

**Overambitious (skip):**
- Full generative world model
- Deep multi-branch planning tree
- Arbitrary compositional generalization
- Long-horizon counterfactual simulation

## Critical Analysis (Claude)

### Concerns
1. **"CA1 future state readout"**: Our hippocampus has place_cells + food_memory, NOT a full predictive CA1. GPT overestimates our hippocampal sophistication.
2. **"WM recent-rich trace"**: No explicit rich-zone trace in WM. Need to create or repurpose Temporal_Recent.
3. **Motor interference**: Pred_FoodSoon → BG must follow "gentle modulator" pattern (≤2.0).
4. **Omission PE**: Nice-to-have but adds complexity. Defer to C4.1.

### Adaptations for Our Architecture
- Use existing food_memory + place_cells as "hippocampal context" (not SR)
- Use Temporal_Recent or WM_Context_Binding as "temporal context"
- Sound_food populations as "auditory context"
- Pred_FoodSoon → D1 (approach bias) with gentle weight
- Learning: R-STDP with food-arrival as reward signal

### References (from GPT)
- Bono et al.: SR learned in hippocampal weights, readout downstream
- Barry: hippocampal framing, conjunctive coding
- Spiking predictive coding: explicit error units optional
