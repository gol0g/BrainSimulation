# C5 Curiosity-Driven Exploration — GPT Design Consultation

> Date: 2026-04-03
> Source: ChatGPT (Curiosity-Driven Exploration in SNN conversation)
> Status: Received, pending critical analysis

## GPT Core Recommendation

**"Do not make C5 a pure novelty module."**
Smallest biologically plausible version: **novelty-gated, uncertainty-reduction-seeking curiosity**

- **Novelty** = "this may be worth sampling" (trigger)
- **Metacognitive uncertainty** = "I do not know enough yet" (amplifier)  
- **Dopamine** reinforces **only if** sampling actually reduces uncertainty (teacher)

### Formula
```
Curiosity ≈ novelty × uncertainty × safety × expected_learnability
```
"expected_learnability" emerges because states that reduced uncertainty get R-STDP reinforced.

## New Populations (14 neurons total)

1. **Curiosity_Gate** (6 excitatory LIF)
   - ACC-like gate: "novel + unresolved + potentially informative"
   - Light recurrent: E→E +0.2~0.4 (brief persistence)

2. **Safety_Gate** (4 inhibitory LIF)
   - Pain/Fear/urgent survival → suppresses curiosity
   - Safety_Gate → Curiosity_Gate: -1.5 ~ -2.0

3. **InfoPE** (4 excitatory LIF)
   - 2 positive: fires when uncertainty DECREASED (U_pre > U_post)
   - 2 negative: fires when uncertainty NOT decreased
   - InfoPE+ → VTA burst (information gain = reward)
   - InfoPE- → DA dip (no information = no reward)

## Concrete Wiring

### A. Curiosity_Gate inputs
- V4_Novel → Curiosity_Gate: +0.8
- Assoc_Novelty → Curiosity_Gate: +1.0
- Meta_Uncertainty → Curiosity_Gate: +1.2
- ACC_Conflict → Curiosity_Gate: +0.8
- KC sparse context → Curiosity_Gate: +0.3~0.6

### B. Safety_Gate inputs
- Pain/Fear → Safety_Gate (strong)
- Safety_Gate → Curiosity_Gate: -1.5 ~ -2.0

### C. BG influence (NOT Motor)
- Curiosity_Gate → BG explore/orient channels: +0.8~1.5
- Uses existing KC→BG to carry context/action eligibility

### D. InfoPE circuit
- WM(U_pre) → InfoPE+ excitatory
- U_post → InfoPE+ inhibitory
- U_post → InfoPE- excitatory
- WM(U_pre) → InfoPE- inhibitory
- InfoPE+ → VTA burst
- InfoPE- → DA dip

## Learning Rule (3-stage)

1. **Candidate generation**: Novelty + uncertainty activates Curiosity_Gate
2. **Exploratory action**: Curiosity_Gate biases BG explore channels
   - Set eligibility traces on KC→BG_explore, KC→Curiosity_Gate
   - Tau_e = 500-1200ms
3. **Reinforcement**: After sample:
   - InfoPE+ fires → DA burst → R-STDP strengthens exploratory connections
   - InfoPE- fires → DA dip → no reinforcement

## Validation Tests

### Test A: Informative vs Junk Novelty
- Zone A: novel combination that predicts food (informative)
- Zone B: equally novel but no predictive value (junk)
- Expected: early sampling of both → later preference for A → decline after A becomes predictable

### Test B: Safe vs Risky Information
- Informative cue near mild risk
- Expected: curiosity only when threat is low

### Test C: Hidden-state disambiguation
- Two contexts with same reward but different uncertainty
- Expected: curiosity biases toward uncertainty-reducing context

## Critical Analysis (Claude)

### Strengths
- Extremely minimal (14 neurons)
- Clean safety gating (no Motor interference)
- InfoPE mechanism is biologically sound
- Reuses all existing populations

### Concerns
1. **InfoPE implementation**: Comparing U_pre vs U_post requires temporal storage — can we use WM for this?
2. **InfoPE→VTA**: Adding a new input to existing dopamine could change baseline DA behavior. Need very weak weight.
3. **"BG explore channels"**: We don't have explicit explore channels. D1/D2 are Go/NoGo for food approach. Would need Curiosity_Gate → Goal_Food or → D1 with very low weight.
4. **6 neurons for Curiosity_Gate**: Very small — may not produce reliable spiking. Consider 20-30.
5. **Validation tests require environment changes**: Need to add informative/junk novelty zones.

### Adaptations for Our Architecture
- Scale up to ~40 neurons (Curiosity_Gate: 20, Safety_Gate: 10, InfoPE: 10)
- Route through Goal_Food/Goal_Safety (existing PFC goals) instead of "BG explore"
- Use meta_uncertainty rate as U signal, store in process() as self.prev_uncertainty
- InfoPE→VTA as very gentle (+0.5~1.0) bonus DA, not full burst
- Safety_Gate uses existing fear_rate + pain activity (no new sensory input)

## References (from GPT)
- Hippocampal-VTA novelty loop
- Active-sampling neuroscience (Gottlieb, Oudeyer)
- ACC-BG circuits for information anticipation
- Dopamine responses to information itself
