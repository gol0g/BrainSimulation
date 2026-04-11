# SWR Replay: Consolidation → Revaluation Architecture

> Date: 2026-04-11
> Source: ChatGPT (SWR Replay Test Evaluation, follow-up #2)

## Core Diagnosis

"Consolidation-only replay cannot solve replanning after contingency change."
Current SWR = re-strengthen already-experienced place→food associations.
Missing: backward credit assignment from new rewards to predecessor states.

## Minimum Viable Change

### Add 1: Place→Place Transition Synapse (W_pp)
- Online learning: when agent moves PC_i → PC_j, strengthen W_pp[i,j] via STDP
- Stores spatial topology: "where can I go from here?"
- Sparse recurrent synapse within place_cells population

### Add 2: Reverse Replay + Value Backup
- After new food discovery: start reverse replay from rewarded place cell
- Traverse W_pp backward: rewarded_PC → predecessor_PCs
- Update PC→Value and PC→BG connections with dopamine-gated 3-factor rule
- Result: "being at location X means food is reachable via Y steps"

### Buffer Change
- Current: (pos_x, pos_y, food_type, step, reward) — location memory
- Needed: (place_t, place_t+1, reward) — transition memory
- At minimum: store consecutive place cell activations, not just food positions

## Implementation Mapping (28K architecture)

| Component | Existing | New |
|-----------|---------|-----|
| Place cells (400) | ✓ | — |
| Place→Food_Memory | ✓ Hebbian | Keep (consolidation) |
| Place→Place (W_pp) | ✗ | **ADD**: STDP sparse recurrent |
| Value population | ✗ | **ADD**: 20-30 neurons |
| PC→Value | ✗ | **ADD**: DA-gated 3-factor |
| Value→BG | ✗ | **ADD**: gentle D1 bias |
| Reverse replay | ✗ | **ADD**: in replay_swr(), reverse sequence |
| Transition buffer | ✗ | **ADD**: (PC_t, PC_t+1, reward) storage |

## Phase 2 (later): Sequence Composition
- CA3 recurrent chain + PFC reservoir for snippet stitching
- A→B experienced + B→C experienced → offline A→B→C composition
- Requires hippocampus replay output → PFC recurrent input
- Already have WM/PFC populations — connect replay to them

## Key Insight

"The single smallest necessary change: learn place→place transitions,
then after new reward discovery, run reverse replay to update
place→value/place→BG. This turns SWR from a 'past location
re-strengthener' into a replanning-capable offline backup device."
