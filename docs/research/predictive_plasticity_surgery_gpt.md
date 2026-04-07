# Predictive Plasticity Surgery — GPT Follow-up Consultation

> Date: 2026-04-07
> Source: ChatGPT (SNN Project Review, follow-up)
> Status: Concrete implementation plan received

## Root Cause (one sentence)

"Your place_cells → Pred_FoodSoon pathway is trying to learn a predictive world-model feature using a global action-reward learning rule."

## The Single Surgery

**Stop training `place_cells → Pred_FoodSoon` with DA-modulated R-STDP.**
Replace with **local predictive synapse** + **per-postsynaptic input budget**.

### Key Split
- **Representation learning** = predictive local rule (self-supervised)
- **Action/value learning** = dopamine / RPE (keep R-STDP only here)

## Concrete Implementation

### 1. Change what teaches Pred_FoodSoon

**Plastic dendritic/context inputs** (predictive rule):
- `place_cells(400)` → Pred_FoodSoon
- `sound_food / Wernicke_Food` → Pred_FoodSoon
- `WM_Context / recent_rich_zone_trace` → Pred_FoodSoon

**Fixed somatic teacher input** (non-plastic):
- `Food_Visible` / `Food_Close` / `Food_Obtained` → Pred_FoodSoon
- Driven by delayed food event, NOT reward
- Pick earliest event that is still behaviorally useful

### 2. New Synapse Rule (Predictive STDP + Heterosynaptic Budget)

```python
# Synapse: Context -> Pred_FoodSoon
# No dopamine term here.

# state per synapse
x_pre[j]         # pre trace
w[i,j]

# parameters
tau_pre = 150     # ms
eta_ltp = 2e-4
eta_ltd = 1e-4
w_min   = 0.0
w_max   = 1.5
W_budget_per_post = 12.0  # total incoming excitatory weight budget per Pred neuron

on_pre_spike(j):
    x_pre[j] += 1.0
    # weight-dependent presynaptic depression
    for each post neuron i connected from j:
        w[i,j] -= eta_ltd * w[i,j]

every_dt:
    x_pre[j] *= exp(-dt / tau_pre)

on_post_spike(i):
    # post spike caused mainly by teacher / somatic food event
    for each presyn j:
        w[i,j] += eta_ltp * x_pre[j]
    # immediate heterosynaptic / resource normalization
    S = sum_j w[i,j]
    if S > W_budget_per_post:
        scale = W_budget_per_post / S
        for each presyn j:
            w[i,j] *= scale
    # clip
    for each presyn j:
        w[i,j] = min(max(w[i,j], w_min), w_max)
```

### 3. Connection Map (exact population names)

| From | To | Rule | Notes |
|------|----|------|-------|
| `place_cells(400)` | `Pred_FoodSoon(30)` | Predictive STDP + budget | Main change |
| `sound_food / Wernicke_Food` | `Pred_FoodSoon(30)` | Same predictive rule | Smaller init_w |
| `WM_Context / rich_zone_trace` | `Pred_FoodSoon(30)` | Same predictive rule | |
| `Food_Visible / Food_Close` | `Pred_FoodSoon(30)` | **Fixed teacher** (static, strong) | Drives post spikes |
| `Pred_FoodSoon` | `NAc / BG D1` | **Keep DA/RPE** | Action interface |

### 4. Add Competition Inside Pred_FoodSoon

**Missing**: WTA everywhere else but NOT in Pred_FoodSoon itself.

Option A: Lateral inhibition `Pred_FoodSoon ↔ Pred_FoodSoon: -4 to -6`

Option B: Tiny interneuron pool:
- `Pred_FoodSoon(30) → Pred_Inh(4)` excitatory
- `Pred_Inh(4) → Pred_FoodSoon(30)` inhibitory -6 to -8

Without local competition, all 30 neurons become duplicates even with budget.

## Success Criteria (NOT survival — check these first)

1. Pred_FoodSoon fires **before** Food_Visible/Food_Close, not everywhere
2. Incoming weights maintain **bounded distribution** with different receptive profiles (not uniform ceiling)
3. Lesioning Pred_FoodSoon **measurably increases** first-food latency

## Why Other Options Are Lower Priority

- **BCM/sliding-threshold**: useful later, not the main fix now
- **Synaptic scaling**: too slow for the timescale where saturation happens
- **Fast/slow splitting**: helps consolidation, doesn't fix wrong credit signal
- **eta/decay/w_max tuning**: already tried, doesn't create selectivity

## Implementation in PyGeNN

Since PyGeNN doesn't have built-in predictive STDP, implement in `process()`:
1. Track pre-trace per synapse (or approximate with population-level trace)
2. On food event (teacher): strengthen active-context→pred connections
3. On every step: weak LTD on all active synapses
4. After each update: normalize incoming weights to budget per post neuron
5. Clip to [w_min, w_max]

This is similar to existing Hebbian code but with:
- Teacher signal instead of DA
- Weight-dependent LTD instead of constant decay
- Hard budget normalization instead of soft decay toward w_rest
