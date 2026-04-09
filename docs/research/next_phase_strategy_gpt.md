# Next Phase Strategy — GPT Consultation

> Date: 2026-04-09
> Source: ChatGPT (Replay-driven Replanning Milestone)
> Status: Strategic direction received

## Core Recommendation

**"Replay-driven replanning under uncertainty"** — not more cortex, not 3D.

## 1. Next Milestone: First-Trial Detour/Reversal Planning

After latent environmental change, the agent improves its NEXT behavior because
offline replay updates policy/value, not because it brute-force relearns online.

**Concrete protocol:**
- Train stable food route with known predator landscape
- During rest: silently change world (block route, move risk zone, alter call reliability)
- Give offline replay window (no reward)
- Test first resumed trial
- Compare: intact replay vs replay-silenced vs no-rest

**Why this matters:**
- Demonstrates internal prospective computation, not just reactive behavior
- Ties to hippocampus-PFC planning literature (Nature Neuroscience 2024)
- Scientifically sharper than adding regions/neurons

## 2. Environment vs Brain: 70/30 Environment

**70% environment investment:**
- NOT 3D or prettier rendering
- Partial observability, volatility, social signals, latent-state switches
- Blocked passages, changing predator territories, probabilistic food reliability
- Deceptive auditory/social cues, delayed consequences

**30% brain mechanisms:**
- Only add what's needed for replanning + continual adaptation

**Why not 3D:**
- 28K neurons ≈ fly central brain (32,388 intrinsic neurons per 2024 atlas)
- 3D spends compute on perception plumbing, not cognition
- Task structure matters more than visual complexity

## 3. Scaling Reality Check

**Realistic ceiling: insect-grade flexible embodied cognition**
- Achievable: detour behavior, reversal learning, latent-state inference, call grounding, social cue use, uncertainty-sensitive explore/exploit, few-task continual learning
- NOT achievable: robust open-ended language, deep abstract reasoning, large-scale compositional planning

**Main ceiling is NOT neuron count, but:**
- Point-neuron LIF limits branch-specific context binding
- Synaptic-only plasticity struggles with rapid task switching
- Need selective consolidation beyond what trace+replay provides

## 4. Missing Mechanisms (priority ranked)

### 1st: Active Dendritic / Compartmental Computation
- Add to small subset: PFC, hippocampal prediction units, association cortex
- One branch for sensory, one for replay/memory, one for goal/social
- Nonlinear conjunctions + cleaner context gating
- Recent SNN results: active dendrites reduce interference in sequential learning

### 2nd: Uncertainty-Specific Neuromodulation (ACh/NE beyond DA)
- Missing: "surprising but don't overwrite" vs "world changed, switch to fast-learning mode"
- Acetylcholine in mPFC modulates how strongly surprising outcomes change future decisions
- Maps to volatility handling and catastrophic overreaction prevention

### 3rd: Synaptic Tagging and Capture (STC) / Multi-Timescale Consolidation
- Fast/slow memory split: many candidate changes stay temporary
- Only replayed or behaviorally important ones get consolidated
- Principled selective long-term memory formation

### 4th: Inhibitory Plasticity
- Suppress shared features, improve pattern separation
- Reduces interference between similar food/risk/social contexts

### 5th (later): Structural Plasticity
- Pruning/regrowth for representational crowding
- GPU-accelerated frameworks exist (GeNN-relevant)

## 5. Best Single Demo

**"Sleep-dependent first-trial detour replanning after hidden contingency change"**

Readouts:
- First-trial success rate after change
- Path inefficiency / latency
- Replay content vs chosen detour correlation
- Ablations: hippocampus, PFC gating, uncertainty neuromodulation

**Why compelling:**
- Neuroscience: replay + uncertainty + consolidation causally shape adaptive behavior
- AI: fully local-learning SNN shows rapid reconfiguration without backprop/pretraining
- Much stronger than "survival went from 70% to 78%"

## GPT's Top 3 Engineering Tasks

1. **ACh/NE-style uncertainty gate** (volatility detection → learning rate modulation)
2. **Compartmental PFC/hippocampal neurons** (active dendrites for context binding)
3. **STC-based fast/slow consolidation** (selective memory formation)

All tested in **2D latent-state detour environment**.
