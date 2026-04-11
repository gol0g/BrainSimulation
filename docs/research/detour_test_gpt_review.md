# Detour Test GPT Review — Replay Content Control

> Date: 2026-04-11
> Source: ChatGPT (SWR Replay Test Evaluation)

## Verdict

"Your current result does NOT support 'replay helps reversal.' Your replay condition is systematically handicapped by design."

## Core Problem

SWR buffer still contains OLD food-location experiences after the latent switch.
Replaying these = **maladaptive consolidation** (strengthening obsolete memories).

"That is almost a textbook setup for memory anchoring, interference, and re-consolidation of obsolete attractors."

## Required Redesign: 4-Condition Test

1. **No replay** — baseline
2. **Old-only replay** — current (harmful?)
3. **New-only replay** — replay after discovering new locations
4. **Mixed with recency weighting** — biologically realistic

## Measurement Fixes

- **First 100 steps only** for old/new zone bias (full episode averages hide initial bias)
- **First switched episode** analyzed separately from episodes 2-5
- **20-30 independent seeds** per condition (not 5)
- Report **paired deltas per seed**, not aggregate means

## What the Test Actually Measures

Current: "what happens if I reinforce obsolete memories after a contingency reversal?"
Wanted: "does memory-guided replanning improve first-trial adaptation?"

These are fundamentally different questions.

## Action Items

1. Clear/update experience_buffer after latent switch for "new-only" condition
2. Add recency weighting to replay sampling
3. Measure first-100-step heading bias to old vs new zones
4. Run with 20+ seeds per condition
