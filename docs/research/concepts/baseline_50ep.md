# 개념 형성 baseline (50ep 훈련, 2026-07-25)

훈련: forager_brain 50ep, Survival 66%/Reward 2.53%. 가중치 brain_concepts_50ep.npz.
평가: evaluate_concepts.py --test all.

| 테스트 | 점수 | 기준 | 판정 |
|---|---|---|---|
| 공간 기억 (rich zone) | 29.5% (무작위 14%) | — | ✅ PASS |
| 먹이 변별 (selectivity) | 0.62 (good661/bad405) | >기준 | ✗ FAIL (부분변별) |
| call semantics (소리만) | 50.0% (15/30) | >60% (무작위50) | ✗ FAIL (우연) |

**해석**: 공간 개념 형성됨. 먹이 변별 부분적(구별하나 기준미달). 소리 단서 개념 없음(우연).
**다음**: 훈련부족 vs 실제결함 판별 — 더 길게(120ep) 훈련 후 재평가. 점수 오르면 훈련량,
정체하면 특정회로(특히 청각→행동, call semantics) 결함.
