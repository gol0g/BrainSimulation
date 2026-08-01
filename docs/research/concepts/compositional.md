# 합성/관계 개념 (C3) — 다음 프론티어

개념 형성 3개(공간·good/bad 양식초월·범주일반화) 확정 후 다음 층위: **학습된 원시개념이
조합되어 새 범주를 이루나** = 인간형 개념형성의 다음 rung.

## 프로브 (test_compositional, evaluate_concepts.py)
강제선택: 양쪽 다 good food(동일 시각). 한쪽만 위험(predator_rays + sound_danger + danger_signal)과 결합.
올바른 선택 = **(good∧safe) 접근, (good∧danger) 회피** = value가 맥락(위험)에 조건화되나.
good/bad·위험 원시개념이 조합돼 "안전한 좋은음식"이라는 합성 범주를 이루는지 측정. 기준>60%(무작위50).

## Baseline (250ep, 위험-가치 미훈련, 2026-08-01)
3회: **48.3 / 41.7 / 47.5%** = 전부 우연(50%) 수준, FAIL.
- **예상된 결과**: 250ep는 good/bad foraging만 훈련 → food value가 위험 맥락에 조건화된 적 없음.
- 프로브 정상작동 확인(미훈련=우연). 합성 개념은 아직 없음 = **위험-조건화 훈련 레짐 필요.**

## 프로브 위험-양식 수정 → 97% (그러나 대조가 반전, 2026-08-01)
**핵심**: 250ep는 이미 danger_food_enabled(기본 True, 30% food가 pain zone 근처)로 훈련됨.
첫 baseline 48%는 프로브가 predator/sound_danger를 썼기 때문 = 훈련 위험양식(pain zone)과 불일치
(call_semantics와 동형 오류). 프로브 위험을 **pain_rays**로 교체 → **97.5/96.7/98.3% PASS.**

## 충돌 대조 = danger-only 반사로 판명 (성급결론 방지)
97%가 "합성"인지 "단순 위험회피"인지 판별: 충돌배치(안전쪽 bad food, 위험쪽 good food),
위험 무릅쓰는 비율(brave) 측정. test_compositional_conflict.
- brave-for-GOOD = 6.7/8.3%, brave-for-BAD = 0.0/2.5%. **합성(good−bad) = +6~7%pt** (기준15 미달).
- **결론(정직)**: 97%는 대부분 **위험회피 반사** — 음식가치 무관하게 pain 쪽 회피. value×danger
  균형 통합 아님(위험 압도). 좋은음식 위한 약한 값조절(+6%pt)만 존재.
- **위험-조건화 행동은 real·robust**(97% 회피), 그러나 **균형 합성개념은 미확정.** 4번째 개념 아님.
- 교훈: 97% 단독이면 오판. 충돌 대조가 danger-dominance 폭로 = 성급결론 규율의 실효.

## Graded sweep = 약하지만 실재하는 합성 확인 (2026-08-01)
단일 0.8 충돌테스트가 너무 가혹(둘 다 회피)해 "danger-only"로 보였음. **위험강도 sweep**하니 반전:
| 위험 | brave_GOOD | brave_BAD | diff | (2차 재현) |
|---|---|---|---|---|
| **0.10** | 26/30% | 3/5% | **+23/+25%pt** | 재현✅ |
| 0.25 | 6/6% | 0/3% | +6/+3 | |
| 0.40 | 3/12% | 1/1% | +2/+11 | |
| 0.55~0.70 | 4~17% | 0~2% | +4~+15 | noisy |

- **낮은 위험(0.10)서 good food 위해 26~30% 무릅씀 vs bad food 3~5%(+23~25%pt), 2회 재현.**
  위험 커지면 good도 회피로 수렴 = **음식가치가 위험 감수도를 graded 조절 = value×danger 통합.**
- **정직한 규모**: 약함(26%도 소수, 위험회피 전반 우세). 강한 균형합성 아니나 **실재·재현**하는 graded 신호.
- **방법론 교훈(재확인)**: 단일 강도(0.8)테스트는 오도("danger-only") → 강도 sweep이 진실 드러냄.
  개념 selectivity(0.64)를 강제선택이 뚫은 것과 동형. 단일점 판정 금지, 축을 훑어라.

## 결론: 4번째 개념층위 = 약한 graded 합성 (real, qualified)
개념형성 프로빙 4층위: ①공간 ②good/bad 양식초월 ③범주일반화 ④**graded value×danger 합성(약함)**.
④는 재훈련 없이 기존 250ep서 발현 — 원시개념(good/bad, 위험)이 조합돼 맥락조건화 가치 형성.
강화는 다음: graded 위험/가치 훈련레짐으로 합성강도↑(brave-for-good 다수화) 유도.
