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

## 다음: 합성 훈련 레짐 (미착수)
food + predator 공존 환경에서 훈련 → 위험 결합 good food의 가치가 하락하도록 학습.
필요: 훈련 중 food·predator 동시제시 + 위험맥락 value 하강 신호(amygdala fear→value 경로).
그 후 재프로브: 합성 개념 형성(>60%)되면 = 원시개념 조합 능력 확증(개념형성 4번째, 최상위 층위).
분리검증: (good∧safe) 접근 AND (bad∧safe) 회피 여전 유지 = 위험축이 good/bad축과 독립 조합.
