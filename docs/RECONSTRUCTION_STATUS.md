# 재건 최종 상태 (2026-07-28)

2026-07 디스크 사고 후 폐허에서 재건. 이 문서 = 정직한 최종 요약. 상세는 각 문서/git log.

## 확정된 성과 (real, 검증됨)

- **실행 환경 복구** ✅ — Ubuntu 24.04 + CUDA 12.3 + PyGeNN 5.4.0 (소스빌드). 함정 4개 우회(RECONSTRUCTION.md).
- **4월판 baseline 재현** ✅ — Survival 55%/Reward 2.63% (원본 정합).
- **A-트랙 8회로 재유도** ✅ — place_pref/biletaxis/brake/hunger-gate/factored/klino/olf/multicap.
  **단 다중시드 검증서 헤드라인 정정**: biletaxis align 진짜값 ~0.67(후반, 전시드>0.5 robust)이지
  seed0 "0.75" 아님. dwell 등 seed0 헤드라인 전반 낙관적. 회로는 real·robust하나 효과크기 겸손.
  (DESIGN_RECOVERY §D1~D9, verification/multiseed_a2.md)
- **개념 형성 2개** ✅ — **최종목표("개념 형성") 실제 달성:**
  - 공간 개념: rich zone 무작위 2배, 3체크포인트(50/120/250ep) 26~31% robust.
  - **good/bad 시각 변별**: 강제선택 프로브 81%(77/83/83, 무작위50) robust. selectivity 지표(0.64)에
    가려져 있던 개념 — 자유forage 행동역학이 confound였음. concepts/trajectory.md.

## 정직한 열린 문제 (억지 통과 안 시킴)

- **seq-wm (순서 학습)** — 창발 안 함(행동 order_rate≈0 확정). PBWM dopamine-gated WM 배선했으나
  판정 불가(내부 측정 실패). 7/12 원본 끊긴 지점. §D10~D13.
- **자유forage selectivity** — 0.64(base-rate 0.60). **변별결함 아님**(시각 강제선택 81% 확인). forage
  행동역학(구별 가능해도 접촉먹이 먹음)이 자유forage 비율을 낮춤. param 3반증(훈련/회피/변별학습).
  개념최적화 아닌 행동최적화 문제(별도). concepts/trajectory.md.
- **call semantics (소리로 good/bad)** — 우연. 부호 아님(A/B 반증). **시각 변별은 되는데 청각만 안 됨**
  = 특정된 청각→변별 갭(시각처럼 강제선택 통과 못함). concepts/auditory_diagnosis.md.

## 방법론 교훈 (세션 관통)

성급판단 4회 전부 후속실험이 반증: WM+0.42(brake트랩착시) / ON==OFF(SPARSE API오용) /
call50→60(n30노이즈) / biletaxis0.75(seed0낙관). **근본원인 = 단일시드·작은표본·미확정규약.**
→ **단일시드 헤드라인 금지, 다중시드 필수, 규약은 경험적 A/B로.** 정직한 음성 > 억지 양성.

## 보호

전부 `gol0g/BrainSimulation` (public, 연구코드) 원격 push. 금융 PII 없음. 디스크 사고 재발 대비 완료.
