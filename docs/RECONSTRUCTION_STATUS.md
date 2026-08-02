# 재건 상태 — 실행 앵커 (repo)

> **연구 지식(무엇을 배웠나)의 장기기억 정본 = `llm_wiki` 볼트 `sessions/brainsim/`.**
> 이 repo = 코드 + 실험 실행로그(무엇을 돌렸나·수치·커밋). 결론/지식은 볼트를 볼 것.
> 개념형성 4층위·seq-wm 분해·방법론 교훈 등 모든 결론 = 볼트.

## 재현 앵커 (repo 실행 정보)
- **환경**: Ubuntu 24.04 + CUDA 12.3 + PyGeNN 5.4.0 (소스빌드). 함정 우회 = `docs/RECONSTRUCTION.md`.
- **baseline 회귀 기준**: 4월판 20ep = Survival 55.0% / Reward 2.63%.
- **실행법**: `wsl -d Ubuntu -u root -- bash scripts/<exp>.sh` (전부 `scripts/cuda_env.sh` source, gcc-12 강제).
- **개념 스위트**: `scripts/eval_concept_suite.sh` (250ep 정본 가중치, 6프로브).
- **seq-wm 프로브**: `backend/genesis/wm_latch_probe.py`. 러너 플래그: `--inhib-wm`/`--seq-pattern-latch`/`--seq-no-curiosity`/`--danger-food-ratio`.
- 현재 원격 push = 커밋 `650e1ca` 이후 갱신.

## 실험 실행로그 (repo 상세, 결론은 볼트)
- 개념 실험 런기록: `docs/research/concepts/` (SCORECARD 수치표, trajectory·compositional·auditory_diagnosis)
- seq-wm 실험 런기록: `docs/research/DESIGN_RECOVERY.md` (§D1~D19b)
- 회로 설계·CLI 카탈로그: `docs/research/DESIGN_RECOVERY.md` 상단
