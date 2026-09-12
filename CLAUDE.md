# BrainSimulation — 프로젝트 지침

> 이 파일은 2026-07 디스크 사고로 Chrome 번역 문자열 93KB에 덮여 있었다(수정일 Jul 5, git 미추적).
> 원본은 복구 불가로 확정돼 2026-09-12에 새로 작성했다. 손상본은 스크래치패드에 백업.

## 최종 목표
**인간의 뇌처럼 학습하고 개념을 형성하는 인공 뇌.** (PyGeNN + CUDA 스파이킹 신경망, 28,323 뉴런)

## ★ 연구를 시작하기 전에 — 연구실부터 읽어라
작업 대상은 이 디렉터리가 아니라 **`../BrainSimulation-rebuild/`** 다.
그 안의 **`research/`** 가 연구 제도이며, **즉흥으로 실험하지 마라.**

**재개 순서 (고정):**
1. `research/current-state.md` — 지금 무엇을 믿는가 (역사 아님. 이것만 읽으면 재개 가능)
2. `research/invariants.md` — 어기면 결론이 **무효**가 되는 설정 8개. 실험 전 필수
3. `research/protocols.md` — 측정 규약 P1~P12
4. `bash scripts/lab/audit.sh` — 미완 등록·결과 미기입·신선도 점검

## 실험 실행 규칙 (게이트가 강제한다)
```
scripts/lab/new_experiment.sh E### "제목"   # 사전등록 생성
  → E###.md 의 1~5번을 채운다
scripts/lab/run_experiment.sh E### "커맨드"  # 게이트 통과분만 실행
```
게이트가 요구하는 것: 검증할 가설 + **경쟁 가설** + **조작 검증 방법** + **사전 판정 기준** +
"이 결과로 새로 알게 되는 것" + 불변식 8개 확인. **마지막에 답할 수 없으면 실험하지 않는다.**

거부당하면 체크박스를 거짓으로 채우지 말고, 실제로 채우거나 `[~] ... (사유: ...)`로 이탈을 선언하라.

**훅이 우회를 막는다**: `lab_gate_guard.sh`(PreToolUse)가 사전등록 없는 실험 진입점 직접 실행을 차단하고,
`lab_p9_check.sh`(Stop)가 결과 미반영 상태의 응답 종료를 차단한다.
짧은 진단(`--episodes 0~9`)과 조작검증 프로브는 통과한다.

## 실험 후 (규약 P9) — 세 곳을 갱신한다
1. `research/experiments/E###.md` — 결과와 **사전기준 대비** 판정
2. `research/hypotheses/H###.md` — 신뢰도와 증거
3. `research/current-state.md` — 현재 믿음 (덮어쓰기, 최종 갱신일 포함)

역사는 `docs/research/DESIGN_RECOVERY.md`에 append. **역사와 현재 믿음을 섞지 마라.**

## 이 프로젝트에서 반복된 실패 (읽고 시작하라)
한 세션에서 결론을 **5회 뒤집었다.** 원인은 뇌가 아니라 **측정**이었다.
- 체크포인트 로드가 학습의 63.6%를 조용히 파괴 (시드 미고정)
- 무시드 런 오프셋 → 같은 뇌가 100%도 0%도 냄
- 이분법 임계 지표 → "0%"가 사실은 "결정 안 함"
- 상태 미정규화 → 도파민 없이도 같은 표류
- **조작이 듣는지 확인 안 함 5회** — 두 조건 값이 소수점까지 같으면 **조작 무효를 먼저 의심하라**
- 내 수리가 경로를 막은 사례 2회(`d1→direct=1`, `direct_inhibition=-400`) → 규약 P10

**결론이 안 나오면 뇌보다 측정을 먼저 의심하라. 기각과 검출불가는 다르다.**

## 환경
- WSL2 Ubuntu, PyGeNN 5.4.0 + CUDA 12.3, `source scripts/cuda_env.sh` → `source ~/pygenn_wsl/bin/activate`
- 실행 디렉터리는 `/root/*_run/` (런마다 분리해 GeNN CODE 충돌 방지)
- **WSL 인라인 명령에 `$VAR`·`for` 루프 금지** — Git Bash가 소비해 조용히 실패한다. `scripts/*.sh` 파일로.
- 파이프에 `head -N` 금지 — SIGPIPE로 실험이 중간에 죽는다.

## 저장소
- 코드·실행로그: `gol0g/BrainSimulation` (**로컬 커밋만 하지 말고 반드시 push** — 2026-07 소실 원인)
- 연구 지식 정본: `llm_wiki/sessions/brainsim/`
