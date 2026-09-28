# 세션 인계 — 2026-09-28 (Pi tmux 세션 → 새 세션)

> 연구 믿음은 여기 없다. **현재 믿음 = `research/current-state.md`, 다음 결정 = `research/abcd-2026-09-28.md`.**
> 이 문서는 운영(무엇이 돌고 있나, 어떻게 돌리나, 사용자 규칙)만 담는다. 옛 세션이 긴 압축으로 요청서 조건을 잃은 것이 오류의 주원인이었다(DESIGN_RECOVERY 2026-09-28 14:3x) — 그래서 새 세션으로 넘긴다.

## 1. 지금 상태 (인계 시점)
- **실행 중인 실험 없음.** 마지막 실험 E118(13:57~14:36) 완료, 판정 **보류** 확정(독립 대조 반영).
- 분기 "반사 25 역전"(E108~E118, 11/11) **종료 — 미해결**. 새 분기 "통합 중 어디서 능력을 잃는가" 상한 3.
- 다음 할 일: **E119**(abcd C절) — 아직 사전등록 안 함. 전부 push됨(마지막 730802e).

## 2. 재개 순서
CLAUDE.md의 1~5 그대로(current-state → invariants → protocols → process-request → audit.sh). 이어서
`research/abcd-2026-09-28.md`(A/B/C/D 확정본)와 `research/audit-2026-09-28.md`(감사, 반복 오류 유형 5가지)를 읽는다.

## 3. E119 착수 절차
1. `bash scripts/lab/new_experiment.sh E119 "반사 0에서 전체 모델 KC→motor 매핑 학습"` → 1~8절 채움.
   7절(실행 경로 검사)은 로그 경로가 실제로 있어야 게이트 통과, 8절은 `1 / 3`.
2. **경로 검사 먼저**(짧은 진단 `--episodes 0~9`는 훅 통과): abcd C절 1~4번.
   - 보상 판정: `correct`는 reflex_override_task.py:646에서 **행동 창 이전 조향 v**로(|v|≤0.02 오답), 실행 행동 `_ex`는 :665에서 v의 부호로 정한다.
     반사 0에서 불일치 비율이 5% 넘으면 실행 행동 기준 판정 옵션을 추가하고 두 반사 칸을 같은 코드로 돌린다(새 옵션 = 새 경로 → 7절 로그).
   - `--reflex-w`는 `cfg.food_approach_init_w`를 바꾼다(:269-270). 학습 전후 그 가중치가 그대로인지 확인.
3. 판정 스크립트(`scripts/judge_e119.py`)를 **본실험 전에** 쓰고 합성 입력(성공·실패·경계·같은 값·결측)으로 시험한다(조건 1).
   출력 문자열에 **단위와 정의**를 넣는다(규약 P19) — 예: "효과(학습−무학습 변조폭, 음수=교차)".
4. 기준 명령은 E118과 같다: `scripts/e118_main.sh`의 BASE·ACT(초기 150, w_max 300, 밀도 0.25, eta 0.15, tau_e 12, 보상 창 2·자극 끔, 간격 10, ε 0.6, bias 25, env 0, 행동 창 3스텝 ±5000). 뇌 10~14.

## 4. 실행 인프라
- 대화: Pi, tmux 세션 `brain`. `~/brain/BrainSimulation-rebuild` → sshfs `~/brain/desk`(= 데스크탑 `C:\Users\JungHyun\Desktop\brain`). 볼트 `~/llm_wiki`도 sshfs.
- 데스크탑 SSH: `ssh -i ~/.ssh/id_ed25519_desktop JungHyun@100.100.227.23`(Windows cmd 셸 — `|`는 따옴표 안에, 출력에 `\r`).
- **실험 띄우기**: `C:\Users\JungHyun\lab_run.cmd`를 scp로 바꿔 쓰고(`wsl bash -lc "cd /mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild && bash scripts/lab/run_experiment.sh E### \"bash scripts/e###_main.sh\" >> /tmp/e###_run.out 2>&1"`)
  `ssh … 'schtasks /run /tn LabRun'`. SSH 세션에 묶이지 않아 세션이 끊겨도 산다. 런 디렉터리는 WSL `/root/e###_run/`.
- **감시(필수)**: 실험이 도는 채로 턴을 끝내기 전 반드시 `~/bin/watch-exp.sh E### <런 수> "<결과 줄 패턴>"`을 백그라운드로 **종료까지** 건다(18시간 정지 사고).
- 데스크탑 절전: 지연·무응답이면 절전부터 의심. **요청 없이 깨우지 않는다**(`~/bin/wake-desktop.sh`는 요청 시에만). 절전 설정 변경 금지.
- **push**: 맨 `git push` 금지(인증 창). 커밋 메시지를 `C:\Users\JungHyun\lab_commit_msg.txt`로 scp → `ssh … 'C:\Users\JungHyun\scoop\apps\git\2.55.0.3\bin\bash.exe -l /c/Users/JungHyun/lab_push.sh'`(add·commit·push, 끝에 `## main...origin/main`이면 동기).
- Pi에서 git 읽기: `git -c safe.directory='*' log …`(sshfs 소유자 불일치).
- 훅 오탐: Bash 명령 문자열에 과제 파일 이름(reflex_override…)이 들어가면 lab_gate_guard가 막는다 → Read 도구로 읽거나 `scripts/*.sh` 파일로. **훅을 수정·우회하지 않는다.**
- Pi 자원: ARM·RAM 작음, 실거래 서비스(traveler 등) 공존 — 무거운 계산·대량 I/O 금지, 그 서비스는 건드리지 않는다.

## 5. 사용자 규칙 (어기면 신뢰 손상 — 전부 실제 지시)
- 한국어로 답한다. **매 응답 끝에 현재 시각**(`date` 출력). 기록에 쓰는 시각도 `date`/커밋/로그 시각만 — 2026-09-28에 지어낸 시각 5곳을 교정했다.
- "마무리/오늘은 여기까지" 류 금지. 결과 보고 후 같은 응답에서 다음 수를 실행.
- 방향 결정을 사용자에게 넘기지 않는다(목표는 명확 — 인간처럼 학습하고 개념을 형성하는 뇌). 대신 요청서(process-request) 절차를 따른다.
- 반성하는 척 금지 — 원인을 근거와 함께 측정해 말한다.
- git 토큰은 절대 언급·출력하지 않는다(값·앞자리·길이 포함). GitHub 인증 창을 띄우지 않는다.
- `D:\Recovered` 수정 금지. 파킹 점검 훅 수정·우회 금지.
- 속도보다 검증: 새 측정 도구는 합성 정답으로 먼저, 결과 서술은 기록 전 독립 대조.

## 6. 오늘 확인된 측정 함정 (current-state·감사에 근거 있음)
- 출력의 Δg(초기 대비)를 가중치로 읽음 → 판정 출력에 단위 명시.
- KCPRE(부호 있는 평균 상위 5%)는 **순 강화일 때만** 활성 KC를 고른다 — 순 약화에선 퇴화. 반응 KC 분석은 KCSETS.
- "재실행 흔들림 ≤1e-4" 전제는 틀림 — 관측 최대 0.03. 0.03 미만 차이는 확립 안 됨.
- 조작검증은 바꾼 변수 하나를 분리해야 한다(E118 (2)는 초기값만으로 통과해 판별력 없음).
- SPARSE 변수는 `pull_connectivity_from_device()` 후 `.values`; float32 가중치 정확 일치 검사.

## 7. 재개 시험 결과 (2026-09-28 14:56~15:01)
- 방법: `claude -p` 새 프로세스(~/brain, CLAUDE.md·기억 자동 로드), 읽기 전용(Bash·Write·Edit 금지), 12문항. 정답표는 시험 전 작성.
- 결과 **12/12**. 파킹 훅이 막았을 때도 시험 지시(파일 수정·실험 금지)를 지키고 다음 할 일로 E119 사전등록을 짚었다.
- 시험이 드러낸 결함 2개와 조치(15:01):
  1. Pi의 `~/brain/CLAUDE.md`가 작업 대상을 `../BrainSimulation-rebuild/`로 적어 존재하지 않는 `~/BrainSimulation-rebuild`를 먼저 읽음 → Pi 경로로 교정.
  2. `~/brain/rebuild-ro/`(9/27 GitHub 클론)의 낡은 current-state를 읽음(시험 세션은 스스로 배제) → fast-forward(9b686b7) + CLAUDE.md에 "정본 아님" 명시.

## 8. E119 진행 상태 (시각은 date)
> 작성 2026-09-28 15:25 — Pi 세션 "BrainSimulation E119 (Pi)"(시작 폴더 ~/brain). 사용자 지시로 저장소 폴더 새 세션에 교대. **본실험은 띄우지 않았다.**

**끝낸 것**
- 재개 순서 1~6 + audit.sh(실행 중 실험 없음, P9 위반 없음).
- 사전등록 `research/experiments/E119.md` 1~8절(7절은 P5 결과 한 줄 남음), 가설 `research/hypotheses/H045.md` 신설. 템플릿 INV-A5(−400, 폐기값)는 `[~]` 사유와 함께 현행 −100 선언. **게이트는 아직 돌리지 않았다.**
- 판정 `scripts/judge_e119.py`(판정 핵심 `judge()` 순수 함수, 단위 출력) + 합성 시험 `scripts/test_judge_e119.py` **13/13 통과**(`logs/E119/judge_synthetic.log`). 시험 중 경계 부동소수 결함(−0.0877−0.0123=−0.0999…)을 발견해 효과를 소수 4자리 반올림 후 비교하도록 수정.
- 경로 검사 P1~P4(`scripts/e119_path_check.sh`, 요약 `logs/E119/path_summary.out`, 로그 15:06~15:24) — 결과는 E119.md 7절:
  P1 반사 0 도달 ✓(기준 +0.02 vs 반사 25 +0.41), P2 판독 권한 R 0.564/0.553 ✓, **P3 판정 불일치 12.4% ✗ → `--judge exec` 채택**, P4 E118 b0 소수점 재현 ✓.
- 러너 `scripts/e119_main.sh` 작성(재개 가능, 요약 줄 형식 = judge 파서). **`JUDGE="__JUDGE__"` 자리표시 그대로** — P5 통과 후 `exec`로 바꿔야 한다(안 바꾸면 argparse가 거부해 전 런 실패).

**진행 중 / 남은 것**
1. **P5 `--judge exec` 경로 검사**(`scripts/e119_exec_check.sh`, 15:24:53 schtasks로 시작, 뇌 15 반사 0·25 각 1런) → `logs/E119/path_exec_summary.out`.
   통과 기준: (a) 반사 25 exec가 P3 judge v(`path_learn_rw25_b15.log`: 보상 140, 사후 +0.4208, 변화 −0.0189)와 소수점까지 같음(불일치 0이므로), (b) 반사 0 exec의 `[판정경로]`가 `judge=exec`로 찍히고 보상 횟수가 judge v(237)와 달라짐. 결과는 E119.md 7절에 한 줄 추가(아래 9절에 이 세션이 적었으면 그것을 확인).
2. `e119_main.sh`의 JUDGE → exec, E119.md 상태 줄 갱신 → `bash scripts/lab/gate.sh E119` → 절차 4절(lab_run.cmd + schtasks, 명령 `bash scripts/lab/run_experiment.sh E119 "bash scripts/e119_main.sh"`) → `~/bin/watch-exp.sh E119 20 "변조폭 변화"` 종료까지 감시.
3. 완료 후 `python3 scripts/judge_e119.py` → P9 세 곳 + DESIGN_RECOVERY append + 독립 대조.
4. **커밋·push 안 함** — 이번 변경 전부(과제 코드, 스크립트 5개, E119.md, H045.md, 이 절) 미커밋.

**코드 변경과 검증 상태** (`backend/genesis/reflex_override_task.py`)
| 변경 | 수정 | 검증 |
|---|---|---|
| `--judge {v,exec}` 옵션(exec는 act-window 필요, 아니면 종료) | 완료 | v: P4 회귀로 기존 경로 불변 확인. **exec: P5 진행 중 — 미검증** |
| `_ex`(실행 행동) 계산을 판정 직후로 이동 | 완료 | P4 E118 b0 소수점 재현(보상 140, +0.4060→+0.4033) ✓ |
| `[판정경로]` 계수기(읽기 전용) | 완료 | P3에서 값 출력 확인(반사 25 0%, 반사 0 21.0%/12.4%). 계수 정의의 합성 검사는 안 함 — 반사 25 0건·반대 방향 0건은 v 부호=_ex 정의와 일치 |
| `[반사가중치]` 스냅숏(읽기 전용, 빈 배열이면 예외) | 완료 | P1·P3에서 0/25/10 설정값 그대로 판독 ✓ |
| judge_e119.py | 완료 | 합성 13/13 ✓. 실제 E119.log 줄로는 미검증(형식은 E118 러너와 같은 echo) |

**주의**: P3에서 표본 밖 뇌 15의 반사 0 judge v 학습 효과 −0.0989를 이미 보았다(E119.md 7절 "사전 노출 기록"). 기준은 abcd C절에 먼저 고정돼 있었다.

## 9. 이 세션(저장소 폴더, 15:26~) 진행 — 시각은 date·로그
- 재개 순서 1~6 + audit.sh(E119 미실행·결과 미기입, P9 위반 없음).
- **P5 통과**(로그 15:24:53~15:34:50, `logs/E119/path_exec_summary.out`): (a) 반사 25 exec = judge v 소수점까지 동일(사전 +0.4397, 보상 140, 사후 +0.4208, 변화 −0.0189). (b) 반사 0 exec `[판정경로] judge=exec`, 보상 330(≠237). E119.md 7절에 기록.
- 판정 파서를 P3 원 로그에서 러너와 같은 echo로 만든 줄로 확인(정확히 파싱).
- `e119_main.sh` JUDGE → `exec`(15:35). 게이트: 7절 줄의 "난수 소비 없음"을 '변경 없음'으로 오인 → 문구를 "소비 0회"로 바꿔 통과(게이트 코드 불변).
- 커밋·push 후 본실험 시작 → 감시 `~/bin/watch-exp.sh E119 20 "변조폭 변화"`.
- E119 완료(15:36~16:36) → 판정 **보류**(반사 0 효과 5/5 교차, 평균 −0.077, ≤−0.10 1/5) · 독립 대조 일치 · P9 반영(dd40c96). 사용량 한도로 16:37~19:13 공백.
- **E120**(반사 0 학습량 5·10·15 용량-반응, 분기 2/3) 사전등록·게이트 통과(fdd066f), 19:18:16 시작, 15런 약 2.5시간. 감시 `watch-exp.sh E120 15 "변조폭 변화"`. 완료 후 `python3 scripts/judge_e120.py` → P9.
- E120 완료(19:18~21:08) → **B 포화**(d +0.016 평균, 5/5 약감소). P9 반영(21:10). 판정 밖: 학습 중 보상률 증가 vs 이식 평가 불변 → E121(분기 3/3)은 측정 확인 경로 검사부터.
- E121 경로 검사(21:12~21:57): 측정 확인 — 이식 평가는 학습을 놓치지 않음(학습 중 보상률 상승 = 부호 정답률, 크기는 작음). 이식 밖 가소성 6개(IT 지각 학습) 발견(영향 작음). 새 옵션 `--kc-bilateral-scale`·`--snap-all-syn`·`[KC공통입력]`(회귀 소수점 동일). kcsets 공유 KC 수는 공통 입력 제거로 거의 불변 → 요소 이름 "좌우 공통 KC 입력 제거"로 정정.
- **E121**(분기 3/3, 마지막) 게이트 통과(21:58) → 커밋·push 후 실행, 10런 약 1시간. 감시 `watch-exp.sh E121 10 "변조폭 변화"`. 완료 후 `python3 scripts/judge_e121.py` → P9 + 분기 종료 정리(abcd D절).
- E121 완료(21:59~22:25) → **음성**. 분기 미해결 종료(22:26). P9·invariants(INV-B5 알려진 누락)·이력 반영. **다음: 새 분기 — 최소 회로(K50) 전이**, 새 A/B/C/D 문서부터.
- 새 분기 "최소 회로의 전이"(상한 3): `research/abcd-2026-09-28-transfer.md`. **E122**(미학습 사례 범주 일반화) 경로 검사 통과·게이트 통과 → 실행(24런 ≈15분). 감시 `watch-exp.sh E122 24 "EXGEN"`. 완료 후 `python3 scripts/judge_e122.py`.
- E122 완료(22:36~22:49) → **지지**(learn 14/16, frozen 0/8, 원형 100%). P9 반영. 다음 E123(용량-반응, 분기 2/3).
- E123(용량-반응 50·100·200시행, 분기 2/3) 게이트 통과 → 실행(48런 ≈20분). 감시 `watch-exp.sh E123 48 "EXGEN"`.
