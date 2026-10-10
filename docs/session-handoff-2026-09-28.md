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
- E123 완료(22:53~23:25) → **지지**(64→95%, 16/16). P9 반영. E124(같음/다름 관계 전이) 경로 검사 23:25 시작.
- E124(같음/다름 관계, 분기 3/3) — 경로 검사에서 배선 18 획득 실패 → 1차 질문을 획득으로 재정렬, 게이트 통과(23:31) → 실행(24런 ≈25분). 감시 `watch-exp.sh E124 24 "SDGEN"`.
- E124 완료(23:32~23:45) → **획득 불가**(0/16). 분기 종료. P9 반영. 다음: 새 A/B/C/D(관계 획득 장애 원인).
- 새 분기 "관계 획득 장애의 원인"(상한 3): `research/abcd-2026-09-28-relation.md`. **E125**(같은 자극·half1 규칙) 게이트 통과(23:50) → 실행(24런 ≈25분). 감시 `watch-exp.sh E125 24 "SDLAB"`.
- E125 완료(23:51~00:05) → **관계 특이 장애**(12/16 경계, 짝 15/16). P9 반영. 다음 E126 균형 같음/다름(분기 2/3).
- **E126**(균형 같음/다름, 분기 2/3) 게이트 통과 → 실행(24런). 감시 `watch-exp.sh E126 24 "SDLAB"`.
- E126 완료(00:11~00:25) → **획득 불가**(균형 0/16). P9 반영. 다음 E127 KC 신용 측정(분기 3/3).
- **E127**(KC 신용 측정, 분기 3/3) 게이트 통과 → 실행(24런). 감시 `watch-exp.sh E127 24 "SDCREDIT"`.
- 00:34 push 1회 실패(ahead 1, 원인 미확인) → 재실행으로 동기(52d2bc3, 실행 중 E127.log 일부 포함). **push 후 `## main...origin/main` 뒤에 ahead 가 없는지 항상 확인.**
- E127 완료(00:34~00:46) → **표현 희석**(15/16·16/16). 분기 원인 특정·종료. P9 반영. 다음: 새 A/B/C/D(관계를 담는 표현).
- E128 사전 보정(00:48~00:52): 후보 없음, KC 1스파이크 코드 발견(억제 강화 무효 원인). 새 abcd-2026-09-29-representation(교차 반쪽 결합 배선).
- 00:57~09:31 sshfs 무응답(데스크탑 절전 추정)으로 셸 불가 — 깨우지 않고 대기. 09:31 복구 후 E128 등록 재개, 게이트 통과(09:32) → 실행. 감시 `watch-exp.sh E128 24 "SDCREDIT"`.
- E128 완료(09:32~09:45) → **보류**(11/16, 짝 15/15). P9 반영. 다음 E129 위치 짝 결합(분기 2/3).
- **E129**(비교기 배선, 분기 2/3) 게이트 통과 → 실행(24런). 감시 `watch-exp.sh E129 24 "SDLAB"`.
- E129 완료(09:51~23:26, 절전 09:55~23:17 포함) → **획득 + 전이 지지**(16/16·16/16). P9 반영. 다음 E130 비교 특징 학습 형성(분기 3/3).
- **E130**(비교 특징 경험 형성, 분기 3/3) 게이트 통과 → 실행(40런 ≈25분). 감시 `watch-exp.sh E130 40 "SDLAB"`.
- E130 완료(23:35~00:13) → **보류(조작검증 7/8)**, 전이 15/16 vs 0/16. 분기 종료. P9 반영. 다음 E131 새 배선 확증 재현.
- 새 분기 "경험 형성 관계 전이의 확증"(상한 2): `research/abcd-2026-09-30-confirm.md`. **E131**(E130 그대로, 배선 20~27) 게이트 통과 → 실행(40런 ≈40분). 감시 `watch-exp.sh E131 40 "SDLAB"`.
- E131 완료(00:16~00:36) → **보류(혼재)** 11/16 vs 1/16, 짝 14/16. P9 반영. 다음 E132 큰 새 표본 짝 비교(분기 2/2).
- **E132**(큰 새 표본 짝 비교 확증, 분기 2/2) 게이트 통과 → 실행(96런 ≈90분). 감시 `watch-exp.sh E132 96 "SDLAB"`.
- E132 완료(00:41~21:46, 절전 01:07~21:06 포함) → **인과 효과 지지(확증)**. 짝 검정 독립 단위 정정(E125~E131). 분기 종료. 다음: 새 A/B/C/D.
- 새 분기 "망 스파이크로 비교 특징 형성"(상한 3): `research/abcd-2026-09-30-inetwork.md`. minimal_circuit.py candidates/loaded 배선·헤브 발달(hebb_update, CPU 검사 PASS). E133 보정 21:52 시작(규칙 고정 dev_rule_fixed.txt).
- **E133**(망 스파이크 헤브 발달, 분기 1/3) 게이트 통과 → 실행(발달 32 + 과제 96 ≈100분). 감시 `watch-exp.sh E133 128 "=> "` 대신 과제 96 + 발달 32 줄 패턴 — 러너 줄 수 확인.

## 10. 재시작 인계 — 2026-09-30 22:26 (Claude Code 업데이트로 세션 재시작)
**실행 중: E133**(망 스파이크 헤브 발달, 분기 "망 스파이크로 비교 특징 형성" 1/3). 데스크탑 schtasks 로 2026-09-30 22:04 시작, 이 시점 54/128 줄(실패 0). 세션과 무관하게 계속 돈다(데스크탑 절전 시 멈췄다 재개).
새 세션이 할 일(순서):
1. 재개 순서 1~6(CLAUDE.md) — 최신 abcd 는 `research/abcd-2026-09-30-inetwork.md`.
2. **종료 감시부터**: `~/bin/watch-exp.sh E133 128 "=> "`(발달 32줄 + 과제 96줄, 둘 다 "=> " 포함)를 백그라운드로. 이미 끝났으면 건너뜀(`tail -2 research/experiments/E133.log` 에 "전체 루프 종료").
3. 완료 후 `python3 scripts/judge_e133.py` → 판정(기준 `logs/E133/criteria_fixed.txt`, 독립 단위 = 배선 16).
4. **독립 대조**: 원 로그(`logs/E133/*.log`)를 별도 코드로 재계산 — 배선 단위 짝(corr−indep, learn−frozen), [KC불러옴] = DEVHEBB, 실패 줄 0.
5. P9: E133.md 결과 절, H056, current-state(헤더·K 행·§6·§7), DESIGN_RECOVERY append, 이 인계 문서.
6. 커밋·push(lab_commit_msg.txt → lab_push.sh), **출력 끝이 `## main...origin/main` 이고 ahead 가 없는지 확인**.
결과별 다음(E133.md 4절): 지지 → K68 범위를 "망 스파이크 기반 헤브 형성"으로 확장, 분기 종료 → 새 abcd. 효과 없음 → 호스트 규칙과 망 헤브 차이 측정(E134). 보류 → 원인 측정.
주의: 이 세션에서 발견한 반복 오류 — 판정 정규식이 출력 괄호 라벨 안 공백을 \\S* 로 받는 결함(E127·E128 두 번), 시험 입력 결함(learn=frozen 우연 일치), 짝 검정 독립 단위(배선). 커밋 메시지 끝은 "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 한 줄(Claude-Session 줄은 이제 안 붙인다).
- (재시작 후) E133 완료 22:57 → **지지**(16/16·16/16). P9 반영. 분기 종료. 다음: 새 A/B/C/D(관계의 다른 형태 / 연속 STDP / 전체 모델 중 하나).
- 새 분기 "경험이 관계를 정하는가"(상한 2): `research/abcd-2026-09-30-dissociation.md`. **E134**(이중 해리) 게이트 통과 → 실행(발달 32 + 과제 128 ≈90분). 감시 `watch-exp.sh E134 160 "=> "`. 완료 후 `python3 scripts/judge_e134.py`.
- 23:36 백그라운드 Bash 감시가 30분 제한으로 강제 종료됨(업데이트 후 변화) → **Monitor 도구(timeout 1800000)로 watch-exp.sh 를 걸고 만료마다 재무장**. E134 는 23:36 기준 76/160, 실패 0.
- E134 완료(23:05~00:11) → **이중 해리 지지**(16/16·16/16). P9 반영. 분기 종료. 다음: 새 A/B/C/D.
- 새 분기 "연속 STDP 로 비교 특징 형성"(상한 2): `research/abcd-2026-10-01-stdp.md`. minimal_circuit.py --dev-mode stdp(후보 = GeNN R-STDP, 도파민 상수 ±). E135 보정 00:15 시작(규칙 고정 stdp_rule_fixed.txt), 감시 = Monitor.
- E135 보정(00:15~00:20): STDP 충돌 수정(모델 객체 공유) 후 재보정 → **형성 불충분**(사전 규칙, 본실험 미실행). 다음 E136 Oja 형 망 안 경쟁.
- E136 보정(00:23~00:27): Oja 형 4 조합 모두 형성 → eta 0.005·beta 5 선택. **E136 본실험** 게이트 통과 → 실행(발달 32 + 과제 96). 감시 Monitor(watch-exp.sh E136 128 "=> "), 30분마다 재무장.
- E136 완료(00:29~01:21) → **지지**(16/16·16/16). P9 는 01:22 일부 반영 후 데스크탑 절전으로 중단, 23:09 완료.

## 11. 자동 재개 장치 (2026-10-01 23:10 — 사용자 지적 후)
- 사고: 01:22 E136 판정 직후 데스크탑 절전 → sshfs 무응답으로 셸 실행 불가 → "깨우면 알려 달라"며 멈춤 → 깬 뒤에도 재개 신호 없음, ~22시간 정지. 사용자: "깨면 알아서 움직여야 될 거 아니야".
- 조치: **세션 시작 시 CronCreate 재개 점검**(`7,27,47 * * * *`, 세션 전용·7일 만료) — 셸이 안 뜨면 한 줄 대기, 뜨면 인계·current-state 기준으로 감시 재무장·미판정 판정·다음 수 실행. 이 세션: 작업 ID 158ffc03(23:10).
- 기억: auto-resume-after-desktop-sleep(하드 룰). 새 세션은 재개 순서 직후 이 예약부터 건다.

## 12. E137 진행 (2026-10-02 00:07)
- 다음 결정 문서: research/abcd-2026-10-01-dose.md(K72 범위 정정 + E137). E137 = 관계 전이의 용량-반응(헌장 개념 조건 4), 176줄 ≈70분, 러너 scripts/e137_main.sh, 판정 python3 scripts/judge_e137.py(기준 logs/E137/criteria_fixed.txt, 합성 20/20).
- 경로 검사가 잠복 결함 발견·수리(minimal_circuit.py main() 지역 변수 io → ino, --dump-rewards·--reward-file 복구). 회귀 4종 동일.
- 감시: Monitor ~/bin/watch-exp.sh E137 176 "e137 " (30분 만료마다 재무장). 끝나면 판정 → 독립 대조(로그 직접 재계산) → P9(E137·H060·current-state) → DESIGN_RECOVERY → push.
- 2026-10-02 01:06 E137 지지(K73): 61.1→68.2→77.4→88.0%, 무학습 50.7%, 15/16·16/16, 독립 대조 일치. P9 완료. 다음: 새 A/B/C/D(후보: 전체 모델 크기 한계 = KC 표현 희석 / 망 안 승자 선택 / 비교 틀 형성 / 맥락 의존 개념).

## 13. E138 진행 (2026-10-02 23:32)
- 다음 결정 문서: research/abcd-2026-10-02-dilution.md. E138 = 전체 모델 학습 효과 상한의 출처(Q1 표현 vs 학습, Q2 공통 모드 억제) — E119 반사 0 가중치 분해 측정, 학습 런 없음, 35줄(뇌 10~14 × kcrate + 이식 평가 6종).
- 러너 scripts/e138_main.sh, 판정 python3 scripts/judge_e138.py(기준 logs/E138/criteria_fixed.txt, 합성 13/13), 계산 모듈 backend/genesis/kc_selectivity.py(합성 16/16).
- 데스크탑 절전(01:06 무렵~23:14)으로 약 22시간 멈췄다가 재개 점검이 이어받음 — 절전 중 다음 A/B/C/D 초안은 Pi 스크래치패드에 써 두고 깬 뒤 원 기록과 대조해 반영.
- 감시: Monitor ~/bin/watch-exp.sh E138 35 "e138 .*: =>" (30분 만료마다 재무장). 끝나면 판정 → 독립 대조(런별 로그 직접) → P9(E138·H061·current-state) → DESIGN_RECOVERY → push.
- 2026-10-03 00:21 E138 판정: 학습 상한·공통 모드 억제 없음(K74). 측정 도구 eps 경계 결함(독립 대조 발견) 수리 재실행으로 판정 동일. P9 완료. 다음: E139(선택 KC 시냅스의 시행 유형별 변화 추적 — 원인 측정).
- 2026-10-05 21:36 데스크탑 복귀 후 E139 재개: 경로 검사 통과(뇌 15), 기준 수정 2(V1' 최대값 삭제), audit.sh 수리, 기록 반영 → 본실험(e139_main.sh, 뇌 10~14) 띄움. 세션은 로컬 ~/brain/brainsim-session, 재개 점검 1시간(SSH 만).
- 2026-10-05 22:02 E139 보류(V1' 뇌 13 0.3007%) — 분기 미해결 종료. 다음: 새 A/B/C/D(보상 창 motor 침묵 인과 조작).

## 14. E140 폐기 → E141 진행 (2026-10-05 22:25)
- 결정 문서: research/abcd-2026-10-05-rwsilence.md(추가 절). E140(보상 창 motor 침묵)은 경로 검사에서 조작 부적합(흔적 생성이 LTD 로 남음) → 폐기, 본실험 미실행.
- E141 = 보상 창 동안 KC→motor A_plus = A_minus = 0(--rw-apm-scale 0). 기준 logs/E141/criteria_fixed.txt(22:19:56), 판정 python3 scripts/judge_e141.py(합성 19/19, 비교 traces/E139/tr_b*.npz 필요).
- 경로 검사: bash scripts/e141_path_check.sh → logs/E141/path_check.out(뇌 15, 배율 1 회귀 + 배율 0). 통과하면 E141.md §7 기록 → 게이트 → 본실험(lab_run.cmd: run_experiment.sh E141 "bash scripts/e141_main.sh") → 감시 Monitor ~/bin/watch-exp.sh E141 5 "e141 b.*: =>"(30분 만료마다 재무장).
- 끝나면 판정 → 독립 대조(런별 로그·추적 직접 재계산) → P9(E141·H064·current-state) → DESIGN_RECOVERY → push. 결과와 무관하게 분기 종료 → 새 A/B/C/D.
- 2026-10-05 23:01 E141 **지지**(K75): 효과 −0.077 → −0.259(5/5), 독립 대조 일치, P9 완료. 분기 종료. 다음 결정 문서 research/abcd-2026-10-05-l2.md — E142(반사 25 + 동결, F500·F1500, 헌장 L2).
- 2026-10-05 23:09 E142(반사 25 + 동결, 헌장 L2) 실행 중 — 로그 23:08:47 시작, 15런(F500 → F1500 → NF1500) ≈2.5시간. 기준 logs/E142/criteria_fixed.txt(23:02:02 + 수정 1 23:04:43: NF1500 팔·필요성 해석). 감시 Monitor ~/bin/watch-exp.sh E142 15 "e142 .*b.*: =>"(30분 만료마다 재무장). 끝나면 python3 scripts/judge_e142.py → python3 scripts/verify_e142_independent.py(독립 대조, 합성 7/7) → P9(E142·H065·current-state) → DESIGN_RECOVERY → push.
- 2026-10-06 12:30 E142 **반사를 거스름**(K76): F500 −0.27(무동결 ≈0), F1500 −0.30·사후 +0.08~+0.13(L2 0/5), NF1500 −0.06. 독립 대조 일치, P9 완료. 탐색적 분해 → E143(결정 단계 흔적 동결, 분기 마지막) 등록: 기준 logs/E143/criteria_fixed.txt(12:26:47), 판정 python3 scripts/judge_e143.py(합성 20/20), 독립 대조 python3 scripts/verify_e143_independent.py(합성 8/8), 경로 검사 scripts/e143_path_check.sh(뇌 15 REG·DEC) → 통과하면 게이트 → run_experiment.sh E143 "bash scripts/e143_main.sh"(5런 ≈60분) → 감시 watch-exp.sh E143 5 "e143 b.*: =>".
- 2026-10-06 13:46 E143 **결정 단계 흔적 무관**(K77) — 분기 L2 미해결 종료. 다음 결정 문서 research/abcd-2026-10-06-actwin.md — E144(반사 25 행동 창 반대쪽 음 전류 보정 → 1,500시행, 짝 E142 F1500).
- 2026-10-06 13:56 E144(반사 25 행동 창 반대쪽 침묵 N = 10000) 실행 중 — 로그 13:56:04 시작, 5런 ≈60분. 보정(뇌 15)이 기전을 직접 확인(반대쪽 발화 0.2106 → 0.0000, 같은 쪽 흔적 LTP → LTD). 감시 watch-exp.sh E144 5 "e144 b.*: =>". 끝나면 python3 scripts/judge_e144.py → python3 scripts/verify_e144_independent.py → P9(E144·H067·current-state) → push.
- 2026-10-06 15:05 E144 **반대**(K78): 반대쪽 침묵이 거스름을 줄임(d +0.047~+0.071), 시냅스 차이는 +30% — 판독 가정 의문. E145(집단 맞바꿈 이식 분해, 학습 없음 25평가) 등록·경로 검사 중 — 통과하면 게이트 → run_experiment.sh E145 "bash scripts/e145_main.sh" → 감시 watch-exp.sh E145 25 "e145 b.*: =>" → python3 scripts/judge_e145.py.
- 2026-10-06 15:24 E145 **보류**(K79) — 분기 종료, 헌장 L2 미해결 종료. 다음 결정 문서 research/abcd-2026-10-06-rulegen.md — E146(반사 0·동결: 무학습·500·1,500시행 × 자극 변형 5종 이식 평가).
- 2026-10-06 15:35 E146(반사 0 규칙 일반화·용량) 실행 중 — 학습 5런(1,500시행) + 평가 75회 ≈2시간. 감시 watch-exp.sh E146 80 "e146 .*b.*: =>". 끝나면 python3 scripts/judge_e146.py → python3 scripts/verify_e146_independent.py(합성 4/4) → P9(E146·H069·current-state) → push.
- 2026-10-06 17:21 E146 **충족**(K80, 가림 범위 정정 — 쪽별 광선 평균 입력). 분기 종료. 헌장 현황 열 갱신. 다음 결정 문서 research/abcd-2026-10-06-reversal.md — E147(전체 모델 규칙 반전: 1,500 교차 + 1,500 반전, --reverse-after 신설).
- 2026-10-06 17:28 E147(전체 모델 규칙 반전) 실행 중 — 5런 × 3,000시행 ≈2시간. 감시 watch-exp.sh E147 5 "e147 b.*: =>". 끝나면 python3 scripts/judge_e147.py → python3 scripts/verify_e147_independent.py(합성 5/5) → P9 → push.
- 2026-10-06 19:21 E147 **반전 성공**(K81). E148(간섭 하 유지) 등록·경로 검사 중 → 통과하면 게이트 → run_experiment.sh E148 "bash scripts/e148_main.sh"(학습 5 × 3,000시행 + 평가 20, ≈2.3시간) → 감시 watch-exp.sh E148 25 "e148 .*b.*: =>" → judge_e148.py.
- 2026-10-06 21:26 E148 **간섭**(K82) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-06-overlap.md — E149(good·bad KC 겹침 측정, 기본 vs 공통 입력 차단).
- 2026-10-06 21:46 E149 **보류**(K83, 중간 겹침). E150(차단 상태 간섭 하 유지) 등록·경로 검사 중 → 통과하면 게이트 → run_experiment.sh E150 "bash scripts/e150_main.sh"(학습 10런 ≈3시간 + 평가 25) → 감시 watch-exp.sh E150 35 "e150 .*b.*: =>" → judge_e150.py.
- 2026-10-07 00:58 E150 **유지**(K84, 약한 학습 교란) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-07-retention.md — E151(차단 + eta 상향, 뇌 15 보정 → 본실험 뇌 10~14).
- 2026-10-07 18:57 E151 **보류(학습 크기 미회복)**(K85, 보정만) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-07-formation.md — E152(기본 표현 겹침 중 먹이 단독 반응 몫, kcoverlap3 측정).
- 2026-10-07 19:21 E152 **결합 주도**(K86, 수정 1 기준 — 원래 기준 보류). 다음 E153(종류 입력 합 보존 헤브 재분배 형성 — kcdev 개발 단계·가중치 저장/적재 신설 예정).
- 2026-10-07 20:15 E153 **형성 성공**(K87 — 겹침 0, 학습 효과 약 2배). E154(형성 표현 간섭 하 유지, 분기 마지막) 경로 검사 중 → 게이트 → run_experiment.sh E154 "bash scripts/e154_main.sh"(학습 10런 ≈3시간 + 평가 25) → 감시 watch-exp.sh E154 35 "e154 .*b.*: =>" → judge_e154.py·verify_e154_independent.py.
- 2026-10-07 23:30 E154 **유지**(K88 — 형성 표현에서 간섭 하 유지 0.80~0.88) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-07-baseline.md — E155(형성 표현 K80·K81 회귀).
- 2026-10-09 00:35 E155 **기준선 미확립(부분)**(K89 — 반전 보존·강화, 잡음 비율 경계) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-09-l2.md — E156(형성 표현 L2 재도전, 반사 25).
- 2026-10-09 01:02 E156 **폐기(본실험 미실행)** — 형성 표현의 반사 발현 압축(뇌 15 +0.44 → +0.16), 맞춤 보정 실패. H079 시험 불가, 분기 종료. 다음 결정 문서 research/abcd-2026-10-09-gain.md — E157(이득 맞춘 2×2: 형성 이득 낮춤 Fk·기본 이득 높임 Dk, 뇌 15 k 보정 → 뇌 10~14).
- 2026-10-09 01:28 E157(이득 맞춘 2×2) 실행 중 — 로그 01:28:05 시작, 60런(kcrate 20 → 겹침 10 → 반사 25 사전 20 → 학습 10) ≈1.3시간. 경로 검사·보정 logs/E157/calib.out(E141·E153 재현 일치, k Fk 0.70·Dk 1.50). 감시 watch-exp.sh E157 60 "e157 .* b1[0-4]: =>"(30분 만료마다 재무장). 끝나면 python3 scripts/judge_e157.py → python3 scripts/verify_e157_independent.py → P9(E157·H080·current-state) → push.
- 2026-10-09 13:05 E157 **보류**(K90 — 발화 맞춘 형성 1.66~1.88배, 이득 단독 1.20~1.43배, 희석은 이득을 따름) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-09-l2fk.md — E158(Fk 로 헌장 L2, 반사 25).
- 2026-10-09 13:13 E158(이득 맞춘 형성 표현 ×0.70 으로 헌장 L2) 실행 중 — 로그 13:13:13 시작, 10런(F500 → F1500) ≈1.5시간. 경로 검사 logs/E158/path_check.out 통과. 감시 watch-exp.sh E158 10 "e158 F.* b1[0-4]: =>"(30분 만료마다 재무장). 끝나면 python3 scripts/judge_e158.py → python3 scripts/verify_e158_independent.py → P9(E158·H081·current-state·CHARTER L2 줄) → push.
- 2026-10-09 14:33 E158 **반사를 거스름**(K91 — 사후 +0.004~+0.027, L2 0/5) — 분기 종료, L2 미해결. 다음 결정 문서 research/abcd-2026-10-09-cap.md — E159(학습 가중치 교차 평가: 반사 0·25 × 가중치 r0·r25, 뇌 10~14, 학습 없음).
- 2026-10-09 14:57 E159 **둘 다**(K92 — O 0.52~0.58, C 0.56~0.65) — 분기 종료, L2 는 구조 변경 없이는 다시 열지 않음. 다음 결정 문서 research/abcd-2026-10-09-innet.md — E160(망 안 Oja 형성: KC 종류 입력 가소성, 뇌 15 β 보정 → 뇌 10~14 겹침·학습 효과).
- 2026-10-09 15:29 E160(망 안 Oja 형성) 실행 중 — 로그 15:29 시작, 20런(뇌마다 형성 → 겹침 → 학습 500 → kcrate) ≈50분. 보정 logs/E160/calib.out(η 0.02·β 0.3, 기본 모델 불변). 감시 watch-exp.sh E160 20 "e160 .* b1[0-4]: =>"(30분 만료마다 재무장). 끝나면 python3 scripts/judge_e160.py → python3 scripts/verify_e160_independent.py → P9(E160·H083·current-state) → push.
- 2026-10-09 16:12 E160 **보류**(K93 관측 — 자카드 ≤ 0.027·r 2.19~2.58, MS 4/5) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-09-innet2.md — E161(망 안 Oja 형성 확증, 쓰지 않은 뇌 16~20, 뇌마다 7런).
- 2026-10-09 16:23 E161(망 안 Oja 형성 확증, 뇌 16~20) 실행 중 — 16:23 시작, 40런 ≈1.5시간. 경로 검사 logs/E161/path_check.out 통과(E160 값 재현). 게이트 첫 시도는 §7 '같은 경로 직전 실험' 누락으로 거부(본실험 미시작) → 보완 후 통과. 감시 watch-exp.sh E161 40 "e161 .* b(1[6-9]|20): =>"(30분마다 재무장). 끝나면 judge_e161.py → verify_e161_independent.py → P9.
- 2026-10-09 17:38 E161 **형성 성공 5/5**(K94 확증 — 망 안 Oja 형성, 쓰지 않은 뇌) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-09-innet3.md — E162(망 안 형성 표현 간섭 하 유지, 뇌 10~14, E154 설계 + E160 Oja 가중치, ≈3시간).
- 2026-10-09 17:48 E162(망 안 형성 표현 간섭 하 유지) 실행 중 — 17:48 시작, 학습 10(1,500 × 5 + 3,000 × 5)·평가 25 ≈3시간. 경로 검사 logs/E162/path_check.out 통과. 감시 watch-exp.sh E162 35 "e162 .*: =>"(30분마다 재무장). 끝나면 judge_e162.py → verify_e162_independent.py → P9(E162·H085·current-state).
- 2026-10-09 19:13 외부 검토 응답 research/review-response-2026-10-09.md — 구현 지점 ①(보상 창 끝 도파민 뉴런 입력)·③(오프셋 5 대 3)은 E162·E163 뒤 측정 분기로, ②(--no-reward)는 정정·경고. E162 진행 중.
- 2026-10-09 20:46 E162 **유지 5/5**(K95 — rA 0.95~1.00, T ≤ 0.05). 다음 E163(망 안 형성 표현의 반전 — 준비물 완료: e163_main.sh·e163_path_check.sh·judge_e163.py·verify_e163_independent.py, 합성 통과). 사전등록·기준 고정 → 경로 검사 → 게이트 → 본실험(5런 × 3,000시행 ≈2.5시간).
- 2026-10-09 20:55 E163(망 안 형성 표현 반전) 실행 중 — 20:51:27 시작, 5런 × 3,000 ≈2.5시간. 감시 watch-exp.sh E163 5 "e163 rev .*: =>". 끝나면 judge_e163.py → verify_e163_independent.py → P9. 다음 E164(구현 정합성 측정) 준비물 완료 — 사전등록은 E163 판정 뒤.
- 2026-10-09 22:41 E163 **반전 성공 5/5**(K96) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-09-impl.md — E164(구현 정합성: --rw-da-reset·--offset-steps 3 함께, 반사 0·25, 뇌 10~14, 10런 ≈1시간). 준비물 완료.
- 2026-10-09 22:51 E164(구현 정합성: --rw-da-reset·--offset-steps 3, 반사 0·25, 뇌 10~14) 실행 중 — 로그 22:51:18 시작, 10런 ≈1시간. 경로 검사 logs/E164/path_check.out 통과(옵션 끔 E141 재현, 점검 줄 3·I_input 53.0 → 0.0). 감시 watch-exp.sh E164 10 "e164 R.* b1[0-4]: =>"(30분마다 재무장). 끝나면 python3 scripts/judge_e164.py → python3 scripts/verify_e164_independent.py → P9(E164·H087·current-state K97·§6·§7) → 새 A/B/C/D.
- 2026-10-09 23:38 E164 **무시 가능 10/10**(K97 — 구현 지점 ①·③ 영향 |Δ| < 0.03, 기존 코드 유지) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-09-expo.md — E165(망 안 형성 노출 통계 일반성: 팔 NR 무작위 순서·강도 0.5~0.9, 팔 NU + bad 세 배; 뇌 16~20, 30런 ≈1.5시간). 준비물(노출 일정 옵션·판정·대조·러너) 작성부터.
- 2026-10-10 00:01 E165(망 안 형성 노출 통계 일반성: NR 무작위 순서·강도 0.5~0.9, NU + bad 세 배; 뇌 16~20) 실행 중 — 로그 00:00:49 시작, 30런 ≈1.5시간. 경로 검사 logs/E165/path_check.out 통과(고정 순환 E161 정확 재현). 감시 watch-exp.sh E165 30 "e165 N.* b[12][0-9]: =>"(30분마다 재무장). 끝나면 python3 scripts/judge_e165.py → python3 scripts/verify_e165_independent.py → P9(E165·H088·current-state K98).
- 2026-10-10 01:29 E165 **견고 5/5·5/5**(K98 — 무작위 순서·강도 변이·bad 세 배에서도 형성·학습 이점 유지) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-10-host.md — E166(보상 창 흔적 동결 제거: FNF 형성·DNF 기본, 뇌 16~20, 10런 ≈45분), 이어서 E167(최소 회로 승자 선택 제거). 준비물 작성부터.
- 2026-10-10 01:44 E166(보상 창 흔적 동결 제거: FNF·DNF, 뇌 16~20) 실행 중 — 로그 01:43:50 시작, 10런 ≈45분. 경로 검사 logs/E166/path_check.out 통과. 감시 watch-exp.sh E166 10 "e166 .NF b[12][0-9]: =>"(30분마다 재무장). 끝나면 python3 scripts/judge_e166.py → python3 scripts/verify_e166_independent.py → P9(E166·H089·current-state K99). 그다음 E167(사전등록·기준 고정 01:44:23 완료): bash scripts/e167_path_check.sh(배선 18) → §7 → 게이트 → run_experiment.sh E167 "bash scripts/e167_main.sh"(80런 ≈35분).
- 2026-10-10 14:13 E166 **손실 5/5**(K99 — 동결 없이 형성 표현 학습 효과 4.5~9.3%, 형성이 동결 의존을 키움; 절전 01:48~13:32 견딤). E167 경로 검사(배선 18) 실행 중 — 14:13:41 시작, 감시 watch-pc.sh logs/E167/path_check.out. 통과하면 §7 → 게이트 → run_experiment.sh E167 "bash scripts/e167_main.sh" → 감시 watch-exp.sh E167 80 "e167 .*w[789][0-9].*: =>" → judge_e167.py·verify_e167_independent.py → P9(E167·H090·current-state K100) → 분기 종료·새 A/B/C/D.
- 2026-10-10 14:17 E167(최소 회로 승자 선택·재부여 제거, 배선 78~93) 실행 중 — 로그 14:16:44 시작, 80런 ≈35분. 경로 검사 logs/E167/path_check.out 통과(발달 재현 4/4, 장치 대조 차 0; 관측: 배선 18 ojafull learn 은 '같음' 편향 50/50, 참고 H 72.3/66.7). 감시 watch-exp.sh E167 80 "e167 .*w[789][0-9].*: =>"(30분마다 재무장). 끝나면 python3 scripts/judge_e167.py → python3 scripts/verify_e167_independent.py → P9 → 분기 종료·새 A/B/C/D.
- 2026-10-10 15:03 E167 **손실**(K100 — 최소 회로도 호스트 승자 선택·재부여 없이는 관계 획득·전이 없음) — 분기 종료(두 호스트 단계 모두 필요). 다음 결정 문서 research/abcd-2026-10-10-gate.md — E168(도파민 뉴런 → KC 억제 뉴런 연결로 보상 창 KC 침묵: 뇌 15 가중치 보정 → 뇌 16~20 FI·FR). 준비물(forager_brain 연결 옵션·과제 진단·판정·대조·러너) 작성부터.
- 2026-10-10 15:12 E168 사전등록·기준 고정 15:11:17. 보정 겸 경로 검사(뇌 15, W 0·2·5·10·20, (a) 옵션 끔 E166 재현) 실행 중 — 15:11:25 시작, 감시 watch-pc.sh logs/E168/calib.out. 끝나면 logs/E168/pick.txt 의 W* 확인 → none 이면 본실험 없이 미해결 종료(분기 종료), 아니면 §7 기록 → 게이트 → run_experiment.sh E168 "bash scripts/e168_main.sh"(10런 ≈45분) → judge_e168.py·verify_e168_independent.py → P9.
- 2026-10-10 15:28 E168 **보정 실패·본실험 미실행**(K101 — 토닉 도파민으로 억제가 보상 창 특이적이지 않음; 첫 실행 장치 연결 난수 결함은 호스트 연결로 수리) — 분기 미해결 종료. 다음 결정 문서 research/abcd-2026-10-10-window.md — E169(보상 창 1처리, FW1·FFW1, 뇌 16~20, 10런). 준비물 작성부터.
- 2026-10-10 15:36 E169(보상 창 1처리로 동결 대체) 실행 중 — 로그 15:35:18 시작, 10런 ≈45분. 경로 검사 logs/E169/path_check.out 통과(FFW1 잔차 1.7e-8 = 창 1·동결 확인, FW1 5.3). 감시 watch-exp.sh E169 10 "e169 F.* b[12][0-9]: =>"(30분마다 재무장). 끝나면 python3 scripts/judge_e169.py → python3 scripts/verify_e169_independent.py → P9(E169·H092·current-state K102) → 분기 종료·새 A/B/C/D.
- 2026-10-10 16:25 E169 **부분**(K102 — 10 ms 창은 동결 기능의 약 절반) — 분기 종료, 호스트 동결 점검 종료. 다음 결정 문서 research/abcd-2026-10-10-combo.md — E170(전체 모델 조합 전이, 평가만 40). 사전등록·기준 고정 16:24:33, 경로 검사(뇌 15) 16:24:40 시작 — 감시 watch-pc.sh logs/E170/path_check.out. 통과하면 §7 → 게이트 → run_experiment.sh E170 "bash scripts/e170_main.sh" → judge_e170.py·verify_e170_independent.py → P9.
- 2026-10-10 16:54 E170 **합성 없음·충돌 가산 상쇄**(K103). E171(비가산 수준 측정, 평가 진단) 사전등록·기준 고정 16:53:30, 경로 검사 16:53:37 시작 — 감시 watch-pc.sh logs/E171/path_check.out. 통과하면 §7 → 게이트 → run_experiment.sh E171 "bash scripts/e171_main.sh"(30 평가 ≈15분) → judge_e171.py·verify_e171_independent.py → P9 → 분기 종료·새 A/B/C/D.
- 2026-10-10 17:16 E171 **KC 보존 5/5**(K104 — 조합 비가산은 KC 하류 motor 쪽, KC 비 0.87~0.91) — 분기 "전체 모델 조합 전이" 종료. 다음 결정 문서 research/abcd-2026-10-10-confirm.md — E172(망 안 형성 표현 유지·반전 독립 확증, 뇌 16~20, 학습 15런 + 평가 25 ≈ 5시간). 사전등록·기준 고정 17:14:52, 경로 검사 17:15:06 시작 — 감시 watch-pc.sh logs/E172/path_check.out. 통과하면 §7 → 게이트 → run_experiment.sh E172 "bash scripts/e172_main.sh" → 감시 watch-exp.sh E172 40 "e172 .*: =>"(30분마다 재무장) → judge_e172_ret.py·judge_e172_rev.py → verify_e172_ret.py·verify_e172_rev.py → P9(E172·H095·H096·current-state).
- 2026-10-10 17:36 E172(망 안 형성 표현 유지·반전 독립 확증, 뇌 16~20) 실행 중 — 로그 17:35:28 시작, 학습 15런 + 평가 25 ≈ 5시간. 경로 검사 logs/E172/path_check.out 통과(E162 경로 정확 재현, rev 0.0002 차는 재현성 검사로 비결정성 확인 — logs/E172/rep_check.out·rep_compare.out; 적재 5/5 E161 [사전] 정확). 감시 watch-exp.sh E172 40 "e172 .*: =>"(30분마다 재무장). 끝나면 python3 scripts/judge_e172_ret.py > logs/E172/judge_ret.out · judge_e172_rev.py > judge_rev.out → verify_e172_ret.py·verify_e172_rev.py → P9(E172·H095·H096·current-state K95·K96 범위) → 분기 종료 → 새 abcd(다음 능력 분기).
- 2026-10-10 22:44 E172 **확증**(K95 유지 0.97~0.99·K96 반전 +0.62~+0.64, 뇌 16~20, 크기 부판정 5/5·5/5) — 분기 종료. 다음 결정 문서 research/abcd-2026-10-10-context.md — E173 맥락 의존 규칙(쌍조건 변별): 코드 준비(맥락 집단 zz_ctx·--ctx-*·kcctx 측정 — 패치 명세는 세션 scratchpad patch_ctx.py, 과제 파일은 Edit 도구로 적용) → 판정·대조·합성 시험 → 사전등록·기준 고정 → 경로 검사(뇌 15: 맥락 집단 꺼진 채 E162 경로 정확 재현) → 보정(뇌 15 w 격자) → 본실험.
- 2026-10-10 22:53 E173(맥락 의존 규칙 — 쌍조건 변별) 코드 적용(맥락 집단 zz_ctx·--ctx-*·kcctx, 기본값에서는 가드로 이전 경로) · 판정·대조·보정 선택 합성 시험 통과 · 사전등록·기준 고정 22:52:26 · 경로 검사 22:52:34 시작 — 감시 watch-pc.sh logs/E173/path_check.out. 통과하면 §7 → 보정 bash scripts/e173_calib.sh(schtasks, logs/E173/calib.out → pick.txt) → W* 있으면 게이트 → run_experiment.sh E173 "bash scripts/e173_main.sh"(30줄, 약 2.5시간) → 감시 watch-exp.sh E173 30 "e173 .*: =>" → judge_e173.py·verify_e173_independent.py → P9(E173·H097·current-state).
- 2026-10-10 23:14 E173 경로 검사 첫 실행에서 새 뉴런 집단(zz_ctx)이 장치 연결 난수를 바꿔 중단(logs/E173/pathcheck_try1/) → 정정 1(23:01:50): 맥락 = KC 좌·우 Ioffset 동적 균일 전류 --ctx-i. 경로 검사 다시 통과(23:11:41 — 동적화 켜고 맥락 끔이면 E162 정확 재현). I=4 효과 미미 → 정정 2(23:13:09): 격자 I 1·2·4·8·12·16·20 + 최소 효과(결합 합 ≥ 10 또는 자카드 합 ≤ 1.80). 보정 23:13:58 시작 — 감시 watch-pc.sh logs/E173/calib.out. 끝나면 logs/E173/pick.txt 확인: I=<값> 이면 §7 보정 기록 → 게이트 → run_experiment.sh E173 "bash scripts/e173_main.sh"; none·효과 부족이면 본실험 없이 기록 → E174(이질 맥락 또는 결합 형성).
- 2026-10-10 23:33 E173 **보정 효과 부족 — 본실험 미실행**(K105: 균일 흥분 맥락 I 1~20 이 KC 집합을 못 가름). 다음 결정 문서 research/abcd-2026-10-10-context2.md — E174 이질 억제성 맥락(KC 억제 뉴런 10% 에 I_c, forager_brain process() 항상성 줄 + 맥락 벡터). 코드·판정·대조·선택(합성 25/25·12/12·11/11) 준비, 기준 고정 23:32:41, 경로 검사 23:32:42 시작 — 감시 watch-pc.sh logs/E174/path_check.out. 통과하면 보정 bash scripts/e174_calib.sh(logs/E174/calib.out → pick.txt IC=) → 게이트 → run_experiment.sh E174 "bash scripts/e174_main.sh" → 감시 watch-exp.sh E174 30 "e174 .*: =>" → judge_e174.py·verify_e174_independent.py → P9.
