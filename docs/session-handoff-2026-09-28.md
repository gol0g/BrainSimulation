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
