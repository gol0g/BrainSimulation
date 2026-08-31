#!/bin/bash
# 사전등록된 실험만 실행한다. 로그는 파일로 직접 쓴다.
#
# P8 대응(실측 사고 2026-08-31): 이전 판은 `| tee`를 써서, 호출부가 `| head -6`을 붙이면
# SIGPIPE로 **실험 전체가 죽었다**(종료코드 13, 첫 런에서 중단). 규약만으로는 막지 못했으므로
# 스크립트가 하류 독자와 무관하게 동작하도록 고쳤다. stdout에는 요약만 낸다.
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"
ID="${1:?사용: run_experiment.sh E### \"커맨드\"}"
shift
CMD="${*:?실행할 커맨드가 필요하다}"

bash "$R/scripts/lab/gate.sh" "$ID" || exit $?

LOG="$R/research/experiments/$ID.log"
{
  echo "=== $ID 실행: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "커맨드: $CMD"
  echo "---"
} >> "$LOG"

echo "[실행] $ID → 로그: $LOG"
bash -c "$CMD" >> "$LOG" 2>&1
RC=$?
echo "=== 종료코드 $RC ===" >> "$LOG"

echo "[실행] 완료(rc=$RC), 로그 $(wc -l < "$LOG") 줄"
echo "[다음] 결과를 research/experiments/$ID.md 에 기록하고 hypotheses/·current-state.md 를 갱신하라(P9)."
exit $RC
