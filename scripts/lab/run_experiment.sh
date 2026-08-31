#!/bin/bash
# 사전등록된 실험만 실행한다. 로그를 실험 파일 옆에 남긴다.
# 사용: bash scripts/lab/run_experiment.sh E070 "실행할 커맨드..."
set -eu
R="$(cd "$(dirname "$0")/../.." && pwd)"
ID="${1:?사용: run_experiment.sh E### \"커맨드\"}"
shift
CMD="${*:?실행할 커맨드가 필요하다}"

bash "$R/scripts/lab/gate.sh" "$ID" || exit $?

LOG="$R/research/experiments/$ID.log"
echo "[실행] $ID"
echo "  커맨드: $CMD"
echo "  로그: $LOG"
{
  echo "=== $ID 실행: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "커맨드: $CMD"
  echo "---"
} >> "$LOG"
set +e
bash -c "$CMD" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
echo "=== 종료코드 $RC ===" >> "$LOG"
echo "[실행] 완료(rc=$RC). 결과를 $R/research/experiments/$ID.md 의 '실행 결과'에 기록하고"
echo "       hypotheses/H###.md 와 current-state.md 를 갱신하라(P9)."
exit $RC
