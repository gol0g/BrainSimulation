#!/bin/bash
# 실험 사전등록 파일 생성. 이것 없이는 run_experiment.sh가 실행을 거부한다.
# 사용: bash scripts/lab/new_experiment.sh E070 "w_max 상한 판별"
set -eu
R="$(cd "$(dirname "$0")/../.." && pwd)"
ID="${1:?사용: new_experiment.sh E### \"제목\"}"
TITLE="${2:?제목이 필요하다}"
F="$R/research/experiments/$ID.md"
[ -e "$F" ] && { echo "이미 존재: $F"; exit 1; }
sed "s|^# E###: <제목>|# $ID: $TITLE|; s|^- \*\*등록일\*\*:|- **등록일**: $(date +%Y-%m-%d)|" \
  "$R/research/experiments/_TEMPLATE.md" > "$F"
echo "생성됨: $F"
echo "→ 1~5번 항목을 모두 채운 뒤 run_experiment.sh 로 실행하라."
