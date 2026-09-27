#!/bin/bash
# E108: D1 경로 행동 권한 보정 스윕(학습 없음) — 설정 6 × 뇌 0~4. 재개 가능(결과 줄 기준).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E108.log"
RAW="$R/research/experiments/logs/E108"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e108_run && cd /root/e108_run
cp $R/backend/genesis/*.py . 2>/dev/null
run () {  # tag d1w dinh brain
  local tag="$1"
  if grep -qF "$tag: => CALIB" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 1800 python reflex_override_task.py --calib-d1-sign 20 --episodes 0 --brain-seed "$4" --env-seed 0 \
    --real-rstdp --crossed --d1-inhib -400 --direct-inhib "$3" --d1-direct-w "$2" --bias 25 > "$f" 2>&1
  grep -E "^=> CALIB" "$f" || { echo "[실패]"; tail -2 "$f" | sed 's/^/      /'; }
}
for S in "S0 20 -100" "S1 40 -100" "S2 80 -100" "S3 20 -50" "S4 40 -50" "S5 80 -50"; do
  set -- $S
  for B in 0 1 2 3 4; do run "$1 b$B" "$2" "$3" "$B"; done
done
echo "[E108] 전체 루프 종료"
