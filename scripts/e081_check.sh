#!/bin/bash
# E081 조작검증: 감마분포가 D1 다양성(CV)을 만드는가.
# 2026-09-16 수정: 이전 판에서 감마 조건이 **빈 줄**로 나왔다. grep 필터가 예외를 삼켰기 때문이다
#   (원인: init_var("Gamma") 파라미터명 shape/scale → GeNN은 a/b).
#   재발 방지로 실패 시 마지막 3줄을 그대로 출력한다. 조용한 실패 금지.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG=/root/kc_run/e081_check.log
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/kc_run
cp $R/backend/genesis/*.py . 2>/dev/null

run_one () {
  local label="$1"; shift
  printf "%s: " "$label"
  timeout 2400 python d1_diversity_probe.py "$@" > "$LOG" 2>&1
  local rc=$?
  local out
  out=$(grep -E "d1_left|d1_right" "$LOG" | tr -s ' ' | tr '\n' ' ')
  if [ -z "$out" ]; then
    echo "[실패 rc=$rc]"
    tail -3 "$LOG" | sed 's/^/    /'
  else
    echo "$out"
  fi
}

rm -rf forager_brain_CODE CODE
for S in 0 1 2; do
  run_one "기본  seed=$S" --kc-rstdp --seed "$S"
done
rm -rf forager_brain_CODE CODE
for S in 0 1 2; do
  run_one "감마  seed=$S" --kc-rstdp --kc-gamma --seed "$S"
done
