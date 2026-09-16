#!/bin/bash
# E081 재실행 — 불변식 INV-A4(d1_inhibition=-400) 적용판.
# 1차 조작검증은 d1_inhibition 기본값 0.0(포화)에서 돌았다. 불변식 위반이므로 무효.
# 포화를 푼 상태에서 (1) 기본 CV (2) 감마 CV (3) KC→D1 용량-반응을 다시 잰다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG=/root/kc_run/e081c.log
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/kc_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

run_one () {
  local label="$1"; shift
  printf "%-22s: " "$label"
  timeout 2400 python d1_diversity_probe.py "$@" > "$LOG" 2>&1
  local rc=$?
  local out
  out=$(grep -E "d1_left|d1_right" "$LOG" | tr -s ' ' | tr '\n' '|')
  if [ -z "$out" ]; then echo "[실패 rc=$rc]"; tail -3 "$LOG" | sed 's/^/    /'; else echo "$out"; fi
}

echo "### A. 기본 vs 감마 (INV-A4 적용, 3시드)"
for S in 0 1 2; do run_one "기본 seed=$S" --kc-rstdp --seed "$S"; done
for S in 0 1 2; do run_one "감마 seed=$S" --kc-rstdp --kc-gamma --seed "$S"; done
echo
echo "### B. KC→D1 용량-반응 (seed=1)"
for W in 0 0.5 5 20; do run_one "kc_d1_w=$W" --kc-rstdp --kc-d1-w "$W" --seed 1; done
