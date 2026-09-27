#!/bin/bash
# E109 조작검증 수정판(학습 없음): KC→motor 권한 — **조건마다 새 뇌 1회 평가**(이력 교란 제거, K22/K24).
# sparsity 0.05·0.25 × w_max 40·200 × {zero, rev} × 뇌 0~4. 결정론 확인: sp0.05 wm40 zero b0 을 두 번.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E109"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e109_run && cd /root/e109_run
cp $R/backend/genesis/*.py . 2>/dev/null
one () { # tag sp wm set brain
  local f="$OUT/c2_$1.log"
  if grep -q "^=> CALIBKM1" "$f" 2>/dev/null; then echo "$1: $(grep '^=> CALIBKM1' "$f")"; return; fi
  timeout 2400 python reflex_override_task.py --calib-kc-motor-set "$4" --kc-motor --kc-motor-sparsity "$2" --kc-motor-w-max "$3" \
    --episodes 0 --trials 40 --brain-seed "$5" --env-seed 0 --real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 > "$f" 2>&1
  echo "$1: $(grep -E '^=> CALIBKM1' "$f" || tail -2 "$f")"
}
one "det_a" 0.05 40 zero 0
one "det_b" 0.05 40 zero 0
for SP_ in 0.05 0.25; do for WM in 40 200; do for ST in zero rev; do for B in 0 1 2 3 4; do
  one "sp${SP_}_wm${WM}_${ST}_b$B" "$SP_" "$WM" "$ST" "$B"
done; done; done; done
