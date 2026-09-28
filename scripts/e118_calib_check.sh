#!/bin/bash
# E118 조작검증(학습 없음): 초기 대칭 KC→motor 가중치별 기준 변조폭(none) — 30(E115)·150·300, 뇌 0·1. E115 설정.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E118"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e118_run && cd /root/e118_run
cp $R/backend/genesis/*.py . 2>/dev/null
for IW in 30 150 300; do for B in 0 1; do
  f="$OUT/calib_iw${IW}_b$B.log"
  timeout 3600 python reflex_override_task.py --real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w $IW --kc-motor-eta 0.15 --tau-e 12 --env-seed 0 \
    --episodes 0 --brain-seed $B --decomp-weights $R/research/experiments/traces/E115/w_b$B.npz --decomp-mode none > "$f" 2>&1
  echo "iw$IW b$B: $(grep -E '^=> DECOMP' "$f" | cut -c1-80 || tail -1 "$f")"
done; done
