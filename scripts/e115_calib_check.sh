#!/bin/bash
# E115 조작검증(학습 없음): 지속 전류 행동 창 세기별 실행/반대 motor 발화 — 뇌 0·1, E110 뇌 설정.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E115"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e115_run && cd /root/e115_run
cp $R/backend/genesis/*.py . 2>/dev/null
for B in 0 1; do
  timeout 3600 python reflex_override_task.py --real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 30 --tau-e 12 \
    --episodes 0 --brain-seed $B --env-seed 0 --act-window 3 --calib-act-current 0,100,300,1000,3000 > "$OUT/calib_b$B.log" 2>&1
  echo "b$B:"; grep -E "^=> CALIBAC|Error" "$OUT/calib_b$B.log" | sed 's/^/  /'
done
