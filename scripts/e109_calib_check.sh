#!/bin/bash
# E109 조작검증(학습 없음): KC→motor 경로의 행동 권한 — 가중치를 극단(완전 역전/반사 정렬/0)으로 넣고 변조폭.
# E098 설정 + 불변식(INV-A4/A5). w_max 2·5·10·20·40 × 뇌 0~4, env 0.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E109"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e109_run && cd /root/e109_run
cp $R/backend/genesis/*.py . 2>/dev/null
for WM in 2 5 10 20 40; do
  for B in 0 1 2 3 4; do
    f="$OUT/calib_wm${WM}_b$B.log"
    if grep -q "^=> CALIBKM" "$f" 2>/dev/null; then echo "wm$WM b$B: $(grep '^=> CALIBKM' "$f")"; continue; fi
    timeout 2400 python reflex_override_task.py --calib-kc-motor --kc-motor --kc-motor-w-max $WM --episodes 0 --trials 40 \
      --brain-seed $B --env-seed 0 --real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 > "$f" 2>&1
    echo "wm$WM b$B: $(grep -E '^=> CALIBKM' "$f" || tail -2 "$f")"
  done
done
