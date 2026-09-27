#!/bin/bash
# E108 조작검증(학습 없음): D1/motor 좌우 편향별 조향 부호 실측 — 행동 흔적 구동 방향 결정용.
# E098 과 같은 뇌 설정(불변식 INV-A4/A5 포함), 뇌 0~4, 환경 시드 0, 편향별 20회.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E108"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e108_run && cd /root/e108_run
cp $R/backend/genesis/*.py . 2>/dev/null
for B in 0 1 2 3 4; do
  timeout 1800 python reflex_override_task.py --calib-d1-sign 20 --episodes 0 --brain-seed $B --env-seed 0 \
    --real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --bias 25 > "$OUT/calib_b$B.log" 2>&1
  echo "b$B: $(grep -E '^=> CALIB' "$OUT/calib_b$B.log" || tail -2 "$OUT/calib_b$B.log")"
done
