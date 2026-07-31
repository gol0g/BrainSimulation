#!/bin/bash
# D14 후속: WM 억제강도 스윕 → 희소성(활성비율) 측정. order_rate 아닌 독립지표(용량 보정).
# 목표 활성비율 0.10~0.20. 포화(-5 기본) 탈출하는 억제강도 탐색.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_concepts_250ep.npz
for iw in -175 -200 -250; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  echo "===== inhibitory_to_wm = $iw ====="
  python -u "$R/backend/genesis/wm_latch_probe.py" --load-weights "$W" \
    --trials 12 --delay 20 --inhib-wm "$iw" 2>&1 \
    | grep -iE "active_frac|SPARSE|corr\(|DISCRIMINATION|A-SPECIFIC" | tail -8
done
echo "===== DONE ====="
