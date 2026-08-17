#!/bin/bash
# C9: STS/social WTA 억제 스윕 → 사회표상 포화 해소되나(WM 희소화와 동형).
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=$R/checkpoints/brain_social_mirror_150ep.npz
for inh in -8 -50 -150; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  echo "===== sts_inhib = $inh ====="
  python -u $R/backend/genesis/social_repr_probe.py --load-weights "$W" --sts-inhib "$inh" --trials 15 2>&1 \
    | grep -iE "mirror_food|social_memory|tom_intention|social_observation|sts_social|motor_left" | head -6
done
echo "===== DONE ====="
