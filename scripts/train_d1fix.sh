#!/bin/bash
# C15: d1 포화 해소(-60)로 350ep에서 +60ep 이어훈련 → 개념 점수 개선되나.
source "$(dirname "$0")/cuda_env.sh"; source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
setsid nohup python -u $R/backend/genesis/forager_brain.py --episodes 60 --render none \
  --d1-inhib -60 --load-weights $R/checkpoints/brain_concepts_350ep.npz \
  --save-weights $R/checkpoints/brain_d1fix_60ep.npz > /tmp/c15.log 2>&1 < /dev/null &
echo "C15 launched (pid $!)"
