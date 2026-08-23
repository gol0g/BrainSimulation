#!/bin/bash
# C13-b: 세션 종료에 영향받지 않게 완전 분리 실행(setsid+nohup). ep50에서 이어받지 않고 새로 200ep.
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
setsid nohup python -u $R/backend/genesis/forager_brain.py --episodes 200 --render none \
  --social-task --mirror-motor 8.0 --sts-inhib -30 --social-kc 8.0 \
  --save-weights $R/checkpoints/brain_social_kc_200ep.npz > /tmp/c13.log 2>&1 < /dev/null &
echo "C13 detached launched (pid $!)"
