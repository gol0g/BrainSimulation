#!/bin/bash
# 재건 run_v2_tasks.py 실행 래퍼
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
REPO=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
cd ~/pygenn_test
rm -rf forager_brain_CODE CODE
python -u "$REPO/backend/genesis/run_v2_tasks.py" "$@"
rc=$?
echo "=== run_v2 종료 (exit=$rc) ==="
exit $rc
