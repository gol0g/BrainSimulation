#!/bin/bash
# 4월판 baseline 회귀 검증 — 재건 기준점 확보
# CLAUDE.md 필수 기준: Survival Rate > 40%, Reward Freq > 2.5%
set -uo pipefail

source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate

REPO=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP=${1:-20}

# GeNN은 CWD에 코드를 생성한다. 원래 관례대로 ~/pygenn_test에서 돌린다.
cd ~/pygenn_test
rm -rf forager_brain_CODE CODE

echo "=== 4월판 baseline 검증 시작: ${EP}ep ==="
date
python -u "$REPO/backend/genesis/forager_brain.py" --episodes "$EP" --render none
rc=$?
echo "=== 종료 (exit=$rc) ==="
date
exit $rc
