#!/bin/bash
# D16: 결정적 검증 — 포화 WM(-5) vs 희소 WM(-200) seq-wm 재훈련. 순서학습 창발 비교.
# 희소에서만 order_rate 오르면 = 포화가 학습 블로커였다 확증. 동일 조건, inhib만 차이.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
BASE="--task integrated --seq-task --seq-nav --seq-wm --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 10 --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 40 --seed 0"
FILT="\[ep  0\]|\[ep 1[05]\]|\[ep 2[05]\]|\[ep 3[05]\]|\[ep 39\]|최종순서율|WM_latch|Traceback|Error"

run () {
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  echo "########## $1 ##########"
  python -u $R $BASE $2 --output "$D/$3" 2>&1 | grep -iE "$FILT" | tail -14
}
run "포화 WM (inhib -5, baseline)" "--inhib-wm -5"   "d16_sat.json"
run "희소 WM (inhib -200, D15)"    "--inhib-wm -200" "d16_sparse.json"
echo "########## DONE ##########"
