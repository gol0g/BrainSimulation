#!/bin/bash
# D18: 희소(-200)+패턴래치+curiosity OFF. D17 결론(order confound=탐색) 검증.
# biletaxis가 target(래치→A후B)으로만 조향 → 무작위탐색 제거. order_rate 뛰면 confound 확증.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
RUN=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
FILT="\[ep  0\]|\[ep 1[05]\]|\[ep 2[05]\]|\[ep 3[05]\]|\[ep 39\]|최종순서율|last_5|Traceback|Error"
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "########## 희소 -200 + 패턴래치 + curiosity OFF, 40ep ##########"
python -u "$RUN" --task integrated --seq-task --seq-nav --seq-wm --seq-pattern-latch \
  --seq-no-curiosity --inhib-wm -200 --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 10 \
  --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 40 --seed 0 \
  --output "$D/d18_nocur.json" 2>&1 | grep -iE "$FILT" | tail -16
echo "########## DONE ##########"
