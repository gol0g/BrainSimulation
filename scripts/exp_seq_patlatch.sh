#!/bin/bash
# D17: 희소 WM(-200) + 패턴기반 래치 감지. D16 둘째 블로커(패턴래치가 rate제어에 안읽힘) 정조준.
# rate 대신 A-패턴 상관으로 latch 판정 → 컨트롤러가 희소 WM 기억 읽음. 순서 창발하나.
# 비교: D16 희소+rate래치 = order 0.006 평탄. 여기서 상승하면 = 둘째 블로커 해결.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
RUN=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/backend/genesis/run_v2_tasks.py
D=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/docs/research/rebuild_baseline
FILT="\[ep  0\]|\[ep 1[05]\]|\[ep 2[05]\]|\[ep 3[05]\]|\[ep 39\]|최종순서율|last_5|Traceback|Error"
cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
echo "########## 희소 -200 + 패턴래치, 40ep ##########"
python -u "$RUN" --task integrated --seq-task --seq-nav --seq-wm --seq-pattern-latch \
  --inhib-wm -200 --zone-cx 0.3 --zone-cy 0.3 --sparse-reward --n-food 10 \
  --biletaxis --biletaxis-gain 0.5 --biletaxis-brake --episodes 40 --seed 0 \
  --output "$D/d17_patlatch.json" 2>&1 | grep -iE "$FILT" | tail -16
echo "########## DONE ##########"
