#!/bin/bash
# E071 조작 검증: (1) 교차 시냅스 실재 (2) D1 주입이 행동을 바꾸는가
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
echo "=== (1) 교차경로 OFF ==="
timeout 2400 python synapse_count_probe.py 2>&1 | grep -E "SPARSITY|cross|없음"
echo "=== (1) 교차경로 ON ==="
timeout 2400 python synapse_count_probe.py --crossed 2>&1 | grep -E "SPARSITY|cross|없음"
echo "=== (2) D1 주입 작동(보상>0 이어야) ==="
timeout 2400 python reflex_override_task.py --real-rstdp --crossed --epsilon 0.6 \
  --bias 300 --bias-at-d1 --d1-inhib -400 --direct-inhib -400 --reflex-w 3 \
  --episodes 6 --seed 0 2>&1 | grep -E "학습\]"
echo "=== (2대조) motor 주입 ==="
timeout 2400 python reflex_override_task.py --real-rstdp --crossed --epsilon 0.6 \
  --bias 25 --d1-inhib -400 --direct-inhib -400 --reflex-w 3 \
  --episodes 6 --seed 0 2>&1 | grep -E "학습\]"
