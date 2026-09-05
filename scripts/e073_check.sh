#!/bin/bash
# E073 조작검증: 시드 분리가 실제로 작동하는가.
# 뇌 고정 → 연결 수 n이 5런 모두 동일해야. 뇌 변화 → n이 달라져야.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
echo "=== 뇌 고정(brain-seed 0), 환경 변화 → n 동일해야 ==="
for E in 0 1 2; do
  printf "  env=%s: " "$E"
  timeout 2400 python reflex_override_task.py --real-rstdp --crossed --epsilon 0.6 --bias 25 \
    --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 1 \
    --brain-seed 0 --env-seed "$E" 2>&1 | grep -E "D1가중치.*food_to_d1_l" | head -1
done
echo "=== 뇌 변화, 환경 고정(env-seed 0) → n 달라져야 ==="
for B in 0 1 2; do
  printf "  brain=%s: " "$B"
  timeout 2400 python reflex_override_task.py --real-rstdp --crossed --epsilon 0.6 --bias 25 \
    --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 1 \
    --brain-seed "$B" --env-seed 0 2>&1 | grep -E "D1가중치.*food_to_d1_l" | head -1
done
