#!/bin/bash
# E082 조작검증 2 (규약 P2·P10): --kc-w-max 750 이 실제로 가중치를 유지하는가.
# rstdp_model.py:70 의 클램프 때문에 상한을 안 올리면 초기 150이 학습 중 30으로 깎인다.
# 짧은 진단(--episodes 3)으로 학습 후 kc_to_d1 가중치 평균을 본다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

echo "### 상한 미지정(기본 30) + 초기 150 — 깎여야 정상"
timeout 1800 python reflex_override_task.py --real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 \
  --epsilon 0.6 --bias 25 --reflex-w 3 --episodes 3 --kc-rstdp --kc-d1-w 150 \
  --brain-seed 0 --env-seed 0 --dump-kc-weights 2>&1 | grep -E "kc_to_d1|^=>" | tail -4

echo "### 상한 750 + 초기 150 — 유지돼야 정상"
timeout 1800 python reflex_override_task.py --real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 \
  --epsilon 0.6 --bias 25 --reflex-w 3 --episodes 3 --kc-rstdp --kc-w-max 750 --kc-d1-w 150 \
  --brain-seed 0 --env-seed 0 --dump-kc-weights 2>&1 | grep -E "kc_to_d1|^=>" | tail -4
