#!/bin/bash
# C32: 일반화 — 이 세션에서 유일하게 살아남은 학습 후보. 반복 8회씩으로 확정 시도.
# 예비(n=3): 학습 6런 전부 >=60.0, 무학습 3런 전부 <=59.2 (겹침 없으나 간격 0.8pp).
# 사전 기준: 학습 평균 - 무학습 평균 > 8%p **그리고** 두 분포 min/max가 겹치지 않아야 학습 기여 인정.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
U=$R/checkpoints/brain_seeded_untrained.npz
T=$R/checkpoints/brain_seeded_typedsound_150ep.npz
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/snd_run

echo "### UNTRAINED x8 ###"
for i in 1 2 3 4 5 6 7 8; do
  timeout 3000 python evaluate_concepts.py --test generalization --typed-sound --load-weights "$U" 2>&1 | grep -i "Generalization"
done
echo "### TRAINED(typedsound 150ep) x8 ###"
for i in 1 2 3 4 5 6 7 8; do
  timeout 3000 python evaluate_concepts.py --test generalization --typed-sound --load-weights "$T" 2>&1 | grep -i "Generalization"
done
