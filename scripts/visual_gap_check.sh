#!/bin/bash
# C31b: 시각·일반화 20%p 격차가 학습분인지 확인.
# 손상 0인 학습 뇌(ctxhard 150ep / typedsound 150ep) vs 무학습 뇌를 같은 조건에서 비교.
# 250ep 완료를 기다리지 않고 지금 좁힌다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
U=$R/checkpoints/brain_seeded_untrained.npz
T1=$R/checkpoints/brain_seeded_typedsound_150ep.npz
T2=$R/checkpoints/brain_seeded_ctxhard_150ep.npz
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/snd_run

for t in visual_discrim generalization; do
  for label in UNTRAINED TYPEDSOUND CTXHARD; do
    case $label in
      UNTRAINED) W=$U ;;
      TYPEDSOUND) W=$T1 ;;
      CTXHARD) W=$T2 ;;
    esac
    echo "=== $t / $label ==="
    for i in 1 2 3; do
      timeout 3000 python evaluate_concepts.py --test "$t" --typed-sound --load-weights "$W" 2>&1 \
        | grep -iE "Visual Discrim|Generalization"
    done
  done
done
