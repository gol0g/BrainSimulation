#!/bin/bash
# C26b: 무학습 뇌 개념 전 항목 측정 (3회 반복)
# 인라인 WSL 명령에서 변수 확장이 반복적으로 깨져(Git Bash가 $var 소비) 스크립트 파일로 고정.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
U=$R/checkpoints/brain_seeded_untrained.npz
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/snd_run

for t in visual_discrim generalization compositional spatial; do
  echo "=== 무학습: $t ==="
  for i in 1 2 3; do
    timeout 3000 python evaluate_concepts.py --test "$t" --typed-sound --load-weights "$U" 2>&1 \
      | grep -iE "Visual Discrim|Generalization|Compositional|Spatial Memory"
  done
done
