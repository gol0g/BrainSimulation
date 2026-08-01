#!/bin/bash
# 개념 형성 통합 검증: 같은 정본 가중치(250ep)에서 4층위 개념이 공존·재현하나.
# ①공간 ②good/bad 시각·소리 변별 ③범주일반화 ④graded 합성. 단일 스코어카드.
set -uo pipefail
source "$(dirname "$0")/cuda_env.sh"
source ~/pygenn_wsl/bin/activate
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
W=${1:-$R/checkpoints/brain_concepts_250ep.npz}
echo "########## 개념 스위트: $W ##########"
for t in spatial visual_discrim sound_discrim generalization compositional compositional_graded; do
  cd ~/pygenn_test && rm -rf forager_brain_CODE CODE
  echo "----- $t -----"
  # sound_discrim은 타입×방향 소리 인코딩(--typed-sound) 필요 (양식전이 확인용, C1-fix)
  FLAG=""; [ "$t" = "sound_discrim" ] && FLAG="--typed-sound"
  python -u $R/backend/genesis/evaluate_concepts.py --load-weights "$W" --test "$t" $FLAG 2>&1 \
    | grep -iE "Spatial|Visual Discrim|Sound Discrim|Generalization|Compositional:|MAX diff|PASS|FAIL" | tail -4
done
echo "########## SUITE DONE ##########"
