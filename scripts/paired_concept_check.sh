#!/bin/bash
# C47: 개념 5층 판정 — **같은 시드로 학습 뇌와 무학습 뇌를 짝지어** 재확인.
#
# 이 세션의 개념 측정은 전부 무시드였다(evaluate_concepts.py에 시드 인자가 아예 없었음).
# 결론(개념 5층 전부 선천)은 각 8회 반복으로 뒷받침됐지만, 짝지은 비교가 훨씬 강력하다:
# 같은 시드면 환경·워밍업이 동일하므로 **차이가 곧 학습분**이다.
#
# 사전 기준: 시드별 (학습−무학습) 차이가 5개 시드 중 4개 이상에서 +8%p 넘으면 학습 기여 인정.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
U=$R/checkpoints/brain_seeded_untrained.npz
T=$R/checkpoints/brain_seeded_typedsound_150ep.npz
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/snd_run
cp $R/backend/genesis/evaluate_concepts.py . 2>/dev/null

for TEST in generalization visual_discrim compositional; do
  echo "########## $TEST ##########"
  for S in 0 1 2 3 4; do
    echo "--- seed=$S ---"
    printf "  무학습: "
    timeout 3000 python evaluate_concepts.py --test "$TEST" --typed-sound --seed "$S" \
      --load-weights "$U" 2>&1 | grep -iE "Generalization|Visual Discrim|Compositional" | tail -1
    printf "  학습  : "
    timeout 3000 python evaluate_concepts.py --test "$TEST" --typed-sound --seed "$S" \
      --load-weights "$T" 2>&1 | grep -iE "Generalization|Visual Discrim|Compositional" | tail -1
  done
done
