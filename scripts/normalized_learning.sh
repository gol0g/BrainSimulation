#!/bin/bash
# C64: 상태 정규화 후 (1) 아티팩트 제거 검증 (2) 학습 재측정.
#
# C63 확정: 도파민 없이도 변조폭이 +0.025~0.038 똑같이 표류(보상 있음과 차이 0.0002~0.0011)
#   = 사전/사후를 서로 다른 적응 상태에서 잰 **측정 아티팩트**. 학습과 무관.
# C64 수리: evaluate() 진입 시 brain.reset() + 동일 안정화 30스텝 → 남는 차이는 가중치뿐.
#
# 검증 기준: 보상없음 조건의 변조폭 변화가 |0.005| 이내로 떨어지면 아티팩트 제거 성공.
# 학습 판정: (수리 − 정적)이 5시드 중 4개에서 음수(반사 역전 방향)면 학습 성립.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

echo "########## (1) 아티팩트 제거 검증 ##########"
for S in 0 1 2; do
  printf "seed=%s 보상없음: " "$S"
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 --no-reward \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
done

echo "########## (2) 학습 재측정 (수리 vs 정적) ##########"
for S in 0 1 2 3 4; do
  echo "--- seed=$S ---"
  printf "  수리: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
  printf "  정적: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --epsilon 0.6 --bias 25 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
done
