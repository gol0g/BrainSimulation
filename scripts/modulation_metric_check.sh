#!/bin/bash
# C50: 변조폭 지표로 학습 판정 — 이분법 정답률이 못 잡는 미세 변화 포착.
#
# C49에서 드러난 지표 결함: 반사를 없애면 조향이 거의 0이라 |v|<0.02 임계에 걸려
# 좌·우 양쪽 다 오답 처리 → 0.0%는 "틀림"이 아니라 "결정 안 함".
# C28b에서 이미 얻은 교훈(좌↔우 차이값=변조폭만 견고)을 이 프로브에 적용.
#
# 변조폭 = mean(조향|good=우) − mean(조향|good=좌).
#   양수=반사방향, 학습이 역전시키면 **음수 방향으로 이동**해야 한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2; do
  echo "--- seed=$S ---"
  printf "  수리전부: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -200 \
    --reflex-w 3 --d1-direct-w 1 2>&1 | grep -E "^=>"
  printf "  정적대조: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --epsilon 0.6 --bias 25 --d1-inhib -200 --reflex-w 3 2>&1 | grep -E "^=>"
done
