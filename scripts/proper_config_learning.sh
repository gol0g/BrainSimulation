#!/bin/bash
# C60: **제대로 구성된 조건**으로 학습 시험 — 지금까지의 학습 실패는 전부 잘못된 설정에서 측정됐다.
#
# 지금까지의 오류:
#   - d1 억제 -200: 5시드 중 2개가 포화(~594)라 변별할 정보가 없었다 (C58)
#   - d1→direct = 1: direct는 탈포화됐으나 **D1 영향력이 0**이 됐다 (C55, 보상 381→0)
#   - direct 포화를 가중치로 해결하려다 위 문제를 만듦
# 올바른 설정(C56/C59로 확정):
#   - d1 억제 **-400** (5/5 탈포화, 변별 평균 30.8로 최고)
#   - d1→direct **20 유지**(영향력 보존) + direct 억제 -400(포화 해소)
#
# 같은 시드로 [수리전부] vs [정적대조]를 짝지어 5시드.
# 사전 기준: 변조폭 변화 차이(수리−정적)가 5시드 중 4개에서 음수(반사 역전 방향)면 학습 성립.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2 3 4; do
  echo "--- seed=$S ---"
  printf "  수리전부: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
  printf "  정적대조: "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --epsilon 0.6 --bias 25 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
done
