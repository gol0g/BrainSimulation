#!/bin/bash
# C61: 탐색을 **D1에 주입**해 학습 방향을 바꾼다 — 이제서야 시험 가능해진 실험.
#
# C52 지적: 탐색 편향을 motor(학습 시냅스 **하류**)에 주입하면 자격흔적이 탐색한 행동을 담지 못하고
#   자극이 만든 반사정렬 패턴이 기록돼 보상이 그걸 강화한다(실측: 변조폭이 반사방향 +0.036).
# C53: 그래서 D1 주입을 시도했으나 보상 0회 → 당시 `d1→direct=1`이라 **조작 자체가 무효**였다.
# C55: `d1→direct=20`(기본)에서는 D1 편향 300이 보상 413회를 만든다 = **조작이 작동한다.**
# → 올바른 설정(C59/C60)에서 D1 주입 탐색을 처음으로 제대로 시험한다.
#
# 사전 기준: (D1주입 − motor주입) 변조폭 차이가 5시드 중 4개에서 **음수**면 C52 가설 성립.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-30}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2 3 4; do
  echo "--- seed=$S ---"
  printf "  D1주입(300)  : "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 300 --bias-at-d1 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "학습\]|^=>" | tr '\n' ' '
  echo
  printf "  motor주입(25): "
  timeout 3600 python reflex_override_task.py --episodes "$EP" --seed "$S" \
    --real-rstdp --crossed --epsilon 0.6 --bias 25 \
    --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "학습\]|^=>" | tr '\n' ' '
  echo
done
