#!/bin/bash
# 실제 과제 경로에서의 이식 항등성 (검토 권고 1).
#
# 단독 transplant_eval.py 검증은 **다른 생성 경로**를 쓰므로 이 확인을 대신하지 못한다
# (2026-09-19: 단독 검증은 통과했는데 과제 경로는 cuda error 로 죽었고, 고친 뒤에도
#  이식 목록 누락·환경 시드 불일치가 남아 있었다).
#
# 훈련 0회(--episodes 0)면 가중치가 안 변하므로, 사후(이식)와 사전(직접)이 **같아야 한다.**
# 다르면 이식 경로 자체가 값을 바꾸는 것이고, 모든 칸 차이에 그 편향이 실린다.
# 쓸 모든 brain seed(0~4)에서 확인한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
RAW=/root/rstdp_run/identity
mkdir -p $RAW
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null

BASE="--d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --episodes 0 --kc-w-max 750 --real-rstdp --crossed --kc-rstdp"
for B in 0 1 2 3 4; do
  f=$RAW/b$B.log
  printf "brain-seed %d: " "$B"
  timeout 1800 python reflex_override_task.py $BASE --kc-d1-w 0.5 \
    --brain-seed "$B" --env-seed 0 --transplant-eval > "$f" 2>&1
  rc=$?
  if grep -q "^\[사전\]" "$f" && grep -q "^\[사후\]" "$f"; then
    grep -E "^[이식]" "$f" || true
    PRE=$(grep "^\[사전\]" "$f" | grep -oE '변조폭 [+-][0-9.]+' | tail -1)
    POST=$(grep "^\[사후\]" "$f" | grep -oE '변조폭 [+-][0-9.]+' | tail -1)
    echo "사전 $PRE | 사후(이식) $POST"
  else
    echo "[실패 rc=$rc]"; tail -3 "$f" | sed 's/^/    /'
  fi
done
echo
echo "판정: 사전과 사후(이식)가 모든 시드에서 같아야 한다. 다르면 이식 경로가 값을 바꾸는 것이다."
