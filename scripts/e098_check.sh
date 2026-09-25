#!/bin/bash
# E098 조작검증 1: --tau-e 가 실제로 모델에 닿는가.
# 빌드 로그의 tau_e 출력(forager_brain.py:2973)과 학습 후 |Δ|평균을 본다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
RAW=/root/rstdp_run/e098chk
mkdir -p $RAW
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --episodes 5 --kc-w-max 750 --transplant-eval --dump-kc-weights"
for TAU in 200 12; do
  echo "### tau_e=$TAU"
  f=$RAW/tau$TAU.log
  timeout 2400 python reflex_override_task.py $BASE --tau-e "$TAU" --brain-seed 0 --env-seed 0 > "$f" 2>&1
  rc=$?
  if grep -q "^\[사후\]" "$f"; then
    grep -E "tau_e=" "$f" | head -2 | sed 's/^/   빌드: /'
    grep -E "^\[이식\]|^\[사후\]|^  가중치" "$f" | sed 's/^/   /'
  else
    echo "   [실패 rc=$rc]"; tail -4 "$f" | sed 's/^/     /'
  fi
done
