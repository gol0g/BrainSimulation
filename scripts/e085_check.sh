#!/bin/bash
# E085 조작검증 1·1b: --no-reward 가 **이식 목록 전체**의 가소성을 멈추는가 + 이식에 교차 경로가 포함되는가.
# 본실험과 같은 설정(--real-rstdp --crossed 포함)으로 확인한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
RAW=/root/rstdp_run/e085chk
mkdir -p $RAW
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null

BASE="--d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --episodes 5 --kc-w-max 750 --transplant-eval --real-rstdp --crossed"
for CASE in "학습끔 --no-reward" "학습켬 "; do
  set -- $CASE
  lbl=$1; shift
  f=$RAW/$lbl.log
  echo "### $lbl"
  timeout 1800 python reflex_override_task.py $BASE ${1:-} --kc-d1-w 0.5 \
    --brain-seed 0 --env-seed 0 --dump-kc-weights > "$f" 2>&1
  rc=$?
  if grep -q "^\[이식\]" "$f"; then
    grep -E "^\[이식\]|^\[사후\]|kc_to_d1_l:" "$f" | sed 's/^/  /'
  else
    echo "  [실패 rc=$rc]"; tail -3 "$f" | sed 's/^/    /'
  fi
done
