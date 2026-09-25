#!/bin/bash
# E098: 전체 모델에 K42 적용 — tau_e 를 시행 길이에 맞춘다. 재개 가능(P13).
# 지표: (학습 − 무학습) 변조폭 차이. 무학습 대조는 --no-reward.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E098.log"
RAW="$R/research/experiments/logs/E098"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --episodes 60 --kc-w-max 750 --transplant-eval --env-seed 0"
done_already () { grep -qF "$1: =>" "$LOG" 2>/dev/null; }
run_one () {   # tag tau seed extra
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 5400 python reflex_override_task.py $BASE --tau-e "$2" --brain-seed "$3" $4 \
    --dump-kc-weights > "$f" 2>&1
  local rc=$?
  grep -E "^=>" "$f" || { echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; }
}
for TAU in 200 50 12 3; do
  echo "########## tau_e=$TAU ##########"
  for S in 0 1 2 3 4; do
    run_one "tau$TAU 학습 b$S" "$TAU" "$S" ""
    run_one "tau$TAU 무학습 b$S" "$TAU" "$S" "--no-reward"
  done
done
echo "[E098] 전체 루프 종료 — 원본 로그: $RAW"
