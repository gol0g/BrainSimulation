#!/bin/bash
# E121: 좌우 공통 KC 입력 배율 0 — 학습·무학습 5 에피소드, 뇌 10~14 (배율 1은 E119 재사용). 재개 가능(P13).
# 요약 줄 형식은 judge_e121.py 의 TR 과 맞춘다: "  s0 learn b10: => <사전> <사후> | ... 변조폭 변화 X ..."
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E121.log"
RAW="$R/research/experiments/logs/E121"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E121"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e121_main_run && cd /root/e121_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --kc-bilateral-scale 0 --episodes 5"
for B in 10 11 12 13 14; do for C in learn nolearn; do
  tag="s0 $C b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/s0_${C}_b$B.log"; EXTRA="--save-weights $WD/w_s0_b$B.npz --snap-all-syn"; [ "$C" = "nolearn" ] && EXTRA="--no-reward"
  printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE $ACT --steps 100 --transplant-eval --brain-seed $B $EXTRA > "$f" 2>&1; rc=$?
  if grep -q "변조폭 변화" "$f"; then echo "=> $(grep -E '^\[사전\]|^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/' | tr '\n' ' ')| $(grep '변조폭 변화' "$f" | tail -1 | sed 's/^=> //')"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done; done
echo "[E121] 전체 루프 종료"
