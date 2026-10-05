#!/bin/bash
# E142: E119 반사 25 학습(뇌 10~14) + 보상 창 KC→motor 흔적 동결(--rw-apm-scale 0) + E139 계층 추적. 팔 F500(5ep)·F1500(15ep)·NF1500(무동결 15ep, 수정 1). 재개 가능(P13).
# 요약 줄: "  e142 F500 b10: => 사전 +0.4148 사후 +0.2000 보상 140 || => KCTRACE ... || => KCTRACE3 ..."  (judge_e142.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E142.log"
RAW="$R/research/experiments/logs/E142"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E142"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e142_main_run && cd /root/e142_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
for ARM in F500 F1500 NF1500; do
  [ "$ARM" = "F500" ] && EP=5 || EP=15
  [ "$ARM" = "NF1500" ] && APM="" || APM="--rw-apm-scale 0"
  for B in 10 11 12 13 14; do
    tag="e142 $ARM b$B"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    f="$RAW/${ARM}_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT --reflex-w 25 --episodes $EP --steps 100 --transplant-eval --brain-seed $B $APM \
      --save-weights $WD/w_${ARM}_b$B.npz --trace-kc-class $WD/tr_${ARM}_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || $(grep '^=> KCTRACE ' "$f" | cut -c1-200) || $(grep '^=> KCTRACE3' "$f")"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E142] 전체 루프 종료"
