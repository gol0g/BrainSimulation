#!/bin/bash
# E158: E142 인자(반사 25·보상 창 흔적 동결·E139 추적) + 같은 뇌 E153 형성 가중치 × 0.70(--kc-type-scale, E157 k). 팔 F500(5ep)·F1500(15ep), 뇌 10~14. 재개 가능(P13).
# 기준 logs/E158/criteria_fixed.txt(2026-10-09 13:07:04). 요약 줄: "  e158 F500 b10: => 사전 +0.3500 사후 +0.1000 보상 140 || 적재 2 배율 2 || => KCTRACE ..."  (judge_e158.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E158.log"
RAW="$R/research/experiments/logs/E158"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E158"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e158_main_run && cd /root/e158_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
LD='^\[E153 종류 입력 적재\].*검증 일치'; SC='^\[E157 종류 입력 배율\].*검증 일치'
for ARM in F500 F1500; do
  [ "$ARM" = "F500" ] && EP=5 || EP=15
  for B in 10 11 12 13 14; do
    tag="e158 $ARM b$B"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    f="$RAW/${ARM}_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT --reflex-w 25 --episodes $EP --steps 100 --transplant-eval --brain-seed $B --rw-apm-scale 0 \
      --kc-type-weights $R/research/experiments/traces/E153/kctype_b$B.npz --kc-type-scale 0.70 \
      --save-weights $WD/w_${ARM}_b$B.npz --trace-kc-class $WD/tr_${ARM}_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$f") 배율 $(grep -c "$SC" "$f") || $(grep '^=> KCTRACE ' "$f" | cut -c1-120)"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E158] 전체 루프 종료"
