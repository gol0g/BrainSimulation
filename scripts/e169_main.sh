#!/bin/bash
# E169 본실험 — 기준 logs/E169/criteria_fixed.txt. 보상 창 1처리(10 ms)로 호스트 동결 대체. 뇌 16~20, 형성 표현(E161 가중치), 반사 0, 500시행.
# E141 인자에서 --reward-window 2 → 1. 팔 FW1 = 동결 없음, FFW1 = 동결(--rw-apm-scale 0). 기준 E161 F(동결, 창 2), 부지표 E166 FNF.
# 요약 줄 "  e169 {FW1|FFW1} b16: => ..." 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E169.log"
RAW="$R/research/experiments/logs/E169"; WD="$R/research/experiments/traces/E169"; mkdir -p "$RAW" "$WD"
W161="$R/research/experiments/traces/E161"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e169_main_run && cd /root/e169_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 1 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
for B in 16 17 18 19 20; do
  for ARM in FW1 FFW1; do
    tag="e169 $ARM b$B"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    [ "$ARM" = "FFW1" ] && XA="--rw-apm-scale 0" || XA=""
    f="$RAW/${ARM}_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT --episodes 5 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $W161/kctype_oja_b$B.npz $XA \
      --save-weights $WD/w_${ARM}_b$B.npz --trace-kc-class $WD/tr_${ARM}_b$B.npz --kc-rate-file $W161/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$f")"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E169] 전체 루프 종료"
