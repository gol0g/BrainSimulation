#!/bin/bash
# E149: good·bad 자극 KC 겹침 측정(학습 없음) — 뇌 10~14 × {기본, 차단}. 재개 가능(P13).
# 요약 줄: "  e149 b10 base: => KCOVERLAP side=l ... | side=r ... | food_eye_scale=..."  (judge_e149.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E149.log"
RAW="$R/research/experiments/logs/E149"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e149_main_run && cd /root/e149_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
for B in 10 11 12 13 14; do
  for C in base block; do
    tag="e149 b$B $C"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    [ "$C" = "block" ] && X="--kc-food-eye-scale 0 --kc-bilateral-scale 0" || X="--kc-food-eye-scale 1 --kc-bilateral-scale 1"
    f="$RAW/b${B}_$C.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --decomp-weights $R/research/experiments/traces/E141/w_b$B.npz --decomp-mode kcoverlap --trials 200 $X > "$f" 2>&1; rc=$?
    if grep -q "^=> KCOVERLAP" "$f"; then
      echo "$(grep '^=> KCOVERLAP' "$f")"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E149] 전체 루프 종료"
