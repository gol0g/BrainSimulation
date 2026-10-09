#!/bin/bash
# E164 구현 정합성 측정 — 외부 검토 2026-10-09 ①(보상 창 끝 도파민 뉴런 입력 0)·③(학습 오프셋 3처리)을 함께 켠 학습(FIX)을
# 같은 뇌 기본 코드 기준(E141 반사 0·E142 F500 반사 25 — 원 로그 재사용)과 짝 비교. 뇌 10~14, 500시행, 보상 창 흔적 동결. 재개 가능(P13).
# 요약 줄: "  e164 R0X b10: => 사전 .. 사후 .. 보상 N || 점검 K" (judge_e164.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E164.log"
RAW="$R/research/experiments/logs/E164"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E164"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e164_main_run && cd /root/e164_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
FIX="--rw-da-reset --offset-steps 3"
for ARM in R0X R25X; do
  [ "$ARM" = "R0X" ] && RW=0 || RW=25
  for B in 10 11 12 13 14; do
    tag="e164 $ARM b$B"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    f="$RAW/${ARM}_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT --reflex-w $RW --episodes 5 --steps 100 --transplant-eval --brain-seed $B --rw-apm-scale 0 $FIX \
      --save-weights $WD/w_${ARM}_b$B.npz --trace-kc-class $WD/tr_${ARM}_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 점검 $(grep -c '^\[구현 점검\]' "$f")"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E164] 전체 루프 종료"
