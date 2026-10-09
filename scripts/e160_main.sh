#!/bin/bash
# E160 본실험 — 기준 logs/E160/criteria_fixed.txt(2026-10-09 15:01:00), η·β = logs/E160/oja_pick.txt(뇌 15 보정). 뇌 10~14. 재개 가능(P13).
# 뇌마다: (1) kcdevoja 망 안 Oja 형성 → traces/E160/kctype_oja_b{B}.npz (2) kcoverlap(정적 뇌, 자카드) (3) 학습 500시행(E153 학습 인자) (4) kcrate(부지표).
# 요약 줄: "  e160 {dev|ov|learn|kcrate} b10: => ..." (판정은 원 로그를 직접 읽는다)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E160.log"
RAW="$R/research/experiments/logs/E160"; WD="$R/research/experiments/traces/E160"; mkdir -p "$RAW" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e160_main_run && cd /root/e160_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT2="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
PE=$(grep -oE '^eta=[0-9.]+' "$RAW/oja_pick.txt" 2>/dev/null | cut -d= -f2)
PB=$(grep -oE 'beta=[0-9.]+' "$RAW/oja_pick.txt" 2>/dev/null | cut -d= -f2)
if [ -z "$PE" ] || [ -z "$PB" ]; then echo "[E160] η·β 없음(보정 실패 또는 oja_pick.txt 없음) — 본실험 안 함"; exit 1; fi
echo "[E160] Oja η $PE β $PB"
LD='^\[E153 종류 입력 적재\].*검증 일치'
skip() { grep -qF "$1: =>" "$LOG" 2>/dev/null && { echo "  $1: [건너뜀]"; return 0; }; return 1; }
fail() { echo "[실패 rc=$1]"; tail -2 "$2" | sed 's/^/      /'; }
for B in 10 11 12 13 14; do
  KW="$WD/kctype_oja_b$B.npz"; WS="$R/research/experiments/traces/E141/w_b$B.npz"
  tag="e160 dev b$B"
  if ! skip "$tag"; then
    f="$RAW/dev_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $WS --decomp-mode kcdevoja --kc-dev-n 100 \
      --kc-type-oja --kc-oja-eta $PE --kc-oja-beta $PB --kc-oja-mmax 32 --kc-oja-tau 20 --kc-dev-save $KW > "$f" 2>&1; rc=$?
    if grep -q "^=> KCDEVOJA" "$f"; then echo "=> $(grep '^=> KCDEVOJA' "$f" | grep -oE 'side=[lr] fired=[0-9]+|sel_med=[0-9.]+|sum_med=[0-9.]+' | tr '\n' ' ')|| Oja $(grep -c '^\[E160 종류 입력 Oja\]' "$f")"; else fail $rc "$f"; continue; fi
  fi
  tag="e160 ov b$B"
  if ! skip "$tag"; then
    f="$RAW/ov_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT2 --brain-seed $B --decomp-weights $WS --decomp-mode kcoverlap --trials 200 --kc-type-weights $KW > "$f" 2>&1; rc=$?
    if grep -q "^=> KCOVERLAP" "$f"; then echo "=> $(grep '^=> KCOVERLAP' "$f" | grep -oE 'side=[lr] good=[0-9]+ bad=[0-9]+ jac=[0-9.]+' | tr '\n' ' ')|| 적재 $(grep -c "$LD" "$f")"; else fail $rc "$f"; fi
  fi
  tag="e160 learn b$B"
  if ! skip "$tag"; then
    f="$RAW/learn_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT2 --episodes 5 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
      --save-weights $WD/w_b$B.npz --trace-kc-class $WD/tr_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$f")"
    else fail $rc "$f"; fi
  fi
  tag="e160 kcrate b$B"
  if ! skip "$tag"; then
    f="$RAW/kcrate_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --judge exec --reflex-w 25 --episodes 0 --steps 100 --brain-seed $B --rw-apm-scale 0 \
      --kc-type-weights $KW --trials 100 --decomp-weights $WS --decomp-mode kcrate --kc-rate-file $WD/kcrate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCRATE kc_r" "$f"; then echo "=> 스파이크 $(grep '^=> KCRATE' "$f" | grep -oE '제시 스파이크 [0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ')|| 적재 $(grep -c "$LD" "$f")"; else fail $rc "$f"; fi
  fi
done
echo "[E160] 전체 루프 종료"
