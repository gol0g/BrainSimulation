#!/bin/bash
# E192 본실험 — 기준 logs/E192/criteria_fixed.txt. 범주 × 맥락 결합 부호 측정(학습 없음): 뇌 10~14 × 맥락 연결 C0·C1·C2 × rawcat --rc-measure-ctx(288 제시).
# 요약 줄 "  e192 ... : => ..." 재개 가능(P13). 15줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E192.log"
RAW="$R/research/experiments/logs/E192"; mkdir -p "$RAW"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e192_main_run && cd /root/e192_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
echo "[E192] 범주 × 맥락 결합 부호 측정 · 맥락 연결 C0(p 0.10·w 4)·C1(p 0.05·w 8)·C2(p 0.02·w 20) · 뇌 10~14"
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  for C in C0 C1 C2; do
    case $C in
      C0) CW=4; CP=0.10 ;;
      C1) CW=8; CP=0.05 ;;
      C2) CW=20; CP=0.02 ;;
    esac
    CTX="--ctx-n 200 --ctx-w $CW --ctx-p $CP --ctx2-n 200 --ctx2-w $CW --ctx2-p $CP --motor-tonic -4000"
    tag="e192 meas b$B $C"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    g="$RAW/meas_b${B}_$C.log"; printf "  %s: " "$tag"
    timeout 7200 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $FEAT $CTX \
      --decomp-weights $R/research/experiments/traces/E179/w_bc_b$B.npz --decomp-mode rawcat --rc-expo-eta 0 --rc-measure-ctx --rc-reps 3 > "$g" 2>&1; rc=$?
    if grep -q "^RCKXCHK " "$g"; then echo "=> RCKX $(grep -c '^RCKX ' "$g") || $(grep '^RCKXCHK' "$g" | grep -oE 'ctx_on_spikes=[0-9]+ ctx_off_spikes=[0-9]+ feat_inactive_spikes=[0-9]+')"
    else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  done
done
echo "[E192] 전체 루프 종료"
