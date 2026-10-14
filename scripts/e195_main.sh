#!/bin/bash
# E195 본실험 — 기준 logs/E195/criteria_fixed.txt(정정 1). KC 희소화와 XOR 결합 순도(측정, 학습 없음): 뇌 10~14 × KC 되먹임 억제 배율 1·1.5(주)·2(보조) × rawcat --rc-measure-ctx(C2, 288 제시).
# 요약 줄 "  e195 ... : => ..." 재개 가능(P13). 15줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E195.log"
RAW="$R/research/experiments/logs/E195"; mkdir -p "$RAW"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e195_main_run && cd /root/e195_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
echo "[E195] KC 희소화와 XOR 결합 순도 측정 · 되먹임 억제 배율 1·1.5·2 · 맥락 C2 · 뇌 10~14"
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  for L in S1 SP SR; do
    case $L in S1) S=1 ;; SP) S=1.5 ;; SR) S=2 ;; esac
    tag="e195 meas b$B $L"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    g="$RAW/meas_b${B}_$L.log"; printf "  %s: " "$tag"
    timeout 7200 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $FEAT $CTX --kc-inh-scale $S \
      --decomp-weights $R/research/experiments/traces/E179/w_bc_b$B.npz --decomp-mode rawcat --rc-expo-eta 0 --rc-measure-ctx --rc-reps 3 > "$g" 2>&1; rc=$?
    if grep -q "^RCKXCHK " "$g"; then echo "=> RCKX $(grep -c '^RCKX ' "$g") || $(grep '^RCKXCHK' "$g" | grep -oE 'ctx_off_spikes=[0-9]+|kc_inh_scale=[0-9.]+ cfg_kc_inh_w=[-0-9.]+ inh_g_l=[-0-9.]+' | tr '\n' ' ')"
    else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  done
done
echo "[E195] 전체 루프 종료"
