#!/bin/bash
# E198 본실험 — 기준 logs/E198/criteria_fixed.txt. L4 학습 규칙 형식(학습 없음): 뇌 10~14 × 팔 none·sgnR·lsgnR·lsgnA × 맥락 X·Y(E196 과 같은 평가).
# 지수 = E196 −4900 측정, 학습값 = traces/E193/w_b{B}.npz(lsgn 팔의 Δ). none·sgnR 은 E197 회귀. 요약 줄 "  e198 ... : => ..." 재개 가능(P13). 40줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E198.log"
RAW="$R/research/experiments/logs/E198"; mkdir -p "$RAW"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e198_main_run && cd /root/e198_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4900"   # E193·E196 동작점
echo "[E198] L4 학습 규칙 형식 — 학습 부호 판독(lsgnR·lsgnA) + 회귀(none·sgnR) × 맥락 X·Y · C2 · 기저값 −4900 · 뇌 10~14"
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  NPZ="$R/research/experiments/traces/E196/sxi_b$B.npz"
  for W in none sgnR lsgnR lsgnA; do
    for X in X Y; do
      tag="e198 b$B $W $X"
      if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
      [ -s "$NPZ" ] || { echo "  $tag: [실패 rc=지수 파일 없음]"; continue; }
      if [ "$W" = "none" ]; then XM="--decomp-mode none"; else XM="--decomp-mode kcxor --kc-xor-file $NPZ --kc-xor-arm $W"; fi
      g="$RAW/ev_b${B}_${W}_$X.log"; printf "  %s: " "$tag"
      timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --brain-seed $B --kc-type-weights $KW --decomp-weights $R/research/experiments/traces/E193/w_b$B.npz $XM --rc-eval --rc-eval-ctx $X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
      if grep -q "^=> DECOMP .*rceval" "$g"; then echo "=> $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+|pushed=[0-9]+' | tr '\n' ' ')"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
    done
  done
done
echo "[E198] 전체 루프 종료"
