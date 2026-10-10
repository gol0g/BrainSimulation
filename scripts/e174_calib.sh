#!/bin/bash
# E174 보정(사전 선택 규칙 — logs/E174/criteria_fixed.txt) — 표본 밖 뇌 15, E160 보정 칸 형성 가중치. 학습 없음(측정만).
# 맥락 세기 I_c ∈ {0.5, 1, 1.5, 2, 3, 4}(KC 억제 뉴런 맥락 부분집합 10% 에 더하는 전류): kcctx 240 제시(good-좌 끔·켬, 맥락 단독, good-우 끔·켬, 맥락 단독 × 40).
# 선택은 scripts/e174_pick.py → logs/E174/pick.txt('IC=<값>' 또는 'IC=none').
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E174/calib"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
W0="$R/research/experiments/traces/E174/pathcheck/w_a_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e174_run && cd /root/e174_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
[ -s "$W0" ] || { echo "[실패] 경로 검사 가중치 없음 $W0"; exit 1; }
for G in "05 0.5" "10 1" "15 1.5" "20 2" "30 3" "40 4"; do
  set -- $G; T=$1; I=$2
  f="$OUT/kcctx_c${T}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 --ctx-inh-i $I --ctx-inh-frac 0.1 \
    --decomp-weights $W0 --decomp-mode kcctx --trials 240 > "$f" 2>&1
  echo "[I_c=$I rc=$?] $(grep '^=> KCCTX' "$f" | cut -c1-420)"
done
cd $R && python3 scripts/e174_pick.py
echo "[E174 보정] 종료"
