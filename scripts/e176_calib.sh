#!/bin/bash
# E176 보정(사전 선택 규칙 — logs/E176/criteria_fixed.txt) — 표본 밖 뇌 15, E160 보정 칸 형성 가중치. 학습 없음(측정만).
# 맥락 세기 w ∈ {1, 2, 4, 8}(맥락 전용 집단 zz_ctx → KC 가중치, 초기 연결 스냅숏 conn_b15): kcctx 240 제시(good-좌 끔·켬, 맥락 단독, good-우 끔·켬, 맥락 단독 × 40).
# 선택은 scripts/e176_pick.py → logs/E176/pick.txt('W=<값>' 또는 'W=none').
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E176/calib"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
W0="$R/research/experiments/traces/E176/pathcheck/w_a_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e176_run && cd /root/e176_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
[ -s "$W0" ] || { echo "[실패] 경로 검사 가중치 없음 $W0"; exit 1; }
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
for G in "1 1" "2 2" "4 4" "8 8"; do
  set -- $G; T=$1; I=$2
  f="$OUT/kcctx_w${T}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 --conn-snapshot $SN15 --ctx-n 200 --ctx-w $I --ctx-p 0.10 \
    --decomp-weights $W0 --decomp-mode kcctx --trials 240 > "$f" 2>&1
  echo "[w=$I rc=$?] $(grep '^=> KCCTX' "$f" | cut -c1-600)"
done
cd $R && python3 scripts/e176_pick.py
echo "[E176 보정] 종료"
