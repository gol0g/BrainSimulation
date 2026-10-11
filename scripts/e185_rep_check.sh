#!/bin/bash
# E185 경로 검사 (a) 보충 — 반복성 확인(2026-10-12): (a) 에서 스냅숏 없음 −0.6136·스냅숏 −0.6137·스냅숏+특징 −0.6136 이 나왔다(1e-4 차).
# 같은 조건을 반복해 실행 간 비결정성인지(같은 조건이 −0.6136/−0.6137 을 오감) 연결 차이인지(스냅숏 조건만 늘 다름) 가린다. 측정 아님(경로 검사 보충).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E185/pathcheck/rep"; mkdir -p "$OUT"
KW="$R/research/experiments/traces/E161/kctype_oja_b16.npz"
SN="$R/research/experiments/traces/E176/snap/conn_b16.npz"
WA="$R/research/experiments/traces/E172/w_A_b16.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e185_run && cd /root/e185_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
echo "[E185 반복성 검사] 시작 $(date '+%F %T')"
for V in snap snap snap nosnap nosnap; do
  case $V in nosnap) X="" ;; snap) X="--conn-snapshot $SN" ;; esac
  i=$((${i:-0} + 1)); g="$OUT/${V}_$i.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 16 --kc-type-weights $KW $X --decomp-weights $WA --decomp-mode all --eval-variant base > "$g" 2>&1; rc=$?
  echo "[$i $V rc=$rc] $(grep '^=> DECOMP' "$g" | grep -oE 'mod=[-+0-9.]+ acc=[0-9.]+ off=[-+0-9.]+')"
done
echo "[E185 반복성 검사] 종료 $(date '+%F %T')"
