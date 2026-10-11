#!/bin/bash
# E183 보정(사전 규칙 — logs/E183/criteria_fixed.txt [보정]) — 표본 밖 뇌 15, O 팔 rawcat 전체 절차, feat_w 격자 {8, 16, 24}(필요하면 한 번 확장).
# 선택은 scripts/e183_pick.py → logs/E183/pick.txt. 요약 줄 "[W=16 rc=0] K..." 은 RCKC n 평균(형성 전·후)만 — 분리 지표는 선택에 안 씀.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E183/calib"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e183_calib_run && cd /root/e183_calib_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
echo "[E183 보정] 시작 $(date '+%F %T')"
run_w() {
  local W=$1; local g="$OUT/rc_w${W}_b15.log"
  if grep -q '^=> RAWCAT ' "$g" 2>/dev/null; then echo "[W=$W] 건너뜀(있음)"; return; fi
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 --conn-snapshot $SN15 \
    --feat-n 200 --feat-w $W --feat-p 0.10 --feat-oja --feat-oja-eta 0 --decomp-weights $WB --decomp-mode rawcat --rc-repeat-pre --rc-expo-eta 0.02 > "$g" 2>&1
  echo "[W=$W rc=$?] 완료 $(grep -c '^=> RAWCAT ' "$g") RCKC $(grep -c '^RCKC ' "$g")"
}
for W in 8 16 24; do run_w $W; done
cd $R && python3 scripts/e183_pick.py
P=$(cat research/experiments/logs/E183/pick.txt)
case "$P" in
  EXTEND=*) cd /root/e183_calib_run; for W in $(echo "${P#EXTEND=}" | tr ',' ' '); do run_w $W; done; cd $R && python3 scripts/e183_pick.py ;;
esac
echo "[E183 보정] 선택 $(cat $R/research/experiments/logs/E183/pick.txt) 종료 $(date '+%F %T')"
