#!/bin/bash
# E086: 고른 행동을 자격흔적에 남기면 학습이 되는가.
# frozen(배선만) 대비로 판단한다 — 50% 대비가 아니다(seed0 배선이 87%).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
rm -rf minimal_circuit_CODE
SEP="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0"
for S in 0 1 2; do
  printf "frozen  seed=%d : " "$S"
  f=/root/minc_run/c_frozen_$S.log
  timeout 2400 python minimal_circuit.py --seed "$S" --trials 300 --block 100 --mode frozen $SEP > "$f" 2>&1
  grep -E "^=> MINCIRC" "$f" || { echo "[실패]"; tail -2 "$f" | sed 's/^/    /'; }
  printf "learn   seed=%d : " "$S"
  f=/root/minc_run/c_learn_$S.log
  timeout 2400 python minimal_circuit.py --seed "$S" --trials 300 --block 100 --mode learn $SEP > "$f" 2>&1
  grep -E "^=> MINCIRC" "$f" || { echo "[실패]"; tail -2 "$f" | sed 's/^/    /'; }
done
