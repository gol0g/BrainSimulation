#!/bin/bash
# E128 경로 검사 1: 기본 배선 회귀 + crosshalf 가중치 보정(학습 없음). 배선 18·19.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E128"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e128_run && cd /root/e128_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2 --trials 800"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
echo "[Q1 회귀 — E124 learn w10 t600 SDGEN 과 같아야]"
f="$OUT/path_regress_sd_w10_t600.log"; timeout 3600 python minimal_circuit.py --mode learn --seed 10 --trial-seed 600 $K50 $SDO > "$f" 2>&1
grep '^=> SDGEN' "$f" || { echo "[실패]"; tail -3 "$f"; }
echo "[Q2 crosshalf 가중치 보정 — probe-sd]"
for S in 18 19; do for W in 2 3 4 6 8; do
  f="$OUT/path_xh_w${S}_sw$W.log"
  timeout 1800 python minimal_circuit.py --mode learn --seed $S $K50 $SDO --kc-wiring crosshalf --sens-kc-w $W --probe-sd > "$f" 2>&1; rc=$?
  if grep -q "^=> SDKC" "$f"; then echo "  w$S sw$W: $(grep '^\[KC배선\]' "$f") $(grep '^=> SDKC' "$f" | sed 's/^=> SDKC seed=[0-9]* //')"; else echo "  w$S sw$W: [실패 rc=$rc]"; tail -3 "$f"; fi
done; done
echo "[E128 경로 검사 1] 종료"
