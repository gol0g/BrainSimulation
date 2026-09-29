#!/bin/bash
# E130 경로 검사: developed 배선(corr/indep) 장치 연결·비교 분리(학습 없음) + 기본 배선 회귀. 배선 18.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E130"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e130_run && cd /root/e130_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
DEV="--kc-wiring developed --mismatch-w 8 --dev-theta 0.15 --dev-rounds 1000 --dev-exposures 200 --dev-items 20"
echo "[회귀 — E124 learn w10 t600 SDGEN 과 같아야]"
f="$OUT/path_regress_sd_w10_t600.log"; timeout 3600 python minimal_circuit.py --mode learn --seed 10 --trial-seed 600 $K50 --trials 800 $SDO > "$f" 2>&1
grep '^=> SDGEN' "$f" || { echo "[실패]"; tail -3 "$f"; }
echo "[developed 표현(학습 없음) — 배선 18]"
for ENV in corr indep; do
  f="$OUT/path_dev_${ENV}_w18.log"
  timeout 1800 python minimal_circuit.py --mode frozen --seed 18 --trial-seed 600 $K50 --trials 1 --eval-trials 20 $SDO --sd-diff cyclic --sd-credit $DEV --dev-env $ENV > "$f" 2>&1; rc=$?
  if grep -q "^=> SDCOMP" "$f"; then echo "  $ENV: $(grep '^\[KC발달\]' "$f") || $(grep '^=> SDCOMP' "$f" | sed 's/^=> SDCOMP seed=[0-9]* | //')"; else echo "  $ENV: [실패 rc=$rc]"; tail -3 "$f"; fi
done
echo "[E130 경로 검사] 종료"
