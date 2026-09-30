#!/bin/bash
# E133 보정: 망 스파이크 헤브 발달(--kc-wiring candidates --dev-hebb-save). 배선 18·19 × corr/indep × w-fix 4/6. + 기존 경로 회귀.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E133"; mkdir -p "$OUT"; WD="$R/research/experiments/traces/E133"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e133_run && cd /root/e133_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
echo "[회귀 — E124 learn w10 t600 SDGEN 과 같아야]"
f="$OUT/path_regress_sd_w10_t600.log"; timeout 3600 python minimal_circuit.py --mode learn --seed 10 --trial-seed 600 $K50 --trials 800 $SDO > "$f" 2>&1
grep '^=> SDGEN' "$f" || { echo "[실패]"; tail -3 "$f"; }
echo "[회귀 — E130 developed corr w10 t600 SDLAB 과 같아야]"
f="$OUT/path_regress_dev_w10_t600.log"; timeout 3600 python minimal_circuit.py --mode learn --seed 10 --trial-seed 600 $K50 --trials 800 $SDO --sd-diff cyclic --kc-wiring developed --mismatch-w 8 --dev-theta 0.15 --dev-rounds 1000 --dev-exposures 200 --dev-items 20 --dev-env corr > "$f" 2>&1
grep '^=> SDLAB' "$f" || { echo "[실패]"; tail -3 "$f"; }
echo "[헤브 발달 보정]"
for S in 18 19; do for WF in 4 6; do for ENV in corr indep; do
  f="$OUT/calib_w${S}_wf${WF}_${ENV}.log"
  timeout 3600 python minimal_circuit.py --mode frozen --seed $S --trial-seed 600 $K50 --trials 1 $SDO --kc-wiring candidates --mismatch-w 8 --dev-items 20 --dev-env $ENV --dev-w-fix $WF --dev-wc-total 4 --dev-wi-total 8 --dev-hebb-eta 1 --dev-hebb-exposures 400 --dev-hebb-save "$WD/calib_w${S}_wf${WF}_${ENV}.npz" > "$f" 2>&1; rc=$?
  if grep -q "^=> DEVHEBB" "$f"; then echo "  $(grep '^=> DEVHEBB' "$f" | sed 's/^=> DEVHEBB //; s/ → .*//')"; else echo "  w$S wf$WF $ENV: [실패 rc=$rc]"; tail -3 "$f"; fi
done; done; done
echo "[E133 보정] 종료"
