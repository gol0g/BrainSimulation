#!/bin/bash
# E134 경로 검사: 회귀(E133 corr learn w46 t600 재현) + shift 발달 형성(배선 18·19, identity/shift) + [KC불러옴이동] 출력(frozen, 학습 결과 아님).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E134"; WD="$R/research/experiments/traces/E134"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e134_run && cd /root/e134_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
DEV="--kc-wiring candidates --mismatch-w 8 --dev-items 20 --dev-w-fix 4 --dev-wc-total 4 --dev-wi-total 8 --dev-hebb-eta 1 --dev-hebb-exposures 400"
echo "[Q1 회귀 — E133 corr learn w46 t600 와 같아야]"
f="$OUT/path_regress_e133_w46.log"
timeout 3600 python minimal_circuit.py --mode learn --seed 46 --trial-seed 600 $K50 --trials 800 $SDO --sd-diff cyclic --mismatch-w 8 --kc-wiring loaded --kc-wiring-file "$R/research/experiments/traces/E133/dev_corr_w46.npz" > "$f" 2>&1
grep '^=> SDLAB' "$f" || { echo "[실패]"; tail -3 "$f"; }
echo "[Q2 발달 형성 — identity/shift k=7, 배선 18·19]"
for S in 18 19; do for ENV in corr shift; do
  f="$OUT/path_dev_${ENV}_w$S.log"
  timeout 3600 python minimal_circuit.py --mode frozen --seed $S --trial-seed 600 $K50 --trials 1 $SDO $DEV --dev-env $ENV --dev-shift 7 --dev-hebb-save "$WD/path_dev_${ENV}_w$S.npz" > "$f" 2>&1; rc=$?
  if grep -q "^=> DEVSHIFT" "$f"; then echo "  $(grep '^=> DEVSHIFT' "$f" | sed 's/^=> DEVSHIFT //')"; else echo "  w$S $ENV: [실패 rc=$rc]"; tail -3 "$f"; fi
done; done
echo "[Q3 출력 경로 — shift 발달 연결을 shift 과제로 불러오기(frozen, trials 1)]"
f="$OUT/path_loaded_shift_w18.log"
timeout 3600 python minimal_circuit.py --mode frozen --seed 18 --trial-seed 600 $K50 --trials 1 --eval-trials 20 $SDO --sd-diff cyclic --sd-shift 7 --mismatch-w 8 --kc-wiring loaded --kc-wiring-file "$WD/path_dev_shift_w18.npz" > "$f" 2>&1
grep -E '^\[KC불러옴' "$f" | sed 's#/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/##' || { echo "[실패]"; tail -3 "$f"; }
echo "[E134 경로 검사] 종료"
