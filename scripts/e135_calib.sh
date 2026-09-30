#!/bin/bash
# E135 보정: 연속 STDP 발달. 회귀(E133 hebb dev corr w46) + 정답 아는 검사(eta 0) + eta 0.005/0.02 × 배선 18·19 × corr/indep.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E135"; WD="$R/research/experiments/traces/E135"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e135_run && cd /root/e135_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
DEV="--kc-wiring candidates --mismatch-w 8 --dev-items 20 --dev-w-fix 4 --dev-wc-total 4 --dev-wi-total 8 --dev-hebb-exposures 400"
STDP="--dev-mode stdp --dev-a-minus 0 --dev-tau-e 20 --dev-stdp-wmax-c 1.0 --dev-stdp-wmax-i 2.0"
echo "[회귀 — E133 hb dev corr w46 DEVHEBB 와 같아야]"
f="$OUT/path_regress_hebb_w46.log"
timeout 3600 python minimal_circuit.py --mode frozen --seed 46 --trial-seed 600 $K50 --trials 1 $SDO $DEV --dev-hebb-eta 1 --dev-env corr --dev-hebb-save "$WD/regress_hebb_w46.npz" > "$f" 2>&1
grep '^=> DEVHEBB' "$f" | sed 's/ → .*//' || { echo "[실패]"; tail -3 "$f"; }
echo "[정답 아는 검사 — stdp eta 0: 가중치 불변]"
f="$OUT/path_stdp_eta0_w18.log"
timeout 3600 python minimal_circuit.py --mode frozen --seed 18 --trial-seed 600 $K50 --trials 1 $SDO $DEV $STDP --dev-stdp-eta 0 --dev-env corr --dev-hebb-save "$WD/eta0_w18.npz" > "$f" 2>&1
grep -E '^\[STDP발달\]|^=> DEVHEBB' "$f" | sed 's/ → .*//' || { echo "[실패]"; tail -3 "$f"; }
echo "[보정]"
for ETA in 0.005 0.02; do for S in 18 19; do for ENV in corr indep; do
  f="$OUT/calib_eta${ETA}_w${S}_${ENV}.log"
  timeout 3600 python minimal_circuit.py --mode frozen --seed $S --trial-seed 600 $K50 --trials 1 $SDO $DEV $STDP --dev-stdp-eta $ETA --dev-env $ENV --dev-hebb-save "$WD/calib_eta${ETA}_w${S}_${ENV}.npz" > "$f" 2>&1; rc=$?
  if grep -q "^=> DEVHEBB" "$f"; then echo "  eta$ETA w$S $ENV: $(grep '^\[STDP발달\]' "$f") || $(grep '^=> DEVHEBB' "$f" | grep -oE '발화율.*같은 위치: 일치형 [0-9.]+ 불일치형 [0-9.]+')"; else echo "  eta$ETA w$S $ENV: [실패 rc=$rc]"; tail -3 "$f"; fi
done; done; done
echo "[E135 보정] 종료"
