#!/bin/bash
# E135 수정 검사: R-STDP 모델 객체 공유 후 — 기존 경로 회귀 2종 + STDP 정답 아는 검사(eta 0) + eta 0.005 1런(배선 18 corr).
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
echo "[회귀 K50 w0 t100 — 기대 first=60.5 reward=60.5 eval=100.0]"
f="$OUT/fix_regress_k50.log"; timeout 3600 python minimal_circuit.py --mode learn --seed 0 --trial-seed 100 $K50 --trials 400 > "$f" 2>&1; grep '^=> MINCIRC' "$f" || { echo "[실패]"; tail -3 "$f"; }
echo "[회귀 E133 corr learn w46 t600 — 기대 train_lbal=85.4 novel_lbal=69.4]"
f="$OUT/fix_regress_e133.log"; timeout 3600 python minimal_circuit.py --mode learn --seed 46 --trial-seed 600 $K50 --trials 800 $SDO --sd-diff cyclic --mismatch-w 8 --kc-wiring loaded --kc-wiring-file "$R/research/experiments/traces/E133/dev_corr_w46.npz" > "$f" 2>&1; grep '^=> SDLAB' "$f" || { echo "[실패]"; tail -3 "$f"; }
echo "[STDP 정답 아는 검사 eta 0]"
f="$OUT/fix_stdp_eta0_w18.log"; timeout 3600 python minimal_circuit.py --mode frozen --seed 18 --trial-seed 600 $K50 --trials 1 $SDO $DEV $STDP --dev-stdp-eta 0 --dev-env corr --dev-hebb-save "$WD/fix_eta0_w18.npz" > "$f" 2>&1; rc=$?
grep -E '^\[STDP발달\]|^=> DEVHEBB' "$f" | sed 's/ → .*//' || { echo "[실패 rc=$rc]"; tail -3 "$f"; }
echo "[STDP eta 0.005 배선 18 corr]"
f="$OUT/fix_stdp_eta0.005_w18.log"; timeout 3600 python minimal_circuit.py --mode frozen --seed 18 --trial-seed 600 $K50 --trials 1 $SDO $DEV $STDP --dev-stdp-eta 0.005 --dev-env corr --dev-hebb-save "$WD/fix_eta0.005_w18.npz" > "$f" 2>&1; rc=$?
grep -E '^\[STDP발달\]|^=> DEVHEBB' "$f" | sed 's/ → .*//' || { echo "[실패 rc=$rc]"; tail -3 "$f"; }
echo "[E135 수정 검사] 종료"
