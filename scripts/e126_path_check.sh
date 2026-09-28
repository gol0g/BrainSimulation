#!/bin/bash
# E126 경로 검사: --sd-diff cyclic(같음 4 vs 다름 4 균형). 회귀(E124 w10 t600) + 배선 18(본 표본 밖).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E126"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e126_run && cd /root/e125_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
run() {
  local n="$1" pat="$2"; shift 2; local f="$OUT/path_$n.log"
  timeout 3600 python minimal_circuit.py "$@" > "$f" 2>&1; local rc=$?
  if grep -qE "$pat" "$f"; then echo "  $n: $(grep -E '^=> ' "$f" | tr '\n' ' ')"; else echo "  $n: [실패 rc=$rc]"; tail -3 "$f"; fi
}
echo "[Q1 회귀 — E124 learn w10 t600 SDGEN 과 같아야 함]"
run regress_sd_w10_t600 '^=> SDGEN' --mode learn --seed 10 --trial-seed 600 $K50 --trials 800 $SDO
echo "[Q2 cyclic — 배선 18]"
for T in 600 601; do run cy_learn_w18_t$T '^=> SDLAB' --mode learn --seed 18 --trial-seed $T $K50 --trials 800 $SDO --sd-diff cyclic; done
run cy_frozen_w18_t600 '^=> SDLAB' --mode frozen --seed 18 --trial-seed 600 $K50 --trials 800 $SDO --sd-diff cyclic
echo "[E126 경로 검사] 종료"
