#!/bin/bash
# E127 경로 검사: --sd-credit 도구. 양성 대조(half1 w10 t600, E125 에서 97% 획득) + 균형 cyclic 배선 18(표본 밖) learn/frozen.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E127"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e127_run && cd /root/e127_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2 --trials 800"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3 --sd-credit"
run() {
  local n="$1"; shift; local f="$OUT/path_$n.log"
  timeout 3600 python minimal_circuit.py "$@" > "$f" 2>&1; local rc=$?
  if grep -q "^=> SDCREDIT" "$f"; then echo "  $n: $(grep -E '^=> SDCREDIT|^=> SDLAB' "$f" | tr '\n' ' ')"; else echo "  $n: [실패 rc=$rc]"; tail -3 "$f"; fi
}
echo "[P1 양성 대조 — half1 w10 t600, E125 SDLAB 과 같아야: train_accL=93.2 train_accR=100.0 train_lbal=96.6]"
run pos_half1_w10_t600 --mode learn --seed 10 --trial-seed 600 $K50 $SDO --sd-rule half1
run pos_half1_frozen_w10 --mode frozen --seed 10 --trial-seed 600 $K50 $SDO --sd-rule half1
echo "[P2 균형 cyclic — 배선 18]"
run cy_learn_w18_t600 --mode learn --seed 18 --trial-seed 600 $K50 $SDO --sd-diff cyclic
run cy_frozen_w18_t600 --mode frozen --seed 18 --trial-seed 600 $K50 $SDO --sd-diff cyclic
echo "[E127 경로 검사] 종료"
