#!/bin/bash
# E124 경로 검사(보정, 학습 없음): KC 입력 밀도·가중치별 결합(AND) KC 비율. 배선 18·19(본 표본 밖).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E124"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e124_run && cd /root/e124_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
REST="--kc-inh 12.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
for S in 18 19; do
  for PW in "0.02 4.0" "0.02 2.0" "0.1 4.0" "0.1 2.0" "0.1 1.5" "0.1 1.0" "0.2 1.0" "0.2 0.7"; do
    set -- $PW; P=$1; W=$2; f="$OUT/calib_w${S}_p${P}_w${W}.log"
    timeout 1800 python minimal_circuit.py --mode learn --seed $S --sens-kc-p $P --sens-kc-w $W $REST $SDO --probe-sd > "$f" 2>&1; rc=$?
    if grep -q "^=> SDKC" "$f"; then echo "  $(grep '^=> SDKC' "$f")"; else echo "  w$S p$P w$W: [실패 rc=$rc]"; tail -2 "$f"; fi
  done
done
echo "[E124 보정] 종료"
