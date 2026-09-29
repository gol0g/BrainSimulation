#!/bin/bash
# E128 경로 검사 2: crosshalf w4 과제 자극 기준 희석(학습 없음) — 배선 18·19.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E128"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e128_run && cd /root/e128_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
REST="--kc-inh 12.0 --sens-kc-p 0.02 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 20 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2 --trials 1"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3 --sd-diff cyclic --sd-credit --kc-wiring crosshalf"
for S in 18 19; do
  f="$OUT/path2_xh_w$S.log"
  timeout 1800 python minimal_circuit.py --mode frozen --seed $S --trial-seed 600 $REST $SDO > "$f" 2>&1; rc=$?
  if grep -q "^=> SDRATE" "$f"; then echo "  xh w$S: $(grep '^=> SDCREDIT' "$f" | grep -oE 'L전용 n=[0-9]+|R전용 n=[0-9]+|양쪽 n=[0-9]+' | tr '\n' ' ')| $(grep '^=> SDRATE' "$f" | sed 's/^=> SDRATE mode=frozen seed=[0-9]* trialseed=[0-9]* | //')"; else echo "  xh w$S: [실패 rc=$rc]"; tail -3 "$f"; fi
done
echo "[E128 경로 검사 2] 종료"
