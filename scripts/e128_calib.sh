#!/bin/bash
# E128 사전 보정(학습 없음): 표현 후보별 균형 같음/다름 8자극 KC 부류 수. 배선 18·19.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E128"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e128_run && cd /root/e128_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
REST="--da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 20 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2 --trials 1"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3 --sd-diff cyclic --sd-credit"
for S in 18 19; do
  for C in "K50|--kc-inh 12.0 --sens-kc-p 0.02 --sens-kc-w 4.0" "inh24|--kc-inh 24.0 --sens-kc-p 0.02 --sens-kc-w 4.0" "inh48|--kc-inh 48.0 --sens-kc-p 0.02 --sens-kc-w 4.0" "p01w15|--kc-inh 12.0 --sens-kc-p 0.1 --sens-kc-w 1.5" "p02w10|--kc-inh 12.0 --sens-kc-p 0.2 --sens-kc-w 1.0"; do
    N="${C%%|*}"; A="${C#*|}"; f="$OUT/calib_${N}_w$S.log"
    timeout 1800 python minimal_circuit.py --mode frozen --seed $S --trial-seed 600 $A $REST $SDO > "$f" 2>&1; rc=$?
    if grep -q "^=> SDRATE" "$f"; then echo "  $N w$S: $(grep '^=> SDCREDIT' "$f" | grep -oE 'L전용 n=[0-9]+|R전용 n=[0-9]+|양쪽 n=[0-9]+' | tr '\n' ' ')| $(grep '^=> SDRATE' "$f" | sed 's/^=> SDRATE mode=frozen seed=[0-9]* trialseed=[0-9]* | //')"; else echo "  $N w$S: [실패 rc=$rc]"; tail -2 "$f"; fi
  done
done
echo "[E128 보정 2차(발화율)] 종료"
