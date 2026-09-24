#!/bin/bash
# 훈련(탐색 있음) / 평가(탐색 없음, 무작위 순서) 분리. frozen 대비로 판정.
# 이월 확인을 위해 무자극 간격도 함께 훑는다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
SEP="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.05"
for GAP in 200 600; do
  for MODE in frozen learn; do
    for S in 0 1 2; do
      printf "gap=%-4s %-7s seed=%d : " "$GAP" "$MODE" "$S"
      f=/root/minc_run/ev2_${GAP}_${MODE}_$S.log
      timeout 2400 python minimal_circuit.py --seed "$S" --trials 400 --block 400 \
        --mode "$MODE" --epsilon 0.6 --gap-steps "$GAP" $SEP > "$f" 2>&1
      rc=$?
      grep -E "^=> MINCIRC" "$f" || { echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/    /'; }
    done
  done
done
