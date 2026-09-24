#!/bin/bash
# E086 기준선: 배선만으로 나오는 정답률은 얼마인가 (C47 재발 방지).
# frozen  = 학습률 0        -> 순수 배선 성능
# noreward= 도파민 안 줌    -> 가소성은 살아있으나 신호 없음
# learn   = 정상
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
rm -rf minimal_circuit_CODE
SEP="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0"
for MODE in frozen noreward learn; do
  for S in 0 1 2; do
    printf "%-9s seed=%d : " "$MODE" "$S"
    f=/root/minc_run/null_${MODE}_$S.log
    timeout 2400 python minimal_circuit.py --seed "$S" --trials 300 --block 100 --mode "$MODE" $SEP > "$f" 2>&1
    rc=$?
    grep -E "^=> MINCIRC" "$f" || { echo "[실패 rc=$rc]"; tail -3 "$f" | sed 's/^/    /'; }
  done
done
