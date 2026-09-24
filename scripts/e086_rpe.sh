#!/bin/bash
# 보상 예측 오차(기준선 차감)가 전역 표류를 없애는가.
# 비교: 비대칭(0.5) / 대칭(1.0) / RPE(기준선 0.05, 0.2)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
rm -rf minimal_circuit_CODE
SEP="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0"
for CFG in "1.0 0.0" "1.0 0.05" "1.0 0.2"; do
  set -- $CFG
  for S in 0 2; do
    echo "### da_neg=$1 baseline=$2 seed=$S"
    f=/root/minc_run/rpe_$1_$2_$S.log
    timeout 2400 python minimal_circuit.py --seed "$S" --trials 400 --block 200 \
      --mode learn --da-neg "$1" --baseline "$2" $SEP > "$f" 2>&1
    rc=$?
    if grep -q "자극 A →" "$f"; then
      grep -E "자극 [AB] →|^=> MINCIRC" "$f" | sed 's/^/  /'
    else
      echo "  [실패 rc=$rc]"; tail -3 "$f" | sed 's/^/    /'
    fi
  done
done
