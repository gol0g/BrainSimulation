#!/bin/bash
# 학습 후 출력이 자극에 따라 갈리는가. 정답률만으로는 원인을 못 본다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
rm -rf minimal_circuit_CODE
SEP="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0"
for MODE in frozen learn; do
  for S in 0 2; do
    echo "### $MODE seed=$S"
    f=/root/minc_run/out_${MODE}_$S.log
    timeout 2400 python minimal_circuit.py --seed "$S" --trials 300 --block 150 --mode "$MODE" $SEP > "$f" 2>&1
    rc=$?
    if grep -q "자극 A →" "$f"; then
      grep -E "자극 [AB] →|kc_out_|^=> MINCIRC" "$f" | sed 's/^/  /'
    else
      echo "  [실패 rc=$rc]"; tail -3 "$f" | sed 's/^/    /'
    fi
  done
done
