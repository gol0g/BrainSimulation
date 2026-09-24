#!/bin/bash
# 최소 회로가 결정론적인가 + noreward가 frozen과 같아야 하는데 다른 이유.
#
# noreward는 도파민을 0으로 두므로 rstdp_model의 `if (dopamine != 0.0)` 가드가 갱신을 막는다.
# 즉 가중치가 안 변하고 frozen과 **같아야** 한다. 그런데 s3/s4에서 크게 달랐다.
# 비결정성이면 E086의 모든 비교가 잡음이다. 먼저 가른다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
CFG="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.05 --epsilon 0.6 --gap-steps 600 --trials 400 --block 400 --eval-trials 100"
echo "### 같은 조건 3회 반복 (결정론 확인)"
for i in 1 2 3; do
  printf "  learn s4 #%d: " "$i"
  timeout 2400 python minimal_circuit.py --mode learn --seed 4 $CFG 2>&1 | grep -E "^=> MINCIRC" || echo "[실패]"
done
echo "### frozen vs noreward (같아야 한다) — 가중치까지 확인"
for M in frozen noreward; do
  printf "  %-9s s4: " "$M"
  f=/root/minc_run/det_${M}_4.log
  timeout 2400 python minimal_circuit.py --mode "$M" --seed 4 $CFG > "$f" 2>&1
  grep -E "^=> MINCIRC" "$f" || echo "[실패]"
  grep -E "kc_out_" "$f" | sed 's/^/      /'
done
