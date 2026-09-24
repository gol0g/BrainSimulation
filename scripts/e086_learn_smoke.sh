#!/bin/bash
# E086 학습 확인: KC 분리가 성립한 설정에서 A->L / B->R 이 학습되는가.
# 고정 설정: p=0.02 inh=12 w=4.0 (양방향 대칭, 자카드 11~14%, KC 발화 11.6%)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
SEP="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0"
for MODE in learn noreward; do
  echo "### mode=$MODE"
  f=/root/minc_run/learn_$MODE.log
  timeout 2400 python minimal_circuit.py --seed 0 --trials 300 --block 50 --mode "$MODE" $SEP > "$f" 2>&1
  rc=$?
  if grep -q "MINCIRC" "$f"; then
    grep -E "시행|kc_out_|첫 구간" "$f" | sed 's/^/  /'
  else
    echo "  [실패 rc=$rc]"; tail -5 "$f" | sed 's/^/    /'
  fi
done
