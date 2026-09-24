#!/bin/bash
# E086 전제: KC가 A/B를 분리하는가. 순서 효과를 제거하고 양방향으로 잰다.
# 목표: 양쪽 다 KC 발화 5~15%, 자카드 30% 이하.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
rm -rf minimal_circuit_CODE
for CFG in "0.05 6.0 2.0" "0.02 6.0 4.0" "0.02 12.0 4.0"; do
  set -- $CFG
  for REV in "" "--probe-reverse"; do
    printf "p=%-5s inh=%-5s w=%-4s %-16s: " "$1" "$2" "$3" "${REV:-A먼저}"
    f=/root/minc_run/sep_$1_$2_$3_${REV:-fwd}.log
    timeout 900 python minimal_circuit.py --seed 0 --probe-kc \
      --sens-kc-p "$1" --kc-inh "$2" --sens-kc-w "$3" $REV > "$f" 2>&1
    rc=$?
    grep -E "^=> KCSEP" "$f" || { echo "[실패 rc=$rc]"; tail -3 "$f" | sed 's/^/    /'; }
  done
done
