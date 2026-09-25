#!/bin/bash
# E091: 보상 예측 오차가 정답을 붙잡게 하는가. 재개 가능(P13).
# 지표는 성공률·파괴율·**유지율**(1600시행/400시행 성공률 비).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E091.log"
RAW="$R/research/experiments/logs/E091"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001"
done_already () { grep -qF "$1: " "$LOG" 2>/dev/null; }
run_one () {   # tag mode seed trialseed baseline trials
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode "$2" --seed "$3" --trial-seed "$4" \
    --baseline "$5" --trials "$6" $BASE > "$f" 2>&1
  local rc=$?
  grep -E "^=> MINCIRC" "$f" || { echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; }
}
for S in 0 1 2 3 4; do run_one "frozen w$S" frozen "$S" 100 0.0 400; done
for TR in 400 1600; do
  for B in 0.0 0.02 0.1; do
    echo "########## baseline=$B trials=$TR ##########"
    for S in 0 1 2 3 4; do
      for T in 100 101 102 103 104 105 106 107; do
        run_one "b${B}n${TR} w$S t$T" learn "$S" "$T" "$B" "$TR"
      done
    done
  done
done
echo "[E091] 전체 루프 종료 — 원본 로그: $RAW"
