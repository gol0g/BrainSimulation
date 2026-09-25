#!/bin/bash
# E096: 최적 tau_e 가 행동 창(act_steps)을 따라가는가. 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E096.log"
RAW="$R/research/experiments/logs/E096"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400 --act-drive 18.0"
done_already () { grep -qF "$1: " "$LOG" 2>/dev/null; }
run_one () {   # tag mode seed trialseed tau actsteps
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode "$2" --seed "$3" --trial-seed "$4" \
    --tau-e "$5" --act-steps "$6" $BASE > "$f" 2>&1
  local rc=$?
  grep -E "^=> MINCIRC" "$f" || { echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; }
}
for S in 0 3 4; do run_one "frozen w$S" frozen "$S" 100 12 15; done
for CFG in "15 3" "15 6" "15 12" "15 25" "5 3" "5 6" "5 12" "45 12" "45 25" "45 50"; do
  set -- $CFG
  echo "########## act_steps=$1 tau_e=$2 ##########"
  for S in 0 3 4; do
    for T in 100 101 102 103 104 105 106 107; do
      run_one "act$1_tau$2 w$S t$T" learn "$S" "$T" "$2" "$1"
    done
  done
done
echo "[E096] 전체 루프 종료 — 원본 로그: $RAW"
