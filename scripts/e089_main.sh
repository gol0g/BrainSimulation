#!/bin/bash
# E089: 시행 난수열만 바꿨을 때 학습 결과가 얼마나 갈리는가. 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E089.log"
RAW="$R/research/experiments/logs/E089"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
CFG="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --trials 400 --block 400 --eval-trials 100 --epsilon 0.6"
done_already () { grep -qF "$1: " "$LOG" 2>/dev/null; }
run_one () {   # tag mode seed trialseed
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode "$2" --seed "$3" --trial-seed "$4" $CFG > "$f" 2>&1
  local rc=$?
  grep -E "^=> MINCIRC" "$f" || { echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; }
}
for S in 0 1 2 3 4; do
  echo "########## 배선 시드 $S ##########"
  run_one "frozen w$S" frozen "$S" 100
  for T in 100 101 102 103 104 105 106 107; do
    run_one "learn w$S t$T" learn "$S" "$T"
  done
done
echo "[E089] 전체 루프 종료 — 원본 로그: $RAW"
