#!/bin/bash
# E097: 행동 창인가 보상 지연인가. --delay-steps 로 교락을 분리. 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E097.log"
RAW="$R/research/experiments/logs/E097"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400 --act-drive 18.0"
done_already () { grep -qF "$1: " "$LOG" 2>/dev/null; }
run_one () {   # tag mode seed trialseed tau act delay
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode "$2" --seed "$3" --trial-seed "$4" \
    --tau-e "$5" --act-steps "$6" --delay-steps "$7" $BASE > "$f" 2>&1
  local rc=$?
  grep -E "^=> MINCIRC" "$f" || { echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; }
}
for S in 0 3 4; do run_one "frozen w$S" frozen "$S" 100 12 15 0; done
echo "########## 축 A: 창 15 고정, 지연만 변경 ##########"
for D in 0 30 90; do
  for TAU in 12 25 50; do
    echo "--- delay=$D tau=$TAU ---"
    for S in 0 3 4; do
      for T in 100 101 102 103 104 105 106 107; do
        run_one "A_d${D}_tau${TAU} w$S t$T" learn "$S" "$T" "$TAU" 15 "$D"
      done
    done
  done
done
echo "########## 축 B: E096 미탐색 범위 보완 ##########"
for CFG in "5 12" "5 25" "5 50" "45 50" "45 100" "45 200"; do
  set -- $CFG
  echo "--- act=$1 tau=$2 ---"
  for S in 0 3 4; do
    for T in 100 101 102 103 104 105 106 107; do
      run_one "B_act$1_tau$2 w$S t$T" learn "$S" "$T" "$2" "$1" 0
    done
  done
done
echo "[E097] 전체 루프 종료 — 원본 로그: $RAW"
