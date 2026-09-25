#!/bin/bash
# E095: 흔적 오염인가 tau 자체인가. 잔존량 exp(-gap/tau)를 맞춘 교차 설계. 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E095.log"
RAW="$R/research/experiments/logs/E095"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400 --act-drive 18.0"
done_already () { grep -qF "$1: " "$LOG" 2>/dev/null; }
run_one () {   # tag mode seed trialseed tau gap
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 7200 python minimal_circuit.py --mode "$2" --seed "$3" --trial-seed "$4" \
    --tau-e "$5" --gap-steps "$6" $BASE > "$f" 2>&1
  local rc=$?
  if grep -q "^=> MINCIRC" "$f"; then
    grep -E "^=> MINCIRC" "$f"
    grep -E "KC → out_L" "$f" | sed 's/^/        /'
  else
    echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'
  fi
}
for S in 0 3 4; do run_one "frozen w$S" frozen "$S" 100 50 600; done
# 교차: (잔존량, tau, gap)
echo "########## 교차 설계 — 잔존량을 맞춘 (tau, gap) ##########"
for CFG in "r05 50 150" "r05 200 600" "r05 400 1200" "r0025 50 300" "r0025 200 1200" "r0025 400 2400"; do
  set -- $CFG
  echo "--- 잔존 $1 / tau $2 / gap $3 ---"
  for S in 0 3 4; do
    for T in 100 101 102 103 104 105 106 107; do
      run_one "X_$1_tau$2_gap$3 w$S t$T" learn "$S" "$T" "$2" "$3"
    done
  done
done
echo "########## 최적점 탐색 — gap 600 고정 ##########"
for TAU in 12 25; do
  echo "--- tau $TAU ---"
  for S in 0 3 4; do
    for T in 100 101 102 103 104 105 106 107; do
      run_one "O_tau${TAU}_gap600 w$S t$T" learn "$S" "$T" "$TAU" 600
    done
  done
done
echo "[E095] 전체 루프 종료 — 원본 로그: $RAW"
