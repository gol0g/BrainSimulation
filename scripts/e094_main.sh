#!/bin/bash
# E094: act_drive·tau_e 용량-반응(P7). 두 축을 따로 훑는다. 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E094.log"
RAW="$R/research/experiments/logs/E094"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400"
done_already () { grep -qF "$1: " "$LOG" 2>/dev/null; }
run_one () {   # tag mode seed trialseed tau act
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode "$2" --seed "$3" --trial-seed "$4" \
    --tau-e "$5" --act-drive "$6" $BASE > "$f" 2>&1
  local rc=$?
  if grep -q "^=> MINCIRC" "$f"; then
    grep -E "^=> MINCIRC" "$f"
    grep -E "KC → out_L" "$f" | sed 's/^/        /'
  else
    echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'
  fi
}
for S in 0 3 4; do run_one "frozen w$S" frozen "$S" 100 200 18.0; done
echo "########## 축 A: tau_e=200 고정, act_drive 훑기 ##########"
for ACT in 6.0 18.0 36.0 72.0; do
  echo "--- act=$ACT ---"
  for S in 0 3 4; do
    for T in 100 101 102 103 104 105 106 107; do
      run_one "A_tau200act${ACT} w$S t$T" learn "$S" "$T" 200 "$ACT"
    done
  done
done
echo "########## 축 B: act_drive=18 고정, tau_e 훑기 ##########"
for TAU in 50 100 400; do
  echo "--- tau=$TAU ---"
  for S in 0 3 4; do
    for T in 100 101 102 103 104 105 106 107; do
      run_one "B_tau${TAU}act18.0 w$S t$T" learn "$S" "$T" "$TAU" 18.0
    done
  done
done
echo "[E094] 전체 루프 종료 — 원본 로그: $RAW"
