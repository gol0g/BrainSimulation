#!/bin/bash
# E088: 탐색·선택 루프를 끊으면 자율 학습이 되는가. 재개 가능(P13).
# 훈련은 --mode learn (정답을 알려주지 않고 보상만), 평가는 탐색·학습 없음.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E088.log"
RAW="$R/research/experiments/logs/E088"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
CFG="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --trials 400 --block 400 --eval-trials 100"
done_already () { grep -qF "$1: " "$LOG" 2>/dev/null; }
run_one () {   # tag mode eps seed
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  echo "  $tag:"
  timeout 3600 python minimal_circuit.py --mode "$2" --epsilon "$3" --seed "$4" $CFG > "$f" 2>&1
  local rc=$?
  if grep -q "^=> MINCIRC" "$f"; then
    grep -E "KC → out_L|kc_out_|^=> MINCIRC" "$f" | sed 's/^/      /'
  else
    echo "      [실패 rc=$rc]"; tail -3 "$f" | sed 's/^/        /'
  fi
}
for S in 0 1 2 3 4; do run_one "frozen s$S" frozen 0.6 "$S"; done
for E in 0.6 0.8 1.0; do
  echo "########## epsilon=$E ##########"
  for S in 0 1 2 3 4; do run_one "eps$E s$S" learn "$E" "$S"; done
done
echo "[E088] 전체 루프 종료 — 원본 로그: $RAW"
