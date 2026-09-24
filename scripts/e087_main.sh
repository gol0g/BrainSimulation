#!/bin/bash
# E087: 지도 조건에서 보상이 올바른 시냅스를 강화하는가. 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E087.log"
RAW="$R/research/experiments/logs/E087"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
CFG="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --sup-correct 0.5 --trials 400 --block 400 --eval-trials 100"
done_already () { grep -qF "$1: " "$LOG" 2>/dev/null; }
run_one () {
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  echo "  $tag:"
  timeout 3600 python minimal_circuit.py --mode "$2" --seed "$3" $CFG > "$f" 2>&1
  local rc=$?
  if grep -q "가중치 구조" "$f"; then
    grep -E "KC → out_L|=== 평가|^=> MINCIRC|kc_out_" "$f" | sed 's/^/      /'
  else
    echo "      [실패 rc=$rc]"; tail -3 "$f" | sed 's/^/        /'
  fi
}
for MODE in frozen supervised; do
  echo "########## $MODE ##########"
  for S in 0 1 2 3 4; do run_one "$MODE s$S" "$MODE" "$S"; done
done
echo "[E087] 전체 루프 종료 — 원본 로그: $RAW"
