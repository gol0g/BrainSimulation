#!/bin/bash
# E101: 반전 고착 원인 — w_max 2 (L4 / rev8 / L8 / rev16) + w_max 20 rev16. 배선 5~7 × 난수열 300~307. 120런.
# 재개 가능 — 결과 줄까지 있어야 완료.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E101.log"
RAW="$R/research/experiments/logs/E101"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12"
done_already () { grep -qF "$1: => MINCIRC" "$LOG" 2>/dev/null; }
run_one () {   # tag mode seed trialseed extra...
  local tag="$1"; local mode="$2"; local sd="$3"; local ts="$4"; shift 4
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 7200 python minimal_circuit.py --mode "$mode" --seed "$sd" --trial-seed "$ts" \
    "$@" $BASE > "$f" 2>&1
  local rc=$?
  grep -E "^=> MINCIRC" "$f" || { echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; }
}
for S in 5 6 7; do
  for T in 300 301 302 303 304 305 306 307; do
    run_one "W2-L4 w$S t$T"     learn    "$S" "$T" --w-max 2  --trials 400
    run_one "W2-rev8 w$S t$T"   reversal "$S" "$T" --w-max 2  --trials 800
    run_one "W2-L8 w$S t$T"     learn    "$S" "$T" --w-max 2  --trials 800
    run_one "W2-rev16 w$S t$T"  reversal "$S" "$T" --w-max 2  --trials 1600
    run_one "W20-rev16 w$S t$T" reversal "$S" "$T" --w-max 20 --trials 1600
  done
done
echo "[E101] 전체 루프 종료 — 원본 로그: $RAW"
