#!/bin/bash
# E086 본실험: 최소 회로 — 보상이 행동을 바꾸는가. 재개 가능(P13).
# 고정 설정(조정 종료 2026-09-20 00:14). 5모드 × 5시드 = 25런.
# 판정은 **같은 시드의 frozen 대비**. 50% 대비가 아니다(배선 기준선이 시드마다 0~87%).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E086.log"
RAW="$R/research/experiments/logs/E086"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null

CFG="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.05 --epsilon 0.6 --gap-steps 600 --trials 400 --block 400 --eval-trials 100"

done_already () { grep -qF "$1: =>" "$LOG" 2>/dev/null; }
run_one () {   # tag mode seed
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode "$2" --seed "$3" $CFG > "$f" 2>&1
  local rc=$?
  grep -E "^=> MINCIRC" "$f" || { echo "[실패 rc=$rc]"; tail -3 "$f" | sed 's/^/      /'; }
}

for MODE in frozen noreward yoked learn reversal; do
  echo "########## $MODE ##########"
  for S in 0 1 2 3 4; do
    run_one "$MODE s$S" "$MODE" "$S"
  done
done
echo "[E086] 전체 루프 종료 — 원본 로그: $RAW"
