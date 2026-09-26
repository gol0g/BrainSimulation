#!/bin/bash
# E100: K38 설정의 규칙 반전 재학습. learn400(전반부 쌍둥이) / reversal800 / learn800(반전 없는 대조).
# 재개 가능(P13) — 결과 줄까지 있어야 완료로 친다(E099 사고). 96런.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E100.log"
RAW="$R/research/experiments/logs/E100"
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
  timeout 3600 python minimal_circuit.py --mode "$mode" --seed "$sd" --trial-seed "$ts" \
    "$@" $BASE > "$f" 2>&1
  local rc=$?
  grep -E "^=> MINCIRC" "$f" || { echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; }
}
for S in 5 6 7 8; do
  for T in 300 301 302 303 304 305 306 307; do
    run_one "L4 w$S t$T" learn "$S" "$T" --trials 400
    run_one "rev w$S t$T" reversal "$S" "$T" --trials 800
    run_one "L8 w$S t$T" learn "$S" "$T" --trials 800
  done
done
echo "[E100] 전체 루프 종료 — 원본 로그: $RAW"
