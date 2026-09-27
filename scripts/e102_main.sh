#!/bin/bash
# E102: 회귀(K38 24/24) → 추적 반전 12런(배선 5·6·7 × 300~303) + 런별 분석. 재개 가능(결과 줄 기준).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E102.log"
RAW="$R/research/experiments/logs/E102"
TR="$R/research/experiments/traces/E102"
mkdir -p "$RAW" "$TR"
if ! grep -q "^\[회귀\] \*\*통과\*\*" "$LOG" 2>/dev/null; then
  echo "########## 회귀 (코드 변경 후 필수) ##########"
  bash $R/scripts/regression_mincirc.sh || { echo "[E102] 회귀 실패 — 중단"; exit 1; }
fi
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 20"
echo "########## 추적 반전 ##########"
for S in 5 6 7; do for T in 300 301 302 303; do
  tag="trace w$S t$T"
  if grep -qF "$tag: => MINCIRC" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/trace_w${S}_t${T}.log"; z="$TR/trace_w${S}_t${T}.npz"
  printf "  %s: " "$tag"
  timeout 7200 python minimal_circuit.py --mode reversal --seed "$S" --trial-seed "$T" --trials 800 --trace-file "$z" $BASE > "$f" 2>&1
  rc=$?
  if grep -qE "^=> MINCIRC" "$f"; then
    python $R/scripts/analyze_e102.py "$z" > "$RAW/trace_w${S}_t${T}_summary.log" 2>&1
    grep -E "^=> MINCIRC" "$f"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done; done
echo "[E102] 전체 루프 종료"
