#!/bin/bash
# E104: 회귀 → w_max 2 반전 시간 경과 F400(400+1200)·F800(800+1200) 각 12런. 재개 가능(결과 줄 기준).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E104.log"
RAW="$R/research/experiments/logs/E104"
TR="$R/research/experiments/traces/E104"
mkdir -p "$RAW" "$TR"
if ! grep -q "^\[회귀\] \*\*통과\*\*" "$LOG" 2>/dev/null; then
  echo "########## 회귀 (코드 변경 후 필수) ##########"
  bash $R/scripts/regression_mincirc.sh || { echo "[E104] 회귀 실패 — 중단"; exit 1; }
fi
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12"
echo "########## 반전 시간 경과 (F400, F800) ##########"
for C in F400 F800; do
  FA=400; TT=1600; [ "$C" = "F800" ] && { FA=800; TT=2000; }
  for S in 5 6 7; do for T in 300 301 302 303; do
    tag="$C w$S t$T"
    if grep -qF "$tag: => MINCIRC" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    f="$RAW/${C}_w${S}_t${T}.log"; z="$TR/${C}_w${S}_t${T}.npz"
    printf "  %s: " "$tag"
    timeout 7200 python minimal_circuit.py --mode reversal --seed "$S" --trial-seed "$T" --trials "$TT" --flip-at "$FA" --w-max 2 --trace-file "$z" $BASE > "$f" 2>&1
    rc=$?
    if grep -qE "^=> MINCIRC" "$f"; then
      python $R/scripts/analyze_e104.py "$z" > "$RAW/${C}_w${S}_t${T}_summary.log" 2>&1
      grep -E "^=> MINCIRC" "$f"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done; done
done
echo "[E104] 전체 루프 종료"
