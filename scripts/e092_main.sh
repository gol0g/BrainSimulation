#!/bin/bash
# E092: 파괴가 성공보다 빈번한 이유 — 학습이 보상 방향 이동인가 무작위 재추첨인가.
# shuffled = learn 이 실제로 받은 보상 계열을 **시행에 무작위 재배정**(총량 동일, 수반성만 제거).
# 재개 가능(P13). 조건당 난수열 8회(P18, 보조 조건 없음).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E092.log"
RAW="$R/research/experiments/logs/E092"
RW=/root/minc_run/rewards
mkdir -p "$RAW" "$RW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400"
done_already () { grep -qF "$1: " "$LOG" 2>/dev/null; }
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
for S in 0 1 2 3 4; do run_one "frozen w$S" frozen "$S" 100; done
echo "########## learn (보상 계열 저장) ##########"
for S in 0 1 2 3 4; do
  for T in 100 101 102 103 104 105 106 107; do
    run_one "learn w$S t$T" learn "$S" "$T" --dump-rewards "$RW/w${S}t${T}.txt"
  done
done
echo "########## shuffled (같은 보상 계열, 수반성만 제거) ##########"
for S in 0 1 2 3 4; do
  for T in 100 101 102 103 104 105 106 107; do
    run_one "shuf w$S t$T" shuffled "$S" "$T" --reward-file "$RW/w${S}t${T}.txt"
  done
done
echo "[E092] 전체 루프 종료 — 원본 로그: $RAW"
