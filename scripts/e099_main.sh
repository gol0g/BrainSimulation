#!/bin/bash
# E099: K38 최종 설정의 미사용 표본 일반화 (배선 5~9, 난수열 200~207) + 대조 + 배선 1·2 파괴 점검.
# 재개 가능(P13). 144런.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E099.log"
RAW="$R/research/experiments/logs/E099"
RW=/root/minc_run/rewards_e099
mkdir -p "$RAW" "$RW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400 --act-drive 18.0 --tau-e 12"
done_already () { grep -qF "$1: => MINCIRC" "$LOG" 2>/dev/null; }   # 결과 줄까지 있어야 완료(끊긴 런 재실행)
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
echo "########## frozen (선천 기준선) ##########"
for S in 5 6 7 8 9 1 2; do for T in 200 201 202 203; do run_one "frozen w$S t$T" frozen "$S" "$T"; done; done
echo "########## learn (보상 계열 저장) ##########"
for S in 5 6 7 8 9; do for T in 200 201 202 203 204 205 206 207; do
  run_one "learn w$S t$T" learn "$S" "$T" --dump-rewards "$RW/w${S}t${T}.txt"; done; done
echo "########## shuffled ##########"
for S in 5 6 7 8 9; do for T in 200 201 202 203 204 205 206 207; do
  run_one "shuf w$S t$T" shuffled "$S" "$T" --reward-file "$RW/w${S}t${T}.txt"; done; done
echo "########## noreward ##########"
for S in 5 6 7 8 9; do for T in 200 201 202 203; do run_one "norew w$S t$T" noreward "$S" "$T"; done; done
echo "########## learn 배선 1·2 (파괴 점검) ##########"
for S in 1 2; do for T in 200 201 202 203 204 205 206 207; do run_one "learn w$S t$T" learn "$S" "$T"; done; done
echo "[E099] 전체 루프 종료 — 원본 로그: $RAW"
