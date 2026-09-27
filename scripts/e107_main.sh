#!/bin/bash
# E107: 간섭 하 유지 — K50, 배선 5~8 × 400~407, 2단계 same/cross + frz + 회귀(K38·K50).
# 재개 가능(P13). 144런.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E107.log"
RAW="$R/research/experiments/logs/E107"
RW=/root/minc_run/rewards_e107
mkdir -p "$RAW" "$RW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400 --act-drive 18.0 --tau-e 12 --w-max 2"
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
echo "########## 회귀 (코드 변경 — K38·K50) ##########"
if ! grep -q "K38 회귀 통과" "$LOG" 2>/dev/null; then bash $R/scripts/regression_mincirc.sh && echo "K38 회귀 통과" || { echo "[E107] K38 회귀 실패 — 중단"; exit 1; }; fi
if ! grep -q "K50 회귀 통과" "$LOG" 2>/dev/null; then bash $R/scripts/regression_mincirc_k50.sh && echo "K50 회귀 통과" || { echo "[E107] K50 회귀 실패 — 중단"; exit 1; }; fi
cd /root/minc_run && cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py .
echo "########## 2단계 same / cross ##########"
for RL in same cross; do for S in 5 6 7 8; do for T in 400 401 402 403 404 405 406 407; do
  run_one "$RL w$S t$T" learn "$S" "$T" --n-stim 4 --phase2-trials 400 --phase2-rule "$RL"; done; done; done
echo "########## frz (C/D 선천) ##########"
for S in 5 6 7 8; do for T in 400 401; do run_one "frz w$S t$T" frozen "$S" "$T" --n-stim 4 --phase2-trials 400; done; done
echo "[E107] 전체 루프 종료 — 원본 로그: $RAW"
