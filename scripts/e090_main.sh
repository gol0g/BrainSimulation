#!/bin/bash
# E090: 학습률이 복권성(성공 8% / 파괴 31%)의 원인인가. 재개 가능(P13).
# 지표는 **성공률·파괴율**이다(K29). 평균 정답률이 아니다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E090.log"
RAW="$R/research/experiments/logs/E090"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
BASE="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6"
done_already () { grep -qF "$1: " "$LOG" 2>/dev/null; }
run_one () {   # tag mode seed trialseed eta trials
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode "$2" --seed "$3" --trial-seed "$4" \
    --eta "$5" --trials "$6" $BASE > "$f" 2>&1
  local rc=$?
  grep -E "^=> MINCIRC" "$f" || { echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; }
}
for S in 0 1 2 3 4; do run_one "frozen w$S" frozen "$S" 100 0.02 400; done
for ETA in 0.02 0.005 0.001; do
  echo "########## eta=$ETA ##########"
  for S in 0 1 2 3 4; do
    for T in 100 101 102 103 104 105 106 107; do
      run_one "eta$ETA w$S t$T" learn "$S" "$T" "$ETA" 400
    done
  done
done
echo "########## 보조: eta=0.001, 시행 4배 (한계 1번) ##########"
for S in 0 3 4; do
  for T in 100 101 102 103; do
    run_one "long w$S t$T" learn "$S" "$T" 0.001 1600
  done
done
echo "[E090] 전체 루프 종료 — 원본 로그: $RAW"
