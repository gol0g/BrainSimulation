#!/bin/bash
# E122: 최소 회로(K50) 미학습 사례 범주 일반화. learn 배선 10~17 × 난수열 600·601, frozen 배선 10~17(난수열 무관). 재개 가능(P13).
# 요약 줄 형식은 judge_e122.py 의 TR 과 맞춘다: "  learn w10 t600: => EXGEN ..."
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E122.log"
RAW="$R/research/experiments/logs/E122"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e122_main_run && cd /root/e122_main_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400 --act-drive 18.0 --tau-e 12 --w-max 2"
EXO="--exemplars 8 --distort-train 0.2 --distort-test 0.1,0.2,0.3,0.4 --n-test-ex 20"
one() {  # 모드 배선 난수열
  local tag="$1 w$2 t$3"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/$1_w$2_t$3.log"; printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode $1 --seed $2 --trial-seed $3 $K50 $EXO > "$f" 2>&1; local rc=$?
  if grep -q "^=> EXGEN" "$f"; then echo "$(grep '^=> EXGEN' "$f" | sed 's/^=> /=> /')"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
for S in 10 11 12 13 14 15 16 17; do
  one learn $S 600; one learn $S 601; one frozen $S 600
done
echo "[E122] 전체 루프 종료"
