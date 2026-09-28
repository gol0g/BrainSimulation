#!/bin/bash
# E123: 미학습 사례 일반화의 용량-반응 — learn 훈련 50·100·200시행 × 배선 10~17 × 난수열 600·601 (400은 E122 재사용). 재개 가능(P13).
# 요약 줄: "  learn n50 w10 t600: => EXGEN ... | ties ... || dw_l=<|Δg| 평균> dw_r=<...>"  (judge_e123.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E123.log"
RAW="$R/research/experiments/logs/E123"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e123_main_run && cd /root/e123_main_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
# E122 K50 명령에서 --trials 만 뺀 것(아래에서 지정)
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
EXO="--exemplars 8 --distort-train 0.2 --distort-test 0.1,0.2,0.3,0.4 --n-test-ex 20"
for N in 50 100 200; do for S in 10 11 12 13 14 15 16 17; do for T in 600 601; do
  tag="learn n$N w$S t$T"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/learn_n${N}_w${S}_t$T.log"; printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode learn --seed $S --trial-seed $T --trials $N $K50 $EXO > "$f" 2>&1; rc=$?
  if grep -q "^=> EXGEN" "$f"; then
    dl=$(grep -E '^  kc_out_l:' "$f" | sed -E 's/.*\|Δ\|평균 ([0-9.]+).*/\1/'); dr=$(grep -E '^  kc_out_r:' "$f" | sed -E 's/.*\|Δ\|평균 ([0-9.]+).*/\1/')
    echo "$(grep '^=> EXGEN' "$f") || dw_l=$dl dw_r=$dr"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done; done; done
echo "[E123] 전체 루프 종료"
