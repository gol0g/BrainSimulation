#!/bin/bash
# E132: 경험 상관 인과 효과 확증 — E130 설정 그대로, 새 배선 30~45, frozen 두 난수열. 재개 가능(P13).
# 요약 줄: "  dv corr learn w10 t600: => [KC발달] ... || => SDLAB ..."  (judge_e132.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E132.log"
RAW="$R/research/experiments/logs/E132"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e132_main_run && cd /root/e132_main_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 800 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3 --sd-diff cyclic"
DEV="--kc-wiring developed --mismatch-w 8 --dev-theta 0.15 --dev-rounds 1000 --dev-exposures 200 --dev-items 20"
one() {  # 환경 모드 배선 난수열
  local tag="dv $1 $2 w$3 t$4"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/dv_$1_$2_w$3_t$4.log"; printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode $2 --seed $3 --trial-seed $4 $K50 $SDO $DEV --dev-env $1 > "$f" 2>&1; local rc=$?
  if grep -q "^\[KC발달\]" "$f" && grep -q "^=> SDLAB" "$f"; then echo "=> $(grep '^\[KC발달\]' "$f") || $(grep '^=> SDLAB' "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
for S in 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45; do
  one corr learn $S 600; one corr learn $S 601; one indep learn $S 600; one indep learn $S 601; one corr frozen $S 600; one corr frozen $S 601
done
echo "[E132] 전체 루프 종료"
