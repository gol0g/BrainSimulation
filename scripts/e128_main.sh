#!/bin/bash
# E128: 교차 반쪽 결합 배선(--kc-wiring crosshalf, w4) 균형 같음/다름 24런 + 신용·발화율 분석(읽기 전용). 재개 가능(P13).
# 요약 줄: "  xh learn w10 t600: => SDCREDIT ... || => SDRATE ... || => SDLAB ..."  (judge_e128.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E128.log"
RAW="$R/research/experiments/logs/E128"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e128_main_run && cd /root/e128_main_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 800 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3 --sd-diff cyclic --sd-credit --kc-wiring crosshalf"
one() {
  local tag="xh $1 w$2 t$3"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/xh_$1_w$2_t$3.log"; printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode $1 --seed $2 --trial-seed $3 $K50 $SDO > "$f" 2>&1; local rc=$?
  if grep -q "^=> SDCREDIT" "$f" && grep -q "^=> SDRATE" "$f" && grep -q "^=> SDLAB" "$f"; then echo "$(grep '^=> SDCREDIT' "$f") || $(grep '^=> SDRATE' "$f") || $(grep '^=> SDLAB' "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
for S in 10 11 12 13 14 15 16 17; do
  one learn $S 600; one learn $S 601; one frozen $S 600
done
echo "[E128] 전체 루프 종료"
