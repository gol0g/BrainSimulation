#!/bin/bash
# E124: 같음/다름 관계 과제(800시행). learn 배선 10~17 × 난수열 600·601, frozen 배선 10~17. 재개 가능(P13).
# 요약 줄 형식은 judge_e124.py 의 TR 과 맞춘다: "  learn w10 t600: => SDGEN ..."
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E124.log"
RAW="$R/research/experiments/logs/E124"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e124_main_run && cd /root/e124_main_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 800 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
one() {  # 모드 배선 난수열
  local tag="$1 w$2 t$3"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/$1_w$2_t$3.log"; printf "  %s: " "$tag"
  timeout 3600 python minimal_circuit.py --mode $1 --seed $2 --trial-seed $3 $K50 $SDO > "$f" 2>&1; local rc=$?
  if grep -q "^=> SDGEN" "$f"; then echo "$(grep '^=> SDGEN' "$f" | sed 's/^=> /=> /')"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
}
for S in 10 11 12 13 14 15 16 17; do
  one learn $S 600; one learn $S 601; one frozen $S 600
done
echo "[E124] 전체 루프 종료"
