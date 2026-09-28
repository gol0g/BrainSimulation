#!/bin/bash
# E124 경로 검사: 같음/다름 관계 과제(--samediff). 회귀 2종 + 배선 18(본 표본 밖) 학습/무학습.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E124"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e124_run && cd /root/e124_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400 --act-drive 18.0 --tau-e 12 --w-max 2"
EXO="--exemplars 8 --distort-train 0.2 --distort-test 0.1,0.2,0.3,0.4 --n-test-ex 20"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
run() {
  local n="$1" pat="$2"; shift 2; local f="$OUT/path_$n.log"
  timeout 3600 python minimal_circuit.py "$@" > "$f" 2>&1; local rc=$?
  if grep -qE "$pat" "$f"; then echo "  $n: $(grep -E '^=> |^\[관계\]' "$f" | tr '\n' ' ')"; else echo "  $n: [실패 rc=$rc]"; tail -3 "$f"; fi
}
echo "[Q1 K50 회귀 — 기대 E105 reg w0 t100: first=60.5 reward=60.5 eval=100.0]"
run regress_w0_t100 '^=> MINCIRC' --mode learn --seed 0 --trial-seed 100 $K50
echo "[Q2 사례 모드 회귀 — 기대 E122 경로 검사 learn w18 t600: proto=100.0 train=89.0 d0.10=88.0 d0.20=94.0 d0.30=80.0 d0.40=70.0]"
run regress_ex_w18_t600 '^=> EXGEN' --mode learn --seed 18 --trial-seed 600 $K50 $EXO
echo "[Q3 관계 과제 — 배선 18]"
for T in 600 601; do run sd_learn_w18_t$T '^=> SDGEN' --mode learn --seed 18 --trial-seed $T $K50 $SDO; done
run sd_frozen_w18_t600 '^=> SDGEN' --mode frozen --seed 18 --trial-seed 600 $K50 $SDO
run sd_learn800_w18_t600 '^=> SDGEN' --mode learn --seed 18 --trial-seed 600 $K50 $SDO --trials 800
echo "[E124 경로 검사] 종료"
