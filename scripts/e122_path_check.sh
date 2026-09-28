#!/bin/bash
# E122 경로 검사: 최소 회로 사례(exemplar) 모드. 회귀는 E105 reg w0 t100(K50), 사례 검사는 배선 18(본 표본 10~17 밖).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E122"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e122_run && cd /root/e122_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --trials 400 --act-drive 18.0 --tau-e 12 --w-max 2"
EXO="--exemplars 8 --distort-train 0.2 --distort-test 0.1,0.2,0.3,0.4 --n-test-ex 20"
run() {  # 이름, 기대 패턴, 인자...
  local n="$1" pat="$2"; shift 2; local f="$OUT/path_$n.log"
  timeout 3600 python minimal_circuit.py "$@" > "$f" 2>&1; local rc=$?
  if grep -qE "$pat" "$f"; then echo "  $n: $(grep -E '^=> |^\[사례\] 전체' "$f" | tr '\n' ' ')"; else echo "  $n: [실패 rc=$rc]"; tail -3 "$f"; fi
}
echo "[Q1 회귀 — 사례 모드 꺼짐, E105 reg w0 t100 기대: first=60.5 reward=60.5 eval=100.0]"
run regress_w0_t100 '^=> MINCIRC' --mode learn --seed 0 --trial-seed 100 $K50
echo "[Q2 KC 표현 — 배선 18]"
run probe_w18 '^=> EXKC' --mode learn --seed 18 $K50 $EXO --probe-ex
echo "[Q3 사례 학습 — 배선 18 × 난수열 600·601]"
for T in 600 601; do
  run learn_w18_t$T '^=> EXGEN' --mode learn --seed 18 --trial-seed $T $K50 $EXO
  run frozen_w18_t$T '^=> EXGEN' --mode frozen --seed 18 --trial-seed $T $K50 $EXO
done
echo "[E122 경로 검사] 종료"
