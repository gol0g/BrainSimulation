#!/bin/bash
# E133 경로 검사: --kc-wiring loaded(보정 발달 파일, 배선 18 corr·indep w-fix 4)로 과제 1런씩 — 불러온 연결 = 발달 결과, 출력 형식.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E133"; WD="$R/research/experiments/traces/E133"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e133_run && cd /root/e133_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3"
for ENV in corr indep; do
  f="$OUT/path_loaded_${ENV}_w18.log"
  timeout 3600 python minimal_circuit.py --mode learn --seed 18 --trial-seed 600 $K50 --trials 800 $SDO --sd-diff cyclic --mismatch-w 8 --kc-wiring loaded --kc-wiring-file "$WD/calib_w18_wf4_${ENV}.npz" > "$f" 2>&1; rc=$?
  if grep -q "^\[KC불러옴\]" "$f" && grep -q "^=> SDLAB" "$f"; then echo "  $ENV: $(grep '^\[KC불러옴\]' "$f" | sed 's#.*/##') || $(grep '^=> SDLAB' "$f" | grep -oE 'train_lbal=[0-9.]+|novel_lbal=[0-9.]+' | tr '\n' ' ')"; else echo "  $ENV: [실패 rc=$rc]"; tail -3 "$f"; fi
done
echo "[E133 경로 검사] 종료"
