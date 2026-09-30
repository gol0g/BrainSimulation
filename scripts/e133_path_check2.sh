#!/bin/bash
# E133 경로 검사 2: 배선 18 — 배선 비교기(E129 설정) 학습 1런 vs 불러온 헤브 연결의 분리(SDCOMP, 학습 없음).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E133"; WD="$R/research/experiments/traces/E133"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e133_run && cd /root/e133_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
K50="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.0 --gap-steps 600 --block 400 --eval-trials 100 --epsilon 0.6 --eta 0.001 --act-drive 18.0 --tau-e 12 --w-max 2"
SDO="--samediff --sd-items 8 --sd-train-items 4 --sd-frac 0.3 --sd-diff cyclic"
f="$OUT/path2_comparator_learn_w18.log"
timeout 3600 python minimal_circuit.py --mode learn --seed 18 --trial-seed 600 $K50 --trials 800 $SDO --kc-wiring comparator --mismatch-w 8 > "$f" 2>&1
echo "  comparator learn w18 t600: $(grep '^=> SDLAB' "$f" | grep -oE 'train_accL=[0-9.]+ train_accR=[0-9.]+ train_lbal=[0-9.]+|novel_lbal=[0-9.]+' | tr '\n' ' ')"
for ENV in corr indep; do
  f="$OUT/path2_loaded_sdcomp_${ENV}_w18.log"
  timeout 3600 python minimal_circuit.py --mode frozen --seed 18 --trial-seed 600 $K50 --trials 1 --eval-trials 20 $SDO --sd-credit --mismatch-w 8 --kc-wiring loaded --kc-wiring-file "$WD/calib_w18_wf4_${ENV}.npz" > "$f" 2>&1
  echo "  loaded $ENV SDCOMP: $(grep '^=> SDCOMP' "$f" | sed 's/^=> SDCOMP seed=[0-9]* | //') || $(grep '^=> SDRATE' "$f" | grep -oE 'active_kc_per_stim 평균 [0-9.]+')"
done
f="$OUT/path2_comparator_sdcomp_w18.log"
timeout 3600 python minimal_circuit.py --mode frozen --seed 18 --trial-seed 600 $K50 --trials 1 --eval-trials 20 $SDO --sd-credit --kc-wiring comparator --mismatch-w 8 > "$f" 2>&1
echo "  comparator SDCOMP: $(grep '^=> SDCOMP' "$f" | sed 's/^=> SDCOMP seed=[0-9]* | //') || $(grep '^=> SDRATE' "$f" | grep -oE 'active_kc_per_stim 평균 [0-9.]+')"
echo "[E133 경로 검사 2] 종료"
