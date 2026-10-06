#!/bin/bash
# E145: 집단 맞바꿈 이식 분해(학습 없음) — 뇌 10~14 × 모드 5. A = E142 F1500 가중치, B = E144 가중치. 재개 가능(P13).
# 요약 줄: "  e145 b10 A_all: => mod +0.1026"  (judge_e145.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E145.log"
RAW="$R/research/experiments/logs/E145"; mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e145_main_run && cd /root/e145_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 25 --rw-apm-scale 0"
SAME="kc_l_to_motor_l,kc_r_to_motor_r"; CROSS="kc_l_to_motor_r,kc_r_to_motor_l"; D1="food_to_d1_l,food_to_d1_r,food_to_d1_cross_lr,food_to_d1_cross_rl"
for B in 10 11 12 13 14; do
  A="$R/research/experiments/traces/E142/w_F1500_b$B.npz"; W2="$R/research/experiments/traces/E144/w_b$B.npz"
  for M in A_all B_all A_sameB A_crossB A_d1B; do
    tag="e145 b$B $M"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    case $M in
      A_all)    X="--decomp-weights $A --decomp-mode all" ;;
      B_all)    X="--decomp-weights $W2 --decomp-mode all" ;;
      A_sameB)  X="--decomp-weights $A --decomp-mode swap --decomp-swap-weights $W2 --decomp-swap-pops $SAME" ;;
      A_crossB) X="--decomp-weights $A --decomp-mode swap --decomp-swap-weights $W2 --decomp-swap-pops $CROSS" ;;
      A_d1B)    X="--decomp-weights $A --decomp-mode swap --decomp-swap-weights $W2 --decomp-swap-pops $D1" ;;
    esac
    f="$RAW/b${B}_$M.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B $X > "$f" 2>&1; rc=$?
    if grep -q "^=> DECOMP" "$f"; then
      echo "=> mod $(grep '^=> DECOMP' "$f" | sed -E 's/.*mod=([-+0-9.]+).*/\1/')"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E145] 전체 루프 종료"
