#!/bin/bash
# E145 경로 검사(조건 2) — 기준 고정(logs/E145/criteria_fixed.txt 15:03:06) 뒤. 표본 밖 뇌 15 의 저장 가중치로 swap 경로를 시험한다.
# A = E142 경로 검사 F500 b15(사후 +0.1459), B = E143 경로 검사 DEC b15(사후 +0.1372).
# 확인: A_all 재현(+0.1459), B_all 재현(+0.1372), 항등 맞바꿈(A 에 A 의 같은쪽 집단) = A_all 정확 일치, 실제 맞바꿈 두 개가 돈다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E145/pathcheck"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e145_run && cd /root/e145_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 25 --rw-apm-scale 0"
A="$R/research/experiments/traces/E142/pathcheck/w_F500_b15.npz"; W2="$R/research/experiments/traces/E143/pathcheck/w_DEC_b15.npz"
SAME="kc_l_to_motor_l,kc_r_to_motor_r"; CROSS="kc_l_to_motor_r,kc_r_to_motor_l"
for M in A_all B_all A_sameA A_sameB A_crossB; do
  case $M in
    A_all)    X="--decomp-weights $A --decomp-mode all" ;;
    B_all)    X="--decomp-weights $W2 --decomp-mode all" ;;
    A_sameA)  X="--decomp-weights $A --decomp-mode swap --decomp-swap-weights $A --decomp-swap-pops $SAME" ;;
    A_sameB)  X="--decomp-weights $A --decomp-mode swap --decomp-swap-weights $W2 --decomp-swap-pops $SAME" ;;
    A_crossB) X="--decomp-weights $A --decomp-mode swap --decomp-swap-weights $W2 --decomp-swap-pops $CROSS" ;;
  esac
  f="$OUT/b15_$M.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 $X > "$f" 2>&1; rc=$?
  echo "[$M] $(grep -E '^\[E145 swap\]' "$f" | cut -c1-90) $(grep '^=> DECOMP' "$f" | cut -c1-160)"
  [ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
done
echo "[E145 경로 검사] 종료"
