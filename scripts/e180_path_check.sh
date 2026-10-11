#!/bin/bash
# E180 경로 검사(조건 2) — 기준 logs/E180/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15, E162 경로 검사 가중치(w_AB_b15: 과제 A 200 + B 200)·형성 가중치(E160 보정 β 0.3).
# 새·변경 코드 없음(E178·E179 의 motor_tonic·eval-diag 그대로) — E170 평가 경로에 --motor-tonic 을 더할 때의 도달 확인.
# (a) 기저값 0(인자 없음): AB·none × base·bad 가 E170 경로 검사 값(−0.4783, +0.5105, +0.0228, +0.0120)을 정확히 재현.
# (b) 기저값 −4000 + --eval-diag: AB·none × base·bad·agree·conflict — '[motor 동작점]' 줄 = −4000, 진단 줄 좌·우, motor 발화율이 포화(4개 모두 ≥ 0.66) 아님,
#     '[E170 자극]' 줄 설계 구성, pushed AB 8·none 0. 변조폭은 기록만.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E180/pathcheck"; mkdir -p "$OUT"
WD="$R/research/experiments/traces/E162/pathcheck"; KW="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e180_run && cd /root/e180_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
B=15
TN='^\s*\[motor 동작점\]'
for W in AB none; do
  [ "$W" = "AB" ] && X="--decomp-weights $WD/w_AB_b$B.npz --decomp-mode all" || X="--decomp-weights $WD/w_AB_b$B.npz --decomp-mode none"
  for S in base bad; do
    f="$OUT/a_ev_b${B}_${W}_$S.log"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $X --eval-variant $S > "$f" 2>&1
    echo "[(a) X0 $W $S rc=$?] 동작점 줄 $(grep -cE "$TN" "$f") | $(grep '^=> DECOMP' "$f" | grep -oE 'mode=[a-z]+ mod=[-+0-9.]+|pushed=[0-9]+' | tr '\n' ' ')"
  done
done
echo "  기대 (a): AB base −0.4783 · AB bad +0.5105 · none base +0.0228 · none bad +0.0120(E170·E162 경로 검사), 동작점 줄 0"
for W in AB none; do
  [ "$W" = "AB" ] && X="--decomp-weights $WD/w_AB_b$B.npz --decomp-mode all" || X="--decomp-weights $WD/w_AB_b$B.npz --decomp-mode none"
  for S in base bad agree conflict; do
    f="$OUT/b_ev_b${B}_${W}_$S.log"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $X --eval-variant $S --motor-tonic -4000 --eval-diag > "$f" 2>&1
    echo "[(b) X-4000 $W $S rc=$?] 동작점 줄 $(grep -cE "$TN" "$f") | $(grep '^=> DECOMP' "$f" | grep -oE 'mode=[a-z]+ mod=[-+0-9.]+|pushed=[0-9]+' | tr '\n' ' ') | $(grep '^\[E171 평가 진단\]' "$f" | sed -E 's/^\[E171 평가 진단\] variant=[a-z]+ //' | grep -oE 'side=[a-z]+ n=[0-9]+ motor L/R [0-9./]+' | tr '\n' ' ')"
    grep '^\[E170 자극\]' "$f" | sed 's/^/    /'
  done
done
echo "  기대 (b): 동작점 줄 ≥ 1(= −4000), 진단 줄 좌·우, motor 4개 모두 ≥ 0.66 아님, 자극 줄 설계 구성, pushed AB 8·none 0"
echo "[E180 경로 검사] 종료 $(date '+%F %T')"
