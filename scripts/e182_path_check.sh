#!/bin/bash
# E182 경로 검사(조건 2) — 기준 logs/E182/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15, E179 경로 검사 가중치(w_bc2_b15 — bicond2 4ep, 기저값 −4000).
# 새 코드: evaluate 시행 맞춤 맥락 모드(--eval-ctx-trial·--eval-ctx-gap, 기본 꺼짐 = 이전 경로).
# (a) 시행 맞춤 끔: 학습 A·B 평가 → E179 경로 검사 값(A −0.0629, B +0.0111) 정확 재현, '[맥락 평가 시행]' 줄 0.
# (b) 시행 맞춤 켬: 학습 A·B·무학습 A·B + 진단 → '[맥락 평가 시행] 시행 100 · 맥락 켬 100회·끔 100회 … 간격 10처리' 줄 1, A·B 표지, 동작점 줄, 진단 좌·우·비포화,
#     진단 KC 발화율(조향 중)이 맥락 켬 수준(시행 맞춤에서도 조향 때 맥락이 켜졌는가 — (a) 학습 A 진단과 비교). 변조폭은 기록만.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E182/pathcheck"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e182_run && cd /root/e182_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CTX="--conn-snapshot $SN15 --ctx-n 200 --ctx-w 4 --ctx-p 0.10 --ctx2-n 200 --ctx2-w 4 --ctx2-p 0.10 --motor-tonic -4000"
TN='^\s*\[motor 동작점\]'
for f_ in "$KW15" "$SN15" "$WB"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E182 경로 검사] 종료"; exit 1; }; done
for C in A B; do
  [ "$C" = "B" ] && XE="--eval-ctx" || XE="--eval-ctx2"
  g="$OUT/a_learn_${C}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX --decomp-weights $WB --decomp-mode all $XE --eval-diag > "$g" 2>&1
  echo "[(a) 시행 맞춤 끔 learn $C rc=$?] 시행 줄 $(grep -c '^\[맥락 평가 시행\]' "$g") | $(grep '^=> DECOMP' "$g" | grep -oE 'mod=[-+0-9.]+') | $(grep '^\[E171 평가 진단\]' "$g" | grep -oE 'side=[a-z]+ n=[0-9]+ motor L/R [0-9./]+ KC L/R [0-9./]+' | tr '\n' ' ')"
done
echo "  기대 (a): learn A −0.0629, learn B +0.0111(E179 경로 검사 (d)), 시행 줄 0"
for WC in "learn A" "learn B" "none A" "none B"; do
  set -- $WC; M=$1; C=$2
  [ "$M" = "learn" ] && XM="all" || XM="none"
  [ "$C" = "B" ] && XE="--eval-ctx" || XE="--eval-ctx2"
  g="$OUT/b_${M}_${C}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX --decomp-weights $WB --decomp-mode $XM $XE --eval-ctx-trial --eval-diag > "$g" 2>&1
  echo "[(b) 시행 맞춤 $M $C rc=$?] $(grep '^\[맥락 평가 시행\]' "$g" | cut -c1-80) | 표지 B $(grep -c '^\[맥락 평가\] ctx=켬' "$g") A $(grep -c '^\[맥락 평가 A\]' "$g") 동작점 $(grep -cE "$TN" "$g") | $(grep '^=> DECOMP' "$g" | grep -oE 'mod=[-+0-9.]+|pushed=[0-9]+' | tr '\n' ' ') | $(grep '^\[E171 평가 진단\]' "$g" | grep -oE 'side=[a-z]+ n=[0-9]+ motor L/R [0-9./]+ KC L/R [0-9./]+' | tr '\n' ' ')"
done
echo "  기대 (b): 시행 줄 '시행 100 · 맥락 켬 100회·끔 100회 … 간격 10처리', A·B 표지 맞음, 동작점 ≥ 1, 진단 좌·우·비포화, 조향 중 KC 발화율이 (a) 맥락 연속 평가와 비슷한 수준"
echo "[E182 경로 검사] 종료 $(date '+%F %T')"
