#!/bin/bash
# E181 경로 검사(조건 2) — 기준 logs/E181/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15, 이득 맞춘 형성 표현(E153 경로 검사 가중치 × 0.70), E158 경로 검사 200시행 가중치 — E159 경로 검사와 같은 칸.
# 새·변경 코드 없음(E178 의 motor_tonic·E171 의 eval-diag 그대로) — E159 평가 경로에 --motor-tonic -4000 --eval-diag 를 더할 때의 도달 확인.
# (a) 기저값 0: none·R25 → +0.3877, W·R25 → +0.1226(E159·E158 경로 검사) 정확 재현, 동작점 줄 0.
# (b) 기저값 −4000 + 진단: none·R25, W·R25, W·R0, none·R0 — '[motor 동작점]' 줄 = −4000, 진단 줄 좌·우, motor 비포화, pushed(W 8·none 0), 적재·배율 줄. 변조폭은 기록만.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E181/pathcheck"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e181_run && cd /root/e181_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --rw-apm-scale 0"
KT="--kc-type-weights $R/research/experiments/traces/E153/pathcheck/kctype_b15_eta01.npz --kc-type-scale 0.70"
WF="$R/research/experiments/traces/E158/pathcheck/w_F200_b15.npz"
LD='^\[E153 종류 입력 적재\].*검증 일치'; SC='^\[E157 종류 입력 배율\].*검증 일치'; TN='^\s*\[motor 동작점\]'
for C in none_R25 W_R25; do
  case $C in none_*) MODE=none ;; W_*) MODE=all ;; esac
  f="$OUT/a_${C}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w 25 --steps 100 --brain-seed 15 $KT --decomp-weights $WF --decomp-mode $MODE > "$f" 2>&1; rc=$?
  echo "[(a) X0 $C rc=$rc] 동작점 줄 $(grep -cE "$TN" "$f") 적재 $(grep -c "$LD" "$f") 배율 $(grep -c "$SC" "$f") | $(grep '^=> DECOMP' "$f" | grep -oE 'mod=[-+0-9.]+|pushed=[0-9]+' | tr '\n' ' ')"
done
echo "  기대 (a): none_R25 mod=+0.3877, W_R25 mod=+0.1226(E158·E159 경로 검사), 동작점 줄 0"
for C in none_R25 W_R25 W_R0 none_R0; do
  case $C in none_*) MODE=none ;; W_*) MODE=all ;; esac
  case $C in *_R0) RW=0 ;; *_R25) RW=25 ;; esac
  f="$OUT/b_${C}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w $RW --steps 100 --brain-seed 15 $KT --decomp-weights $WF --decomp-mode $MODE --motor-tonic -4000 --eval-diag > "$f" 2>&1; rc=$?
  echo "[(b) X-4000 $C rc=$rc] 동작점 줄 $(grep -cE "$TN" "$f") 적재 $(grep -c "$LD" "$f") 배율 $(grep -c "$SC" "$f") | $(grep '^=> DECOMP' "$f" | grep -oE 'mod=[-+0-9.]+|pushed=[0-9]+' | tr '\n' ' ') | $(grep '^\[E171 평가 진단\]' "$f" | grep -oE 'side=[a-z]+ n=[0-9]+ motor L/R [0-9./]+' | tr '\n' ' ')"
done
echo "  기대 (b): 동작점 줄 ≥ 1(= −4000), 적재 ≥ 2·배율 ≥ 2, 진단 줄 좌·우, motor 4개 모두 ≥ 0.66 아님, pushed W 8·none 0"
echo "[E181 경로 검사] 종료 $(date '+%F %T')"
