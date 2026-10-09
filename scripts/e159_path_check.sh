#!/bin/bash
# E159 경로 검사(조건 2) — 기준 고정(logs/E159/criteria_fixed.txt 14:35:05, 수정 1 14:35:46) 뒤. 표본 밖 뇌 15, 이득 맞춘 형성 표현(E153 경로 검사 가중치 × 0.70).
# [1] none·R25 → E158 경로 검사 [사전] +0.3877 재현. [2] W(E158 경로 검사 200시행 가중치)·R25 → E158 경로 검사 [사후] +0.1226 재현(분해 'all' = 학습 런 이식 평가).
# [3] W·R0, none·R0 — 반사 0 평가 경로(값은 판정 밖).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E159/pathcheck"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e159_run && cd /root/e159_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --rw-apm-scale 0"
KT="--kc-type-weights $R/research/experiments/traces/E153/pathcheck/kctype_b15_eta01.npz --kc-type-scale 0.70"
WF="$R/research/experiments/traces/E158/pathcheck/w_F200_b15.npz"
LD='^\[E153 종류 입력 적재\].*검증 일치'; SC='^\[E157 종류 입력 배율\].*검증 일치'
for C in none_R25 W_R25 W_R0 none_R0; do
  case $C in none_*) MODE=none ;; W_*) MODE=all ;; esac
  case $C in *_R0) RW=0 ;; *_R25) RW=25 ;; esac
  f="$OUT/${C}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w $RW --steps 100 --brain-seed 15 $KT --decomp-weights $WF --decomp-mode $MODE > "$f" 2>&1; rc=$?
  echo "[$C rc=$rc] 적재 $(grep -c "$LD" "$f") 배율 $(grep -c "$SC" "$f") | $(grep '^=> DECOMP' "$f" | cut -c1-110)"
done
echo "  기대: none_R25 mod=+0.3877, W_R25 mod=+0.1226 (E158 경로 검사)"
echo "[E159 경로 검사] 종료"
