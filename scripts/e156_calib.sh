#!/bin/bash
# E156 수정 1 보정(규칙 logs/E156/criteria_fixed.txt 수정 1, 2026-10-09 00:44:40 고정) — 표본 밖 뇌 15, 형성 가중치(E153 경로 검사 kctype_b15_eta01.npz), --episodes 0.
# 격자 W ∈ {25, 40, 60, 90, 135, 200, 300} 의 [사전] → e156_wstar.py 가 +0.4397(같은 뇌 기본 반사 25) 최근접 W* 선택(±0.05 밖이면 보정 실패).
# 이식 평가도 돌려 0시행 [사후] = [사전]·반사 W→W·적재 2줄을 확인한다(평가 뇌에 W 와 형성 가중치가 실리는지 — 조건 2).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E156/calib"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e156_run && cd /root/e156_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
for W in 25 40 60 90 135 200 300; do
  f="$OUT/W${W}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w $W --episodes 0 --steps 100 --transplant-eval --brain-seed 15 --rw-apm-scale 0 \
    --kc-type-weights $R/research/experiments/traces/E153/pathcheck/kctype_b15_eta01.npz > "$f" 2>&1; rc=$?
  echo "[W$W rc=$rc] 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') | 반사 $(grep '^\[반사가중치\] good_food_to_motor' "$f" | grep -oE 'w_mean \S+' | tr '\n' ' ')"
done
cd $R && python3 scripts/e156_wstar.py
echo "[E156 보정] 종료"
