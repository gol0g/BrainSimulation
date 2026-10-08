#!/bin/bash
# E156 탐색 진단(판정 밖, 보정 실패 뒤): 기본 표현(형성 가중치 없음)도 반사 가중치를 올리면 [사전] 이 줄어드는가 — 표본 밖 뇌 15, --episodes 0.
# 형성 표현 보정(logs/E156/calib.out: W25 +0.1618 → W300 +0.1206 단조 감소)과 같은 격자 일부. 다음 설계(A/B/C/D)의 입력이다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E156/diag_default"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e156_run && cd /root/e156_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
for W in 0 25 60 135 300; do
  f="$OUT/W${W}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w $W --episodes 0 --steps 100 --brain-seed 15 --rw-apm-scale 0 > "$f" 2>&1; rc=$?
  echo "[기본 W$W rc=$rc] $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') | 반사 $(grep '^\[반사가중치\] good_food_to_motor' "$f" | grep -oE 'w_mean \S+' | tr '\n' ' ')"
done
echo "[E156 기본 W 진단] 종료"
