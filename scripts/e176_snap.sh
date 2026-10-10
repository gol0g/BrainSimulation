#!/bin/bash
# E176 초기 연결 스냅숏 저장 — 기존 구조(맥락 집단 없음)로 뇌를 만든 직후 모든 희소 집단 연결을 npz 로. 뇌 10~15(경로 검사·보정 15, 본실험 10~14).
# 인자는 E162 학습 인자와 같다(구조를 정하는 옵션이 같아야 이름·연결이 맞는다). --episodes 0, 저장 뒤 종료.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
SNAP="$R/research/experiments/traces/E176/snap"; OUT="$R/research/experiments/logs/E176/snap"; mkdir -p "$SNAP" "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e176_run && cd /root/e176_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
for B in 15 10 11 12 13 14; do
  f="$OUT/snap_b$B.log"
  [ -s "$SNAP/conn_b$B.npz" ] && { echo "[b$B] 이미 있음"; continue; }
  timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 0 --brain-seed $B --conn-snapshot-save $SNAP/conn_b$B.npz > "$f" 2>&1
  echo "[b$B rc=$?] $(grep '^  \[E176 초기 연결 스냅숏\] 저장' "$f" | cut -c1-200) | 파일 $(ls -la $SNAP/conn_b$B.npz 2>/dev/null | awk '{print $5}')B"
done
echo "[E176 스냅숏] 종료"
