#!/bin/bash
# E156 탐색 진단(판정 밖, 보정 실패 뒤): 형성 표현의 반사 발현 감소(뇌 15 반사 25 [사전] +0.4397 → +0.1618)가 KC→motor 경로를 거치는가.
# E109 경로(--calib-kc-motor-set): 학습 없이 KC→motor 4집단을 zero(전부 0)·rev(교차 w_max, 같은쪽 0)·ali(같은쪽 w_max, 교차 0)로 넣고 변조폭.
# 기본·형성(E153 경로 검사 kctype_b15_eta01.npz) × zero·rev·ali, 반사 25, 표본 밖 뇌 15. 다음 설계(A/B/C/D)의 입력이다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E156/diag_kcmotor"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e156_run && cd /root/e156_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
KT="$R/research/experiments/traces/E153/pathcheck/kctype_b15_eta01.npz"
for REP in default formed; do
  for SET in zero rev ali; do
    f="$OUT/${REP}_${SET}_b15.log"
    if [ "$REP" = "formed" ]; then XA="--kc-type-weights $KT"; else XA=""; fi
    timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w 25 --episodes 0 --steps 100 --brain-seed 15 --rw-apm-scale 0 $XA --calib-kc-motor-set $SET > "$f" 2>&1; rc=$?
    echo "[$REP $SET rc=$rc] 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$f") | $(grep '^=> CALIBKM1' "$f" | cut -c1-120)"
  done
done
# KC 이득: kcrate 분해(E138 경로 — 새 뇌, KC→motor 초기값, good 좌·우 교대 100회 각 3처리 스텝)의 '제시 스파이크'(쪽별 KC 발화 총수). 가중치 파일은 연결 크기 확인용(분해 sub 비움).
TD="$R/research/experiments/traces/E156/diag"; mkdir -p "$TD"
for REP in default formed; do
  f="$OUT/${REP}_kcrate_b15.log"
  if [ "$REP" = "formed" ]; then XA="--kc-type-weights $KT"; else XA=""; fi
  timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w 25 --episodes 0 --steps 100 --brain-seed 15 --rw-apm-scale 0 $XA --trials 100 \
    --decomp-weights $R/research/experiments/traces/E156/pathcheck/w_F200_b15.npz --decomp-mode kcrate --kc-rate-file $TD/kcrate_${REP}_b15.npz > "$f" 2>&1; rc=$?
  echo "[$REP kcrate rc=$rc] 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$f") | $(grep '^=> KCRATE' "$f" | grep -oE 'kc_[lr] \| 좌선택 [0-9]+ 우선택 [0-9]+|제시 스파이크 [0-9]+ 기준선\(제시창\) 평균 [0-9.]+' | tr '\n' ' ')"
done
echo "[E156 KC→motor 진단] 종료"
