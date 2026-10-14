#!/bin/bash
# E193 동작점 보정(경로 검사 뒤, 결과 전 — 기준 정정 1) — 표본 밖 뇌 15, 무학습(none) 평가 motor 평균.
# 기준: C0(w 4·p 0.10) motor 기저값 −4000 의 무학습 평가 motor 평균(맥락 X·Y, 범주 A·B × train·new, 두 쪽 L/R 평균).
# 후보: C2(w 20·p 0.02) × 기저값 −4000·−4300·−4600·−4900. 규칙(사전 고정): |m_C2(T) − m_C0| 가 가장 작은 T(같으면 −4000 에 가까운 쪽).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E193/calib"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e193_run && cd /root/e193_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
for f_ in "$KW15" "$SN15" "$WB"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E193 보정] 종료"; exit 1; }; done
echo "[E193 보정] 시작 $(date '+%F %T')"
for V in C0_4000 C2_4000 C2_4300 C2_4600 C2_4900; do
  C=${V%_*}; T=${V#*_}
  [ "$C" = "C0" ] && CW="--ctx-w 4 --ctx-p 0.10 --ctx2-w 4 --ctx2-p 0.10" || CW="--ctx-w 20 --ctx-p 0.02 --ctx2-w 20 --ctx2-p 0.02"
  for X in X Y; do
    g="$OUT/${V}_${X}_b15.log"
    timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --ctx-n 200 --ctx2-n 200 $CW --motor-tonic -$T --brain-seed 15 --kc-type-weights $KW15 \
      --decomp-weights $WB --decomp-mode none --rc-eval --rc-eval-ctx $X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
    m=$(grep '^RCEVAL ' "$g" | grep -oE 'motor L/R [0-9.]+/[0-9.]+' | awk '{split($3,a,"/"); s+=a[1]+a[2]; n+=2} END {if (n) printf "%.4f", s/n; else print "nan"}')
    echo "  ($V $X rc=$rc) RCEVAL $(grep -c '^RCEVAL ' "$g") | motor 평균 $m | 최대 $(grep '^RCEVAL ' "$g" | grep -oE 'motor L/R [0-9.]+/[0-9.]+' | tr '/ ' '\n\n' | grep -E '^[0-9.]+$' | sort -g | tail -n 1)"
  done
done
echo "[E193 보정] 종료 $(date '+%F %T')"
