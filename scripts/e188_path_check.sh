#!/bin/bash
# E188 경로 검사(조건 2) — 기준 logs/E188/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15.
# 새 코드: 과제 --rc-reverse-after(catrule 전용, '[범주 반전]' 줄 2개).
# (a) 반전 없음 회귀: E184 경로 검사 (b) 학습(catrule 2 × 20)·평가 learn → A_train +0.0046 A_new +0.0349 B_train +0.0659 B_new +0.0468 재현.
# (b) 반전 20: catrule 2 × 20 + --rc-reverse-after 20 → '[범주 반전] 시행 20 부터 …'·'반전 시행 20 (시행 20 부터, 전체 40)', 범주 수 E184 와 같음(23/17).
# (c) (b) 가중치 평가 learn → RCEVAL 4줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E188/pathcheck"; WD="$R/research/experiments/traces/E188/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e188_run && cd /root/e188_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
for f_ in "$KW15" "$SN15"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E188 경로 검사] 종료"; exit 1; }; done
echo "[E188 경로 검사] 시작 $(date '+%F %T')"
for V in norev rev20; do
  [ "$V" = "rev20" ] && XR="--rc-reverse-after 20" || XR=""
  f="$OUT/${V}_train_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --rc-task catrule $XR --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
    --save-weights $WD/w_${V}_b15.npz > "$f" 2>&1; rc=$?
  echo "[($V) 학습 rc=$rc] 반전 줄 $(grep -c '^\[범주 반전\]' "$f") $(grep '^\[범주 반전\]' "$f" | tr '\n' ' ')| $(grep '^\[범주 과제\] 시행' "$f") | 학습 줄 $(grep -c '^\[학습\]' "$f")"
  g="$OUT/${V}_ev_learn_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_${V}_b15.npz --decomp-mode all --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "[($V) 평가 learn rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"
done
echo "  기대: (norev) 반전 줄 0, 범주 B 23 · A 17, 평가 A_train=+0.0046 A_new=+0.0349 B_train=+0.0659 B_new=+0.0468(E184 경로 검사 (b)); (rev20) 반전 줄 2('시행 20 부터'·'반전 시행 20 (시행 20 부터, 전체 40)'), 범주 23/17, RCEVAL 4"
echo "[E188 경로 검사] 종료 $(date '+%F %T')"
