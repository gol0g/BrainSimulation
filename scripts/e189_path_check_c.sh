#!/bin/bash
# E189 경로 검사 보충 (c') — 본 경로 검사 (c)(뇌 15 기본 시행 시드 18515)는 시행 1·2 가 둘 다 D 라 C 제시·'C 같은 쪽' 정답이 안 보였다.
# 시행 시드 18905(호스트 계산: 시행 0 B 사례 3, 1 A 사례 3 → C, 2 B 사례 5 → D)로 catrule 1 × 5 + --rc-phase2-after 1.
# 기대: 점검 시행 0 = 범주 B 사례 3 같은 쪽, 1 = 범주 C 사례 3 같은 쪽, 2 = 범주 D 사례 5 교차, 행동 창 끝 특징 발화 > 0·시행 끝 0.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E189/pathcheck"; WD="$R/research/experiments/traces/E189/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e189_run && cd /root/e189_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
echo "[E189 경로 검사 보충] 시작 $(date '+%F %T')"
f="$OUT/p2x1s_train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --rc-task catrule --rc-trial-seed 18905 --rc-phase2-after 1 --episodes 1 --steps 5 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_p2x1s_b15.npz > "$f" 2>&1; rc=$?
echo "[(p2x1s) 학습 rc=$rc] $(grep '^\[범주 과제\] catrule' "$f" | grep -oE '배정 지역 난수 [0-9]+') | $(grep '^\[둘째 범주 쌍\]' "$f" | tr '\n' ' ')"
grep '^\[범주 과제 점검\]' "$f" | sed 's/^/    /'
echo "  기대: 시행 0 범주 B 사례 3 같은 쪽 · 1 범주 C 사례 3 같은 쪽 · 2 범주 D 사례 5 교차, 특징 발화 > 0·시행 끝 0"
echo "[E189 경로 검사 보충] 종료 $(date '+%F %T')"
