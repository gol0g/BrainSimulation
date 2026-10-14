#!/bin/bash
# E191 경로 검사(조건 2) — 기준 logs/E191/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15. 코드 변경 없음 — 새 조합(반전 + 형성 가중치)만.
# (a) E190 경로 검사 형성 가중치(feat_ABCD_b15 — 네 범주 2 주기)로 반전 짧은 학습(catrule 2 × 20 + --rc-reverse-after 20) → 반전 줄 2(20/20/40)·적재 줄·둘째 쌍 줄 0·범주 23/17.
# (b) (a) 가중치 평가 learn A·B·C·D → RCEVAL 8·적재·pushed 8.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E191/pathcheck"; WD="$R/research/experiments/traces/E191/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
FF="$R/research/experiments/traces/E190/pathcheck/feat_ABCD_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e191_run && cd /root/e191_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
LDF='^\[E184 특징 가중치 적재\].*검증 일치'
for f_ in "$KW15" "$SN15" "$FF"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E191 경로 검사] 종료"; exit 1; }; done
echo "[E191 경로 검사] 시작 $(date '+%F %T')"
f="$OUT/a_train_V_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $FF --rc-task catrule --rc-reverse-after 20 --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_V_b15.npz > "$f" 2>&1; rc=$?
echo "[(a) V 학습 rc=$rc] 적재 $(grep -cE "$LDF" "$f") | 둘째 쌍 줄 $(grep -c '^\[둘째 범주 쌍\]' "$f") | $(grep '^\[범주 반전\]' "$f" | tr '\n' ' ')| $(grep '^\[범주 과제\] 시행' "$f") | 점검 $(grep '^\[범주 과제 점검\]' "$f" | grep -oE '정답 [^ ]+ [^ ]* ?— 행동 창 끝 특징 발화 [0-9]+ · 시행 끝 [0-9]+' | tr '\n' ',') | 학습 줄 $(grep -c '^\[학습\]' "$f")"
g="$OUT/b_ev_V_learn_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $FF --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_V_b15.npz --decomp-mode all --rc-eval --rc-eval-cats A,B,C,D --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(b) V 평가 learn rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | 적재 $(grep -cE "$LDF" "$g") | $(grep '^=> DECOMP' "$g" | grep -oE 'pushed=[0-9]+') | $(grep '^=> DECOMP' "$g" | grep -oE '[ABCD]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"
echo "  기대: (a) 적재 ≥ 1·둘째 쌍 줄 0·반전 줄 2('시행 20 부터 정답 A → 같은 쪽 · B → 교차'·'반전 시행 20 (시행 20 부터, 전체 40)')·범주 40(23/17)·점검 > 0/0; (b) RCEVAL 8·적재 ≥ 1·pushed 8, 값이 E190 경로 검사 (c) R(반전 없음)과 다름"
echo "[E191 경로 검사] 종료 $(date '+%F %T')"
