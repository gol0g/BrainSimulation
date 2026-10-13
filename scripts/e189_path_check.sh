#!/bin/bash
# E189 경로 검사(조건 2) — 기준 logs/E189/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15.
# 새 코드: 사례 세계 C·D(rawcat), 과제 --rc-phase2-after(catrule 전용, '[둘째 범주 쌍]' 줄 2개), 평가 --rc-eval-cats.
# (a) 둘째 쌍 없음 회귀: E184 경로 검사 (b) 학습(catrule 2 × 20)·평가 learn(기본 A·B) → A_train +0.0046 A_new +0.0349 B_train +0.0659 B_new +0.0468 재현.
# (b) 둘째 쌍 20: catrule 2 × 20 + --rc-phase2-after 20 → '[둘째 범주 쌍] 시행 20 부터 …'·'둘째 쌍 시행 20 (시행 20 부터, 전체 40)', 범주 수 E184 와 같음(23/17);
#     평가 learn --rc-eval-cats A,B,C,D → RCEVAL 8줄.
# (c) 둘째 쌍 1: catrule 1 × 5 + --rc-phase2-after 1 → 점검 줄 시행 0 = A·B, 시행 1·2 = C·D(정답 C 같은 쪽·D 교차), 행동 창 끝 특징 발화 > 0·시행 끝 0.
# (d) (b) 가중치를 기본 범주(A·B)로 평가 → (b) 평가의 A·B 4값과 같음(평가 자리 불변).
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
for f_ in "$KW15" "$SN15"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E189 경로 검사] 종료"; exit 1; }; done
echo "[E189 경로 검사] 시작 $(date '+%F %T')"
python rawcat.py | tail -n 3
for V in nop2 p2x20; do
  [ "$V" = "p2x20" ] && XP="--rc-phase2-after 20" && XC="--rc-eval-cats A,B,C,D" || { XP=""; XC=""; }
  f="$OUT/${V}_train_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --rc-task catrule $XP --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
    --save-weights $WD/w_${V}_b15.npz > "$f" 2>&1; rc=$?
  echo "[($V) 학습 rc=$rc] 둘째 쌍 줄 $(grep -c '^\[둘째 범주 쌍\]' "$f") $(grep '^\[둘째 범주 쌍\]' "$f" | tr '\n' ' ')| $(grep '^\[범주 과제\] 시행' "$f") | 학습 줄 $(grep -c '^\[학습\]' "$f")"
  g="$OUT/${V}_ev_learn_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_${V}_b15.npz --decomp-mode all --rc-eval $XC --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "[($V) 평가 learn rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep '^=> DECOMP' "$g" | grep -oE '[ABCD]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"
done
f="$OUT/p2x1_train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --rc-task catrule --rc-phase2-after 1 --episodes 1 --steps 5 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_p2x1_b15.npz > "$f" 2>&1; rc=$?
echo "[(p2x1) 학습 rc=$rc] $(grep '^\[둘째 범주 쌍\]' "$f" | tr '\n' ' ')"
grep '^\[범주 과제 점검\]' "$f" | sed 's/^/    /'
g="$OUT/p2x20_ev_learnAB_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_p2x20_b15.npz --decomp-mode all --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(p2x20 A·B) 평가 learn rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep '^=> DECOMP' "$g" | grep -oE '[ABCD]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"
echo "  기대: rawcat 자체 시험 통과; (nop2) 둘째 쌍 줄 0, 범주 B 23 · A 17, 평가 A_train=+0.0046 A_new=+0.0349 B_train=+0.0659 B_new=+0.0468(E184 경로 검사 (b));"
echo "        (p2x20) 둘째 쌍 줄 2('시행 20 부터'·'둘째 쌍 시행 20 (시행 20 부터, 전체 40)'), 범주 23/17, RCEVAL 8; (p2x1) 점검 시행 0 = A·B, 1·2 = C(같은 쪽)·D(교차), 특징 발화 > 0·시행 끝 0;"
echo "        (p2x20 A·B) RCEVAL 4, A·B 4값 = (p2x20) 평가의 A·B 4값"
echo "[E189 경로 검사] 종료 $(date '+%F %T')"
