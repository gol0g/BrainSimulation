#!/bin/bash
# E184 경로 검사(조건 2) — 기준 logs/E184/criteria_fixed.txt 고정(2026-10-11 22:17:28) 뒤. 표본 밖 뇌 15.
# 새 코드: 과제 --rc-task catrule(범주·사례 배정, 특징 켬/끔, 정답 규칙, 점검 줄), --rc-eval(evaluate featfn·variant food, RCEVAL), --feat-weights(forager_brain _load_feat_weights), rc_feat.
# (a) 학습 경로 회귀: E179 경로 검사 (a)(A 200, 스냅숏 없음) → 사전 +0.0228 → 사후 −0.5094, 보상 134 정확 재현.
# (b) U: catrule 짧은 학습(2 × 20 = 40시행, 스냅숏 + 특징 w 24 정적) → '[범주 과제] catrule' 1·점검 3줄(행동 창 끝 > 0·시행 끝 0)·'시행 40 중 범주 B …'·[학습];
#     평가 learn·none(--rc-eval --trials 96 --eval-diag) → RCEVAL 4줄 trials 96·feat_on 96, pushed 8/0, 적재 줄 0.
# (c) F: 짧은 형성(rawcat O, 노출 2 주기, --rc-save) → catrule 짧은 학습(--feat-weights) → 평가 learn → '[E184 특징 가중치 적재] … 검증 일치' 줄(학습·평가).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E184/pathcheck"; WD="$R/research/experiments/traces/E184/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
RT15="$R/research/experiments/traces/E138/pathcheck/rate_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e184_run && cd /root/e184_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
LDF='^\[E184 특징 가중치 적재\].*검증 일치'
for f_ in "$KW15" "$RT15" "$SN15" "$WB"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E184 경로 검사] 종료"; exit 1; }; done
echo "[E184 경로 검사] 시작 $(date '+%F %T')"
f="$OUT/a_A200_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 2 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_a_b15.npz --trace-kc-class $WD/tr_a_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
echo "[(a) 학습 회귀 rc=$rc] $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | 범주 과제 줄 $(grep -c '^\[범주 과제\]' "$f")"
echo "  기대 (a): 변조폭 +0.0228 → −0.5094, 보상 134회(E179 경로 검사 (a)), 범주 과제 줄 0"
show_train() {
  local f=$1
  echo "$(grep -c '^\[범주 과제\] catrule' "$f") catrule 줄 | $(grep '^\[범주 과제 점검\]' "$f" | grep -oE '정답 [^ ]+ [^ ]* — 행동 창 끝 특징 발화 [0-9]+ · 시행 끝 [0-9]+' | tr '\n' ' ')| $(grep '^\[범주 과제\] 시행' "$f") | 학습 줄 $(grep -c '^\[학습\]' "$f") 적재 $(grep -c "$LDF" "$f") | $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+')"
}
show_eval() {
  local g=$1
  echo "RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep '^RCEVAL ' "$g" | grep -oE 'cat=[AB] set=[a-z]+ mod=[-+0-9.]+ off=[-+0-9.]+ acc=[0-9.]+ trials=[0-9]+ feat_on=[0-9]+' | tr '\n' ' ')| $(grep '^=> DECOMP' "$g" | grep -oE 'pushed=[0-9]+') 적재 $(grep -c "$LDF" "$g") | 진단 $(grep '^RCEVAL cat=A set=train' "$g" | grep -oE 'motor L/R [0-9./]+' | tr '\n' ' ')"
}
f="$OUT/b_train_U_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --rc-task catrule --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_U_b15.npz > "$f" 2>&1; rc=$?
echo "[(b) U 학습 rc=$rc] $(show_train "$f")"
[ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
for W in learn none; do
  [ "$W" = "learn" ] && XM="all" || XM="none"
  g="$OUT/b_ev_U_${W}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_U_b15.npz --decomp-mode $XM --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "[(b) U 평가 $W rc=$rc] $(show_eval "$g")"
  [ $rc -ne 0 ] && tail -3 "$g" | sed 's/^/      /'
done
echo "  기대 (b): catrule 줄 1, 점검 3줄 행동 창 끝 > 0·시행 끝 0, '시행 40 중 범주 B …', 학습 줄 1, 적재 0; RCEVAL 4 trials=96 feat_on=96, pushed 8/0, 적재 0"
g="$OUT/c_form_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT --feat-oja --feat-oja-eta 0 --decomp-weights $WB --decomp-mode rawcat \
  --rc-expo 2 --rc-expo-eta 0.02 --rc-save $WD/feat_formed_b15.npz > "$g" 2>&1; rc=$?
echo "[(c) F 짧은 형성 rc=$rc] $(grep -c '^=> RAWCAT ' "$g") RAWCAT | 저장 $(ls -la $WD/feat_formed_b15.npz 2>/dev/null | awk '{print $5}') 바이트"
f="$OUT/c_train_F_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $WD/feat_formed_b15.npz --rc-task catrule --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_F_b15.npz > "$f" 2>&1; rc=$?
echo "[(c) F 학습 rc=$rc] $(show_train "$f")"
[ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
g="$OUT/c_ev_F_learn_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $WD/feat_formed_b15.npz --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_F_b15.npz --decomp-mode all --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(c) F 평가 learn rc=$rc] $(show_eval "$g")"
[ $rc -ne 0 ] && tail -3 "$g" | sed 's/^/      /'
echo "  기대 (c): RAWCAT 1·npz 저장, F 학습·평가에 적재 줄 ≥ 1, RCEVAL 4"
echo "[E184 경로 검사] 종료 $(date '+%F %T')"
