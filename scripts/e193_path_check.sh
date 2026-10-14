#!/bin/bash
# E193 경로 검사(조건 2) — 기준 logs/E193/criteria_fixed.txt 고정(2026-10-14 16:34:48) 뒤. 표본 밖 뇌 15. E187 경로 검사와 같은 항목, 맥락 연결만 C2(w 20·p 0.02).
# 코드 변경 없음(E187 이후 오늘의 rawcat 형성·측정 변경이 학습·평가 경로를 안 바꿨는지 (a) 회귀로 확인).
# (a) 회귀: E184 경로 검사 (b) U 평가(w_U_b15, 기저값 0, 맥락 없음) learn·none → 값 정확(learn A_train +0.0046 A_new +0.0349 B_train +0.0659 B_new +0.0468;
#     none +0.0134 +0.0432 +0.0457 +0.0353).
# (b) catctx 짧은 학습(2 × 20 = 40시행, 스냅숏 + 맥락 집단 2 + 특징 w 24 + 기저값 −4000) → catctx 줄·맥락 점검 3줄·특징 점검 3줄·맥락 X/Y 수·학습 줄.
# (c) 평가 learn × 맥락 X·Y(--rc-eval --rc-eval-ctx) → RCEVAL 4줄 ctx 표지·ctx_on 96·진단 motor 비포화.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E193/pathcheck"; WD="$R/research/experiments/traces/E193/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WU="$R/research/experiments/traces/E184/pathcheck/w_U_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e193_run && cd /root/e193_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
for f_ in "$KW15" "$SN15" "$WU"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E193 경로 검사] 종료"; exit 1; }; done
echo "[E193 경로 검사] 시작 $(date '+%F %T')"
for W in learn none; do
  [ "$W" = "learn" ] && XM=all || XM=none
  g="$OUT/a_reg_${W}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WU --decomp-mode $XM --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "[(a) 회귀 $W rc=$rc] $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"
done
echo "  기대 (a): learn A_train=+0.0046 A_new=+0.0349 B_train=+0.0659 B_new=+0.0468 · none +0.0134 +0.0432 +0.0457 +0.0353(E184 경로 검사 (b))"
f="$OUT/b_train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_b15.npz > "$f" 2>&1; rc=$?
echo "[(b) catctx 학습 rc=$rc] catctx 줄 $(grep -c '^\[맥락 범주 과제\] catctx' "$f") | $(grep '^\[맥락 범주 점검\]' "$f" | grep -oE '맥락 [XY] 범주 [AB] 정답 [^ ]+ [^ ]* — 행동 창 끝 켠 맥락 발화 [0-9]+ · 끈 맥락 발화 [0-9]+ · 시행 끝 두 맥락 발화 [0-9]+' | tr '\n' ' ')| 특징 점검 $(grep -c '^\[범주 과제 점검\].*행동 창 끝 특징 발화 [1-9][0-9]* · 시행 끝 0' "$f")/3 | $(grep '^\[맥락 범주 과제\] 시행' "$f") | 학습 줄 $(grep -c '^\[학습\]' "$f") 동작점 $(grep -cE '^\s*\[motor 동작점\]' "$f")"
grep -E '^\s*\[(맥락 입력|맥락 입력 2)\]' "$f" | sort -u | sed 's/^ */    /'
[ $rc -ne 0 ] && tail -3 "$f" | sed 's/^/      /'
for X in X Y; do
  g="$OUT/c_ev_learn_${X}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_b15.npz --decomp-mode all --rc-eval --rc-eval-ctx $X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "[(c) 평가 learn 맥락 $X rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | ctx 표지·ctx_on 96 줄 $(grep -c "^RCEVAL .*trials=96 feat_on=96 ctx=$X ctx_on=96" "$g") | $(grep '^=> DECOMP' "$g" | grep -oE 'pushed=[0-9]+') | 진단 $(grep '^RCEVAL cat=A set=train' "$g" | grep -oE 'motor L/R [0-9./]+' | tr '\n' ' ')"
  [ $rc -ne 0 ] && tail -3 "$g" | sed 's/^/      /'
done
echo "  기대 (b)(c): catctx 줄 1, 맥락 점검 3줄 켠 > 0·끈 0·시행 끝 0, 특징 점검 3/3, '시행 40 중 맥락 X …', 학습 줄 1, 동작점 ≥ 1; RCEVAL 4·ctx 표지 4, pushed 8, motor 비포화"
echo "[E193 경로 검사] 종료 $(date '+%F %T')"
