#!/bin/bash
# E190 경로 검사(조건 2) — 기준 logs/E190/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15.
# 새 코드: rawcat 형성 --rc-expo-cats·--rc-measure-cats(기본 A,B = 이전), RCJ 범주 쌍 겹침 줄.
# (a) 회귀: 기본 범주 짧은 형성(노출 2 주기) = E184 경로 검사 (c) 의 RCW 2줄·노출 줄·RAWCAT 줄 정확 재현, RCJ AB J_pre·J_post = RAWCAT J_AB 앞·뒤.
# (b) A·B·C·D 짧은 형성(노출 2 주기, 측정 A·B·C·D, --rc-save) → '제시 96(주기 2 × 사례 6 × 범주 4 × 쪽 2) … — 노출 범주 A,B,C,D', RCCHK npres 288 × 3, RCJ 12줄, 저장.
# (c) (b) 형성 가중치로 R 짧은 학습(catrule 2 × 20) → 적재 줄·둘째 쌍 줄 0; 평가 learn A·B·C·D → RCEVAL 8·적재 줄·pushed 8.
# (d) 같은 형성 가중치로 P 짧은 학습(catrule 2 × 20 + --rc-phase2-after 20) → 둘째 쌍 줄 2·적재 줄; 평가 none A·B·C·D → RCEVAL 8·적재 줄·pushed 0.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E190/pathcheck"; WD="$R/research/experiments/traces/E190/pathcheck"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
REF="$R/research/experiments/logs/E184/pathcheck/c_form_b15.log"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e190_run && cd /root/e190_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
LDF='^\[E184 특징 가중치 적재\].*검증 일치'
for f_ in "$KW15" "$SN15" "$WB" "$REF"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E190 경로 검사] 종료"; exit 1; }; done
echo "[E190 경로 검사] 시작 $(date '+%F %T')"
g="$OUT/a_form_AB_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT --feat-oja --feat-oja-eta 0 --decomp-weights $WB --decomp-mode rawcat \
  --rc-expo 2 --rc-expo-eta 0.02 --rc-save $WD/feat_AB_b15.npz > "$g" 2>&1; rc=$?
n_same=0
for P_ in '^RCW side=l' '^RCW side=r' '^\[원시 특징 노출\]' '^=> RAWCAT '; do
  a_=$(grep -E "$P_" "$g"); b_=$(grep -E "$P_" "$REF")
  if [ -n "$a_" ] && [ "$a_" = "$b_" ]; then n_same=$((n_same+1)); else echo "    차이 [$P_]"; echo "      새: $a_" | cut -c1-300; echo "      옛: $b_" | cut -c1-300; fi
done
echo "[(a) 기본 범주 형성 rc=$rc] E184 경로 검사 (c) 와 같은 줄 $n_same/4 | $(grep '^RCJ ' "$g" | tr '\n' ' ')"
echo "    RAWCAT J_AB: $(grep '^=> RAWCAT ' "$g" | grep -oE 'J_AB [0-9.]+→[0-9.]+' | tr '\n' ' ')"
g="$OUT/b_form_ABCD_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT --feat-oja --feat-oja-eta 0 --decomp-weights $WB --decomp-mode rawcat \
  --rc-expo 2 --rc-expo-eta 0.02 --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-save $WD/feat_ABCD_b15.npz > "$g" 2>&1; rc=$?
echo "[(b) 네 범주 형성 rc=$rc] $(grep '^\[원시 특징 노출\]' "$g") | RCCHK $(grep '^RCCHK ' "$g" | grep -oE 'tag=[a-z0-9]+ npres=[0-9]+' | tr '\n' ' ')| RCJ $(grep -c '^RCJ ' "$g")줄 | RAWCAT $(grep -c '^=> RAWCAT ' "$g") | 저장 $(ls -la $WD/feat_ABCD_b15.npz 2>/dev/null | awk '{print $5}') 바이트"
grep '^RCW ' "$g" | cut -c1-200 | sed 's/^/    /'
grep '^RCJ ' "$g" | sed 's/^/    /'
FF="$WD/feat_ABCD_b15.npz"
f="$OUT/c_train_R_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $FF --rc-task catrule --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_R_b15.npz > "$f" 2>&1; rc=$?
echo "[(c) R 학습 rc=$rc] 적재 $(grep -cE "$LDF" "$f") | 둘째 쌍 줄 $(grep -c '^\[둘째 범주 쌍\]' "$f") | $(grep '^\[범주 과제\] 시행' "$f") | 점검 $(grep '^\[범주 과제 점검\]' "$f" | grep -oE '특징 발화 [0-9]+ · 시행 끝 [0-9]+' | tr '\n' ',') | 학습 줄 $(grep -c '^\[학습\]' "$f")"
g="$OUT/c_ev_R_learn_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $FF --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_R_b15.npz --decomp-mode all --rc-eval --rc-eval-cats A,B,C,D --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(c) R 평가 learn rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | 적재 $(grep -cE "$LDF" "$g") | $(grep '^=> DECOMP' "$g" | grep -oE 'pushed=[0-9]+') | $(grep '^=> DECOMP' "$g" | grep -oE '[ABCD]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"
f="$OUT/d_train_P_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $FF --rc-task catrule --rc-phase2-after 20 --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_P_b15.npz > "$f" 2>&1; rc=$?
echo "[(d) P 학습 rc=$rc] 적재 $(grep -cE "$LDF" "$f") | $(grep '^\[둘째 범주 쌍\]' "$f" | tr '\n' ' ')| $(grep '^\[범주 과제\] 시행' "$f") | 학습 줄 $(grep -c '^\[학습\]' "$f")"
g="$OUT/d_ev_none_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $FF --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WD/w_P_b15.npz --decomp-mode none --rc-eval --rc-eval-cats A,B,C,D --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(d) 평가 none rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | 적재 $(grep -cE "$LDF" "$g") | $(grep '^=> DECOMP' "$g" | grep -oE 'pushed=[0-9]+') | $(grep '^=> DECOMP' "$g" | grep -oE '[ABCD]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"
echo "  기대: (a) 같은 줄 4/4, RCJ AB 2줄(J_pre·J_post = RAWCAT J_AB 앞·뒤 l 0.3604→0.2072·r 0.2915→0.2045); (b) 제시 96·범주 4·노출 범주 A,B,C,D, RCCHK npres 288 × 3, RCJ 12줄, RAWCAT 1, 저장 > 0;"
echo "        (c) 적재 ≥ 1·둘째 쌍 줄 0·범주 40·점검 > 0/0, 평가 RCEVAL 8·적재 ≥ 1·pushed 8; (d) 적재 ≥ 1·둘째 쌍 줄 2(시행 20 부터·둘째 쌍 시행 20 (시행 20 부터, 전체 40)), 평가 none RCEVAL 8·적재 ≥ 1·pushed 0"
echo "[E190 경로 검사] 종료 $(date '+%F %T')"
