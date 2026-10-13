#!/bin/bash
# E186 경로 검사(조건 2) — 기준 logs/E186/criteria_fixed.txt 고정 뒤. 뇌 16(주 표본 — 원형·먼 사례 값은 출력하지 않고 형식만).
# 새 코드: rawcat 원형·먼 사례 집합, 과제 --rc-eval-sets(기본 train,new = 이전).
# (a) 기본 집합 회귀: 뇌 16 E185 학습1500 평가(--rc-eval 기본) → E185 원 변조폭 A_train −0.2008·A_new −0.1342·B_train +0.2913·B_new +0.1879(실행 간 1e-4 허용).
# (b) 확장 집합: --rc-eval-sets train,new,proto,far → RCEVAL 8줄·96·96, A train·new 가 (a) 와 같음(같은 평가 순서). 원형·먼 사례 변조폭은 출력하지 않음.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E186/pathcheck"; mkdir -p "$OUT"
KW="$R/research/experiments/traces/E161/kctype_oja_b16.npz"
SN="$R/research/experiments/traces/E176/snap/conn_b16.npz"
WF="$R/research/experiments/traces/E185/w_1500_b16.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e186_run && cd /root/e186_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN --feat-n 200 --feat-w 24 --feat-p 0.10"
for f_ in "$KW" "$SN" "$WF"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E186 경로 검사] 종료"; exit 1; }; done
echo "[E186 경로 검사] 시작 $(date '+%F %T')"
g="$OUT/a_default_b16.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed 16 --kc-type-weights $KW --decomp-weights $WF --decomp-mode all --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(a) 기본 집합 rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"
echo "  기대 (a): A_train=-0.2008 A_new=-0.1342 B_train=+0.2913 B_new=+0.1879(E185 ev_b16_learn1500)"
g="$OUT/b_sets_b16.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed 16 --kc-type-weights $KW --decomp-weights $WF --decomp-mode all --rc-eval --rc-eval-sets train,new,proto,far --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(b) 확장 집합 rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g")(96·96 줄 $(grep -c '^RCEVAL .*trials=96 feat_on=96' "$g")) 집합 $(grep -oE '^RCEVAL cat=[AB] set=[a-z]+' "$g" | sed 's/RCEVAL //' | tr '\n' ' ') | A train·new $(grep '^=> DECOMP' "$g" | grep -oE 'A_(train|new)=[-+0-9.]+' | tr '\n' ' ')| pushed $(grep '^=> DECOMP' "$g" | grep -oE 'pushed=[0-9]+')"
echo "  기대 (b): RCEVAL 8(96·96 8), 집합 순서 A train new proto far B train new proto far, A train·new = (a), pushed 8 — 원형·먼 사례 값은 출력 안 함"
echo "[E186 경로 검사] 종료 $(date '+%F %T')"
