#!/bin/bash
# E185 경로 검사(조건 2) — 기준 logs/E185/criteria_fixed.txt 고정(2026-10-12 00:42:15) 뒤. 뇌 16(확증 표본 — 판정에도 쓰이는 뇌이므로 학습 결과값은 보지 않고 경로만).
# 새 입력: 뇌 16~20 스냅숏(conn_b16~20, E176 도구로 00:40:26 저장), E161 종류 가중치를 스냅숏 뇌에 적재. 코드는 E184 와 같다.
# (a) 뇌 16 E172 A base 평가(w_A_b16, decomp all, --eval-variant base) — 스냅숏 없음·스냅숏·스냅숏 + 조용한 특징 집단 셋 다 −0.6136(E172 정확 재현).
# (b) 뇌 16 catrule 짧은 학습(2 × 20 = 40시행, 스냅숏 + 특징 w 24) → 범주 과제·점검 3줄·종류 적재 검증 줄, 평가 learn(--rc-eval) → RCEVAL 4줄·적재 줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E185/pathcheck"; WD="$R/research/experiments/traces/E185/pathcheck"; mkdir -p "$OUT" "$WD"
KW="$R/research/experiments/traces/E161/kctype_oja_b16.npz"
SN="$R/research/experiments/traces/E176/snap/conn_b16.npz"
WA="$R/research/experiments/traces/E172/w_A_b16.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e185_run && cd /root/e185_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
for f_ in "$KW" "$SN" "$WA"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E185 경로 검사] 종료"; exit 1; }; done
echo "[E185 경로 검사] 시작 $(date '+%F %T')"
for V in nosnap snap snapfeat; do
  case $V in nosnap) X="" ;; snap) X="--conn-snapshot $SN" ;; snapfeat) X="--conn-snapshot $SN --feat-n 200 --feat-w 24 --feat-p 0.10" ;; esac
  g="$OUT/a_${V}_b16.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 16 --kc-type-weights $KW $X --decomp-weights $WA --decomp-mode all --eval-variant base > "$g" 2>&1; rc=$?
  echo "[(a) $V rc=$rc] 스냅숏 적재 줄 $(grep -c 'E176 초기 연결 스냅숏\] 적재' "$g") 원시 특징 줄 $(grep -c '\[원시 특징\]' "$g") 종류 적재 $(grep -c "$LD" "$g") | $(grep '^=> DECOMP' "$g" | grep -oE 'mod=[-+0-9.]+')"
done
echo "  기대 (a): 세 줄 모두 mod −0.6136(E172 b16 A base), 종류 적재 검증 ≥ 1"
FEAT="--conn-snapshot $SN --feat-n 200 --feat-w 24 --feat-p 0.10"
f="$OUT/b_train_b16.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --rc-task catrule --episodes 2 --steps 20 --transplant-eval --brain-seed 16 --kc-type-weights $KW \
  --save-weights $WD/w_b16.npz > "$f" 2>&1; rc=$?
echo "[(b) 학습 rc=$rc] catrule 줄 $(grep -c '^\[범주 과제\] catrule' "$f") | 점검 $(grep -c '^\[범주 과제 점검\].*행동 창 끝 특징 발화 [1-9][0-9]* · 시행 끝 0' "$f")/3 | $(grep '^\[범주 과제\] 시행' "$f") | 학습 줄 $(grep -c '^\[학습\]' "$f") 종류 적재 $(grep -c "$LD" "$f")"
g="$OUT/b_ev_learn_b16.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed 16 --kc-type-weights $KW --decomp-weights $WD/w_b16.npz --decomp-mode all --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(b) 평가 learn rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") (trials 96·feat_on 96 줄 $(grep -c '^RCEVAL .*trials=96 feat_on=96' "$g")) | $(grep '^=> DECOMP' "$g" | grep -oE 'pushed=[0-9]+') 종류 적재 $(grep -c "$LD" "$g")"
echo "  기대 (b): catrule 줄 1, 점검 3/3, '시행 40 중 범주 B …', 학습 줄 1, 종류 적재 ≥ 1; RCEVAL 4(96·96 4), pushed 8, 종류 적재 ≥ 1 — 변조폭 값은 출력하지 않음(확증 표본)"
echo "[E185 경로 검사] 종료 $(date '+%F %T')"
