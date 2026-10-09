#!/bin/bash
# E166 경로 검사(조건 2) — 기준 고정 뒤. 표본 밖 뇌 15.
# (a) 동결 없는 기본 경로 회귀: E139 경로 검사 인자 그대로(판정 v, --rw-apm-scale 없음) → [사전] +0.0255 [사후] −0.0734 보상 237(E139·E141 배율 1 재현값).
# (b) FNF 1ep(판정 exec, 동결 없음, E161 경로 검사 형성 가중치): 적재 2, [사전] = E161 경로 검사 F [사전] +0.0228, 보상 창 끝 흔적 잔차 ≥ 0.05(동결 꺼짐).
# (c) DNF 1ep(판정 exec, 동결 없음, 기본 표현): 적재 0, [사전] = E161 경로 검사 D [사전] +0.0255, 잔차 ≥ 0.05.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E166/pathcheck"; WD="$R/research/experiments/traces/E166/pathcheck"; mkdir -p "$OUT" "$WD"
P161="$R/research/experiments/traces/E161/pathcheck"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e166_run && cd /root/e166_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
B=15
f="$OUT/a_b$B.log"
timeout 7200 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --reflex-w 0 --episodes 5 --steps 100 --transplant-eval --brain-seed $B \
  --save-weights $WD/w_a_b$B.npz --trace-kc-class $WD/tr_a_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1
echo "[(a) 동결 없는 기본 rc=$?] $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | 기대 +0.0255 → -0.0734 보상 237회"
for ARM in FNF DNF; do
  [ "$ARM" = "FNF" ] && XA="--kc-type-weights $P161/kctype_oja_b$B.npz" || XA=""
  f="$OUT/${ARM}_b$B.log"
  timeout 3600 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --judge exec --reflex-w 0 --episodes 1 --steps 100 --transplant-eval --brain-seed $B $XA \
    --save-weights $WD/w_${ARM}_b$B.npz --trace-kc-class $WD/tr_${ARM}_b$B.npz --kc-rate-file $P161/rate_b$B.npz > "$f" 2>&1
  echo "[($ARM) 1ep rc=$?] 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | KCTRACE3 $(grep -c '^=> KCTRACE3' "$f")"
done
echo "  기대 [사전]: FNF +0.0228(E161 경로 검사 F), DNF +0.0255(E161 경로 검사 D). 잔차는 judge_e166.stats 로 따로 계산."
echo "[E166 경로 검사] 종료"
