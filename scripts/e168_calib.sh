#!/bin/bash
# E168 경로 검사 + 보정(조건 2) — 기준 logs/E168/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15.
# (a) 옵션 끔 회귀: E166 경로 검사 FNF 1ep 인자 + --kc-rw-diag(읽기 전용) → [사전] +0.0228 [사후] −0.0820 보상 55(E166 경로 검사 재현), 연결 줄 0.
# (b) 보정: 형성 표현(E161 경로 검사 가중치)·동결 없음·--rw-da-reset·--kc-rw-diag, 1ep, 도파민→KC억제 가중치 W ∈ {0, 2, 5, 10, 20}(연결 확률 0.2).
#     각 W: 결정 단계·보상 창(보상·처벌 시행) KC 발화율, [사전], 연결 줄, I_input 점검 줄, 추적. 선택은 scripts/e168_pick.py(기준 파일 규칙).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E168/calib"; WD="$R/research/experiments/traces/E168/calib"; mkdir -p "$OUT" "$WD"
P161="$R/research/experiments/traces/E161/pathcheck"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e168_run && cd /root/e168_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
B=15; KW="$P161/kctype_oja_b$B.npz"
f="$OUT/a_off_b$B.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 1 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW --kc-rw-diag \
  --save-weights $WD/w_a_b$B.npz --trace-kc-class $WD/tr_a_b$B.npz --kc-rate-file $P161/rate_b$B.npz > "$f" 2>&1
echo "[(a) 옵션 끔 rc=$?] 적재 $(grep -c "$LD" "$f") 연결 줄 $(grep -c '^  \[E168 도파민→KC억제\]' "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | 기대 +0.0228 → -0.0820 보상 55회, 연결 줄 0"
echo "    $(grep '^\[E168 KC 발화\]' "$f" | cut -c1-200)"
for W in 0 2 5 10 20; do
  f="$OUT/W${W}_b$B.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 1 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW --kc-rw-diag \
    --rw-da-reset --da-kc-inh $W --da-kc-inh-p 0.2 --save-weights $WD/w_W${W}_b$B.npz --trace-kc-class $WD/tr_W${W}_b$B.npz --kc-rate-file $P161/rate_b$B.npz > "$f" 2>&1
  echo "[W=$W rc=$?] 연결 줄 $(grep -c '^  \[E168 도파민→KC억제\]' "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | $(grep -oE 'I_input [0-9.]+ → [0-9.]+' "$f")"
  echo "    $(grep '^\[E168 KC 발화\]' "$f" | cut -c1-200)"
done
cd $R && python3 scripts/e168_pick.py
echo "[E168 보정] 종료"
