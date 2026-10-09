#!/bin/bash
# E164 경로 검사(조건 2) — 기준 고정 뒤. 표본 밖 뇌 15.
# (a) 옵션 끔: E141 경로 검사 인자 그대로 → [사전] +0.0255 [사후] −0.2103 보상 243(E141·E157 재현) — 새 옵션 코드가 기본 동작을 바꾸지 않았는가.
# (b) 옵션 켬(--rw-da-reset --offset-steps 3), 1ep: '[구현 점검]' 줄 3개(설정·오프셋 값·첫 보상 창 끝 I_input → 0), 추적·동결.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E164/pathcheck"; WD="$R/research/experiments/traces/E164/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e164_run && cd /root/e164_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
RATE="$R/research/experiments/traces/E138/pathcheck/rate_b15.npz"
f="$OUT/off_b15.log"
timeout 7200 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --reflex-w 0 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 --rw-apm-scale 0 \
  --save-weights $WD/w_off_b15.npz --trace-kc-class $WD/tr_off_b15.npz --kc-rate-file $RATE > "$f" 2>&1
echo "[(a) 옵션 끔 rc=$?] $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | 점검 줄 $(grep -c '^\[구현 점검\]' "$f") | 기대 +0.0255 → -0.2103 보상 243회, 점검 0"
f="$OUT/on_b15.log"
timeout 3600 python reflex_override_task.py $BASE --act-window 3 --act-current 5000 --judge exec --reflex-w 0 --episodes 1 --steps 100 --transplant-eval --brain-seed 15 --rw-apm-scale 0 \
  --rw-da-reset --offset-steps 3 --save-weights $WD/w_on_b15.npz --trace-kc-class $WD/tr_on_b15.npz --kc-rate-file $RATE > "$f" 2>&1
echo "[(b) 옵션 켬 1ep rc=$?] $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
grep '^\[구현 점검\]' "$f" | sed 's/^/    /'
echo "[E164 경로 검사] 종료"
