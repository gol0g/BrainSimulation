#!/bin/bash
# E140 경로 검사(조건 2) — 판정 기준 고정(logs/E140/criteria_fixed.txt) 뒤. 표본 밖 뇌 15(E119·E139 경로 검사와 같은 인자 — 판정 v).
# 확인: 보상 창 motor 발화율 ≈0(침묵이 듣는가), [사전] = +0.0255(같은 출발점), 도파민 전 변화 0, B/A·C/P 와 e_da·e_end(기전), 효과(무침묵 뇌 15: −0.0989).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E140/pathcheck"; WD="$R/research/experiments/traces/E140/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e140_run && cd /root/e140_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000"
f="$OUT/silence_b15.log"
echo "[1] 뇌 15 학습 + 보상 창 motor 침묵 5000 + 계층 추적"
timeout 7200 python reflex_override_task.py $BASE $ACT --reflex-w 0 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 --rw-motor-silence 5000 \
  --save-weights $WD/w_b15.npz --trace-kc-class $WD/tr_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
grep -E '^\[사전\]|^\[사후\]|^\[학습\]|^=> KCTRACE|^=> 정답률' "$f" | cut -c1-420
[ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
echo "[E140 경로 검사] 종료"
