#!/bin/bash
# E144: E142 F1500 과 같은 인자(반사 25, 보상 창 동결, 1,500시행, 추적) + 행동 창 반대쪽 음 전류 N(--act-current-neg). 뇌 10~14. 재개 가능(P13).
# N 은 보정(logs/E144/calib.out, 규칙은 criteria_fixed.txt)에서 고른 값 — 인자로 받는다: bash scripts/e144_main.sh <N>
# 요약 줄: "  e144 b10: => 사전 +0.4148 사후 -0.0500 보상 600 || => KCTRACE ..."  (judge_e144.py 와 맞춤)
set -u
NEG="${1:?반대쪽 음 전류 N 이 필요하다(보정 선택값)}"
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E144.log"
RAW="$R/research/experiments/logs/E144"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E144"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e144_main_run && cd /root/e144_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
echo "  [E144] 반대쪽 음 전류 N = $NEG"
for B in 10 11 12 13 14; do
  tag="e144 b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/b$B.log"; printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE $ACT --reflex-w 25 --episodes 15 --steps 100 --transplant-eval --brain-seed $B --rw-apm-scale 0 --act-current-neg $NEG \
    --save-weights $WD/w_b$B.npz --trace-kc-class $WD/tr_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
  if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
    echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || $(grep '^=> KCTRACE ' "$f" | python3 -c 'import sys; print(sys.stdin.read().strip()[:160])')"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done
echo "[E144] 전체 루프 종료"
