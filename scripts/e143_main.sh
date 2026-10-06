#!/bin/bash
# E143: E142 F1500 과 같은 인자(반사 25, 보상 창 동결, 1,500시행, 추적) + 결정 단계 흔적 동결(--dec-apm-scale 0). 뇌 10~14. 재개 가능(P13).
# 요약 줄: "  e143 b10: => 사전 +0.4148 사후 -0.0500 보상 600 || => KCTRACE ... || => KCTRACE3 ..."  (judge_e143.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E143.log"
RAW="$R/research/experiments/logs/E143"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E143"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e143_main_run && cd /root/e143_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
for B in 10 11 12 13 14; do
  tag="e143 b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/b$B.log"; printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE $ACT --reflex-w 25 --episodes 15 --steps 100 --transplant-eval --brain-seed $B --rw-apm-scale 0 --dec-apm-scale 0 \
    --save-weights $WD/w_b$B.npz --trace-kc-class $WD/tr_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
  if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
    # 요약 줄은 앞부분(사전·사후·보상)만 판정에 쓴다. E142 교훈: cut -c 는 바이트 단위라 한글을 반쯤 자른다 → 문자 단위(python)로 자른다.
    echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || $(grep '^=> KCTRACE ' "$f" | python3 -c 'import sys; print(sys.stdin.read().strip()[:160])')"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done
echo "[E143] 전체 루프 종료"
