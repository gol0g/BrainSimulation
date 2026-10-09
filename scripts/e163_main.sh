#!/bin/bash
# E163: 망 안 형성 표현의 반전(K81 회귀, E147·E155 판정 2 규칙) — 뇌 10~14, E148 학습 인자 + 같은 뇌 E160 망 안 Oja 형성 가중치,
# --episodes 30 --reverse-after 1500(교차 1,500 → 같은 쪽 1,500) + 이식 평가. 반전 직전 기준은 E162 A단독(같은 시드·같은 가중치·같은 앞 1,500). 재개 가능(P13).
# 요약 줄: "  e163 rev b10: => 사전 .. 사후 .. 보상 N || 적재 K || ..."  (judge_e163.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E163.log"
RAW="$R/research/experiments/logs/E163"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E163"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e163_main_run && cd /root/e163_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  tag="e163 rev b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/rev_b$B.log"; printf "  %s: " "$tag"
  [ -s "$KW" ] || { echo "[실패 rc=형성 가중치 없음]"; continue; }
  timeout 14400 python reflex_override_task.py $BASE $ACT --episodes 30 --reverse-after 1500 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
    --save-weights $WD/w_rev_b$B.npz --trace-kc-class $WD/tr_rev_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
  if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
    echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$f") || $(grep '^=> KCTRACE ' "$f" | cut -c1-100)"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done
echo "[E163] 전체 루프 종료"
