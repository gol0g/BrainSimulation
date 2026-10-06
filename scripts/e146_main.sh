#!/bin/bash
# E146: 1단계 — 반사 0·동결·1,500시행 학습(뇌 10~14, 추적·가중치 저장). 2단계 — 학습 없는 이식 평가 75회
# (가중치 none·E141·R0F1500 × 자극 base·int05·int07·occ·noise). 재개 가능(P13).
# 요약 줄: "  e146 train b10: => 사전 ... 사후 ... 보상 N || ..." / "  e146 b10 E141 int05: => mod -0.1500"  (judge_e146.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E146.log"
RAW="$R/research/experiments/logs/E146"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E146"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e146_main_run && cd /root/e146_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
for B in 10 11 12 13 14; do
  tag="e146 train b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/train_b$B.log"; printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE $ACT --episodes 15 --steps 100 --transplant-eval --brain-seed $B \
    --save-weights $WD/w_b$B.npz --trace-kc-class $WD/tr_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
  if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
    echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || $(grep '^=> KCTRACE ' "$f" | python3 -c 'import sys; print(sys.stdin.read().strip()[:160])')"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done
for B in 10 11 12 13 14; do
  for W in none E141 R0F1500; do
    case $W in
      none)    X="--decomp-weights $R/research/experiments/traces/E141/w_b$B.npz --decomp-mode none" ;;
      E141)    X="--decomp-weights $R/research/experiments/traces/E141/w_b$B.npz --decomp-mode all" ;;
      R0F1500) X="--decomp-weights $WD/w_b$B.npz --decomp-mode all" ;;
    esac
    for S in base int05 int07 occ noise; do
      tag="e146 b$B $W $S"
      if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
      if [ "$W" = "R0F1500" ] && [ ! -s "$WD/w_b$B.npz" ]; then echo "  $tag: [실패 rc=학습 가중치 없음]"; continue; fi
      f="$RAW/ev_b${B}_${W}_$S.log"; printf "  %s: " "$tag"
      timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B $X --eval-variant $S --eval-vseed 0 > "$f" 2>&1; rc=$?
      if grep -q "^=> DECOMP" "$f" && grep -q "^\[E146 변형\] variant=$S" "$f"; then
        echo "=> mod $(grep '^=> DECOMP' "$f" | sed -E 's/.*mod=([-+0-9.]+).*/\1/')"
      else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
    done
  done
done
echo "[E146] 전체 루프 종료"
