#!/bin/bash
# E155: 형성 표현 기준선의 앞 능력 회귀 — 판정 1 평가 75(none·E153·E154A × 자극 5종, E146 틀) → 판정 2 반전 학습 5런(E147 틀). 뇌 10~14, 모든 뇌에 같은 뇌 E153 형성 가중치. 재개 가능(P13).
# 요약 줄: "  e155 b10 E153 int05: => mod -0.4800" / "  e155 rev b10: => 사전 .. 사후 .. 보상 N || 적재 K || ..."  (judge_e155.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E155.log"
RAW="$R/research/experiments/logs/E155"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E155"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e155_main_run && cd /root/e155_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E153/kctype_b$B.npz"
  for W in none E153 E154A; do
    case $W in
      none)  X="--decomp-weights $R/research/experiments/traces/E153/w_b$B.npz --decomp-mode none" ;;
      E153)  X="--decomp-weights $R/research/experiments/traces/E153/w_b$B.npz --decomp-mode all" ;;
      E154A) X="--decomp-weights $R/research/experiments/traces/E154/w_A_b$B.npz --decomp-mode all" ;;
    esac
    for S in base int05 int07 occ noise; do
      tag="e155 b$B $W $S"
      if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
      f="$RAW/ev_b${B}_${W}_$S.log"; printf "  %s: " "$tag"
      timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $X --eval-variant $S --eval-vseed 0 > "$f" 2>&1; rc=$?
      if grep -q "^=> DECOMP" "$f" && grep -q "^\[E146 변형\] variant=$S" "$f"; then
        echo "=> mod $(grep '^=> DECOMP' "$f" | sed -E 's/.*mod=([-+0-9.]+).*/\1/')"
      else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
    done
  done
done
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E153/kctype_b$B.npz"
  tag="e155 rev b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/rev_b$B.log"; printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE $ACT --episodes 30 --reverse-after 1500 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
    --save-weights $WD/w_rev_b$B.npz --trace-kc-class $WD/tr_rev_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
  if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
    echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$f") || $(grep '^=> KCTRACE ' "$f" | cut -c1-160)"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done
echo "[E155] 전체 루프 종료"
