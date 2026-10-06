#!/bin/bash
# E151: E150(공통 입력 차단) 인자에서 KC→motor eta 만 η*(logs/E151/eta_star.txt — 뇌 15 보정 규칙) — 학습 A단독(1,500)·AB(1,500 → 과제 B 1,500), 뇌 10~14.
# 그다음 평가 25(A×base, AB×base·bad, 무학습×base·bad — 모두 차단 뇌). 재개 가능(P13).
# 요약 줄: "  e151 train A b10: => 사전 ... 사후 ... 보상 N || ..." / "  e151 b10 AB bad: => mod +0.1000"  (judge_e151.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E151.log"
RAW="$R/research/experiments/logs/E151"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E151"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
ETA=$(awk '$1=="eta_star"{print $2}' "$RAW/eta_star.txt" 2>/dev/null)
case "$ETA" in ""|none) echo "[E151] eta_star 없음/none — 본실험 없음"; exit 1 ;; esac
echo "[E151] η* = $ETA"
mkdir -p /root/e151_main_run && cd /root/e151_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta $ETA --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0 --kc-food-eye-scale 0 --kc-bilateral-scale 0"
for ARM in A AB; do
  [ "$ARM" = "A" ] && X="--episodes 15" || X="--episodes 30 --task-b-after 1500"
  for B in 10 11 12 13 14; do
    tag="e151 train $ARM b$B"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    f="$RAW/train_${ARM}_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT $X --steps 100 --transplant-eval --brain-seed $B \
      --save-weights $WD/w_${ARM}_b$B.npz --trace-kc-class $WD/tr_${ARM}_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || $(grep '^=> KCTRACE ' "$f" | python3 -c 'import sys; print(sys.stdin.read().strip()[:160])')"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
for B in 10 11 12 13 14; do
  for WS in "A base" "AB base" "AB bad" "none base" "none bad"; do
    set -- $WS; W=$1; S=$2
    tag="e151 b$B $W $S"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    case $W in
      A) X="--decomp-weights $WD/w_A_b$B.npz --decomp-mode all" ;;
      AB) X="--decomp-weights $WD/w_AB_b$B.npz --decomp-mode all" ;;
      none) X="--decomp-weights $WD/w_A_b$B.npz --decomp-mode none" ;;
    esac
    if [ ! -s "$WD/w_A_b$B.npz" ] || { [ "$W" = "AB" ] && [ ! -s "$WD/w_AB_b$B.npz" ]; }; then echo "  $tag: [실패 rc=학습 가중치 없음]"; continue; fi
    f="$RAW/ev_b${B}_${W}_$S.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B $X --eval-variant $S > "$f" 2>&1; rc=$?
    if grep -q "^=> DECOMP" "$f" && grep -q "^\[E146 변형\] variant=$S" "$f"; then
      echo "=> mod $(grep '^=> DECOMP' "$f" | sed -E 's/.*mod=([-+0-9.]+).*/\1/')"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E151] 전체 루프 종료"
