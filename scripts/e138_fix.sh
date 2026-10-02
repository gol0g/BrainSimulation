#!/bin/bash
# E138 수리 재실행: kc_selectivity eps 경계 결함(지속 발화 KC 오분류, 독립 대조 발견) 수리 뒤 분류 의존 모드만 5뇌 다시.
# 1차 기록(태그 e138)은 그대로 두고 태그 e138f 로 같은 로그에 덧붙인다. 발화 수 npz 는 traces/E138/fix/ 에 새로(1차 npz 보존).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E138.log"
RAW="$R/research/experiments/logs/E138/fix"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E138/fix"; mkdir -p "$WD"
TR="$R/research/experiments/traces/E119"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e138_fix_run && cd /root/e138_fix_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
one() {  # 뇌 모드
  local tag="e138f b$1 $2"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; return; fi
  local f="$RAW/b$1_$2.log"; local extra=""
  [ "$2" = "kcrate" ] && extra="--trials 200 --kc-rate-file $WD/rate_b$1.npz"
  { [ "$2" = "kcsel" ] || [ "$2" = "kcselonly" ]; } && extra="--kc-rate-file $WD/rate_b$1.npz"
  printf "  %s: " "$tag"
  timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w 0 --episodes 0 --brain-seed $1 --decomp-weights $TR/w_rw0_b$1.npz --decomp-mode $2 $extra > "$f" 2>&1; local rc=$?
  if [ "$2" = "kcrate" ]; then
    if [ "$(grep -c '^=> KCRATE' "$f")" = "2" ]; then echo "$(grep '^=> KCRATE' "$f" | sed 's/^=> //' | sed -n 1p | sed 's/^/=> /') || $(grep '^=> KCRATE' "$f" | sed 's/^=> //' | sed -n 2p)"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  else
    if grep -q '^=> DECOMP' "$f"; then echo "$(grep '^=> DECOMP' "$f")$(grep '^\[E138\]' "$f" | sed 's/^/ || /')"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  fi
}
echo "=== E138 수리 재실행(eps 경계) ==="
for B in 10 11 12 13 14; do for M in kcrate kcsel kcselonly; do one $B $M; done; done
echo "[E138 수리] 전체 루프 종료"
