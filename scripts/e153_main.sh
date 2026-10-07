#!/bin/bash
# E153: 종류 입력 합 보존 헤브 재분배(경험 형성) → 겹침(kcoverlap)·과제 A 500시행 학습 효과(E141 인자) — 뇌 10~14. η = logs/E153/eta.txt(경로 검사 보정 규칙). 재개 가능(P13).
# 요약 줄: "  e153 b10 dev: => KCDEV ..." / "  e153 b10 ov: => KCOVERLAP ... || 적재 N" / "  e153 b10 train: => 사전 .. 사후 .. 보상 N || 적재 N || => KCTRACE ..."  (judge_e153.py 와 맞춤)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E153.log"
RAW="$R/research/experiments/logs/E153"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E153"; mkdir -p "$WD"
ETA=$(awk '$1=="eta"{print $2}' "$RAW/eta.txt" 2>/dev/null)
case "$ETA" in ""|none) echo "[E153] eta 없음/none — 본실험 없음"; exit 1 ;; esac
echo "[E153] η = $ETA"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e153_main_run && cd /root/e153_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
for B in 10 11 12 13 14; do
  KW="$WD/kctype_b$B.npz"
  tag="e153 b$B dev"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null && [ -s "$KW" ]; then echo "  $tag: [건너뜀]"
  else
    f="$RAW/dev_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --decomp-weights $R/research/experiments/traces/E141/w_b$B.npz --decomp-mode kcdev \
      --kc-dev-n 100 --kc-dev-eta $ETA --kc-dev-save $KW > "$f" 2>&1; rc=$?
    if grep -q "^=> KCDEV" "$f" && [ -s "$KW" ]; then echo "$(grep '^=> KCDEV' "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; continue; fi
  fi
  tag="e153 b$B ov"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"
  else
    f="$RAW/ov_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --decomp-weights $R/research/experiments/traces/E141/w_b$B.npz --decomp-mode kcoverlap \
      --trials 200 --kc-type-weights $KW > "$f" 2>&1; rc=$?
    if grep -q "^=> KCOVERLAP" "$f"; then echo "$(grep '^=> KCOVERLAP' "$f") || 적재 $(grep -c "$LD" "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  fi
  tag="e153 b$B train"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"
  else
    f="$RAW/train_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT --episodes 5 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
      --save-weights $WD/w_b$B.npz --trace-kc-class $WD/tr_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$f") || $(grep '^=> KCTRACE ' "$f" | cut -c1-160)"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  fi
done
echo "[E153] 전체 루프 종료"
