#!/bin/bash
# E202 본실험(측정) — 기준 logs/E202/criteria_fixed.txt. E200 과 같은 학습(경쟁형 규칙 --kc-motor-hard --kc-motor-comp·η 2.4·도파민 기준선 --da-baseline,
# catctx 3,000시행, 뇌 10~14)에 읽기 전용 보상 창 Δg 계측(--trace-dg)만 더한다. 학습만(이식 평가 없음).
# 요약 줄 "  e202 train b{B}: => ..." 재개 가능(P13). 5줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E202.log"
RAW="$R/research/experiments/logs/E202"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E202"; mkdir -p "$WD"
SNAP="$R/research/experiments/traces/E176/snap"
ETA=2.4   # E199 보정값 그대로(logs/E199/calib_eta.txt) — E200 과 같음
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e202_main_run && cd /root/e202_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta $ETA --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4900"
echo "[E202] 보상 창 Δg 계측(읽기 전용) + E200 학습(도파민 기준선 + 경쟁형 규칙) · η $ETA · catctx 3,000 · C2 · 기저값 −4900 · 뇌 10~14 · 학습만"
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  tag="e202 train b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
  f="$RAW/train_b$B.log"; printf "  %s: " "$tag"
  timeout 14400 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 30 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
    --save-weights $WD/w_b$B.npz --kc-motor-hard --kc-motor-comp --da-baseline --trace-dg $WD/dg_b$B.npz > "$f" 2>&1; rc=$?
  if grep -q "^\[맥락 범주 과제\] 시행" "$f" && grep -q "^\[학습\]" "$f" && grep -q "^\[E199 경쟁 정규화\] 적용" "$f" && grep -q "^\[E200 도파민 통계\]" "$f" \
     && grep -q "^\[E202 계측\] 시행 [0-9]* · " "$f" && [ -s "$WD/w_b$B.npz" ] && [ -s "$WD/dg_b$B.npz" ]; then
    echo "=> $(grep '^\[맥락 범주 과제\] 시행' "$f") || $(grep -oE '보상 [0-9]+회' "$f" | head -1) || $(grep '^\[E199 경쟁 정규화\] 적용' "$f" | grep -oE '적용 [0-9]+회|hard=[01]' | tr '\n' ' ')|| $(grep '^\[E200 도파민 통계\]' "$f" | grep -oE '평균 DA [-+0-9.]+|기준선 [01]' | tr '\n' ' ')|| $(grep '^\[E202 계측\] 시행 [0-9]* · ' "$f" | sed 's/ · 저장.*//')"
  else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
done
echo "[E202] 전체 루프 종료"
