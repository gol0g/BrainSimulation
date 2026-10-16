#!/bin/bash
# E199 학습률 보정(사전 규칙 — logs/E199/criteria_fixed.txt, 경로 검사 뒤·본실험 전) — 표본 밖 뇌 15, 경쟁형 규칙(--kc-motor-hard --kc-motor-comp)으로
# η 0.15·0.6·2.4 각 catctx 3,000시행(본실험과 같은 설정: C2·기저값 −4900) → scripts/calib_e199.py(강한 선택 반응 KC |Δ| ≥ 250 비율 ≥ 0.5 인 가장 작은 η) → logs/E199/calib_eta.txt.
# 평가는 하지 않는다. 요약 줄 "  e199 calib eta … : => …" 재개 가능.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/logs/E199/calib.out"
OUT="$R/research/experiments/logs/E199/calib"; WD="$R/research/experiments/traces/E199/calib"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e199_calib_run && cd /root/e199_calib_run
cp $R/backend/genesis/*.py . 2>/dev/null
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4900"
echo "[E199 보정] 시작 $(date '+%F %T')" >> "$LOG"
for ET in 0.15 0.6 2.4; do
  tag="e199 calib eta $ET"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]" >> "$LOG"; continue; fi
  BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta $ET --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
  f="$OUT/train_b15_eta$ET.log"; printf "  %s: " "$tag" >> "$LOG"
  timeout 14400 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes 30 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
    --save-weights $WD/w_b15_eta$ET.npz --kc-motor-hard --kc-motor-comp > "$f" 2>&1; rc=$?
  if grep -q "^\[맥락 범주 과제\] 시행" "$f" && grep -q "^\[E199 경쟁 정규화\] 적용" "$f" && [ -s "$WD/w_b15_eta$ET.npz" ]; then
    echo "=> $(grep '^\[맥락 범주 과제\] 시행' "$f") || $(grep -oE '보상 [0-9]+회' "$f" | head -1) || $(grep '^\[E199 경쟁 정규화\] 적용' "$f")" >> "$LOG"
  else echo "[실패 rc=$rc]" >> "$LOG"; tail -2 "$f" | sed 's/^/      /' >> "$LOG"; fi
done
python3 $R/scripts/calib_e199.py "$R" >> "$LOG" 2>&1
echo "[E199 보정] 종료 $(date '+%F %T')" >> "$LOG"
