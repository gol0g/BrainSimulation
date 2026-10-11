#!/bin/bash
# E184 본실험 — 기준 logs/E184/criteria_fixed.txt. 원시 특징 범주 의존 행동 학습과 새 사례 일반화(뇌 10~14).
# 뇌마다: (1) F 형성 — E183 O 와 같은 rawcat 절차 + --rc-save(부 팔 가중치), (2) 학습 U·F(catrule 1,500시행), (3) 평가 U·F × 학습·무학습(--rc-eval 4조건, 96시행, --eval-diag).
# 주 팔 U = 특징 가중치 초기값 24(E183 사전 결정 — 형성 없이), 부 팔 F = 형성 가중치(판정 밖). 요약 줄 "  e184 ... : => ..." 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E184.log"
RAW="$R/research/experiments/logs/E184"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E184"; mkdir -p "$WD"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e184_main_run && cd /root/e184_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
echo "[E184] 원시 특징 범주 의존 행동 · 특징 w 24 · 뇌 10~14 × U(주)·F(부)"
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  FF="$WD/feat_formed_b$B.npz"
  tag="e184 form b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; else
    g="$RAW/form_b$B.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $FEAT --feat-oja --feat-oja-eta 0 \
      --decomp-weights $R/research/experiments/traces/E179/w_bc_b$B.npz --decomp-mode rawcat --rc-repeat-pre --rc-expo-eta 0.02 --rc-save $FF > "$g" 2>&1; rc=$?
    if grep -q "^=> RAWCAT " "$g" && [ -s "$FF" ]; then echo "=> $(grep '^=> RAWCAT ' "$g" | cut -c1-70)"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  fi
  for A in U F; do
    FW=""; [ "$A" = "F" ] && FW="--feat-weights $FF"
    tag="e184 train b$B $A"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    if [ "$A" = "F" ] && [ ! -s "$FF" ]; then echo "  $tag: [실패 rc=형성 가중치 없음]"; continue; fi
    f="$RAW/train_b${B}_$A.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT $FEAT $FW --rc-task catrule --episodes 15 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
      --save-weights $WD/w_${A}_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^\[범주 과제\] 시행" "$f" && grep -q "^\[학습\]" "$f" && [ -s "$WD/w_${A}_b$B.npz" ]; then
      echo "=> $(grep '^\[범주 과제\] 시행' "$f") || $(grep -oE '보상 [0-9]+회' "$f" | head -1) || 사후 $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+')"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
  for A in U F; do
    FW=""; [ "$A" = "F" ] && FW="--feat-weights $FF"
    for W in learn none; do
      tag="e184 b$B $A $W"
      if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
      if [ ! -s "$WD/w_${A}_b$B.npz" ]; then echo "  $tag: [실패 rc=학습 가중치 없음]"; continue; fi
      [ "$W" = "learn" ] && XM="all" || XM="none"
      g="$RAW/ev_b${B}_${A}_$W.log"; printf "  %s: " "$tag"
      timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $FW --brain-seed $B --kc-type-weights $KW --decomp-weights $WD/w_${A}_b$B.npz --decomp-mode $XM --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
      if grep -q "^=> DECOMP .*rceval" "$g"; then echo "=> $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
    done
  done
done
echo "[E184] 전체 루프 종료"
