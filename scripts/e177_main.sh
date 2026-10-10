#!/bin/bash
# E177 본실험 — 기준 logs/E177/criteria_fixed.txt(맥락 = 맥락 전용 집단 zz_ctx → KC, 초기 연결 스냅숏 traces/E176/snap). 맥락 세기 w = 4 고정(E176 보정 뇌 15 에서 자카드 합 ≤ 1.60 을 만족하는 가장 약한 칸 — 이 선택 규칙은 보정 자료를 본 뒤 정함, E177.md 1절). 뇌 10~14, E160 망 안 형성 가중치. (e176 본실험 러너에서 보정 선택 파일 대신 w 4 고정 — 뇌마다 스냅숏 conn_b{B})
# 뇌마다: 쌍조건 학습(--ctx-task bicond, 3,000시행, 맥락 50%, 추적) → kcctx(측정, 240 제시, 부지표) → 이식 평가 4(학습·무학습 × 맥락 끔·켬).
# 요약 줄(judge_e177.py 와 맞춤): "  e177 kcctx b10: => KCCTX ..." / "  e177 train b10: => 사전 .. 사후 .. 보상 N || 맥락 켬 n || 적재 K || ..." / "  e177 b10 learn on: => mod +0.1000". 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E177.log"
RAW="$R/research/experiments/logs/E177"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E177"; mkdir -p "$WD"
CI=4
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e177_main_run && cd /root/e177_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
SNAP="$R/research/experiments/traces/E176/snap"
LD='^\[E153 종류 입력 적재\].*검증 일치'
echo "[E177] 맥락 세기 w=$CI(고정)"
for B in 10 11 12 13 14; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  CTX="--conn-snapshot $SNAP/conn_b$B.npz --ctx-n 200 --ctx-w $CI --ctx-p 0.10"
  [ -s "$SNAP/conn_b$B.npz" ] || { echo "  e177 b$B: [실패 rc=스냅숏 없음]"; continue; }
  [ -s "$KW" ] || { echo "  e177 b$B: [실패 rc=형성 가중치 없음]"; continue; }
  tag="e177 train b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; else
    f="$RAW/train_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT $CTX --ctx-task bicond --episodes 30 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
      --save-weights $WD/w_bc_b$B.npz --trace-kc-class $WD/tr_bc_b$B.npz --kc-rate-file $R/research/experiments/traces/E138/fix/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f" && grep -q "^\[맥락 과제\] 시행" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 맥락 켬 $(grep '^\[맥락 과제\] 시행' "$f" | grep -oE '맥락 켬 [0-9]+' | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$f") || $(grep '^=> KCTRACE ' "$f" | cut -c1-90)"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  fi
  tag="e177 kcctx b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; else
    if [ ! -s "$WD/w_bc_b$B.npz" ]; then echo "  $tag: [실패 rc=학습 가중치 없음]"; else
      f="$RAW/kcctx_b$B.log"; printf "  %s: " "$tag"
      timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $CTX --decomp-weights $WD/w_bc_b$B.npz --decomp-mode kcctx --trials 240 > "$f" 2>&1; rc=$?
      if grep -q "^=> KCCTX" "$f"; then echo "$(grep '^=> KCCTX' "$f")"; else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
    fi
  fi
  for WC in "learn off" "learn on" "none off" "none on"; do
    set -- $WC; M=$1; C=$2
    tag="e177 b$B $M $C"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    if [ ! -s "$WD/w_bc_b$B.npz" ]; then echo "  $tag: [실패 rc=학습 가중치 없음]"; continue; fi
    [ "$M" = "learn" ] && XM="all" || XM="none"
    [ "$C" = "on" ] && XE="--eval-ctx" || XE=""
    g="$RAW/ev_b${B}_${M}_$C.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $CTX --decomp-weights $WD/w_bc_b$B.npz --decomp-mode $XM $XE > "$g" 2>&1; rc=$?
    if grep -q "^=> DECOMP" "$g"; then echo "=> mod $(grep '^=> DECOMP' "$g" | sed -E 's/.*mod=([-+0-9.]+).*/\1/')"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  done
done
echo "[E177] 전체 루프 종료"
