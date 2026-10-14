#!/bin/bash
# E191 본실험 — 기준 logs/E191/criteria_fixed.txt. 형성 표현 원시 특징 범주 기준선의 독립 확증: 뇌 16~20(E161 종류 가중치) ×
# (1) 형성(네 범주 노출 17 주기) (2) R catrule 1,500 (3) P 1,500 → 둘째 쌍 1,500 (4) V 1,500 → 반전 1,500 (5) 평가 R·P·V 학습·무학습 × A·B·C·D — 모두 형성 가중치.
# 요약 줄 "  e191 ... : => ..." 재개 가능(P13). 40줄.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E191.log"
RAW="$R/research/experiments/logs/E191"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E191"; mkdir -p "$WD"
SNAP="$R/research/experiments/traces/E176/snap"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e191_main_run && cd /root/e191_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
echo "[E191] 형성 표현 원시 특징 범주 기준선 독립 확증 · 형성(A·B·C·D) → R 1,500 · P 1,500 + 둘째 쌍 1,500 · V 1,500 + 반전 1,500 · 뇌 16~20"
for B in 16 17 18 19 20; do
  KW="$R/research/experiments/traces/E161/kctype_oja_b$B.npz"
  FEAT="--conn-snapshot $SNAP/conn_b$B.npz --feat-n 200 --feat-w 24 --feat-p 0.10"
  FF="$WD/feat_formed_b$B.npz"
  tag="e191 form b$B"
  if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; else
    g="$RAW/form_b$B.log"; printf "  %s: " "$tag"
    timeout 7200 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $FEAT --feat-oja --feat-oja-eta 0 \
      --decomp-weights $R/research/experiments/traces/E185/w_1500_b$B.npz --decomp-mode rawcat --rc-repeat-pre --rc-expo-eta 0.02 \
      --rc-expo-cats A,B,C,D --rc-measure-cats A,B,C,D --rc-save $FF > "$g" 2>&1; rc=$?
    if grep -q "^=> RAWCAT " "$g" && [ -s "$FF" ]; then echo "=> $(grep '^\[원시 특징 노출\]' "$g" | grep -oE '제시 [0-9]+') || $(grep '^RCJ ' "$g" | grep -E 'pair=(AC|BD)' | grep -oE 'side=[lr] pair=[A-D]+ J_pre=[0-9.]+ J_post=[0-9.]+' | tr '\n' ' ')"
    else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  fi
  for A in R P V; do
    tag="e191 train b$B $A"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    if [ ! -s "$FF" ]; then echo "  $tag: [실패 rc=형성 가중치 없음]"; continue; fi
    case $A in R) XT="--episodes 15 --steps 100" ;; P) XT="--rc-phase2-after 1500 --episodes 30 --steps 100" ;; V) XT="--rc-reverse-after 1500 --episodes 30 --steps 100" ;; esac
    f="$RAW/train_${A}_b$B.log"; printf "  %s: " "$tag"
    timeout 14400 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $FF --rc-task catrule $XT --transplant-eval --brain-seed $B --kc-type-weights $KW \
      --save-weights $WD/w_${A}_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^\[범주 과제\] 시행" "$f" && grep -q "^\[학습\]" "$f" && [ -s "$WD/w_${A}_b$B.npz" ]; then
      echo "=> $(grep '^\[범주 과제\] 시행' "$f") || $(grep -E '^\[둘째 범주 쌍\] 둘째 쌍|^\[범주 반전\] 반전 시행' "$f") || $(grep -oE '보상 [0-9]+회' "$f" | head -1)"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
  for W in R_learn P_learn V_learn none; do
    tag="e191 b$B $W"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    case $W in R_learn) WF="$WD/w_R_b$B.npz"; XM=all ;; P_learn) WF="$WD/w_P_b$B.npz"; XM=all ;; V_learn) WF="$WD/w_V_b$B.npz"; XM=all ;; none) WF="$WD/w_P_b$B.npz"; XM=none ;; esac
    [ -s "$WF" ] || { echo "  $tag: [실패 rc=학습 가중치 없음]"; continue; }
    g="$RAW/ev_b${B}_$W.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --feat-weights $FF --brain-seed $B --kc-type-weights $KW --decomp-weights $WF --decomp-mode $XM \
      --rc-eval --rc-eval-cats A,B,C,D --trials 96 --eval-diag > "$g" 2>&1; rc=$?
    if grep -q "^=> DECOMP .*rceval" "$g"; then echo "=> $(grep '^=> DECOMP' "$g" | grep -oE '[ABCD]_(train|new)=[-+0-9.]+' | tr '\n' ' ')"; else echo "[실패 rc=$rc]"; tail -2 "$g" | sed 's/^/      /'; fi
  done
done
echo "[E191] 전체 루프 종료"
