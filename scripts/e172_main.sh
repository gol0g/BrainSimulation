#!/bin/bash
# E172 본실험 — 기준 logs/E172/criteria_fixed.txt. 망 안 형성 표현의 유지(K95)·반전(K96) 독립 확증: 쓰지 않은 뇌 16~20(E161 에서 형성만 씀), E161 망 안 Oja 형성 가중치.
# E162 학습(A단독 1,500 · AB 1,500 → 과제 B 1,500)·평가 5종 + E163 반전(교차 1,500 → 같은 쪽 1,500) 을 인자 그대로 옮김(뇌·형성 가중치·추적 분류 파일만 다름).
# 요약 줄: "  e172 train A b16: => 사전 .. 사후 .. 보상 N || ..." / "  e172 rev b16: => 사전 .. 사후 .. 보상 N || 적재 K || ..." / "  e172 b16 AB bad: => mod +0.1000"
# (judge_e172_ret.py·judge_e172_rev.py 와 맞춤). 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E172.log"
RAW="$R/research/experiments/logs/E172"; mkdir -p "$RAW"
WD="$R/research/experiments/traces/E172"; mkdir -p "$WD"
W161="$R/research/experiments/traces/E161"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e172_main_run && cd /root/e172_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
for B in 16 17 18 19 20; do
  KW="$W161/kctype_oja_b$B.npz"
  for ARM in A AB rev; do
    case $ARM in
      A) X="--episodes 15"; tag="e172 train A b$B"; f="$RAW/train_A_b$B.log" ;;
      AB) X="--episodes 30 --task-b-after 1500"; tag="e172 train AB b$B"; f="$RAW/train_AB_b$B.log" ;;
      rev) X="--episodes 30 --reverse-after 1500"; tag="e172 rev b$B"; f="$RAW/rev_b$B.log" ;;
    esac
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    printf "  %s: " "$tag"
    [ -s "$KW" ] || { echo "[실패 rc=형성 가중치 없음]"; continue; }
    timeout 14400 python reflex_override_task.py $BASE $ACT $X --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $KW \
      --save-weights $WD/w_${ARM}_b$B.npz --trace-kc-class $WD/tr_${ARM}_b$B.npz --kc-rate-file $W161/rate_b$B.npz > "$f" 2>&1; rc=$?
    if grep -q "^=> KCTRACE3" "$f" && grep -q "^\[사후\]" "$f"; then
      echo "=> 사전 $(grep '^\[사전\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 사후 $(grep '^\[사후\]' "$f" | sed -E 's/.*변조폭 ([-+0-9.]+).*/\1/') 보상 $(grep -oE '보상 [0-9]+회' "$f" | grep -oE '[0-9]+') || 적재 $(grep -c "$LD" "$f") || $(grep '^=> KCTRACE ' "$f" | cut -c1-100)"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
for B in 16 17 18 19 20; do
  KW="$W161/kctype_oja_b$B.npz"
  for WS in "A base" "AB base" "AB bad" "none base" "none bad"; do
    set -- $WS; W=$1; S=$2
    tag="e172 b$B $W $S"
    if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
    case $W in
      A) X="--decomp-weights $WD/w_A_b$B.npz --decomp-mode all" ;;
      AB) X="--decomp-weights $WD/w_AB_b$B.npz --decomp-mode all" ;;
      none) X="--decomp-weights $WD/w_A_b$B.npz --decomp-mode none" ;;
    esac
    if [ ! -s "$WD/w_A_b$B.npz" ] || { [ "$W" = "AB" ] && [ ! -s "$WD/w_AB_b$B.npz" ]; }; then echo "  $tag: [실패 rc=학습 가중치 없음]"; continue; fi
    f="$RAW/ev_b${B}_${W}_$S.log"; printf "  %s: " "$tag"
    timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $X --eval-variant $S > "$f" 2>&1; rc=$?
    if grep -q "^=> DECOMP" "$f" && grep -q "^\[E146 변형\] variant=$S" "$f"; then
      echo "=> mod $(grep '^=> DECOMP' "$f" | sed -E 's/.*mod=([-+0-9.]+).*/\1/')"
    else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
  done
done
echo "[E172] 전체 루프 종료"
