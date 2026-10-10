#!/bin/bash
# E170 본실험 — 기준 logs/E170/criteria_fixed.txt. 전체 모델 조합 전이(평가만, 학습 없음): 뇌 10~14, E162 의 두 연합 학습 가중치(w_AB) 와 학습 없음(none),
# 형성 표현 E160 가중치, 평가 변형 base·bad·agree·conflict(이식 평가 — E162 평가 명령과 같은 인자). 요약 줄 "  e170 b10 AB agree: => mod ..." 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E170.log"
RAW="$R/research/experiments/logs/E170"; mkdir -p "$RAW"
WD162="$R/research/experiments/traces/E162"; KWD="$R/research/experiments/traces/E160"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e170_main_run && cd /root/e170_main_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
for B in 10 11 12 13 14; do
  for W in AB none; do
    [ "$W" = "AB" ] && X="--decomp-weights $WD162/w_AB_b$B.npz --decomp-mode all" || X="--decomp-weights $WD162/w_AB_b$B.npz --decomp-mode none"
    for S in base bad agree conflict; do
      tag="e170 b$B $W $S"
      if grep -qF "$tag: =>" "$LOG" 2>/dev/null; then echo "  $tag: [건너뜀]"; continue; fi
      f="$RAW/ev_b${B}_${W}_$S.log"; printf "  %s: " "$tag"
      timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KWD/kctype_oja_b$B.npz $X --eval-variant $S > "$f" 2>&1; rc=$?
      if grep -q "^=> DECOMP" "$f" && grep -q "^\[E146 변형\] variant=$S" "$f"; then
        echo "=> mod $(grep '^=> DECOMP' "$f" | sed -E 's/.*mod=([-+0-9.]+).*/\1/') pushed $(grep '^=> DECOMP' "$f" | grep -oE 'pushed=[0-9]+' | cut -d= -f2) 자극줄 $(grep -c '^\[E170 자극\]' "$f")"
      else echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; fi
    done
  done
done
echo "[E170] 전체 루프 종료"
