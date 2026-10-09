#!/bin/bash
# E162 경로 검사(조건 2) — 기준 고정(logs/E162/criteria_fixed.txt) 뒤. 표본 밖 뇌 15, 망 안 형성 표현(E160 보정 선택 칸 oja_e0.02_b0.3_b15.npz), 짧게:
# A단독 200시행, AB 400시행(200 부터 과제 B) + 평가 5. 확인: 학습·평가 뇌 모두 적재 검증 줄, 동결·규칙 일치, 평가 경로.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E162/pathcheck"; WD="$R/research/experiments/traces/E162/pathcheck"; mkdir -p "$OUT" "$WD"
KW="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e162_run && cd /root/e162_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0 --kc-type-weights $KW"
for ARM in A AB; do
  [ "$ARM" = "A" ] && X="--episodes 2" || X="--episodes 4 --task-b-after 200"
  f="$OUT/train_${ARM}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $X --steps 100 --transplant-eval --brain-seed 15 \
    --save-weights $WD/w_${ARM}_b15.npz --trace-kc-class $WD/tr_${ARM}_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
  echo "[$ARM] 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$f") $(grep -E '^\[과제 B\]' "$f") $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
  [ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
done
for WS in "A base" "AB base" "AB bad" "none base" "none bad"; do
  set -- $WS; W=$1; S=$2
  case $W in
    A) X="--decomp-weights $WD/w_A_b15.npz --decomp-mode all" ;;
    AB) X="--decomp-weights $WD/w_AB_b15.npz --decomp-mode all" ;;
    none) X="--decomp-weights $WD/w_A_b15.npz --decomp-mode none" ;;
  esac
  g="$OUT/ev_b15_${W}_$S.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 $X --eval-variant $S > "$g" 2>&1
  echo "[$W $S] 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$g") $(grep '^=> DECOMP' "$g" | cut -c1-60)"
done
cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e162 as J
for a, sw in (("A", None), ("AB", 200)):
    s = J.stats(np.load("research/experiments/traces/E162/pathcheck/tr_%s_b15.npz" % a)["rows"], sw)
    print("  %s 추적: 시행 %d 규칙 일치 %.4f 동결 잔차 %.2e 도파민전 %.1e" % (a, s["n"], s["agree"], s["res"], s["pre_ratio"]))
PY
echo "[E162 경로 검사] 종료"
