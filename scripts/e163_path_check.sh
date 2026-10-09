#!/bin/bash
# E163 경로 검사(조건 2) — 기준 고정 뒤. 표본 밖 뇌 15, 망 안 형성 표현(E160 보정 선택 칸), 짧게: --episodes 4 --reverse-after 200.
# 확인: '[반전] 시행 200 부터' 줄, 적재 2줄, 반사 0→0, 추적 규칙 일치(200 전 교차·뒤 같은 쪽)·동결 잔차.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E163/pathcheck"; WD="$R/research/experiments/traces/E163/pathcheck"; mkdir -p "$OUT" "$WD"
KW="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e163_run && cd /root/e163_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
f="$OUT/rev_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 4 --reverse-after 200 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW \
  --save-weights $WD/w_rev_b15.npz --trace-kc-class $WD/tr_rev_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
echo "[rev rc=$rc] $(grep -E '^\[반전\]' "$f" | cut -c1-60) | 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') | 반사 $(grep '^\[반사가중치\] good_food_to_motor' "$f" | grep -oE 'w_mean \S+' | tr '\n' ' ')"
cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e163 as J
J.REV_AT = 200
s = J.rev_stats(np.load("research/experiments/traces/E163/pathcheck/tr_rev_b15.npz")["rows"])
print("  추적: 시행 %d 규칙 일치 %.4f 동결 잔차 %.2e 도파민전 %.1e 블록 보상 %s" % (s["n"], s["agree"], s["res"], s["pre_ratio"], s["rew_blk"]))
PY
echo "[E163 경로 검사] 종료"
