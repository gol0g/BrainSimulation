#!/bin/bash
# E172 경로 검사 보충(재현성) — 경로 검사 rev 400(뇌 15) 사후 +0.2726 이 E163 경로 검사 +0.2724 와 0.0002 달랐다(보상 221 같음, A·AB 는 4자리 일치).
# 같은 현재 코드로 같은 rev 400 을 한 번 더 돌려, 같은 코드끼리도 KC→motor 가중치가 비슷한 정도로 갈리는지(비결정성) 본다. 인자는 e172_path_check.sh 의 rev 와 같다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E172/pathcheck"; WD="$R/research/experiments/traces/E172/pathcheck/rep1"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
RT15="$R/research/experiments/traces/E138/pathcheck/rate_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e172_run && cd /root/e172_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
f="$OUT/rep1_rev_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 4 --reverse-after 200 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
  --save-weights $WD/w_rev_b15.npz --trace-kc-class $WD/tr_rev_b15.npz --kc-rate-file $RT15 > "$f" 2>&1; rc=$?
echo "[rep1 rev rc=$rc] $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
echo "  비교 대상: E172 경로 검사 rev +0.2726 보상 221회 · E163 경로 검사 rev +0.2724"
echo "[E172 재현성 검사] 종료"
