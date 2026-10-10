#!/bin/bash
# E175 조작검증 프로브(경로 검사 보충, 보정 전) — 정정 1 경로 검사 (c)에서 I_ab −40 도 assoc_binding 발화를 못 줄임(끔 = 켬 20000, KC 수치도 +2 때와 동일).
# assoc_binding 은 assoc_edible(120)·assoc_context(100)에서 DENSE w 10 입력을 받아 크게 구동되는 것으로 보인다 — 침묵에 필요한 전류 크기를 찾는다.
# 표본 밖 뇌 15, kcctx 60 제시, I_ab ∈ {−500, −2000, −8000}. 판정 아님(조작 도달 범위 확인).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E175/probe"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
W0="$R/research/experiments/traces/E175/pathcheck/w_a_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e175_run && cd /root/e175_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
[ -s "$W0" ] || { echo "[실패] 경로 검사 가중치 없음 $W0"; exit 1; }
for I in -500 -2000 -8000; do
  f="$OUT/kcctx_i${I}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 --ctx-ab-i $I \
    --decomp-weights $W0 --decomp-mode kcctx --trials 60 > "$f" 2>&1
  echo "[I_ab=$I rc=$?] $(grep '^=> KCCTX' "$f" | cut -c1-560)"
done
echo "[E175 프로브] 종료"
