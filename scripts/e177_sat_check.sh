#!/bin/bash
# E177 조작검증(측정 타당성, 정정 1 — logs/E177/criteria_fixed.txt) — 맥락 켬 평가가 조향 차이를 표현할 수 있는가(motor 포화 여부).
# 계기: 뇌 10 맥락 켬 평가 두 개(학습·무학습)가 acc 0.0·off ±0.0000·변조폭 약 0 — 조향 = (motor 우 − 좌) × 0.5 이고 발화율 상한이 약 0.6667 이라 양쪽 포화면 0 에 붙는다.
# 뇌마다 E177 가중치(traces/E177/w_bc_b{B}.npz)로 학습·무학습 × 맥락 끔·켬 평가를 --eval-diag(쪽별 평균 motor·KC 발화율, E171 경로 검사된 읽기 전용 옵션)와 함께 다시 잰다.
# 인자는 e177_main.sh 평가 줄과 같다(+ --eval-diag). 본실험과 겹쳐 돌 수 있으므로 실행 디렉터리를 따로 쓴다.
# 사용: bash scripts/e177_sat_check.sh <출력 하위 폴더> <뇌...>   예) probe 10 (01:33 프로브) / main 10 11 12 13 14 (본실험 뒤 판정용)
# 요약 줄(judge_e177_sat.py 와 맞춤): "b10 all on rc=0 | => DECOMP mode=all mod=... | variant=base side=left n=250 motor L/R a/b KC L/R c/d variant=base side=right ..."
set -u
SUB=${1:-probe}; shift
[ $# -gt 0 ] || set -- 10
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E177/sat_check/$SUB"; mkdir -p "$OUT"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e177_satcheck_run && cd /root/e177_satcheck_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
for B in "$@"; do
  KW="$R/research/experiments/traces/E160/kctype_oja_b$B.npz"
  CTX="--conn-snapshot $R/research/experiments/traces/E176/snap/conn_b$B.npz --ctx-n 200 --ctx-w 4 --ctx-p 0.10"
  W="$R/research/experiments/traces/E177/w_bc_b$B.npz"
  [ -s "$W" ] || { echo "b$B [실패 rc=학습 가중치 없음]" | tee -a "$OUT/summary.out"; continue; }
  for M in all none; do
    for C in off on; do
      [ "$C" = "on" ] && XE="--eval-ctx" || XE=""
      g="$OUT/ev_b${B}_${M}_$C.log"
      timeout 1800 python reflex_override_task.py $BASE $ACT --brain-seed $B --kc-type-weights $KW $CTX --decomp-weights $W --decomp-mode $M $XE --eval-diag > "$g" 2>&1; rc=$?
      echo "b$B $M $C rc=$rc | $(grep '^=> DECOMP' "$g" | cut -c1-70) | $(grep '^\[E171 평가 진단\]' "$g" | sed -E 's/^\[E171 평가 진단\] //' | tr '\n' ' ')" | tee -a "$OUT/summary.out"
    done
  done
done
echo "[E177 sat_check] 종료 $(date '+%F %T')" | tee -a "$OUT/summary.out"
