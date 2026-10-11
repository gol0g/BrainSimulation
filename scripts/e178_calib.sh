#!/bin/bash
# E178 보정(사전 선택 규칙 — logs/E178/criteria_fixed.txt) — 표본 밖 뇌 15, E160 보정 칸 형성 가중치, 초기 연결 스냅숏 conn_b15, 맥락 전용 집단 zz_ctx w 4(E177 과 같음). 학습 없음(평가만).
# motor 긴장성 억제 X ∈ {0(기준), −500, −1000, −2000, −4000}(motor 좌·우 Ioffset 기저값, --motor-tonic)마다 이식 평가 4개 + --eval-diag(쪽별 motor·KC 발화율):
#   무학습(none) 맥락 끔·켬 — motor 동작점, 학습 A(E176 경로 검사 (a) 가중치 w_a_b15 — 교차 규칙 200시행) 맥락 끔·켬 — 학습한 사상이 조향으로 나타나는가.
# 선택은 scripts/e178_pick.py → logs/E178/pick.txt('X=<값>' 또는 'X=none').
# 요약 줄: "[X=-1000 none on rc=0] => DECOMP mode=none mod=... | side=left n=250 motor L/R a/b KC L/R c/d side=right ..."
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E178/calib"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WA="$R/research/experiments/traces/E176/pathcheck/w_a_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e178_run && cd /root/e178_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
CTX="--conn-snapshot $SN15 --ctx-n 200 --ctx-w 4 --ctx-p 0.10"
for f_ in "$KW15" "$SN15" "$WA"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; exit 1; }; done
for X in 0 -500 -1000 -2000 -4000; do
  for M in none all; do
    for C in off on; do
      [ "$C" = "on" ] && XE="--eval-ctx" || XE=""
      g="$OUT/ev_x${X}_${M}_$C.log"
      timeout 1800 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $CTX --motor-tonic $X \
        --decomp-weights $WA --decomp-mode $M $XE --eval-diag > "$g" 2>&1; rc=$?
      echo "[X=$X $M $C rc=$rc] $(grep '^=> DECOMP' "$g" | cut -c1-70) | $(grep '^\[E171 평가 진단\]' "$g" | sed -E 's/^\[E171 평가 진단\] variant=base //' | tr '\n' ' ') | 동작점 줄 $(grep -c '^\s*\[motor 동작점\]' "$g") 맥락 평가 줄 $(grep -c '^\[맥락 평가\]' "$g")"
    done
  done
done
cd $R && python3 scripts/e178_pick.py
echo "[E178 보정] 종료 $(date '+%F %T')"
