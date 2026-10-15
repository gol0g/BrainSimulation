#!/bin/bash
# E197 경로 검사(결과 전) — 표본 밖 뇌 15. (a) E196 경로 검사의 −4900 측정 로그(logs/E196/pathcheck/meas_b15_T4900.log)로 지수 파일
# (b) 평가 맥락 X × none·sgnR·sgnA(E196 과 같은 평가 커맨드 + --decomp-mode kcxor --kc-xor-arm sgnR|sgnA): 오라클 줄(수 = 파일)·장치 검증 줄·pushed·RCEVAL·포화 점검.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E197/pathcheck"; mkdir -p "$OUT"
WD="$R/research/experiments/traces/E197/pathcheck"; mkdir -p "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e197_run && cd /root/e197_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4900"
echo "[E197 경로 검사] 시작 $(date '+%F %T')"
python3 $R/scripts/kcxor_build.py "$R/research/experiments/logs/E196/pathcheck/meas_b15_T4900.log" "$WD/sxi_b15.npz" | sed 's/^/[(a)] /'
for W in none sgnR sgnA; do
  if [ "$W" = "none" ]; then XM="--decomp-mode none"; else XM="--decomp-mode kcxor --kc-xor-file $WD/sxi_b15.npz --kc-xor-arm $W"; fi
  g="$OUT/ev_b15_${W}_X.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WB $XM --rc-eval --rc-eval-ctx X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "[(b) $W X rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep -oE 'pushed=[0-9]+' "$g") | 오라클 $(grep '^\[E196 오라클\] ' "$g" | grep -oE 'arm=[A-Za-z]+ theta=[0-9.]+|cross_l=[0-9]+ same_l=[0-9]+ resp_l=[0-9]+ cross_r=[0-9]+ same_r=[0-9]+ resp_r=[0-9]+' | tr '\n' ' ')| 검증 $(grep -c '^\[E196 오라클 장치 검증\] 일치' "$g") | $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ')| motor $(grep '^RCEVAL ' "$g" | grep -oE 'motor L/R [0-9.]+/[0-9.]+' | awk '{print $3}' | tr '\n' ' ')"
  [ $rc -ne 0 ] && tail -3 "$g" | sed 's/^/      /'
done
echo "[E197 경로 검사] 종료 $(date '+%F %T')"
