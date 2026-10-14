#!/bin/bash
# E195 경로 검사 2(정정 뒤 — 결과 전) — 표본 밖 뇌 15. 1차 경로 검사에서 --kc-target 0.05·0.02 의 측정값이 소수점까지 같음(조작 무효 — PI 제어기는 실제 KC 발화율이 목표보다 낮아 구동 0).
# 새 조작: --kc-inh-scale(KC 되먹임 억제 가중치 배율, 1 = 이전 −15). 배율 1·1.5·2·3·4 측정(rc-reps 1 = 96 제시) → 장치 억제 g 평균(inh_g)·제시당 활성 KC·기저 발화.
# 주 수준 규칙(사전 고정, 기준 정정 1): 활성 KC 가 배율 1 의 0.8배 이하가 되는 가장 작은 배율 = 주 수준 Sp, 그다음 큰 후보 = 보조 Sr.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E195/pathcheck2"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e195_run && cd /root/e195_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
echo "[E195 경로 검사 2] 시작 $(date '+%F %T')"
for S in 1 1.5 2 3 4; do
  g="$OUT/meas_S${S}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT $CTX --kc-inh-scale $S --decomp-weights $WB --decomp-mode rawcat --rc-expo-eta 0 --rc-measure-ctx --rc-reps 1 > "$g" 2>&1; rc=$?
  n=$(grep '^RCKX ' "$g" | awk '{for(i=1;i<=NF;i++) if($i ~ /^n=/){split($i,a,"="); s+=a[2]; c++}} END {if (c) printf "%.1f", s/c; else print "nan"}')
  echo "[배율 $S rc=$rc] RCKX $(grep -c '^RCKX ' "$g") | 제시당 활성 KC $n | $(grep '^RCKXCHK' "$g" | grep -oE 'base_l=[0-9]+ base_r=[0-9]+|kc_inh_scale=[0-9.]+ cfg_kc_inh_w=[-0-9.]+ inh_g_l=[-0-9.]+ inh_g_r=[-0-9.]+' | tr '\n' ' ') | 배율 줄 $(grep -c '^\[E195 KC 되먹임 억제\]' "$g")"
  [ $rc -ne 0 ] && tail -3 "$g" | sed 's/^/      /'
done
echo "[E195 경로 검사 2] 종료 $(date '+%F %T')"
