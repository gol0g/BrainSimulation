#!/bin/bash
# E195 경로 검사(조건 2) — 기준 logs/E195/criteria_fixed.txt 고정 뒤. 표본 밖 뇌 15.
# 새 코드: forager_brain kc_target(PI 목표, 기본 0.05 = 이전), 과제 --kc-target, RCKXCHK kc_target·cfg_kc_target.
# (a) 기본값 회귀: E184 경로 검사 (b) U 평가 learn(w_U_b15) → A_train +0.0046 A_new +0.0349 B_train +0.0659 B_new +0.0468 정확(kc_target 기본 경로 불변).
# (b) 측정 C2 × kc_target 0.05·0.02(rc-reps 1 = 96 제시) → RCKXCHK kc_target = cfg_kc_target, 0.02 목표 줄, 제시당 활성 KC 가 0.02 에서 줄어드는가.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E195/pathcheck"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WU="$R/research/experiments/traces/E184/pathcheck/w_U_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e195_run && cd /root/e195_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
for f_ in "$KW15" "$SN15" "$WU" "$WB"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E195 경로 검사] 종료"; exit 1; }; done
echo "[E195 경로 검사] 시작 $(date '+%F %T')"
g="$OUT/a_reg_learn_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WU --decomp-mode all --rc-eval --trials 96 --eval-diag > "$g" 2>&1; rc=$?
echo "[(a) 회귀 learn rc=$rc] $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ') | E195 줄 $(grep -c '^\[E195 KC 항상성\]' "$g")"
echo "  기대 (a): A_train=+0.0046 A_new=+0.0349 B_train=+0.0659 B_new=+0.0468, E195 줄 0"
for T in 0.05 0.02; do
  g="$OUT/b_meas_T${T}_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT $CTX --kc-target $T --decomp-weights $WB --decomp-mode rawcat --rc-expo-eta 0 --rc-measure-ctx --rc-reps 1 > "$g" 2>&1; rc=$?
  n=$(grep '^RCKX ' "$g" | awk '{for(i=1;i<=NF;i++) if($i ~ /^n=/){split($i,a,"="); s+=a[2]; c++}} END {if (c) printf "%.1f", s/c; else print "nan"}')
  echo "[(b) 측정 T $T rc=$rc] RCKX $(grep -c '^RCKX ' "$g") | 제시당 활성 KC 평균 $n | $(grep '^RCKXCHK' "$g" | grep -oE 'kc_target=[0-9.]+ cfg_kc_target=[0-9.]+|base_l=[0-9]+ base_r=[0-9]+' | tr '\n' ' ') | E195 줄 $(grep '^\[E195 KC 항상성\]' "$g" | head -n 1)"
done
echo "  기대 (b): RCKX 96, kc_target = cfg_kc_target(0.050·0.020), 0.02 에서 E195 줄 1·제시당 활성 KC 감소"
echo "[E195 경로 검사] 종료 $(date '+%F %T')"
