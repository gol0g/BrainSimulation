#!/bin/bash
# E196 경로 검사(결과 전) — 표본 밖 뇌 15. (a) C2 S1 측정(rc-reps 3 = 288 제시) motor 기저값 −4000·−4900 → KC 별 발화가 기저값과 무관한가(같으면 E195 −4000 측정 지수를 −4900 평가에 써도 됨)
# (b) 지수 파일(scripts/kcxor_build.py) (c) 평가 맥락 X × none·cat·sel·eq(E193 과 같은 평가 커맨드 + --decomp-mode kcxor): 오라클 줄·장치 검증 줄·pushed·RCEVAL·포화 점검.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E196/pathcheck"; mkdir -p "$OUT"
WD="$R/research/experiments/traces/E196/pathcheck"; mkdir -p "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e196_run && cd /root/e196_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTXB="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02"
echo "[E196 경로 검사] 시작 $(date '+%F %T')"
for T in 4000 4900; do
  g="$OUT/meas_b15_T$T.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --kc-type-weights $KW15 $FEAT $CTXB --motor-tonic -$T --decomp-weights $WB --decomp-mode rawcat --rc-expo-eta 0 --rc-measure-ctx --rc-reps 3 > "$g" 2>&1; rc=$?
  echo "[(a) 측정 기저값 −$T rc=$rc] RCKX $(grep -c '^RCKX ' "$g") | $(grep '^RCKXCHK' "$g" | grep -oE 'npres=[0-9]+|ctx_on_spikes=[0-9]+|ctx_off_spikes=[0-9]+|feat_inactive_spikes=[0-9]+|g_same=[0-9]+' | tr '\n' ' ')"
  [ $rc -ne 0 ] && tail -3 "$g" | sed 's/^/      /'
done
python3 - "$OUT/meas_b15_T4000.log" "$OUT/meas_b15_T4900.log" <<'PY'
import sys
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/scripts")
import numpy as np, rcxor
a, b = rcxor.parse(sys.argv[1]), rcxor.parse(sys.argv[2])
same = all(len(a[s]) == len(b[s]) and all(x[:5] == y[:5] and np.array_equal(x[5], y[5]) for x, y in zip(a[s], b[s])) for s in ("l", "r"))
md = max(float(np.max(np.abs(x[5] - y[5]))) for s in ("l", "r") for x, y in zip(a[s], b[s]))
print("[(a) 기저값 무관] KC 별 발화(제시 순서·칸 포함) −4000 = −4900: %s (최대 차 %.0f)" % ("같음" if same else "다름", md))
PY
python3 $R/scripts/kcxor_build.py "$OUT/meas_b15_T4000.log" "$WD/sxi_b15.npz" | sed 's/^/[(b)] /'
CTX="$CTXB --motor-tonic -4900"
for W in none cat sel eq; do
  if [ "$W" = "none" ]; then XM="--decomp-mode none"; else XM="--decomp-mode kcxor --kc-xor-file $WD/sxi_b15.npz --kc-xor-arm $W"; fi
  g="$OUT/ev_b15_${W}_X.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WB $XM --rc-eval --rc-eval-ctx X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "[(c) $W X rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep -oE 'pushed=[0-9]+' "$g") | 오라클 $(grep '^\[E196 오라클\] ' "$g" | grep -oE 'arm=[a-z]+ theta=[0-9.]+|cross_l=[0-9]+ same_l=[0-9]+ resp_l=[0-9]+ cross_r=[0-9]+ same_r=[0-9]+ resp_r=[0-9]+' | tr '\n' ' ')| 검증 $(grep -c '^\[E196 오라클 장치 검증\] 일치' "$g") | $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ')| motor $(grep '^RCEVAL ' "$g" | grep -oE 'motor L/R [0-9.]+/[0-9.]+' | awk '{print $3}' | tr '\n' ' ')"
  [ $rc -ne 0 ] && tail -3 "$g" | sed 's/^/      /'
done
echo "[E196 경로 검사] 종료 $(date '+%F %T')"
