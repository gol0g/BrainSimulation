#!/bin/bash
# E202 진단(판정 밖 — 본실험 b10 계측 파일의 재구성 실패(상관 0.58·기울기 0.13)를 본 뒤 2026-10-17 작성): 계측 Σx 와 학습 Δ 가 어디서 갈리는가.
# 뇌 15(표본 밖), E202 경로 검사와 같은 커맨드에서 η 와 시행 구조만 바꾼다 — 짧은 진단(에피소드 ≤ 9).
#   D1 η 2.4 · 2×20(40시행)  — η 만 바꿈(경로 검사는 η 0.15 에서 정확)
#   D2 η 2.4 · 1×200        — 에피소드 경계 없음
#   D3 η 2.4 · 4×50         — 에피소드 경계 3번(같은 200시행)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E202/diag"; WD="$R/research/experiments/traces/E202/diag"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e202_diag && cd /root/e202_diag
cp $R/backend/genesis/*.py . 2>/dev/null
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
echo "[E202 진단] 시작 $(date '+%F %T')"
for spec in "D1 2.4 2 20" "D2 2.4 1 200" "D3 2.4 4 50"; do
  set -- $spec; N=$1; ETA=$2; EP=$3; ST=$4
  BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta $ETA --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
  f="$OUT/${N}_train_b15.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task catctx --episodes $EP --steps $ST --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
    --save-weights $WD/w_${N}_b15.npz --kc-motor-hard --kc-motor-comp --da-baseline --trace-dg $WD/dg_${N}_b15.npz > "$f" 2>&1; rc=$?
  echo "[$N η $ETA ${EP}×${ST} rc=$rc] $(grep '^\[E202 계측\] 시행 [0-9]* · ' "$f" | sed 's/ · 저장.*//')"
  python3 - "$WD/w_${N}_b15.npz" "$WD/dg_${N}_b15.npz" "$SN15" "$N" <<'PY'
import sys
import numpy as np
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/scripts")
import verify_e198_independent as V
w, dgp, sn, nm = sys.argv[1:5]
D = np.load(dgp)
dl, _ = V.learned(w, sn)
m = np.concatenate([D["tot_x_" + s] for s in "lr"]); l = np.concatenate([np.asarray(dl[s], float) for s in "lr"])
nz = (m != 0) | (l != 0)
r = float(np.corrcoef(m[nz], l[nz])[0, 1]) if nz.sum() > 2 else float("nan")
sl = float((m * l).sum() / max((m * m).sum(), 1e-300))
print("    %s 재구성: 최대 오차 %.3e · 상관 %.4f · 기울기 %.4f · 계측 |평균| %.3f · 학습 |평균| %.3f" % (nm, float(np.max(np.abs(m - l))), r, sl, float(np.abs(m).mean()), float(np.abs(l).mean())))
PY
done
echo "[E202 진단] 종료 $(date '+%F %T')"
