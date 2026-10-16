#!/bin/bash
# E198 경로 검사(결과 전) — 표본 밖 뇌 15. 맥락 X × none·lsgnR·lsgnA(E197 과 같은 평가 + --kc-xor-arm lsgnR|lsgnA, 학습값 = E179 경로 검사 가중치 — L4 학습 아님, 효과 값은 의미 없음)
# 점검: 오라클 줄 수·'[E198 학습 부호]' agree·'[E198 연결 확인]' n·pre_sum = 판정기 함수(judge_e198.deltas·expect)로 다시 센 값, 장치 검증 줄·pushed·RCEVAL·포화.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E198/pathcheck"; mkdir -p "$OUT"
NPZ="$R/research/experiments/traces/E197/pathcheck/sxi_b15.npz"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E179/pathcheck/w_bc2_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e198_run && cd /root/e198_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4900"
echo "[E198 경로 검사] 시작 $(date '+%F %T')"
for W in none lsgnR lsgnA; do
  if [ "$W" = "none" ]; then XM="--decomp-mode none"; else XM="--decomp-mode kcxor --kc-xor-file $NPZ --kc-xor-arm $W"; fi
  g="$OUT/ev_b15_${W}_X.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CTX --brain-seed 15 --kc-type-weights $KW15 --decomp-weights $WB $XM --rc-eval --rc-eval-ctx X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
  echo "[$W X rc=$rc] RCEVAL $(grep -c '^RCEVAL ' "$g") | $(grep -oE 'pushed=[0-9]+' "$g") | 오라클 $(grep '^\[E196 오라클\] ' "$g" | grep -oE 'arm=[A-Za-z]+|cross_l=[0-9]+ same_l=[0-9]+ resp_l=[0-9]+ cross_r=[0-9]+ same_r=[0-9]+ resp_r=[0-9]+' | tr '\n' ' ')| $(grep -oE 'agree_l=[0-9]+ agree_r=[0-9]+' "$g") | 검증 $(grep -c '^\[E196 오라클 장치 검증\] 일치' "$g") | $(grep '^=> DECOMP' "$g" | grep -oE '[AB]_(train|new)=[-+0-9.]+' | tr '\n' ' ')| motor $(grep '^RCEVAL ' "$g" | grep -oE 'motor L/R [0-9.]+/[0-9.]+' | awk '{print $3}' | tr '\n' ' ')"
  [ $rc -ne 0 ] && tail -3 "$g" | sed 's/^/      /'
done
python3 - "$OUT" "$NPZ" "$WB" "$SN15" <<'PY'
import sys, os, re
sys.path.insert(0, "/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild/scripts")
import judge_e198 as J
from judge_e196 import read_eval
out, npz, wb, sn = sys.argv[1:5]
D, conn = J.deltas(wb, sn)
for w in ("lsgnR", "lsgnA"):
    p = os.path.join(out, "ev_b15_%s_X.log" % w)
    r = read_eval(p, "X"); ec, ea = J.expect(npz, w, D); lr, cn = J.extra_lines(p)
    o = r["orc"] or {}
    good = all(int(o.get(k, -1)) == v for k, v in ec.items()) and lr is not None and (lr[1], lr[2]) == (ea["agree_l"], ea["agree_r"]) and cn == conn and r["ver"] >= 1 and r["pushed"] == 4 and not r["sat"] and len(r["rc"]) == 4 and not r["bad"]
    print("[대조 %s] 오라클 수·부호 일치·연결 확인 = 판정기 재계산: %s (일치율 %.3f·%.3f)" % (w, "같음" if good else "다름", ea["agree_l"] / max(ea["den_l"], 1), ea["agree_r"] / max(ea["den_r"], 1)))
PY
echo "[E198 경로 검사] 종료 $(date '+%F %T')"
