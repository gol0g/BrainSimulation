#!/bin/bash
# E204 동작점 보정(기준 고정·경로 검사 뒤, 결과 전) — 표본 밖 뇌 15, 무학습(none) 평가 motor 평균(E193 보정과 같은 측정).
# 기준 m_ref: 게이팅 없음 C2 기저값 −4900(E200 동작점)의 motor 평균(맥락 X·Y, 범주 A·B × train·new, L/R 평균). 측정 가중치 traces/E193/pathcheck/w_b15.npz.
# 후보: 게이팅 켬(w −20·p 0.10·frac 0.5) × 기저값 −4900·−4600·−4300·−4000·−3700·−3400 + [정정 1 — 격자 확장, 본실험 전] −3100·−2800·−2500·−2200.
# 이미 RCEVAL 4줄이 있는 로그는 다시 재지 않는다(같은 커맨드 — 결정론적 평가).
# 규칙(사전 고정): |m_G(T) − m_ref| 가 가장 작은 T(같으면 −4900 에 가까운 쪽) → logs/E204/calib_tonic.txt 'T_G=-XXXX'.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E204/calib"; mkdir -p "$OUT"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e204_run && cd /root/e204_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
for f_ in "$KW15" "$SN15" "$WB"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E204 보정] 종료"; exit 1; }; done
echo "[E204 보정] 시작 $(date '+%F %T')"
for V in U_4900 G_4900 G_4600 G_4300 G_4000 G_3700 G_3400 G_3100 G_2800 G_2500 G_2200; do
  C=${V%_*}; T=${V#*_}
  [ "$C" = "G" ] && EX="$GATE" || EX=""
  for X in X Y; do
    g="$OUT/${V}_${X}_b15.log"
    if [ "$(grep -c '^RCEVAL ' "$g" 2>/dev/null)" = "4" ]; then echo "  ($V $X) [이미 잼 — 건너뜀] motor 평균 $(grep '^RCEVAL ' "$g" | grep -oE 'motor L/R [0-9.]+/[0-9.]+' | awk '{split($3,a,"/"); s+=a[1]+a[2]; n+=2} END {if (n) printf "%.4f", s/n; else print "nan"}')"; continue; fi
    timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $EX --motor-tonic -$T --brain-seed 15 --kc-type-weights $KW15 \
      --decomp-weights $WB --decomp-mode none --rc-eval --rc-eval-ctx $X --trials 96 --eval-diag > "$g" 2>&1; rc=$?
    m=$(grep '^RCEVAL ' "$g" | grep -oE 'motor L/R [0-9.]+/[0-9.]+' | awk '{split($3,a,"/"); s+=a[1]+a[2]; n+=2} END {if (n) printf "%.4f", s/n; else print "nan"}')
    echo "  ($V $X rc=$rc) RCEVAL $(grep -c '^RCEVAL ' "$g") | motor 평균 $m | 구성 줄 $(grep -c 'E204 맥락 게이팅 KC\] 무리' "$g")"
  done
done
python3 - "$OUT" "$R/research/experiments/logs/E204/calib_tonic.txt" <<'PY'
import os, re, sys
out, dst = sys.argv[1:3]
def mean_of(V):
    vals = []
    for X in "XY":
        for ln in open(os.path.join(out, "%s_%s_b15.log" % (V, X)), encoding="utf-8", errors="replace"):
            if ln.startswith("RCEVAL "):
                for a, b in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", ln):
                    vals += [float(a), float(b)]
    if len(vals) != 32:
        raise SystemExit("RCEVAL motor 값 수 %d(32 이어야 — 맥락 2 × 4 조건 × 두 쪽 × L/R)" % len(vals))
    return sum(vals) / len(vals)
ref = mean_of("U_4900")
cand = [(abs(mean_of("G_%d" % T) - ref), abs(T - 4900), T, mean_of("G_%d" % T)) for T in (4900, 4600, 4300, 4000, 3700, 3400, 3100, 2800, 2500, 2200)]
for d, _, T, m in cand:
    print("  후보 −%d: m_G %.4f · |m_G − m_ref| %.4f" % (T, m, d))
best = min(cand)
line = "T_G=-%d m_ref=%.4f m_G=%.4f diff=%.4f" % (best[2], ref, best[3], best[0])
open(dst, "w").write(line + "\n")
print("[E204 보정 선택] " + line + " (m_ref = 게이팅 없음 −4900)")
PY
echo "[E204 보정] 종료 $(date '+%F %T')"
