#!/bin/bash
# E206 동작점 격자 확장(criteria_fixed.txt [정정 1] — 본실험 전, 뇌 15 표본 밖): 처음 격자(−4900 … −2200)의 최소가 격자 끝(−2200)이고 |m_F − m_U| 가 문턱 0.0170 을 넘을 때만.
# F15·게이팅·맥락 X·Y 무학습 평가(경로 검사 (m) 과 같은 측정) × 기저값 −1900·−1600·−1300·−1000·−700·−400 → 전체 격자(−4900 … −400)에서 |m_F(T) − m_U(−2800)| 최소(같으면 −2800 에 가까운 쪽).
# 처음 결과는 logs/E206/calib_tonic_grid1.txt 로 옮겨 두고(지우지 않음) 새 결과를 logs/E206/calib_tonic.txt 에 쓴다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E206/pathcheck"; WD="$R/research/experiments/traces/E206/pathcheck"
CAL="$R/research/experiments/logs/E206/calib_tonic.txt"; CAL1="$R/research/experiments/logs/E206/calib_tonic_grid1.txt"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
WB="$R/research/experiments/traces/E193/pathcheck/w_b15.npz"
F15="$WD/feat_formed_b15.npz"
for f_ in "$KW15" "$SN15" "$WB" "$F15" "$CAL1"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; echo "[E206 격자 확장] 종료"; exit 1; }; done
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e206_run && cd /root/e206_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CW="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
echo "[E206 격자 확장] 시작 $(date '+%F %T') — 처음 결과: $(cat "$CAL1")"
for T in 1900 1600 1300 1000 700 400; do
  for X in X Y; do
    g="$OUT/m_F_${T}_${X}_b15.log"
    if [ "$(grep -c '^RCEVAL ' "$g" 2>/dev/null)" = "4" ]; then continue; fi
    timeout 3600 python reflex_override_task.py $BASE $ACT $FEAT $CW $GATE --feat-weights $F15 --motor-tonic -$T --brain-seed 15 --kc-type-weights $KW15 \
      --decomp-weights $WB --decomp-mode none --rc-eval --rc-eval-ctx $X --rc-seed 38300 --trials 96 --eval-diag > "$g" 2>&1
    echo "  (m F_$T $X) RCEVAL $(grep -c '^RCEVAL ' "$g") | 특징 적재 $(grep -c '^\[E184 특징 가중치 적재\]' "$g") | 동작점 $(grep -oE '기저값 -?[0-9.]+' "$g" | sort -u | tr '\n' ' ') | 포화 칸 $(grep '^RCEVAL ' "$g" | grep -cE 'motor L/R 0\.(6[6-9]|[7-9])[0-9]*/0\.(6[6-9]|[7-9])')"
  done
done
python3 - "$OUT" "$CAL" <<'PY'
import os, re, sys
out, cal = sys.argv[1:3]
def mean_of(V):
    v = []
    for X in "XY":
        for ln in open(os.path.join(out, "m_%s_%s_b15.log" % (V, X)), encoding="utf-8", errors="replace"):
            if ln.startswith("RCEVAL "):
                for a, b in re.findall(r"motor L/R ([0-9.]+)/([0-9.]+)", ln):
                    v += [float(a), float(b)]
    if len(v) != 32:
        raise SystemExit("[실패] %s motor 값 수 %d(32 이어야)" % (V, len(v)))
    return sum(v) / len(v)
mu = mean_of("U_2800")
cand = []
for T in (4900, 4600, 4300, 4000, 3700, 3400, 3100, 2800, 2500, 2200, 1900, 1600, 1300, 1000, 700, 400):
    m = mean_of("F_%d" % T)
    cand.append((abs(m - mu), abs(T - 2800), T, m))
    print("  후보 −%d: m_F %.4f · |m_F − m_U| %.4f" % (T, m, abs(m - mu)))
d, _, T, m = min(cand)
line = "T_F=-%d m_U=%.4f m_F=%.4f diff=%.4f (규칙: [정정 1] 넓힌 격자 −4900 … −400 최소 — 처음 격자 −4900 … −2200 의 최소가 끝 −2200 이고 문턱 0.0170 초과)" % (T, mu, m, d)
open(cal, "w").write(line + "\n"); print("[동작점 규칙 — 정정 1] " + line)
if d > 0.0170:
    print("[주의] 넓힌 격자에서도 |m_F − m_U| %.4f > 0.0170 — 동작점 맞춤 미달(본실험 전에 기록·결정)" % d)
PY
echo "[E206 격자 확장] 종료 $(date '+%F %T')"
