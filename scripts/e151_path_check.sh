#!/bin/bash
# E151 보정·경로 검사(조건 2) — 기준 고정(logs/E151/criteria_fixed.txt 01:02:17) 뒤. 표본 밖 뇌 15, 차단 상태(E150 인자).
# 규칙(기준 그대로): η ∈ {0.45, 0.9, 1.8, 3.6} 각 A단독 1,500(병렬, 런 디렉터리 분리) → eA1(η) = base(A_η) − base(무학습).
# eA1 ≤ −0.28 인 후보마다 AB 3,000(병렬) → eB(η) = bad(AB_η) − bad(무학습). eA1 ≤ −0.28·eB ≥ +0.20 인 가장 작은 η = η* → logs/E151/eta_star.txt.
# 보정에서 AB base 평가는 하지 않는다(유지 몫 비노출). 무학습 평가는 eta 0.15·3.6 두 번(동결 평가에 eta 가 무관한지 확인).
set -u
R=${E151_R:-/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild}   # 합성 시험(test_e151_path_check.sh)만 바꾼다
RUNROOT=${E151_RUNROOT:-/root}
OUT="$R/research/experiments/logs/E151/pathcheck"; WD="$R/research/experiments/traces/E151/pathcheck"; mkdir -p "$OUT" "$WD"
STAR="$R/research/experiments/logs/E151/eta_star.txt"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
[ -f /root/pygenn_wsl/bin/activate ] && source /root/pygenn_wsl/bin/activate
BASE0="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0 --kc-food-eye-scale 0 --kc-bilateral-scale 0"
RATE="$R/research/experiments/traces/E138/pathcheck/rate_b15.npz"
ETAS="0.45 0.9 1.8 3.6"
tag() { echo "$1" | tr -d '.'; }

train() {  # $1 arm(A|AB) $2 eta
  local t; t=$(tag "$2"); local d="$RUNROOT/e151_cal_${1}_$t"
  mkdir -p "$d" && cd "$d" && cp $R/backend/genesis/*.py . 2>/dev/null
  local X; [ "$1" = "A" ] && X="--episodes 15" || X="--episodes 30 --task-b-after 1500"
  timeout 10800 python reflex_override_task.py $BASE0 --kc-motor-eta $2 $ACT $X --steps 100 --transplant-eval --brain-seed 15 \
    --save-weights $WD/w_${1}_${t}_b15.npz --trace-kc-class $WD/tr_${1}_${t}_b15.npz --kc-rate-file $RATE > "$OUT/train_${1}_${t}_b15.log" 2>&1
}
done_ok() { grep -q "^\[사후\]" "$OUT/train_${1}_$(tag "$2")_b15.log" 2>/dev/null && [ -s "$WD/w_${1}_$(tag "$2")_b15.npz" ]; }
ev() {  # $1 weights-arm(A|AB|none) $2 eta $3 variant  — 무학습은 A_0.45 가중치 파일을 none 모드로(가중치 미적용)
  local t; t=$(tag "$2"); local d="$RUNROOT/e151_cal_ev"
  mkdir -p "$d" && cd "$d" && cp $R/backend/genesis/*.py . 2>/dev/null
  local X
  case $1 in
    none) X="--decomp-weights $WD/w_A_045_b15.npz --decomp-mode none" ;;
    *) X="--decomp-weights $WD/w_${1}_${t}_b15.npz --decomp-mode all" ;;
  esac
  local g="$OUT/ev_b15_${1}_${t}_$3.log"
  timeout 3600 python reflex_override_task.py $BASE0 --kc-motor-eta $2 $ACT --brain-seed 15 $X --eval-variant $3 > "$g" 2>&1
  grep '^=> DECOMP' "$g" | sed -E 's/.*mod=([-+0-9.]+).*/\1/'
}

echo "[E151 보정] 시작 $(date '+%F %T') — 1단계 A단독 4개 병렬"
for E in $ETAS; do ( train A $E ) & done; wait
for E in $ETAS; do done_ok A $E || { echo "  A η=$E 병렬 실패 → 순차 재실행"; tail -2 "$OUT/train_A_$(tag $E)_b15.log"; ( train A $E ); }; done
for E in $ETAS; do
  f="$OUT/train_A_$(tag $E)_b15.log"
  echo "  A η=$E: $(grep 'E109 R-STDP 4방향' "$f" | grep -oE 'eta=[0-9.]+' | sort -u | tr '\n' ' ')| $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f")"
done
NB=$(ev none 0.15 base); NB36=$(ev none 3.6 base); NBAD=$(ev none 0.15 bad)
echo "  무학습 base(eta 0.15) $NB · base(eta 3.6) $NB36 · bad $NBAD"
if [ -z "$NB" ] || [ -z "$NBAD" ]; then echo "[실패] 무학습 평가 없음 — eta_star 미기록"; exit 1; fi
: > "$OUT/a_mods.txt"
for E in $ETAS; do echo "$E $(ev A $E base)" >> "$OUT/a_mods.txt"; done
CANDS=$(python3 - "$OUT/a_mods.txt" "$NB" <<'PY'
import sys
nb = int(round(float(sys.argv[2]) * 1e4)); out = []
for ln in open(sys.argv[1]):
    t = ln.split()
    if len(t) != 2:
        print("ERROR"); sys.exit(0)
    e, m = t
    ea1 = int(round(float(m) * 1e4)) - nb
    print("  eA1(η=%s) = %+.4f %s" % (e, ea1 / 1e4, "후보" if ea1 <= -2800 else ""), file=sys.stderr)
    if ea1 <= -2800:
        out.append(e)
print(" ".join(out))
PY
)
echo "  eA1 ≤ −0.28 후보: [${CANDS}]"
case "$CANDS" in *ERROR*) echo "[실패] A 평가 결측 — eta_star 미기록"; exit 1 ;; esac
if [ -z "$CANDS" ]; then echo "eta_star none" > "$STAR"; echo "[E151 보정] 해당 η 없음 → eta_star none(본실험 없음) $(date '+%F %T')"; exit 0; fi
echo "[E151 보정] 2단계 AB 병렬: $CANDS"
for E in $CANDS; do ( train AB $E ) & done; wait
for E in $CANDS; do done_ok AB $E || { echo "  AB η=$E 병렬 실패 → 순차 재실행"; tail -2 "$OUT/train_AB_$(tag $E)_b15.log"; ( train AB $E ); }; done
: > "$OUT/ab_mods.txt"
for E in $CANDS; do
  f="$OUT/train_AB_$(tag $E)_b15.log"
  echo "  AB η=$E: $(grep 'E109 R-STDP 4방향' "$f" | grep -oE 'eta=[0-9.]+' | sort -u | tr '\n' ' ')| $(grep -E '^\[과제 B\]' "$f" | cut -c1-30) $(grep -oE '보상 [0-9]+회' "$f")"
  echo "$E $(ev AB $E bad)" >> "$OUT/ab_mods.txt"
done
python3 - "$OUT/ab_mods.txt" "$NBAD" "$STAR" <<'PY'
import sys
nbad = int(round(float(sys.argv[2]) * 1e4)); star = None
L = [ln.split() for ln in open(sys.argv[1])]
if any(len(t) != 2 for t in L):
    print("[실패] AB 평가 결측 — eta_star 미기록"); sys.exit(1)
for e, m in L:
    eb = int(round(float(m) * 1e4)) - nbad
    print("  eB(η=%s) = %+.4f %s" % (e, eb / 1e4, "충족" if eb >= 2000 else ""))
    if eb >= 2000 and star is None:
        star = e
open(sys.argv[3], "w").write("eta_star %s\n" % (star if star else "none"))
print("  η* = %s" % (star if star else "none(본실험 없음)"))
PY
cd $R && python3 - <<'PY'
import sys, glob, os, numpy as np
sys.path.insert(0, "scripts")
import judge_e151 as J
for f in sorted(glob.glob("research/experiments/traces/E151/pathcheck/tr_*_b15.npz")):
    ab = "/tr_AB_" in f
    s = J.stats(np.load(f)["rows"], 1500 if ab else None)
    print("  %s 추적: 시행 %d 규칙 일치 %.4f 동결 잔차 %.2e 도파민전 %.1e Σ|Δg| %.3g" % (os.path.basename(f), s["n"], s["agree"], s["res"], s["pre_ratio"], s["sabs"]))
PY
echo "[E151 보정] 종료 $(date '+%F %T')"
