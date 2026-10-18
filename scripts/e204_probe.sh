#!/bin/bash
# E204 개발 진단(사전등록 전 — 짧은 진단 40시행, 판정 자료 아님) — 표본 밖 뇌 15. 새 코드(--ctx-gate-kc)가 도는지·게이팅이 듣는지 먼저 본다.
# p1: 게이팅 끔(E200 플래그, η 0.15, catctx) → traces/E200/pathcheck/w_base_b15.npz 와 배열 정확(새 코드가 E200 경로를 안 바꿈)
# p2: 게이팅 켬(w −20·p 0.10·frac 0.5, catctx, η 0.15) → '[E204 맥락 게이팅 점검]' 3줄·요약(비활성/활성 비)
# p3o·p3n: catrule(맥락 안 씀) 게이팅 끔·켬 → 저장 가중치 배열 정확(새 억제 시냅스가 맥락이 꺼졌을 때 동역학을 안 바꿈)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E204/probe"; WD="$R/research/experiments/traces/E204/probe"; mkdir -p "$OUT" "$WD"
KW15="$R/research/experiments/traces/E160/calib/oja_e0.02_b0.3_b15.npz"
SN15="$R/research/experiments/traces/E176/snap/conn_b15.npz"
REF9="$R/research/experiments/traces/E200/pathcheck/w_base_b15.npz"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e204_run && cd /root/e204_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
FEAT="--conn-snapshot $SN15 --feat-n 200 --feat-w 24 --feat-p 0.10"
CTX="--ctx-n 200 --ctx-w 20 --ctx-p 0.02 --ctx2-n 200 --ctx2-w 20 --ctx2-p 0.02 --motor-tonic -4000"
GATE="--ctx-gate-kc --ctx-gate-w -20 --ctx-gate-p 0.10 --ctx-gate-frac 0.5"
for f_ in "$KW15" "$SN15" "$REF9"; do [ -s "$f_" ] || { echo "[실패] 입력 없음 $f_"; exit 1; }; done
echo "[E204 진단] 시작 $(date '+%F %T')"
for spec in "p1 catctx 0" "p2 catctx 1" "p3o catrule 0" "p3n catrule 1"; do
  set -- $spec; N=$1; TK=$2; ON=$3
  EXTRA=""; [ "$ON" = "1" ] && EXTRA="$GATE"
  f="$OUT/${N}_train_b15.log"
  timeout 1800 python reflex_override_task.py $BASE $ACT $FEAT $CTX --rc-task $TK --episodes 2 --steps 20 --transplant-eval --brain-seed 15 --kc-type-weights $KW15 \
    --save-weights $WD/w_${N}_b15.npz --kc-motor-hard --kc-motor-comp --da-baseline $EXTRA > "$f" 2>&1; rc=$?
  echo "[($N) $TK 게이팅 $ON rc=$rc] $(grep -E '^\[(맥락 )?범주 과제\] 시행' "$f" | cut -c1-90) | $(grep '^\[E204 맥락 게이팅 KC\] 시행' "$f") | 무리 줄 $(grep -c 'E204 맥락 게이팅 KC\] 무리' "$f")"
  grep '^\[E204 맥락 게이팅 점검\]\|E204 맥락 게이팅 KC\] 무리' "$f" | sed 's/^/    /'
  [ $rc -ne 0 ] && tail -4 "$f" | sed 's/^/      /'
done
python3 - "$REF9" "$WD" <<'PY'
import os, sys
import numpy as np
ref, wd = sys.argv[1:3]
def same(x, y):
    if not (os.path.exists(x) and os.path.exists(y)):
        return None, 0
    X, Y = np.load(x), np.load(y)
    return sorted(X.files) == sorted(Y.files) and all(np.array_equal(X[k], Y[k]) for k in X.files), len(X.files)
s1, n1 = same(ref, os.path.join(wd, "w_p1_b15.npz"))
print("[대조 p1] 게이팅 끔 = E200 경로 검사 (b) 가중치(배열 %d): %s" % (n1, {True: "같음", False: "다름", None: "파일 없음"}[s1]))
s3, n3 = same(os.path.join(wd, "w_p3o_b15.npz"), os.path.join(wd, "w_p3n_b15.npz"))
print("[대조 p3] catrule 게이팅 끔 = 켬(배열 %d): %s" % (n3, {True: "같음", False: "다름", None: "파일 없음"}[s3]))
s2, _ = same(os.path.join(wd, "w_p1_b15.npz"), os.path.join(wd, "w_p2_b15.npz"))
print("[대조 p2] catctx 게이팅 켬 ≠ 끔: %s" % ({True: "같음(조작 무효 의심)", False: "다름", None: "파일 없음"}[s2]))
PY
echo "[E204 진단] 종료 $(date '+%F %T')"
