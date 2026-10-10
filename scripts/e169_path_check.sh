#!/bin/bash
# E169 경로 검사(조건 2) — 기준 고정 뒤. 표본 밖 뇌 15, E161 경로 검사 형성 가중치.
# FW1(보상 창 1, 동결 없음)·FFW1(보상 창 1, 동결) 1ep: 적재 2, [사전] = E161 경로 검사 F [사전] +0.0228, 보상 창 끝 흔적 잔차(r₁ = (11/12)^10) — FFW1 ≈ 0(창 1·동결), FW1 ≥ 0.05.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E169/pathcheck"; WD="$R/research/experiments/traces/E169/pathcheck"; mkdir -p "$OUT" "$WD"
P161="$R/research/experiments/traces/E161/pathcheck"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e169_run && cd /root/e169_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 1 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0"
LD='^\[E153 종류 입력 적재\].*검증 일치'
B=15
for ARM in FW1 FFW1; do
  [ "$ARM" = "FFW1" ] && XA="--rw-apm-scale 0" || XA=""
  f="$OUT/${ARM}_b$B.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 1 --steps 100 --transplant-eval --brain-seed $B --kc-type-weights $P161/kctype_oja_b$B.npz $XA \
    --save-weights $WD/w_${ARM}_b$B.npz --trace-kc-class $WD/tr_${ARM}_b$B.npz --kc-rate-file $P161/rate_b$B.npz > "$f" 2>&1
  echo "[($ARM) 1ep rc=$?] 적재 $(grep -c "$LD" "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | KCTRACE3 $(grep -c '^=> KCTRACE3' "$f")"
done
cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e169 as J
for a in ("FW1", "FFW1"):
    s = J.stats(np.load("research/experiments/traces/E169/pathcheck/tr_%s_b15.npz" % a)["rows"])
    r2 = float(np.abs(np.load("research/experiments/traces/E169/pathcheck/tr_%s_b15.npz" % a)["rows"][:, 21:25] - ((11/12)**20) * np.load("research/experiments/traces/E169/pathcheck/tr_%s_b15.npz" % a)["rows"][:, 13:17]).sum())
    print("    %s 추적 %d 도파민 전 %.2e 잔차(r₁ 창 1 기준) %.3e | 창 2 기준이라면 잔차 합 %.1f" % (a, s["n"], s["pre_ratio"], s["res1"], r2))
PY
echo "  기대: 적재 2, [사전] +0.0228, FFW1 잔차 ≤ 1e-3(창 1·동결), FW1 잔차 ≥ 0.05"
echo "[E169 경로 검사] 종료"
