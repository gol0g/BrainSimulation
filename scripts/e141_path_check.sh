#!/bin/bash
# E141 경로 검사(조건 2) — 판정 기준 고정(logs/E141/criteria_fixed.txt) 뒤. 표본 밖 뇌 15(E139·E140 경로 검사와 같은 인자 — 판정 v).
# [1] --rw-apm-scale 1(동적화만, 값 불변): E139 경로 검사 뇌 15(보상 237, [사후] −0.0734)와 같아야 한다 — 동적 파라미터 경로 회귀.
# [2] --rw-apm-scale 0(동결): 잔차 Σ|e_end − r*·e_da|/Σ|e_da| ≈ 0, 결정 단계 흔적 살아 있음, B/A·C/P 부호, 효과(무동결 뇌 15: −0.0989).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E141/pathcheck"; WD="$R/research/experiments/traces/E141/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e141_run && cd /root/e141_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000"
for S in 1 0; do
  f="$OUT/scale${S}_b15.log"
  echo "[scale $S] 뇌 15 학습 + 보상 창 A± 배율 $S + 계층 추적"
  timeout 7200 python reflex_override_task.py $BASE $ACT --reflex-w 0 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 --rw-apm-scale $S \
    --save-weights $WD/w_s${S}_b15.npz --trace-kc-class $WD/tr_s${S}_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
  grep -E '^\[사전\]|^\[사후\]|^\[학습\]|^=> KCTRACE|^=> 정답률' "$f" | cut -c1-420
  [ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
done
echo "[요약] 판정 코드 stats() 로 — 무동결 기준 = E139 경로 검사 뇌 15(traces/E139/pathcheck/tr_b15.npz)"
cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e141 as J
base = J.stats(np.load("research/experiments/traces/E139/pathcheck/tr_b15.npz")["rows"])
for nm, f in (("E139 무동결", "research/experiments/traces/E139/pathcheck/tr_b15.npz"),
              ("scale 1", "research/experiments/traces/E141/pathcheck/tr_s1_b15.npz"),
              ("scale 0", "research/experiments/traces/E141/pathcheck/tr_s0_b15.npz")):
    try:
        s = J.stats(np.load(f)["rows"])
    except Exception as ex:
        print("  %s: 읽기 실패 %s" % (nm, ex)); continue
    print("  %-11s 잔차 %.2e | 결정흔적/E139 %.3f | B/A %+.3f C/P %+.3f | ΔD %+.4g | 도파민전 %.1e | 시행 %d"
          % (nm, s["res"], s["eda_abs"] / base["eda_abs"], s["BA"], s["CP"], s["dD"], s["pre_ratio"], s["n"]))
w1 = "research/experiments/traces/E141/pathcheck/w_s1_b15.npz"; w0 = "research/experiments/traces/E139/pathcheck/w_b15.npz"
try:
    a, b = np.load(w1), np.load(w0)
    nd = sum(int((np.asarray(a[k]) != np.asarray(b[k])).sum()) for k in a.files if k in b.files)
    mx = max(float(np.abs(np.asarray(a[k], dtype=np.float64) - np.asarray(b[k], dtype=np.float64)).max()) for k in a.files if k in b.files and np.asarray(a[k]).size)
    print("  scale 1 vs E139 경로 검사 가중치: 다른 시냅스 %d, 최대 |차| %.4g (잡음 바닥: E139 재현 검사 2374~4130, 1.5002)" % (nd, mx))
except Exception as ex:
    print("  가중치 비교 실패 %s" % ex)
PY
echo "[E141 경로 검사] 종료"
