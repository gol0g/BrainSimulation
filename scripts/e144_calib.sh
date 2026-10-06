#!/bin/bash
# E144 보정(조건 2 겸 경로 검사) — 기준 고정(logs/E144/criteria_fixed.txt 13:47:56) 뒤. 표본 밖 뇌 15, --episodes 1(100시행, 짧은 진단).
# R0 = 반사 0 · 반대쪽 5000(기준), R25 = 반사 25 · 반대쪽 5000/10000/20000/40000. 행동 창 실행·반대 motor 발화율(추적 열 35·36)과
# 보상 시행 도파민 직전 교차·같은쪽 흔적(열 13·14)을 잰다. 선택 규칙: 열 36 평균 ≤ R0 + 0.01 인 가장 작은 N.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E144/calib"; WD="$R/research/experiments/traces/E144/calib"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e144_run && cd /root/e144_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
for C in R0_5000 R25_5000 R25_10000 R25_20000 R25_40000; do
  RW=${C%%_*}; RW=${RW#R}; N=${C##*_}
  f="$OUT/${C}_b15.log"
  echo "[$C] 뇌 15 반사 $RW · 반대쪽 −$N · 보상 창 동결 · 100시행"
  timeout 3600 python reflex_override_task.py $BASE $ACT --reflex-w $RW --episodes 1 --steps 100 --brain-seed 15 --rw-apm-scale 0 --act-current-neg $N \
    --trace-kc-class $WD/tr_${C}_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
  grep -E '^\[학습\]|^=> KCTRACE3' "$f" | cut -c1-200
  [ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
done
echo "[요약] 열 35(실행)·36(반대) 시행 평균, 보상 시행 흔적 합(교차·같은쪽)"
cd $R && python3 - <<'PY'
import numpy as np
res = {}
for c in ("R0_5000", "R25_5000", "R25_10000", "R25_20000", "R25_40000"):
    try:
        R = np.load("research/experiments/traces/E144/calib/tr_%s_b15.npz" % c)["rows"]
    except Exception as ex:
        print("  %s 읽기 실패 %s" % (c, ex)); continue
    rw = R[:, 7] == 1
    res[c] = float(R[:, 36].mean())
    print("  %-10s 열 %d 시행 %d | 실행 %.4f 반대 %.4f | 보상 시행(%d) 흔적 교차 %+.3e 같은쪽 %+.3e"
          % (c, R.shape[1], len(R), R[:, 35].mean(), R[:, 36].mean(), rw.sum(), R[rw, 13].sum(), R[rw, 14].sum()))
if "R0_5000" in res:
    r0 = res["R0_5000"]
    ok = [int(c.split("_")[1]) for c in ("R25_5000", "R25_10000", "R25_20000", "R25_40000") if c in res and res[c] <= r0 + 0.01]
    print("  기준 R0 = %.4f, 문턱 %.4f → 선택값 %s" % (r0, r0 + 0.01, (min(ok) if ok else "없음(본실험 미실행)")))
PY
echo "[E144 보정] 종료"
