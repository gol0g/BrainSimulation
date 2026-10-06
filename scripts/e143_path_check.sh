#!/bin/bash
# E143 경로 검사(조건 2) — 판정 기준 고정(logs/E143/criteria_fixed.txt) 뒤. 표본 밖 뇌 15, 반사 25, 500시행, 판정 exec.
# [1] 회귀: E142 경로 검사(F500 b15)와 같은 인자 — 새 흔적 읽기(시행 시작·결정 단계 끝)가 동역학을 바꾸지 않는가
#     (E142 경로 검사: 보상 140, 사후 +0.1459, 효과 −0.2938, ΔD +1,595,934).
# [2] 결정 단계 + 보상 창 동결(--dec-apm-scale 0): M1·M1d 잔차 ≈0, 결정 흔적 살아 있음, 반사 25→25, 판독 정상.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E143/pathcheck"; WD="$R/research/experiments/traces/E143/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e143_run && cd /root/e143_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
for C in REG DEC; do
  [ "$C" = "DEC" ] && X="--dec-apm-scale 0" || X=""
  f="$OUT/${C}_b15.log"
  echo "[$C] 뇌 15 반사 25 + 보상 창 동결 $X + 계층 추적, 500시행"
  timeout 7200 python reflex_override_task.py $BASE $ACT --reflex-w 25 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 --rw-apm-scale 0 $X \
    --save-weights $WD/w_${C}_b15.npz --trace-kc-class $WD/tr_${C}_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
  grep -E '^\[사전\]|^\[사후\]|^\[학습\]|^=> KCTRACE |^=> KCTRACE3|^=> 정답률|^\[반사가중치\] good_food' "$f" | cut -c1-300
  [ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
done
echo "[요약] 판정 코드 stats()·reflex_ok() 로"
cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e143 as J
for c in ("REG", "DEC"):
    try:
        s = J.stats(np.load("research/experiments/traces/E143/pathcheck/tr_%s_b15.npz" % c)["rows"])
        print("  %s 열 %d | 보상 창 잔차 %.2e | 결정 단계 잔차 %.2e | 흔적 살아 있음 %.3f | B/A %+.3f C/P %+.3f | ΔD %+.4g | 블록 ΔD %s | 도파민전 %.1e | 시행 %d"
              % (c, s["ncol"], s["res"], s["resd"], s["alive"], s["BA"], s["CP"], s["dD"], " ".join("%+.0f" % x for x in s["blk"]), s["pre_ratio"], s["n"]))
    except Exception as ex:
        print("  %s 추적 읽기 실패 %s" % (c, ex))
    print("  %s 반사 가중치 25→25: %s" % (c, J.reflex_ok("research/experiments/logs/E143/pathcheck/%s_b15.log" % c)))
a, b = np.load("research/experiments/traces/E143/pathcheck/w_REG_b15.npz"), np.load("research/experiments/traces/E142/pathcheck/w_F500_b15.npz")
nd = sum(int((np.asarray(a[k]) != np.asarray(b[k])).sum()) for k in a.files if k in b.files)
mx = max(float(np.abs(np.asarray(a[k], dtype=np.float64) - np.asarray(b[k], dtype=np.float64)).max()) for k in a.files if k in b.files and np.asarray(a[k]).size)
print("  REG vs E142 경로 검사 가중치: 다른 시냅스 %d, 최대 |차| %.4g (잡음 바닥 참고: E139 재현 검사 2374~4130·1.5002)" % (nd, mx))
PY
echo "[E143 경로 검사] 종료"
