#!/bin/bash
# E142 경로 검사(조건 2) — 판정 기준 고정(logs/E142/criteria_fixed.txt) 뒤. 표본 밖 뇌 15, 반사 25 + 동결 500시행(판정 exec — 본실험과 같은 인자).
# 비교: E119 P5 뇌 15 반사 25 exec 무동결 — 사전 +0.4397, 보상 140, 사후 +0.4208, 변화 −0.0189 (logs/E119/path_exec_rw25_b15.log).
# 확인: 동결 잔차 ≈0, 결정 흔적 살아 있음, 반사 가중치 25→25, [사전] 같음, 판독 값 정상(0/nan 아님).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E142/pathcheck"; WD="$R/research/experiments/traces/E142/pathcheck"; mkdir -p "$OUT" "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e142_run && cd /root/e142_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec"
f="$OUT/F500_b15.log"
echo "[1] 뇌 15 반사 25 + 보상 창 동결 + 계층 추적, 500시행"
timeout 7200 python reflex_override_task.py $BASE $ACT --reflex-w 25 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 --rw-apm-scale 0 \
  --save-weights $WD/w_F500_b15.npz --trace-kc-class $WD/tr_F500_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
grep -E '^\[사전\]|^\[사후\]|^\[학습\]|^=> KCTRACE|^=> 정답률|^\[반사가중치\] good_food' "$f" | cut -c1-420
[ $rc -ne 0 ] && { echo "[실패 rc=$rc]"; tail -3 "$f"; }
echo "[요약] 판정 코드 stats()·reflex_ok() 로"
cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e142 as J
try:
    s = J.stats(np.load("research/experiments/traces/E142/pathcheck/tr_F500_b15.npz")["rows"])
    print("  잔차 %.2e | 결정 흔적 살아 있는 시행 %.3f | B/A %+.3f C/P %+.3f | ΔD %+.4g | 블록 ΔD %s | 도파민전 %.1e | 시행 %d"
          % (s["res"], s["alive"], s["BA"], s["CP"], s["dD"], " ".join("%+.0f" % x for x in s["blk"]), s["pre_ratio"], s["n"]))
except Exception as ex:
    print("  추적 읽기 실패 %s" % ex)
print("  반사 가중치 25→25: %s" % J.reflex_ok("research/experiments/logs/E142/pathcheck/F500_b15.log"))
PY
echo "[E142 경로 검사] 종료"
