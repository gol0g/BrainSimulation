#!/bin/bash
# E153 경로 검사·η 보정(조건 2) — 기준 고정(logs/E153/criteria_fixed.txt 19:25:45) 뒤. 표본 밖 뇌 15.
# 1) kcdev η 0.1 → sel_med 두 쪽 ≥ 0.8 이면 η=0.1, 아니면 η 0.3 한 번 더(사전 규칙), 그래도 미달이면 eta none(본실험 없음).
# 2) 고른 η 의 형성 가중치를 실은 kcoverlap(J) — 적재 줄. 3) 학습 100시행에 적재 — 학습·이식 평가 뇌 적재 줄, 반사 0, 동결.
# 4) 음성 대조: 뇌 14 에 뇌 15 형성 가중치 → '연결이 저장본과 다르다' 로 중단해야 한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E153/pathcheck"; WD="$R/research/experiments/traces/E153/pathcheck"; mkdir -p "$OUT" "$WD"
ETAF="$R/research/experiments/logs/E153/eta.txt"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e153_run && cd /root/e153_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000 --judge exec --reflex-w 0 --rw-apm-scale 0"
WS="$R/research/experiments/traces/E141/pathcheck/w_s0_b15.npz"
dev() {   # $1 eta $2 tag
  local f="$OUT/dev_b15_$2.log"
  timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --decomp-weights $WS --decomp-mode kcdev --kc-dev-n 100 --kc-dev-eta $1 \
    --kc-dev-save $WD/kctype_b15_$2.npz > "$f" 2>&1
  grep '^=> KCDEV' "$f" || { echo "[실패] kcdev η $1"; tail -5 "$f"; }
}
selok() { echo "$1" | python3 -c 'import re,sys; v=[float(x) for x in re.findall(r"sel_med=([0-9.]+)", sys.stdin.read())]; print(1 if len(v) == 2 and min(v) >= 0.8 else 0)'; }
echo "[E153 경로 검사] 시작 $(date '+%F %T')"
L=$(dev 0.1 eta01); echo "[dev η 0.1] $L"
if [ "$(selok "$L")" = "1" ]; then ETA=0.1; TAG=eta01
else
  L3=$(dev 0.3 eta03); echo "[dev η 0.3] $L3"
  if [ "$(selok "$L3")" = "1" ]; then ETA=0.3; TAG=eta03; else ETA=none; TAG=; fi
fi
echo "eta $ETA" > "$ETAF"; echo "  η = $ETA"
if [ "$ETA" = "none" ]; then echo "[E153 경로 검사] 형성 안 됨 → eta none(본실험 없음)"; echo "[E153 경로 검사] 종료"; exit 0; fi
KW="$WD/kctype_b15_$TAG.npz"
f="$OUT/ov_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 15 --decomp-weights $WS --decomp-mode kcoverlap --trials 200 --kc-type-weights $KW > "$f" 2>&1
echo "[ov] 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$f") | $(grep '^=> KCOVERLAP' "$f" | cut -c1-330)"
f="$OUT/train_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --episodes 1 --steps 100 --transplant-eval --brain-seed 15 --kc-type-weights $KW \
  --save-weights $WD/w_b15.npz --trace-kc-class $WD/tr_b15.npz --kc-rate-file $R/research/experiments/traces/E138/pathcheck/rate_b15.npz > "$f" 2>&1; rc=$?
echo "[train rc=$rc] 적재 $(grep -c '^\[E153 종류 입력 적재\].*검증 일치' "$f") | $(grep '^\[사전\]' "$f" | grep -oE '변조폭 [-+0-9.]+') → $(grep '^\[사후\]' "$f" | grep -oE '변조폭 [-+0-9.]+') $(grep -oE '보상 [0-9]+회' "$f") | $(grep '^\[반사가중치\]' "$f" | grep -oE 'w_mean \S+' | tr '\n' ' ')"
f="$OUT/neg_b14_with_b15.log"
timeout 3600 python reflex_override_task.py $BASE $ACT --brain-seed 14 --decomp-weights $R/research/experiments/traces/E141/w_b14.npz --decomp-mode kcoverlap --trials 8 --kc-type-weights $KW > "$f" 2>&1
echo "[neg 뇌 14 ← 뇌 15 가중치] 중단 메시지 $(grep -c '연결이 저장본과 다르다' "$f")개(1 이어야 함)"
cd $R && python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "scripts")
import judge_e153 as J
s = J.stats(np.load("research/experiments/traces/E153/pathcheck/tr_b15.npz")["rows"])
print("  추적: 시행 %d 규칙 일치 %.4f 동결 잔차 %.2e 도파민전 %.1e" % (s["n"], s["agree"], s["res"], s["pre_ratio"]))
PY
echo "[E153 경로 검사] 종료"
