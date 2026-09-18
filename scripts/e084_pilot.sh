#!/bin/bash
# E084 pilot: --transplant-eval 이 본 과제 경로에서 도는가 + 네 칸 + 재현성.
# 2026-09-19: grep 패턴이 빌드 로그의 "Error Signal"을 잡아 실제 출력을 가렸다. 파일로 받는다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --episodes 3 --real-rstdp --crossed --kc-w-max 750 --transplant-eval"
run () {
  local f=/root/rstdp_run/e084p_$(echo "$1" | tr ' .' '__').log
  printf "%-16s: " "$1"
  timeout 1800 python reflex_override_task.py $BASE --kc-d1-w "$2" $3 \
    --brain-seed 0 --env-seed 0 > "$f" 2>&1
  local rc=$?
  if grep -q "^\[사후\]" "$f"; then
    grep -E "^\[사후\]" "$f"
  else
    echo "[실패 rc=$rc]"; grep -E "Traceback|^[A-Za-z]*Error:|Exception" "$f" | head -3 | sed 's/^/    /'
    tail -2 "$f" | sed 's/^/    /'
  fi
}
run "A_w0.5_KC끔"  0.5 ""
run "B_w0.5_KC켬"  0.5 "--kc-rstdp"
run "C_w150_KC끔"  150 ""
run "D_w150_KC켬"  150 "--kc-rstdp"
echo "--- 재현성: D 재실행 ---"
run "D_재실행"     150 "--kc-rstdp"
