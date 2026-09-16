#!/bin/bash
# E082 조작검증: KC→D1 기여율을 30% 이상으로 올릴 수 있는가 (규약 P7 용량-반응).
# K19: 기본 가중치 0.5에서 KC가 D1 발화의 +0.6%만 움직인다.
# 기여율 = (d1 발화 at w) / (d1 발화 at w=0) - 1.
# 붕괴 감시: d1 절대발화가 w=0의 0.5배 미만이면 과도.
# INV-A4/A5는 d1_diversity_probe의 argparse 기본값(-400/-100)으로 박혀 있고 여기서도 명시한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG=/root/kc_run/e082.log
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/kc_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

run_one () {
  printf "kc_d1_w=%-5s seed=%s : " "$1" "$2"
  timeout 2400 python d1_diversity_probe.py --kc-rstdp --kc-d1-w "$1" --seed "$2" \
    --d1-inhib -400 --direct-inhib -100 > "$LOG" 2>&1
  local rc=$?
  local out
  out=$(grep -E "d1_left|d1_right" "$LOG" | tr -s ' ' | tr '\n' '|')
  if [ -z "$out" ]; then echo "[실패 rc=$rc]"; tail -3 "$LOG" | sed 's/^/    /'; else echo "$out"; fi
}

for W in 0 20 60 150 400; do
  run_one "$W" 1
done
