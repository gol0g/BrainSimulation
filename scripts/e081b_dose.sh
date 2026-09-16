#!/bin/bash
# E081 후속(조작 도달 확인, 규약 P7): KC→D1 가중치 용량-반응.
# 감마 조작이 가중치엔 확실히 적용됐는데(평균0.496 std0.714 max10.7) D1 발화가 시드1에서
# 소수점까지 동일했다. 하드룰: 두 조건이 소수점까지 같으면 조작 무효를 먼저 의심하라.
# 여기서는 조작이 아니라 **경로 자체**를 의심한다 — 가중치를 0과 20으로 벌려도 D1이 같으면
# KC→D1은 D1 발화에 도달하지 않는 것이고, E079(KC는 학습하는데 행동 무변화)가 설명된다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG=/root/kc_run/e081b.log
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/kc_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

for W in 0 0.5 5 20; do
  printf "kc_d1_w=%-4s : " "$W"
  timeout 2400 python d1_diversity_probe.py --kc-rstdp --kc-d1-w "$W" --seed 1 > "$LOG" 2>&1
  rc=$?
  out=$(grep -E "d1_left|d1_right|kc_left" "$LOG" | tr -s ' ' | tr '\n' '|')
  if [ -z "$out" ]; then echo "[실패 rc=$rc]"; tail -3 "$LOG" | sed 's/^/    /'; else echo "$out"; fi
done
