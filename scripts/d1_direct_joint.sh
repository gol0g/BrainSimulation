#!/bin/bash
# C45: 내 실험 행렬의 빈 칸 — d1 탈포화 **와** d1→direct 약화를 **동시에**.
#
# C40: d1→direct를 낮췄으나 그때 d1이 포화(~667 고정)라 보낼 신호가 없었음 → 포화→침묵으로 건너뜀.
# C41/C44: d1을 탈포화했으나 d1→direct=20 유지라 direct가 666으로 포화.
# → 두 조건을 동시에 건 적이 없다. 여기서 채운다.
#
# 판정: direct 측성차이가 반복 가능하게 >20 이면 기저핵 출력단이 변별을 담는 것.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for DD in 20 8 3 1; do
  echo "### d1억제 -200 + d1_to_direct = $DD ###"
  for i in 1 2; do
    timeout 3000 python pathway_transfer_probe.py --real-rstdp --set-d1-weight 30 \
      --d1-inhib -200 --d1-direct-w "$DD" --reflex-w 0.5 2>&1 \
      | grep -E "^d1|^direct|^motor|^조향"
    echo "  ---"
  done
done
