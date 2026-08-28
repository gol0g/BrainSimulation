#!/bin/bash
# C41: D1 포화 해소 — 기저핵 입구에서 자극 변별이 살아나는가.
#
# C40 정정: d1 측성차이가 5런에서 30.9/-0.0/-0.2/0.3/0.4 → 첫 런만 컸고 나머지는 0(= 단일런 결론은 오류).
# 절대발화가 자극·설정과 무관하게 ~667로 고정 = **d1 자체가 포화**.
# 포화된 집단은 정보를 담을 수 없으므로 하류(direct)를 아무리 조정해도 소용없다.
#
# C14에서 만들어둔 d1 E/I 배선(d1→억제뉴런→d1)의 강도를 스윕해 변별이 나타나는지 본다.
# 각 설정 3회 반복 — 단일런 결론이 방금 틀렸으므로.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for INH in 0 -30 -100 -200; do
  echo "### d1_inhibition = $INH ###"
  for i in 1 2 3; do
    timeout 3000 python pathway_transfer_probe.py --real-rstdp --set-d1-weight 30 \
      --d1-inhib "$INH" 2>&1 | grep -E "^d1|^direct|^조향"
    echo "  ---"
  done
done
