#!/bin/bash
# C56: direct 포화를 **억제로** 푼다 — 영향력과 변별을 동시에.
#
# 상충 관계(실측):
#   d1→direct=20 : D1 영향력 있음(보상 381회) / direct 666 포화 → 변별 없음
#   d1→direct=1  : direct 탈포화(139~197) / **D1 영향력 0**(보상 0회) ← C53 오결론의 원인
# 가중치를 줄이는 대신 direct에 E/I 억제를 걸면 둘 다 만족할 수 있다(d1에서 성공한 방식).
#
# 판정: d1→direct=20 유지하면서 direct 절대발화가 666에서 내려오고 측성차이가 살아나면 성공.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for DI in 0 -30 -100 -200 -400; do
  echo "### direct_inhibition = $DI (d1억제 -200, d1→direct 20 유지) ###"
  timeout 3000 python pathway_transfer_probe.py --real-rstdp --set-d1-weight 30 \
    --d1-inhib -200 --direct-inhib "$DI" --reflex-w 0.5 --seed 0 2>&1 \
    | grep -E "^d1|^direct|^motor|^조향"
done
