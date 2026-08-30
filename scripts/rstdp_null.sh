#!/bin/bash
# C68: **R-STDP 조건의 귀무 분포** — C65 결론의 전제 재검증.
#
# C65에서 "실행 간 잡음 0"을 확정했으나 그건 **정적 조건끼리** 비교한 값이었다.
# C67에서 같은 R-STDP 조건(60ep, seed1, eta 0.02)이 C66의 +0.0031 vs 지금 +0.0026으로 **달랐다.**
# R-STDP 시냅스 동역학은 GPU 원자적 연산을 쓰므로 비결정적일 수 있고,
# 그렇다면 내 귀무 분포는 R-STDP 런의 잡음을 **과소평가**한 것이다.
#
# 방법: 완전히 동일한 R-STDP 조건을 3회 반복해 흩어짐을 측정한다.
# 판정: 반복 간 차이가 |0.005| 이상이면 C64/C65/C66의 효과(-0.005)는 잡음과 구분되지 않는다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

for S in 0 1 2; do
  echo "--- seed=$S (동일 R-STDP 조건 3회) ---"
  for R2 in 1 2 3; do
    printf "  반복%s: " "$R2"
    timeout 3600 python reflex_override_task.py --episodes 60 --seed "$S" \
      --real-rstdp --crossed --epsilon 0.6 --bias 25 \
      --d1-inhib -400 --direct-inhib -400 --reflex-w 3 2>&1 | grep -E "^=>"
  done
done
