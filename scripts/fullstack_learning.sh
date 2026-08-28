#!/bin/bash
# C42: 네 차단요인을 **모두** 해소한 상태의 반사역전 학습 — 이 프로젝트에서 학습이 행동을 바꾸는가.
#
# ① 학습기계: --real-rstdp   (시냅스별 자격흔적, C36 검증: std 0→7.5)
# ② 위상감금: --crossed      (food_eye_L→D1_R 교차 R-STDP, C36)
# ③ 탐색부재: --epsilon/--bias (무작위 ε-탐욕, C38b)
# ④ 기저핵포화: --d1-inhib -200 (C41: d1 667고정→168~590, direct 변별 88.9/204.0 출현)
#
# 대조: ④만 뺀 조건 = 포화가 진짜 결정적이었는지 귀속.
# 사전기준: 3회 평균 (사후−사전) > +10%p 이면 학습 성립.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-40}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

run3 () {
  local label="$1"; shift
  echo "=== $label ==="
  for i in 1 2 3; do
    timeout 3600 python reflex_override_task.py --episodes "$EP" "$@" 2>&1 \
      | grep -E "사전|학습\]|D1가중치|사후|=>"
    echo "  ---"
  done
}

run3 "①②③④ 전부(포화해소 포함)" --real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -200
run3 "①②③ (④ 포화해소 없음)"    --real-rstdp --crossed --epsilon 0.6 --bias 25
