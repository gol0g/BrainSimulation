#!/bin/bash
# C43: 네 수리 + **선천반사 약화** — 행렬의 마지막 칸.
#
# C42: ①②③④를 다 갖춰도 6런 전부 0.0%. 학습 전후 오프셋도 동일(+0.001→+0.001).
#      원인 후보: 이 시험에서 선천반사를 25.0 그대로 뒀다. 반사가 15000시냅스×25로 운동을 독점하면
#      기저핵이 학습해도 그 위에 얹히는 수준에 그친다.
# 반사를 낮춰가며 학습된 기저핵이 행동을 잡는 지점이 있는지 본다.
# 대조: 같은 반사 약화 + 학습기계 없음(정적) → 효과가 학습 때문인지 반사 약화 때문인지 귀속.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
EP="${1:-40}"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run

run2 () {
  local label="$1"; shift
  echo "=== $label ==="
  for i in 1 2; do
    timeout 3600 python reflex_override_task.py --episodes "$EP" "$@" 2>&1 \
      | grep -E "사전|학습\]|사후|=>"
    echo "  ---"
  done
}

run2 "전부+반사3"   --real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -200 --reflex-w 3
run2 "전부+반사0.5" --real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -200 --reflex-w 0.5
run2 "대조:정적+반사0.5" --epsilon 0.6 --bias 25 --d1-inhib -200 --reflex-w 0.5
