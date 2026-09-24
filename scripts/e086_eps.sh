#!/bin/bash
# 선택을 통한 양성 피드백이 원인인가.
#
# 관측: 대칭 도파민도 RPE 기준선도 전역 폭주를 못 막았다(편향 220~326, 자극 간 차이 2~72).
# 재진단: 한쪽이 우세해지면 선택이 쏠리고 -> 행동 주입도 그쪽만 -> 그쪽만 학습 -> 더 우세.
#   epsilon 0.3 이면 우세한 쪽이 시행의 85%를 가져간다.
# epsilon 1.0 = 완전 무작위 행동. 양쪽이 동등하게 경험되므로, 학습이 되려면
#   **자극-행동-보상 수반성만으로** 되어야 한다. 이것이 학습 규칙 자체의 능력 시험이다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
SEP="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0 --da-neg 1.0 --baseline 0.05"
for EPS in 0.3 0.6 1.0; do
  for S in 0 2; do
    echo "### epsilon=$EPS seed=$S"
    f=/root/minc_run/eps_${EPS}_$S.log
    timeout 2400 python minimal_circuit.py --seed "$S" --trials 400 --block 200 \
      --mode learn --epsilon "$EPS" $SEP > "$f" 2>&1
    rc=$?
    if grep -q "자극 A →" "$f"; then
      grep -E "자극 [AB] →|^=> MINCIRC" "$f" | sed 's/^/  /'
    else
      echo "  [실패 rc=$rc]"; tail -3 "$f" | sed 's/^/    /'
    fi
  done
done
