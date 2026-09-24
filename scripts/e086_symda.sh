#!/bin/bash
# 보상/벌 비대칭이 전역 폭주의 원인인가.
#
# 관측: learn 조건에서 자극별 변별(11->37)보다 **전역 편향**(11->258)이 훨씬 크게 자랐다.
# 산수: 보상 +1.0 / 벌 -0.5 이면 정답률 50%에서 기댓값이 +0.25로 **양수**다.
#   -> 모든 가중치가 올라가고, 흔적이 큰 쪽이 더 빨리 올라가 폭주한다.
# 대칭(-1.0)으로 바꾸면 기댓값이 0이 되어 폭주가 사라져야 한다.
# 전체 모델도 같은 비대칭(+1.0 / -0.5)을 쓴다 — 여기서 확인되면 그쪽도 봐야 한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/minc_run && cd /root/minc_run
cp $R/backend/genesis/minimal_circuit.py $R/backend/genesis/rstdp_model.py . 2>/dev/null
SEP="--sens-kc-p 0.02 --kc-inh 12.0 --sens-kc-w 4.0"
for NEG in 0.5 1.0; do
  for S in 0 2; do
    echo "### da_neg=$NEG seed=$S"
    f=/root/minc_run/sym_${NEG}_$S.log
    timeout 2400 python minimal_circuit.py --seed "$S" --trials 300 --block 150 \
      --mode learn --da-neg "$NEG" $SEP > "$f" 2>&1
    rc=$?
    if grep -q "자극 A →" "$f"; then
      grep -E "자극 [AB] →|^=> MINCIRC" "$f" | sed 's/^/  /'
    else
      echo "  [실패 rc=$rc]"; tail -3 "$f" | sed 's/^/    /'
    fi
  done
done
