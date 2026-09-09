#!/bin/bash
# E076 조작검증: 좌우 대칭 강제가 실제로 asym을 0에 가깝게 만드는가.
# 다른 특성(개수·품질비율)은 보존되어야 한다 — 선택적 조작이어야 인과 검증이 성립.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
for E in 0 2 5 6 7; do
  printf "env%s 원본  : " "$E"
  timeout 600 python env_characteristics.py --env-seed "$E" 2>&1 | grep -E "n_good=|asym" | tr '\n' ' '
  echo
  printf "env%s 대칭  : " "$E"
  timeout 600 python env_characteristics.py --env-seed "$E" --symmetric 2>&1 | grep -E "n_good=|asym" | tr '\n' ' '
  echo
done
