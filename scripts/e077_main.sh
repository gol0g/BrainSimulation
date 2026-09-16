#!/bin/bash
# E077: env6의 무엇이 다른가 — 난이도 축 조작. 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E077.log"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

echo "=== 조작검증: 전달 생존 ==="
timeout 2400 python reflex_override_task.py --real-rstdp --crossed --epsilon 0.6 \
  --bias 300 --bias-at-d1 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 \
  --episodes 6 --brain-seed 0 --env-seed 0 2>&1 | grep -E "학습\]"

COMMON="--real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"
STATIC="--epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"

done_already () { grep -qF "$1: =>" "$LOG" 2>/dev/null; }

run_one () {  # $1=tag $2=kind $3=brain $4=extra
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  printf "  %s: " "$tag"
  local ARGS="$COMMON"; [ "$2" = "정적" ] && ARGS="$STATIC"
  local OUT
  OUT=$(timeout 3600 python reflex_override_task.py $ARGS --brain-seed "$3" --env-seed 0 $4 2>&1)
  echo "$OUT" | grep -E "^=>" || echo "[실패] $(echo "$OUT" | tail -2 | tr '\n' ' ')"
}

for CFG in "A기준 " "B개수 --n-food 70" "C품질 --food-ratio 0.85" "D둘다 --n-food 70 --food-ratio 0.85"; do
  set -- $CFG
  LBL="$1"; shift; EXTRA="$*"
  echo "########## $LBL ##########"
  for B in 0 1 2 3 4; do
    run_one "$LBL b$B 수리" 수리 "$B" "$EXTRA"
    run_one "$LBL b$B 정적" 정적 "$B" "$EXTRA"
  done
done
echo "[E077] 전체 루프 종료"
