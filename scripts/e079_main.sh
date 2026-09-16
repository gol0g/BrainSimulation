#!/bin/bash
# E079 본 실험: KC 신용할당이 행동을 바꾸는가. 재개 가능(P13).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E079.log"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

COMMON="--real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"
STATIC="--epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60"

done_already () { grep -qF "$1: =>" "$LOG" 2>/dev/null; }
run_one () {  # tag kind brain extra
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  printf "  %s: " "$tag"
  local ARGS="$COMMON"; [ "$2" = "정적" ] && ARGS="$STATIC"
  local OUT
  OUT=$(timeout 3600 python reflex_override_task.py $ARGS --brain-seed "$3" --env-seed 0 $4 2>&1)
  echo "$OUT" | grep -E "^=>" || echo "[실패] $(echo "$OUT" | tail -2 | tr '\n' ' ')"
}

for MODE in 기본 KC학습; do
  EXTRA=""; [ "$MODE" = "KC학습" ] && EXTRA="--kc-rstdp"
  echo "########## $MODE ##########"
  for B in 0 1 2 3 4; do
    run_one "$MODE b$B 수리" 수리 "$B" "$EXTRA"
    run_one "$MODE b$B 정적" 정적 "$B" "$EXTRA"
  done
done
echo "[E079] 전체 루프 종료"
