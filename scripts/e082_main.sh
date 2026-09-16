#!/bin/bash
# E082 본 실험 (b): KC 경로 영향력을 올리면 행동이 바뀌는가. 재개 가능(P13).
#
# 두 팔 모두 --kc-rstdp + --kc-w-max 750. 다른 것은 --kc-d1-w 0.5(기본) vs 150(고영향력)뿐.
# w_max를 양쪽 동일하게 두는 이유: 한쪽만 올리면 영향력과 학습 여지가 동시에 바뀐다.
# INV-A4/A5는 COMMON/STATIC에 명시 (규약 P15).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E082.log"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

BASE="--d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --reflex-w 3 --episodes 60 --kc-rstdp --kc-w-max 750"
COMMON="--real-rstdp --crossed $BASE"
STATIC="$BASE"

done_already () { grep -qF "$1: =>" "$LOG" 2>/dev/null; }
run_one () {  # tag kind brain kcw
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  printf "  %s: " "$tag"
  local ARGS="$COMMON"; [ "$2" = "정적" ] && ARGS="$STATIC"
  local OUT
  OUT=$(timeout 3600 python reflex_override_task.py $ARGS --brain-seed "$3" --env-seed 0 --kc-d1-w "$4" 2>&1)
  echo "$OUT" | grep -E "^=>" || echo "[실패] $(echo "$OUT" | tail -2 | tr '\n' ' ')"
}

for MODE in 기본0.5 고영향력150; do
  KCW=0.5; [ "$MODE" = "고영향력150" ] && KCW=150
  echo "########## $MODE (kc_d1_w=$KCW) ##########"
  for B in 0 1 2 3 4; do
    run_one "$MODE b$B 수리" 수리 "$B" "$KCW"
    run_one "$MODE b$B 정적" 정적 "$B" "$KCW"
  done
done
echo "[E082] 전체 루프 종료"
