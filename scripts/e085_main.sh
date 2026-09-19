#!/bin/bash
# E085: 이득 효과가 다른 경로 학습의 증폭인가. 재개 가능(P13).
#
# P  kc_d1_w=0.5  가소성 전부 끔 (--no-reward)
# Q  kc_d1_w=0.5  다른경로 학습 켬 (--real-rstdp --crossed)
# R  kc_d1_w=150  가소성 전부 끔
# S  kc_d1_w=150  다른경로 학습 켬
# KC 학습(--kc-rstdp)은 네 칸 모두 끈다.
# 출력은 파일로 받고 기대한 줄이 없으면 실패로 찍는다(규약 P17).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E085.log"
RAW="$R/research/experiments/logs/E085"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null

# 네 칸 공통으로 --real-rstdp --crossed 를 켜 **배선을 동일하게** 둔다(검토 지적 #3).
BASE="--d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --episodes 60 --kc-w-max 750 --transplant-eval --real-rstdp --crossed"

done_already () { grep -qF "$1: =>" "$LOG" 2>/dev/null; }
run_one () {  # tag  가소성(off/other)  brain  kcw
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local PL="--no-reward"; [ "$2" = "other" ] && PL=""
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 3600 python reflex_override_task.py $BASE $PL --kc-d1-w "$4" \
    --brain-seed "$3" --env-seed 0 --dump-kc-weights > "$f" 2>&1
  local rc=$?
  grep -E "^=>" "$f" || { echo "[실패 rc=$rc]"; tail -2 "$f" | sed 's/^/      /'; }
}

for KCW in 0.5 150; do
  for PL in off other; do
    echo "########## kc_d1_w=$KCW / 가소성=$PL ##########"
    for B in 0 1 2 3 4; do
      run_one "w$KCW $PL b$B" "$PL" "$B" "$KCW"
    done
  done
done
echo "[E085] 전체 루프 종료 — 원본 로그: $RAW"
