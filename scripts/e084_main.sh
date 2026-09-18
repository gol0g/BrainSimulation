#!/bin/bash
# E084: KC 경로 영향력 × KC 학습 — 2×2 요인설계.  ★E082를 대체한다
#
# E082의 설계 결함(외부 검토 2026-09-18, 4/4 사실 확인):
#   (2) STATIC 에도 --kc-rstdp 가 있어 (수리−정적) 뺄셈에서 KC 효과가 상쇄된다.
#       => 여기서는 **KC 학습 자체를 요인으로** 올린다. 다른 경로 가소성은 두 팔에서 동일하게 고정.
#   (3) 탐색이 motor 에 주입돼 학습 시냅스 자격흔적에 안 남는다(C52 교훈 미적용).
#       => --bias-at-d1 로 학습 시냅스 하류 첫 단계에 주입.
#   (4) grep "^=>" 한 줄만 남겨 절대 변조폭·보상횟수·가중치가 소실된다.
#       => **런별 원본 stdout 전량**을 logs/E084/ 에 파일로 남기고, 요약만 화면에 낸다.
# INV-B4(평가 중 가소성 동결)는 reset() 수정으로 확보됨. INV-A4/A5 명시.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E084.log"
RAW="$R/research/experiments/logs/E084"
mkdir -p "$RAW"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

# 두 팔 공통. 다른 경로 가소성(real_rstdp/crossed)은 **양쪽 동일하게 켠다** — 고정 요인.
BASE="--d1-inhib -400 --direct-inhib -100 --epsilon 0.6 --bias 25 --bias-at-d1 --reflex-w 3 --episodes 60 --real-rstdp --crossed --kc-w-max 750 --transplant-eval"

done_already () { grep -qF "$1: =>" "$LOG" 2>/dev/null; }
run_one () {  # tag  kc학습(on/off)  brain  kcw
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  local KC=""; [ "$2" = "on" ] && KC="--kc-rstdp"
  local f="$RAW/$(echo "$tag" | tr ' ' '_').log"
  printf "  %s: " "$tag"
  timeout 3600 python reflex_override_task.py $BASE $KC --kc-d1-w "$4" \
    --brain-seed "$3" --env-seed 0 --dump-kc-weights > "$f" 2>&1
  local rc=$?
  grep -E "^=>" "$f" || echo "[실패 rc=$rc] $(tail -2 "$f" | tr '\n' ' ')"
}

for KCW in 0.5 150; do
  for KCL in off on; do
    echo "########## kc_d1_w=$KCW / KC학습=$KCL ##########"
    for B in 0 1 2 3 4; do
      run_one "w$KCW KC$KCL b$B" "$KCL" "$B" "$KCW"
    done
  done
done
echo "[E084] 전체 루프 종료 — 원본 로그: $RAW"
