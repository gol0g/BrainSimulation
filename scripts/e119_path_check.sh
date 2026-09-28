#!/bin/bash
# E119 실행 경로 검사(abcd-2026-09-28 C절 1~4 + 코드 변경 회귀). 본실험 전 짧은 검사.
#  P1 기준선: 반사 0/25 × 뇌 10·11, --episodes 0 → [사전]/[사후] 무학습 변조폭(반사 0 도달 확인)
#  P2 판독 권한 R: 반사 0 × 뇌 10·11, KC→motor 손 설정 rev(교차 300·같은쪽 0) → R = 기준(대칭 150) − rev
#  P3 학습 경로(뇌 15 — 본실험 표본 밖): 반사 0 학습 / 반사 25 학습 / 반사 0 무학습 → [판정경로]·[반사가중치]·Δg·무학습 0
#  P4 회귀: E118 학습 b0 명령 그대로 → 사후 +0.4033·보상 140 재현(판정 계수·_ex 이동이 경로를 바꾸지 않았는가)
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
OUT="$R/research/experiments/logs/E119"; mkdir -p "$OUT"
WD="$R/research/experiments/traces/E119"; mkdir -p "$WD"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
mkdir -p /root/e119_run && cd /root/e119_run
cp $R/backend/genesis/*.py . 2>/dev/null
BASE="--real-rstdp --crossed --d1-inhib -400 --direct-inhib -100 --kc-motor --kc-motor-sparsity 0.25 --kc-motor-w-max 300 --kc-motor-init-w 150 --kc-motor-eta 0.15 --tau-e 12 --reward-window 2 --reward-stim none --trial-gap 10 --epsilon 0.6 --bias 25 --env-seed 0"
ACT="--act-window 3 --act-current 5000"
run() {  # run <태그> <기대 줄 패턴> <인자...>
  local tag="$1" pat="$2"; shift 2
  local f="$OUT/path_$tag.log"
  timeout 3600 python reflex_override_task.py "$@" > "$f" 2>&1; local rc=$?
  if grep -qE "$pat" "$f"; then
    echo "  $tag: => $(grep -E '^\[사전\]|^\[사후\]|^=> |^\[판정경로\]|^\[반사가중치\]|^\[학습\]|^\[이식\]' "$f" | sed -E 's/^\[이식\] ([0-9]+)개 경로.*/[이식] \1개/' | tr '\n' ' ')"
  else
    echo "  $tag: [실패 rc=$rc]"; tail -3 "$f" | sed 's/^/      /'
  fi
}
echo "[P1 기준선]"
for RW in 0 25; do for B in 10 11; do
  run "base_rw${RW}_b$B" '^\[사후\]' $BASE $ACT --reflex-w $RW --episodes 0 --steps 100 --transplant-eval --brain-seed $B
done; done
echo "[P2 판독 권한]"
for B in 10 11; do
  run "rev_rw0_b$B" '^=> CALIBKM1' $BASE --reflex-w 0 --episodes 0 --brain-seed $B --calib-kc-motor-set rev
done
run "rev_rw25_b10" '^=> CALIBKM1' $BASE --reflex-w 25 --episodes 0 --brain-seed 10 --calib-kc-motor-set rev
echo "[P3 학습 경로 — 뇌 15]"
run "learn_rw0_b15" '^=> 정답률' $BASE $ACT --reflex-w 0 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 --save-weights $WD/path_rw0_b15.npz --dump-kc-weights
run "learn_rw25_b15" '^=> 정답률' $BASE $ACT --reflex-w 25 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 --dump-kc-weights
run "noreward_rw0_b15" '^=> 정답률' $BASE $ACT --reflex-w 0 --episodes 5 --steps 100 --transplant-eval --brain-seed 15 --no-reward --dump-kc-weights
echo "[P4 회귀 — E118 학습 b0 명령]"
run "regress_e118_b0" '^=> 정답률' $BASE $ACT --episodes 5 --steps 100 --transplant-eval --brain-seed 0
echo "[E119 경로 검사] 종료"
