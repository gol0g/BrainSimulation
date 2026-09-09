#!/bin/bash
# E076: H008 인과검증 — 좌우 대칭 강제. E075 완료를 기다렸다가 실행한다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate

# E075가 끝날 때까지 대기 (GPU 경합 방지)
while pgrep -f "reflex_override_task.py" >/dev/null 2>&1; do sleep 120; done
echo "[E076] E075 종료 확인, 시작"

cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null
rm -rf forager_brain_CODE CODE

echo "=== 조작검증: 전달 생존 ==="
timeout 2400 python reflex_override_task.py --real-rstdp --crossed --epsilon 0.6 \
  --bias 300 --bias-at-d1 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 \
  --episodes 6 --brain-seed 0 --env-seed 0 2>&1 | grep -E "학습\]"

COMMON="--real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60 --learn-sparsity 0.08 --reflex-sparsity 0.15"
STATIC="--epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60 --learn-sparsity 0.08 --reflex-sparsity 0.15"

for E in 0 5 6 7; do
  for SYM in "원본 " "대칭 --symmetric-env"; do
    set -- $SYM
    LBL="$1"; FLAG="${2:-}"
    echo "########## env=$E $LBL ##########"
    for B in 0 1 2 3 4; do
      printf "  env%s %s b%s 수리: " "$E" "$LBL" "$B"
      OUT=$(timeout 3600 python reflex_override_task.py $COMMON --brain-seed "$B" --env-seed "$E" $FLAG 2>&1)
      echo "$OUT" | grep -E "^=>" || { echo "[실패]"; echo "$OUT" | tail -3; }
      printf "  env%s %s b%s 정적: " "$E" "$LBL" "$B"
      OUT=$(timeout 3600 python reflex_override_task.py $STATIC --brain-seed "$B" --env-seed "$E" $FLAG 2>&1)
      echo "$OUT" | grep -E "^=>" || { echo "[실패]"; echo "$OUT" | tail -3; }
    done
  done
done
