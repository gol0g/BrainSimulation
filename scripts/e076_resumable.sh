#!/bin/bash
# E076 재개 가능 실행기.
# 배경: E074·E075·E076이 모두 장시간 실행 중 종료코드 1로 사망(OOM 흔적 없음, 셸 자체가 죽음).
#       매번 완료분을 잃고 처음부터 돌렸다. 이제 로그를 읽어 **이미 끝난 런은 건너뛴다.**
# 재실행해도 안전하며(멱등), 죽으면 그냥 다시 실행하면 된다.
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
LOG="$R/research/experiments/E076.log"
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
cd /root/rstdp_run
cp $R/backend/genesis/*.py . 2>/dev/null

COMMON="--real-rstdp --crossed --epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60 --learn-sparsity 0.08 --reflex-sparsity 0.15"
STATIC="--epsilon 0.6 --bias 25 --d1-inhib -400 --direct-inhib -100 --reflex-w 3 --episodes 60 --learn-sparsity 0.08 --reflex-sparsity 0.15"

done_already () {  # $1=tag  예: "env5 대칭 b2 수리"
  grep -qF "$1: =>" "$LOG" 2>/dev/null
}

run_one () {  # $1=tag $2=cond(수리|정적) $3=env $4=brain $5=flag
  local tag="$1"
  if done_already "$tag"; then echo "  $tag: [건너뜀]"; return 0; fi
  printf "  %s: " "$tag"
  local ARGS="$COMMON"; [ "$2" = "정적" ] && ARGS="$STATIC"
  local OUT
  OUT=$(timeout 3600 python reflex_override_task.py $ARGS --brain-seed "$4" --env-seed "$3" $5 2>&1)
  echo "$OUT" | grep -E "^=>" || { echo "[실패] $(echo "$OUT" | tail -2 | tr '\n' ' ')"; }
}

for E in 0 5 6 7; do
  for MODE in 원본 대칭; do
    FLAG=""; [ "$MODE" = "대칭" ] && FLAG="--symmetric-env"
    echo "########## env=$E $MODE ##########"
    for B in 0 1 2 3 4; do
      run_one "env$E $MODE b$B 수리" 수리 "$E" "$B" "$FLAG"
      run_one "env$E $MODE b$B 정적" 정적 "$E" "$B" "$FLAG"
    done
  done
done
echo "[E076] 전체 루프 종료"
