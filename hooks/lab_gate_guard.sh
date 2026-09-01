#!/bin/bash
# PreToolUse(Bash) 훅 — 연구실 게이트 우회 차단.
#
# 왜: research/ 연구실을 만들었지만 게이트는 내가 자발적으로 run_experiment.sh를 부를 때만 작동했다.
# 그런데 이 세션이 증명한 것은 **내가 내 규칙을 안 지킨다**는 것이다(규약 P8을 작성 1시간 만에 위반).
# 그래서 실험 진입점을 직접 실행하는 것을 훅으로 막는다.
#
# 차단 대상: 장기 훈련/측정 진입점을 사전등록 없이 직접 실행하는 것.
# 통과: 짧은 진단(--episodes 1~9), 조작검증 프로브, run_experiment.sh 경유 실행.
set -u
INPUT=$(cat)
CMD=$(printf '%s' "$INPUT" | grep -o '"command"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/^"command"[[:space:]]*:[[:space:]]*"//; s/"$//')

# run_experiment.sh 경유면 통과 (게이트를 이미 거쳤다)
case "$CMD" in *run_experiment.sh*|*lab/gate.sh*|*lab/audit.sh*|*new_experiment.sh*) exit 0 ;; esac

# 실험 진입점 목록
ENTRY=""
case "$CMD" in
  *reflex_override_task.py*) ENTRY="reflex_override_task.py" ;;
  *run_v2_tasks.py*)         ENTRY="run_v2_tasks.py" ;;
  *evaluate_concepts.py*)    ENTRY="evaluate_concepts.py" ;;
esac
[ -z "$ENTRY" ] && exit 0

# 짧은 진단은 허용 (--episodes 0~9)
if printf %s "$CMD" | grep -qE -- "--episodes[= ]+[0-9]{2,}"; then :; elif printf %s "$CMD" | grep -qE -- "--episodes[= ]+[0-9]"; then exit 0; fi

# e070_check 류 조작검증 스크립트는 허용
case "$CMD" in *_check.sh*|*synapse_count_probe.py*|*pathway_transfer_probe.py*|*load_damage_audit.py*) exit 0 ;; esac

cat <<'MSG' >&2
[연구실 게이트] 사전등록 없이 실험 진입점을 직접 실행하려 했다.

이 프로젝트에서 사전등록 없이 돌린 실험은 결론을 5회 뒤집었다.
조작이 듣는지 확인하지 않은 채 낸 결론이 4건, 그중 2건은 발표 후 철회했다.

절차:
  1) bash scripts/lab/new_experiment.sh E### "제목"
  2) research/experiments/E###.md 의 1~5번을 채운다
     (가설 / 조작검증 방법 / 불변식 확인 / 사전 판정기준 / 이 결과로 새로 알게 되는 것)
  3) bash scripts/lab/run_experiment.sh E### "커맨드"

짧은 진단은 --episodes 0~9 로 실행하면 통과한다.
MSG
exit 2
