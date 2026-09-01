#!/bin/bash
# Stop 훅 — 규약 P9(결과 반영) 미이행 차단.
# 실험 로그가 current-state.md 보다 새로우면, 결과를 반영하지 않고 응답을 끝내려는 것이다.
set -u
[ "${CLAUDE_STOP_HOOK_ACTIVE:-}" = "1" ] && exit 0
R="C:/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild"
CS="$R/research/current-state.md"
[ -e "$CS" ] || exit 0
NEWEST=$(ls -t "$R"/research/experiments/E*.log 2>/dev/null | head -1)
[ -n "$NEWEST" ] || exit 0
[ "$NEWEST" -nt "$CS" ] || exit 0
# 아직 실행 중이면 통과 (결과가 안 나왔다)
if wsl bash -lc 'ps aux | grep -c "[p]ython"' 2>/dev/null | grep -qvE '^0$'; then exit 0; fi
cat <<MSG >&2
[연구실 P9] 실험 결과를 반영하지 않았다.

  최근 실험 로그: $(basename "$NEWEST")  (current-state.md 보다 새로움)

규약 P9: 실험이 끝나면 **세 곳**을 갱신한다.
  1) research/experiments/E###.md  — 결과와 사전기준 대비 판정
  2) research/hypotheses/H###.md   — 신뢰도와 증거
  3) research/current-state.md     — 현재 믿음 (덮어쓰기, 최종 갱신일 포함)
그리고 docs/research/DESIGN_RECOVERY.md 에 실행 로그를 append.

갱신하지 않으면 다음 세션은 옛 믿음 위에서 연구를 재개하게 된다.
MSG
exit 2
