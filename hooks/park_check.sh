#!/bin/bash
# Stop 훅: 파킹 점검. [잠금 — 어시스턴트 자의 수정 금지. 해시 자체검사 내장]
input=$(cat)
if echo "$input" | grep -q '"stop_hook_active"[[:space:]]*:[[:space:]]*true'; then exit 0; fi

D="C:/Users/JungHyun/Desktop/brain/BrainSimulation/.claude"
SELF="$D/hooks/park_check.sh"
REF="$D/hooks/.park_check.sha256"

# 무결성 자체검사: 본문(해시라인 제외) 해시가 기준과 다르면 사용자에게 노출
CUR=$(sha256sum "$SELF" 2>/dev/null | cut -d' ' -f1)
if [ -f "$REF" ] && [ "$CUR" != "$(cat "$REF")" ]; then
  printf '{"decision":"block","reason":"[훅 무결성 경고] park_check.sh가 잠금 시점과 다르다(해시 불일치). 어시스턴트가 감시 장치를 수정했을 수 있다. 사용자에게 즉시 알리고 승인 없이 되돌리거나 재수정하지 마라."}\n'
  exit 0
fi

# 실행 중인 실험/훈련/빌드면 정당한 대기
if wsl -d Ubuntu -u root -- bash -c "pgrep -f 'probe.py|_check.sh|_main.sh|pygenn_test|BrainSimulation-rebuild|forager_brain.py|run_v2_tasks.py|seq_choice_probe|evaluate_concepts|wm_latch_probe|social_repr_probe|cc1plus|nvcc|make'" >/dev/null 2>&1; then
  exit 0
fi

printf '{"decision":"block","reason":"[파킹 점검 훅] 실행 중인 실험이 없다 = 파킹. 우회로는 없다. 즉시 (1) DESIGN_RECOVERY.md와 vault SUMMARY.md에서 \\"다음|미착수|필요\\"를 grep으로 실제 검색하고 (2) 나온 항목 하나를 지금 실행하라. 검색이 정말 비었을 때만 그 grep 출력을 증거로 사용자에게 제시하라. 방향을 사용자에게 떠넘기지 마라."}\n'
exit 0
