#!/bin/bash
# 연구실 위생 점검: 미완 등록, 결과 미기입, current-state 신선도.
set -eu
R="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$R"
echo "=== 연구실 감사 ==="

echo "-- 사전등록 상태 --"
for f in research/experiments/E*.md; do
  [ -e "$f" ] || { echo "  (등록된 실험 없음)"; break; }
  id=$(basename "$f" .md)
  if bash scripts/lab/gate.sh "$id" >/dev/null 2>&1; then st="등록완료"; else st="**미완**"; fi
  # 2026-10-04 수정: E119 이후 기록은 "## 실행 결과" 아래 "- **판정…**" 글머리로 적는다 — 제목(### 판정)만 찾으면 채운 기록도 미기입으로 보였다.
  # 결과 있음 = "### 판정|결과" 제목, 또는 실행 결과 절에 서식 원문("사전 기준 대비 지지/기각")이 아닌 "- **판정" 줄.
  if grep -qE "^### (판정|결과)" "$f" || awk '/^## 실행 결과/{s=1} s && /^- \*\*판정/ && $0 !~ /사전 기준 대비 지지\/기각$/ {f=1} END{exit !f}' "$f"; then res="결과있음"; else res="결과미기입"; fi
  [ -e "research/experiments/$id.log" ] && lg="로그있음" || lg="미실행"
  printf "  %-8s %-10s %-10s %s\n" "$id" "$st" "$lg" "$res"
done

echo "-- 활성 가설 --"
grep -l "상태\*\*: \*\*활성" research/hypotheses/H*.md 2>/dev/null | while read -r h; do
  echo "  $(basename "$h" .md): $(head -1 "$h" | sed 's/^# //')"
done || echo "  (활성 가설 없음 — 연구가 멈춰 있다는 뜻이다)"

echo "-- current-state 신선도 --"
CS_DATE=$(grep -oE '\*\*최종 갱신\*\*: [0-9-]+' research/current-state.md | grep -oE '[0-9-]+$' || echo "없음")
echo "  최종 갱신: $CS_DATE"
LAST_EXP=$(ls -t research/experiments/E*.log 2>/dev/null | head -1 || true)
[ -n "$LAST_EXP" ] && echo "  최근 실험 로그: $(basename "$LAST_EXP") ($(date -r "$LAST_EXP" +%Y-%m-%d 2>/dev/null || echo '?'))"
echo "  → 최근 실험이 current-state 갱신보다 새로우면 P9 위반이다."
