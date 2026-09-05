#!/bin/bash
# 사전등록 완결성 검사. 미완이면 non-zero로 종료한다.
# 불변식 표기: [x] 준수 / [~] 선언된 이탈(사유 필수) / [ ] 미확인 → 거부
# 사용: bash scripts/lab/gate.sh E070
set -u
R="$(cd "$(dirname "$0")/../.." && pwd)"
ID="${1:?사용: gate.sh E###}"
F="$R/research/experiments/$ID.md"

FAILED=0
fail() { echo "  X $1"; FAILED=1; }

echo "[게이트] $ID 사전등록 검사"
if [ ! -e "$F" ]; then
  echo "  X 사전등록 파일 없음: $F"
  echo "  -> new_experiment.sh 로 먼저 등록하라."
  exit 2
fi

grep -q '<무엇을 바꾸는가' "$F" && fail "1. 조작이 비어 있다"
grep -q '<조작이 실제로 듣는지' "$F" && fail "2. 조작 검증 방법이 비어 있다 (조작 무효 4회 발생)"
grep -q '<구체 수치>' "$F" && fail "4. 사전 판정 기준이 비어 있다 (사후 해석으로 잡음을 신호로 읽은 사례 3건)"
grep -q '<여기 답할 수 없으면' "$F" && fail "5. 새로 알게 되는 것이 비어 있다 -> 실험하지 않는다"

# H### = 현상 가설, M### = 측정 가설(측정 설계 자체를 검증). 둘 다 허용한다.
grep -qE '^- \*\*검증할 가설\*\*:.*[HM][0-9]{3}' "$F" || fail "검증할 가설(H###/M###)이 지정되지 않았다"
grep -qE '^- \*\*경쟁 가설\*\*:.*[HM][0-9]{3}' "$F" || fail "경쟁 가설이 없다 (하나만 보면 그것에 매달린다)"

UNCHECKED=$(grep -cE '^- \[ \] INV-' "$F")
if [ "$UNCHECKED" -gt 0 ]; then
  fail "불변식 확인란 ${UNCHECKED}개가 미확인 (INV-A3 위반으로 결론 2건 철회한 전례)"
  grep -E '^- \[ \] INV-' "$F" | sed 's/^/      /'
fi

DEV=$(grep -cE '^- \[~\] INV-' "$F")
if [ "$DEV" -gt 0 ]; then
  if grep -E '^- \[~\] INV-' "$F" | grep -qv '사유:'; then
    fail "선언된 이탈에 '사유:'가 없다"
  fi
  echo "  ! 선언된 불변식 이탈 ${DEV}건 - 결과 해석에 반드시 명시하라:"
  grep -E '^- \[~\] INV-' "$F" | sed 's/^/      /'
fi

if [ "$FAILED" -ne 0 ]; then
  echo "[게이트] 거부됨. 위 항목을 채운 뒤 다시 실행하라."
  exit 1
fi
echo "[게이트] 통과"
