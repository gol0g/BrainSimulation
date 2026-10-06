#!/bin/bash
# e151_path_check.sh 합성 시험: 가짜 python(답을 아는 학습·평가 출력)으로 보정 규칙 전체 경로를 돌린다(GPU 불필요, Pi 에서도 실행).
# 시나리오 ok: eA1(0.45/0.9/1.8/3.6) = −0.20/−0.30/−0.35/−0.40, eB(0.9/1.8/3.6) = +0.15/+0.25/+0.30 → η* = 1.8.
# nocand: eA1 전부 −0.10 → eta_star none. evfail: 무학습 평가 실패 → eta_star 미기록·종료코드 1.
# 실행: bash scripts/test_e151_path_check.sh (저장소 루트에서)
set -u
SRC=$(pwd)
TMP=$(mktemp -d)
mkdir -p "$TMP/bin"
cat > "$TMP/bin/python" <<'PY'
#!/usr/bin/env python3
import os, re, sys
import numpy as np
a = sys.argv[1:]
val = lambda f: a[a.index(f) + 1] if f in a else None
eta = val("--kc-motor-eta"); scen = os.environ.get("E151_SCEN", "ok")
F = {"0.45": 0.20, "0.9": 0.30, "1.8": 0.35, "3.6": 0.40}
G = {"0.45": 0.10, "0.9": 0.15, "1.8": 0.25, "3.6": 0.30}
if scen == "nocand":
    F = {k: 0.10 for k in F}
print("    KC→motor [E109 R-STDP 4방향]: init_w=150.0, w_max=300.0, eta=%s, tau_e=12.0, sparsity=0.25" % eta)
if "--decomp-mode" in a:
    mode, var, w = val("--decomp-mode"), val("--eval-variant"), val("--decomp-weights")
    if scen == "evfail" and mode == "none":
        sys.exit(1)
    if mode == "none":
        m = 0.0300 if var == "base" else 0.0250
    else:
        g = re.search(r"w_(A|AB)_(\d+)_b15", w); key = {"045": "0.45", "09": "0.9", "18": "1.8", "36": "3.6"}[g.group(2)]
        m = (0.0300 - F[key]) if (g.group(1) == "A" and var == "base") else ((0.0250 + G[key]) if (g.group(1) == "AB" and var == "bad") else 0.0)
    print("[E146 변형] variant=%s vseed=0" % var)
    print("=> DECOMP mode=%s mod=%+.4f acc=0.0" % (mode, m))
else:
    n = int(val("--episodes")) * 100
    if "--task-b-after" in a:
        print("[과제 B] 시행 1500 부터 자극 = bad food, 정답 = 같은 쪽")
    print("[사전] 오프셋 +0.0 | 정답률 0.0% | **변조폭 +0.0300**")
    print("[사후] 오프셋 +0.0 | 정답률 50.0% | **변조폭 -0.2000**")
    print("[학습] 15ep 완료, 보상 900회 (탐색 주입 0회, ε=0.60)")
    open(val("--save-weights"), "w").write("x")
    R = np.zeros((n, 37)); R[:, 6] = -1; R[:, 13:17] = 1.0; R[:, 21:25] = (11 / 12) ** 20; R[:, 12] = 1.0
    np.savez_compressed(val("--trace-kc-class"), rows=R)
PY
chmod +x "$TMP/bin/python"
ALL=0
for SCEN in ok nocand evfail; do
  TD="$TMP/$SCEN"; mkdir -p "$TD/scripts" "$TD/backend/genesis" "$TD/research/experiments/logs/E151" "$TD/run"
  cp "$SRC/scripts/e151_path_check.sh" "$SRC/scripts/judge_e151.py" "$TD/scripts/"; echo "#" > "$TD/backend/genesis/dummy.py"
  E151_SCEN=$SCEN E151_R="$TD" E151_RUNROOT="$TD/run" PATH="$TMP/bin:$PATH" bash "$TD/scripts/e151_path_check.sh" > "$TD/out.txt" 2>&1; rc=$?
  STAR=$(cat "$TD/research/experiments/logs/E151/eta_star.txt" 2>/dev/null || echo "(없음)")
  case $SCEN in
    ok) WANT="eta_star 1.8"; WRC=0 ;;
    nocand) WANT="eta_star none"; WRC=0 ;;
    evfail) WANT="(없음)"; WRC=1 ;;
  esac
  if [ "$STAR" = "$WANT" ] && [ $rc -eq $WRC ]; then echo "$SCEN → $STAR rc=$rc ✓"; else echo "$SCEN → $STAR rc=$rc ✗ (기대 $WANT rc=$WRC)"; cat "$TD/out.txt"; ALL=1; fi
done
grep -E "eA1\(|eB\(|η\* =|후보" "$TMP/ok/out.txt"
rm -rf "$TMP"
[ $ALL -eq 0 ] && echo "전체: 통과" || echo "전체: 실패"
exit $ALL
