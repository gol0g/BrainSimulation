#!/bin/bash
# 네 칸의 연결 구조 동일성. 칸마다 별도 프로세스 + 별도 CODE 디렉터리(GeNN 충돌 회피).
set -u
R=/mnt/c/Users/JungHyun/Desktop/brain/BrainSimulation-rebuild
source $R/scripts/cuda_env.sh >/dev/null 2>&1
source /root/pygenn_wsl/bin/activate
for C in A B C D; do
  D=/root/conn_$C
  mkdir -p $D && cd $D
  cp $R/backend/genesis/*.py . 2>/dev/null
  CELL=$C OUT=/root/conn_$C.json timeout 2400 python connectivity_check.py \
    --seed 0 --d1-inhib -400 --direct-inhib -100 2>&1 | grep -E "^\[|^   " || echo "  [$C 실패]"
done
echo
echo "=== 비교 ==="
cd /root
python3 - <<'PY'
import json
cells = {}
for c in "ABCD":
    try:
        cells[c] = json.load(open("/root/conn_%s.json" % c, encoding="utf-8"))
    except Exception as e:
        print("  %s 읽기 실패: %s" % (c, e))
if len(cells) == 4:
    names = list(cells["A"].keys())
    allsame = True
    print("%-24s %-12s %-12s %-12s %-12s  판정" % ("시냅스", "A(w.5끔)", "B(w.5켬)", "C(150끔)", "D(150켬)"))
    for nm in names:
        vs = [str(cells[c][nm][0])[:8] for c in "ABCD"]
        same = len(set(vs)) == 1
        allsame = allsame and same
        print("%-24s %-12s %-12s %-12s %-12s  %s" % (nm, *vs, "동일" if same else "**다름**"))
    print("\n판정:", "네 칸 연결 동일 — 이식·비교 가능" if allsame
          else "**연결이 다르다. 칸 차이에 배선이 섞인다 — 설계 수정 필요**")
PY
