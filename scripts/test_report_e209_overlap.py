#!/usr/bin/env python3
"""report_e209_overlap 합성 시험 — 정답을 손으로 아는 가짜 측정 로그(범주마다 6사례가 같은 집합 → p = 1, 코사인 = 겹침 수/집합 크기)로
남은 몫 q·읽기 규칙 경계·점검(측정 무효)·결측이 사전 문장대로인지. 실행: python3 test_report_e209_overlap.py (scripts/ 에서)"""
import contextlib
import io
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_e209_overlap as R  # noqa: E402

DEV = {"kc_l_to_inh_l": (59718, 5.0), "kc_inh_l_to_kc_l": (95826, -15.0), "kc_r_to_inh_r": (60064, 5.0), "kc_inh_r_to_kc_r": (96092, -15.0)}   # 재배선 판(가중치 그대로)


def cat_sets(n, k):
    """A = 0..n−1, C = A 와 k 개 겹침, B = 1000.., D = B 와 k 개 겹침 → cos(AC) = cos(BD) = k/n, A·B 겹침 0."""
    A = set(range(n)); C = set(range(n - k, 2 * n - k))
    B = set(range(1000, 1000 + n)); D = set(range(1000 + n - k, 1000 + 2 * n - k))
    return {"A": A, "B": B, "C": C, "D": D}


def write_meas(path, n, k, ctx=None, gate_act=60000, gate_ina=300, im=None, dev=None, ctx_w=0.0, bad_rcj=False, dev_extra=None, tail=True):
    """ctx = None(맥락 없음) 또는 'X'. im = (활성, 비활성) 억제 뉴런 발화(None 이면 줄 없음). dev = 장치 확인 사전(None 이면 줄 없음)."""
    S = cat_sets(n, k)
    lines = ["[원시 특징 노출] 제시 0(노출 없음)",
             "  [맥락 입력] 맥락 전용 집단 zz_ctx 200 → KC 좌·우, w=%.2f, p=0.020, 연결 좌 12106·우 11860" % ctx_w,
             "  [맥락 입력 2] 맥락 전용 집단 zz_ctx2 200 → KC 좌·우, w=%.2f, p=0.020, 연결 좌 11825·우 11895" % ctx_w]
    for d in ([dev] if dev is not None else []) + ([dev_extra] if dev_extra is not None else []):
        lines.append("  [E209 구획 되먹임 억제 확인] 장치 — " + " · ".join("%s %d(같음·무리 밖 0·g %.2f)" % (g, d[g][0], d[g][1]) for g in R.GROUPS)
                     + (" · " + R.RW_TAIL if tail else ""))
    for tag in ("pre", "post"):
        for s in "lr":
            for c in "ABCD":
                for i in range(6):
                    ids = sorted(S[c])
                    lines.append("RCKC tag=%s side=%s set=train cat=%s i=%d n=%d ids=%s" % (tag, s, c, i, len(ids), ",".join(map(str, ids))))
    for s in "lr":
        for pr in ("AB", "AC", "AD", "BC", "BD", "CD"):
            x, y = S[pr[0]], S[pr[1]]
            j = len(x & y) / len(x | y)
            if bad_rcj and pr == "AC" and s == "r":
                j += 0.01
            lines.append("RCJ side=%s pair=%s J_pre=%.4f J_post=%.4f n_pre=%d/%d n_post=%d/%d" % (s, pr, j, j, len(x), len(y), len(x), len(y)))
    if ctx is not None:
        for tag in ("pre", "post"):
            lines.append("[E207 측정 맥락] tag=%s 맥락 %s 켬 — 제시 288 · 켠 맥락 발화 576000 · 끈 맥락 발화 0 · KC 발화 활성 무리 %d · 비활성 무리 %d · 게이팅 있음" % (tag, ctx, gate_act, gate_ina))
            if im is not None:
                lines.append("[E209 구획 억제 측정] tag=%s 맥락 %s 켬 — 제시 288 · KC 억제 뉴런 발화 활성 무리 %d · 비활성 무리 %d" % (tag, ctx, im[0], im[1]))
    lines.append("=> RAWCAT done")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build(root, spec):
    """spec[b] = dict(n, k7, k8, kc, kn, 그 밖 C·X 인자) — 뇌마다 다섯 로그."""
    for e in ("E207", "E208", "E209"):
        os.makedirs(os.path.join(root, "logs", e), exist_ok=True)
    for b, sp in spec.items():
        n = sp["n"]
        L = lambda e, f: os.path.join(root, "logs", e, f % b)  # noqa: E731
        write_meas(L("E207", "meas_F_none_b%d.log"), n, sp["k7"])
        write_meas(L("E207", "meas_F_X_b%d.log"), n, 9 if n == 10 else 18, ctx="X", gate_act=130000, gate_ina=800)
        write_meas(L("E208", "meas_F_X_b%d.log"), n, sp["k8"], ctx="X", gate_act=64000, gate_ina=800)
        if sp.get("miss"):
            continue
        write_meas(L("E209", "meas_C_X_b%d.log"), n, sp["kc"], ctx="X", gate_act=sp.get("ga", 30000), gate_ina=sp.get("gi", 700),
                   im=sp.get("im", (5000, 10)), dev=sp.get("dev", DEV), ctx_w=sp.get("cw", 0.0), bad_rcj=sp.get("bad_rcj", False), tail=sp.get("tail", True))
        write_meas(L("E209", "meas_C_none_b%d.log"), n, sp["kn"], dev=sp.get("devn", DEV))


def run(spec):
    root = tempfile.mkdtemp(prefix="r209_")
    try:
        build(root, spec)
        R.EXP = root
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            R.main()
        return buf.getvalue()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def row(out, b):
    for ln in out.splitlines():
        if ln.startswith("b%d:" % b):
            return ln
    return ""


def main():
    ok = tot = 0

    def chk(name, cond, info=""):
        nonlocal ok, tot
        tot += 1; ok += bool(cond)
        print(("OK   " if cond else "실패 ") + name + ((" — " + info) if (info and not cond) else ""))

    # 순수 함수
    chk("q4: (0.3 − 0.2)/(0.5 − 0.2) = 3333", R.q4(0.3, 0.2, 0.5) == 3333)
    chk("q4: 분모 0 → None", R.q4(0.3, 0.5, 0.5) is None)
    chk("classify 경계: 3333 사라짐 · 3334 가운데 · 6666 가운데 · 6667 남음 · None 판단 불가",
        [R.classify(x) for x in (3333, 3334, 6666, 6667, None)] == ["겹침 사라짐", "가운데", "가운데", "겹침 남음", "판단 불가"])
    base = dict(n=10, k7=2, k8=5, kn=2)
    # 1) 다섯 뇌: 경계 사라짐(q 1/3)·경계 남음(q 2/3)·가운데(n 20, q 1/2)·판단 불가(분모 0)·사라짐(q 0)
    spec = {16: dict(base, kc=3), 17: dict(base, kc=4), 18: dict(n=20, k7=4, k8=10, kn=4, kc=7), 19: dict(base, k8=2, kc=3), 20: dict(base, kc=2)}
    out = run(spec)
    chk("b16 q 0.3333 → 겹침 사라짐", "q 0.3333 → 겹침 사라짐" in row(out, 16), row(out, 16))
    chk("b17 q 0.6667 → 겹침 남음", "q 0.6667 → 겹침 남음" in row(out, 17), row(out, 17))
    chk("b18 q 0.5000 → 가운데", "q 0.5000 → 가운데" in row(out, 18), row(out, 18))
    chk("b19 분모 0 → 판단 불가", "q — → 판단 불가" in row(out, 19), row(out, 19))
    chk("b20 q 0.0000 → 겹침 사라짐", "q 0.0000 → 겹침 사라짐" in row(out, 20), row(out, 20))
    chk("요약 줄(사라짐 2·남음 1·가운데 1·판단 불가 1)", "요약: 겹침 사라짐 2/5 · 겹침 남음 1/5 · 가운데 1/5 · 판단 불가 1/5 · 측정 무효 0/5 · 결측 0/5" in out, out.splitlines()[-1])
    chk("b16 코사인 값(구획·X 0.300 · 짝 0.200·0.500 · 구획·없음 0.200)", "구획·X 코사인 0.300" in row(out, 16) and "E207 맥락 없음 0.200 · E208 흥분 0·X 0.500 · 구획·맥락 없음 0.200" in row(out, 16), row(out, 16))
    chk("b18 사례당 반응 KC 20.0 · 비 1.00", "사례당 반응 KC 구획·X 20.0 · 구획·없음 20.0(비 1.00)" in row(out, 18), row(out, 18))
    chk("b16 집합 S(로그 RCJ 4 × 0.1765 = 7060)", "집합 S 첫/둘째 구획·X 7060/7060" in row(out, 16), row(out, 16))
    chk("억제 뉴런 발화 줄 옮김(활성 5000/5000 · 비활성 10/10)", "활성 5000/5000 · 비활성 10/10" in row(out, 16), row(out, 16))
    # 2) 점검 — 측정 무효 다섯 갈래·결측
    spec = {16: dict(base, kc=3, im=None), 17: dict(base, kc=3, tail=False), 18: dict(base, kc=3, gi=3001),
            19: dict(base, kc=3, devn={**DEV, "kc_l_to_inh_l": (30001, 10.0)}), 20: dict(base, kc=3, miss=True)}
    out = run(spec)
    chk("b16 억제 측정 줄 없음 → 측정 무효", "b16: 측정 무효" in out, row(out, 16))
    chk("b17 장치 줄에 재배선 꼬리 없음(연결 절반 판 꼴) → 측정 무효", "b17: 측정 무효" in out, row(out, 17))
    chk("b18 비활성 KC 발화 3001 × 10 > 같은 측정 활성 30000 → 측정 무효", "b18: 측정 무효" in out, row(out, 18))
    chk("b19 두 측정 구획 연결 수 다름 → 측정 무효", "b19: 측정 무효" in out, row(out, 19))
    chk("b20 C 로그 없음 → 결측", "b20: 결측" in out, row(out, 20))
    spec = {16: dict(base, kc=3, gi=3000), 17: dict(base, kc=3, im=(5000, 2500)), 18: dict(base, kc=3, dev={**DEV, "kc_inh_r_to_kc_r": (96092, -30.0)}, devn={**DEV, "kc_inh_r_to_kc_r": (96092, -30.0)}),
            19: dict(base, kc=3, cw=20.0), 20: dict(base, kc=3, bad_rcj=True)}
    out = run(spec)
    chk("b16 경계 3000 × 10 = 30000 → 유효", "b16: 구획·X" in out, row(out, 16))
    chk("b17 비활성 억제 발화 큰 몫(2500/5000)도 유효(보고만)", "b17: 구획·X" in out and "비활성 2500/2500" in row(out, 17), row(out, 17))
    chk("b18 장치 g −30(재배선은 가중치 그대로여야) → 측정 무효", "b18: 측정 무효" in out, row(out, 18))
    chk("b19 맥락 무게 20 → 측정 무효", "b19: 측정 무효" in out, row(out, 19))
    chk("b20 RCJ ≠ 집합 J(자기 점검) → 측정 무효", "b20: 측정 무효(자기 점검" in out, row(out, 20))
    # 3) 자기 점검 반올림 경계 — 집합 J = 7/160 = 0.04375(로그는 과제와 같은 '%.4f' → '0.0437'): self_check 통과·brain_stats 정수 비교는 거짓 불일치
    root = tempfile.mkdtemp(prefix="r209b_")
    try:
        pth = os.path.join(root, "m.log")
        write_meas(pth, 83, 7)   # |A∩C| = 7, |A∪C| = 159 → J = 7/159 — 아래에서 직접 7/160 꼴을 만든다
        L_ = open(pth, encoding="utf-8").read().splitlines()
        # A = 0..82(83개), C = 76..158(83개) 대신 C 를 한 칸 넓혀 |A∪C| = 160 이 되게: C 에 KC 9999 추가
        out_l = []
        for ln in L_:
            if ln.startswith("RCKC ") and " cat=C " in ln:
                n0 = int(ln.split(" n=")[1].split()[0])
                ln = ln.replace(" n=%d " % n0, " n=%d " % (n0 + 1)) + ",9999"
            out_l.append(ln)
        jt = 7 / 160
        out_l = [("RCJ side=%s pair=AC J_pre=%.4f J_post=%.4f n_pre=83/84 n_post=83/84" % (ln.split()[1][5:], jt, jt)) if (ln.startswith("RCJ ") and "pair=AC" in ln) else ln for ln in out_l]
        open(pth, "w", encoding="utf-8").write("\n".join(out_l) + "\n")
        import posthoc_e208_graded as PG
        chk("경계 J 0.04375: report self_check 통과", R.self_check(pth) is True)
        chk("경계 J 0.04375: brain_stats 정수 비교는 거짓 불일치(알려진 결함 재현)", PG.brain_stats(pth)["self_check"] is False)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("합성 시험 %d/%d" % (ok, tot))
    return 0 if ok == tot else 1


if __name__ == "__main__":
    sys.exit(main())
