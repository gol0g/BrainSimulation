"""verify_e202_independent 합성 시험 — judge 시험과 같은 정답 트리(e202_synth)로 독립 대조의 분해·기전·판정을 검사."""
import contextlib
import io
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e202_synth as SY  # noqa: E402
import verify_e202_independent as VI  # noqa: E402

ok = bad = 0


def chk(name, cond):
    global ok, bad
    if cond:
        ok += 1
    else:
        bad += 1
        print("실패", name)


def run(scens, pb=None):
    d = tempfile.mkdtemp()
    SY.write_tree(d, scens, pb)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        VI.verify(d)
    return buf.getvalue()


for nm, (do, df, dm, mk) in SY.ANSWER.items():
    out = run([nm] * 5)
    m = re.search(r"^b10 \| D_obs ([-+0-9.]+) D_freq ([-+0-9.]+) D_mag ([-+0-9.]+) (\S+)", out, re.M)
    chk("%s 줄" % nm, m is not None)
    if m:
        chk("%s D_obs" % nm, abs(float(m.group(1)) - do) < 1e-3)
        chk("%s D_freq" % nm, abs(float(m.group(2)) - df) < 1e-3)
        chk("%s D_mag" % nm, abs(float(m.group(3)) - dm) < 1e-3)
        chk("%s 기전" % nm, m.group(4) == mk)
    chk("%s 통과" % nm, out.count("| 통과") == 5)
chk("M1 5/5", "독립 판정: 결과 빈도 비대칭(H123-alt)" in run(["M1"] * 5))
chk("M2 4/5", "독립 판정: 흔적 크기 비대칭(H123)" in run(["M2", "M2", "M2", "M2", "MIX"]))
chk("혼합", "독립 판정: 혼합" in run(["M1", "M1", "M1", "M2", "M2"]))
out = run(["M1"] * 5, {12: {"weak_scale": 2.0}})
chk("재구성 실패", "독립 판정: 보류(계측 검증 실패)" in out and "실패 재구성" in out)
out = run(["M1"] * 5, {13: {"flip200": True}})
chk("회귀 실패", "독립 판정: 보류(계측 검증 실패)" in out and "E200 회귀" in out)
chk("결측", "독립 판정: 보류(결측 — b14" in run(["M1"] * 5, {14: {"drop": "dg"}}))
out = run(["M1"] * 5, {10: {"log_drop": "[E202 계측] 시행 3000"}})
chk("계측 요약 없음", "독립 판정: 보류(계측 검증 실패)" in out and "계측 요약" in out)
out = run(["M1"] * 5, {10: {"n_rows": 2999}})
chk("행 2999", "독립 판정: 보류(계측 검증 실패)" in out and "계측 행" in out)
# 정정 1 — 분류 오염
out = run(["M1"] * 5, {12: {"own": 0.6}})
chk("분류 오염", "독립 판정: 보류(계측 검증 실패)" in out and "분류 오염" in out and "자기 쪽 몫 0.600" in out)
chk("자기 쪽 몫 0.8 통과", "독립 판정: 결과 빈도 비대칭(H123-alt)" in run(["M1"] * 5, {12: {"own": 0.8}}))
# reduceat Δ = 정답 Δ
d = tempfile.mkdtemp()
SY.write_tree(d, ["M1"] * 5)
ex = os.path.join(d, "research", "experiments")
dd = VI.kc_delta(os.path.join(ex, "traces", "E202", "w_b10.npz"), os.path.join(ex, "traces", "E176", "snap", "conn_b10.npz"))
chk("Δ 약한 무리 = −16", abs(dd["l"][0] + 16) < 1e-6 and abs(dd["r"][399] + 16) < 1e-6)
chk("Δ 강한 교차 = +200", abs(dd["l"][450] - 200) < 1e-6)
chk("Δ 비반응 0", abs(dd["r"][2000]) < 1e-6)
print("합성 %d/%d" % (ok, ok + bad))
sys.exit(0 if bad == 0 else 1)
