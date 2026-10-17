"""judge_e202 합성 시험 — e202_synth 의 정답(분해 D_obs·D_freq·D_mag, 기전, 판정)과 계측 검증 실패·회귀 실패·결측을 검사."""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e202_synth as SY  # noqa: E402
import judge_e202 as J  # noqa: E402

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
        J.judge(d)
    return buf.getvalue(), d


# 1) 분해 정답(시나리오별)
for nm, (do, df, dm, mk) in SY.ANSWER.items():
    out, d = run([nm] * 5)
    D = np.load(os.path.join(d, "research", "experiments", "traces", "E202", "dg_b10.npz"))
    Z = np.load(os.path.join(d, "research", "experiments", "traces", "E196", "sxi_b10.npz"))
    dc = J.decompose(D, J.groups(Z), "weak")
    chk("%s D_obs" % nm, abs(dc["Dobs"] - do) < 1e-9)
    chk("%s D_freq" % nm, abs(dc["Dfreq"] - df) < 1e-9)
    chk("%s D_mag" % nm, abs(dc["Dmag"] - dm) < 1e-9)
    chk("%s 항등식" % nm, abs(dc["Dobs"] - dc["Dfreq"] - dc["Dmag"]) < 1e-9)
    chk("%s 기전" % nm, J.mech(dc) == mk)
    chk("%s 계측 검증 통과" % nm, "계측 검증 통과 |" in out)
# 2) 종합 판정
out, _ = run(["M1"] * 5)
chk("M1 5/5 → 결과 빈도(H123-alt)", "판정: 결과 빈도 비대칭(H123-alt)" in out and "M1(결과 빈도) 5/5" in out)
out, _ = run(["M2", "M2", "M2", "M2", "MIX"])
chk("M2 4/5 → 흔적 크기(H123)", "판정: 흔적 크기 비대칭(H123)" in out and "M2(흔적 크기) 4/5" in out and "혼합 1/5" in out)
out, _ = run(["M1", "M1", "M1", "M2", "M2"])
chk("3·2 → 혼합", "판정: 혼합" in out)
# 3) 계측 검증 실패 — 약한 무리 학습 Δ 가 계측의 2배(재구성 상대 오차 1.0)
out, _ = run(["M1"] * 5, {12: {"weak_scale": 2.0}})
chk("재구성 실패 → 보류", "판정: 보류(계측 검증 실패)" in out and "실패(b12)" in out and "재구성" in out)
# 4) E200 회귀 실패 — 반응 KC 일부 부호 반대
out, _ = run(["M1"] * 5, {13: {"flip200": True}})
chk("회귀 실패 → 보류", "판정: 보류(계측 검증 실패)" in out and "E200 회귀" in out)
# 5) 결측
out, _ = run(["M1"] * 5, {14: {"drop": "dg"}})
chk("계측 파일 결측 → 보류(결측)", "판정: 보류(결측 — b14" in out)
# 6) 로그 줄 결측
out, _ = run(["M1"] * 5, {10: {"log_drop": "[E202 계측] 시행 3000"}})
chk("계측 요약 줄 없음 → 보류", "판정: 보류(계측 검증 실패)" in out and "계측 요약 줄" in out)
out, _ = run(["M1"] * 5, {11: {"log_drop": "[E199 경쟁 정규화]"}})
chk("규칙 줄 없음 → 보류", "판정: 보류(계측 검증 실패)" in out and "규칙 줄" in out)
# 7) 계측 파일 행 수
out, _ = run(["M1"] * 5, {10: {"n_rows": 2999}})
chk("행 2999 → 보류", "판정: 보류(계측 검증 실패)" in out and "행 2999" in out)
# 8) 무리 균형 평균 — 교차·같은 쪽 KC 수가 달라도 각 평균의 평균
Z = {"resp_l": np.array([1, 1, 1, 0], np.int8), "resp_r": np.array([0, 0, 0, 0], np.int8),
     "sxi_l": np.array([0.05, 0.05, -0.05, 0.05]), "sxi_r": np.zeros(4)}
g = J.groups(Z)
chk("균형 평균", abs(J.bal({"l": np.array([2.0, 4.0, -10.0, 99.0]), "r": np.zeros(4)}, g, "weak") - (3.0 - 10.0) / 2) < 1e-12)
chk("빈 무리 nan", np.isnan(J.bal({"l": np.zeros(4), "r": np.zeros(4)}, g, "strong")))
print("합성 %d/%d" % (ok, ok + bad))
sys.exit(0 if bad == 0 else 1)
