#!/usr/bin/env python3
"""평가전용 스위치 감사 (C23b) — C22에서 드러난 구조적 구멍.

C23 감사는 "어딘가에서 켜지는가"만 봤다. 그러나 **평가 스크립트에서만 켜지고 훈련 러너에는
배선이 없는** 스위치는 "그 회로를 켠 채로 학습시킬 수 없다"는 뜻이라 실질적으로 반쪽이다.
v9 context hard gate가 정확히 그 경우였고(평가전용), 그 탓에 4월 돌파 설정을 한 번도 재훈련하지 못했다.
"""
import re, os

d = os.path.dirname(os.path.abspath(__file__))
brain = open(os.path.join(d, "forager_brain.py"), encoding="utf-8", errors="ignore").read()
runner = open(os.path.join(d, "run_v2_tasks.py"), encoding="utf-8", errors="ignore").read()
evals = ""
for f in sorted(os.listdir(d)):
    if f.endswith(".py") and f not in ("forager_brain.py", "run_v2_tasks.py"):
        evals += open(os.path.join(d, f), encoding="utf-8", errors="ignore").read()

m = re.search(r"class ForagerBrainConfig.*?(?=\nclass |\Z)", brain, re.S).group(0)
flags = re.findall(r"^\s{4}(\w+):\s*bool\s*=\s*(True|False)", m, re.M)

out = []
for name, default in flags:
    if default != "False":
        continue
    in_eval = re.search(r"\." + name + r"\s*=", evals) is not None
    in_runner = re.search(r"\." + name + r"\s*=", runner) is not None
    if in_eval and not in_runner:
        out.append(name)

print("평가전용(훈련 러너 미배선) 스위치: %d개" % len(out))
for n in out:
    print("  - %s" % n)
