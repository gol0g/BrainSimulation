#!/usr/bin/env python3
"""죽은 스위치 감사 (C23) — C21(WM 게이트)·C22(v9 맥락 hard gate)가 같은 병리였으므로 전수 점검.

병리 정의: ForagerBrainConfig에 bool 필드가 기본 False로 있고, 뇌 코드는 그 필드를 실제로 참조해
회로를 구동하는데, **러너(run_v2_tasks.py)가 어디서도 True로 설정하지 않음** = 그 회로는 영원히 꺼져 있음.
"구현됨"으로 기록된 기능이 실행되지 않는 상태.
"""
import re, os, sys

d = os.path.dirname(os.path.abspath(__file__))
brain = open(os.path.join(d, "forager_brain.py"), encoding="utf-8", errors="ignore").read()
runner = open(os.path.join(d, "run_v2_tasks.py"), encoding="utf-8", errors="ignore").read()
# 러너 외 평가·프로브 스크립트도 설정처로 인정(그쪽에서만 켜지면 '훈련에서 죽음'은 별도 표시)
setters = [runner]
for f in sorted(os.listdir(d)):
    if f.endswith(".py") and f not in ("forager_brain.py", "run_v2_tasks.py", os.path.basename(__file__)):
        try:
            setters.append(open(os.path.join(d, f), encoding="utf-8", errors="ignore").read())
        except Exception:
            pass

m = re.search(r"class ForagerBrainConfig.*?(?=\nclass |\Z)", brain, re.S)
body = m.group(0)
flags = re.findall(r"^\s{4}(\w+):\s*bool\s*=\s*(True|False)", body, re.M)

dead, live, unused = [], [], []
for name, default in flags:
    # 설정처: 러너 + 평가 스크립트들(속성대입 또는 setattr)
    set_anywhere = any(
        re.search(r"\." + name + r"\s*=", s) or re.search(r'setattr\([^,]+,\s*["\']' + name, s)
        for s in setters)
    # 사용처: config.X / self.X 뿐 아니라 getattr(self.config,"X",...) 형태도 포함(거짓음성 방지)
    uses = len(re.findall(
        r"config\." + name + r"|self\." + name + r"|getattr\([^,]+,\s*[\"']" + name, brain))
    if uses == 0:
        unused.append(name)
    elif default == "False" and not set_anywhere:
        dead.append((name, uses))
    else:
        live.append(name)

print("bool 설정필드 %d개 = 죽은스위치 후보 %d / 정상 %d / 뇌미사용 %d\n"
      % (len(flags), len(dead), len(live), len(unused)))
print("[죽은 스위치 후보] 기본 False + 러너 미설정 + 뇌가 실제 참조:")
for n, u in sorted(dead, key=lambda x: -x[1]):
    print("  %-44s (뇌에서 %d회 참조)" % (n, u))
if unused:
    print("\n[뇌에서 미사용 = 껍데기 필드] " + ", ".join(unused))
