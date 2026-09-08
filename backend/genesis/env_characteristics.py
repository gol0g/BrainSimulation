#!/usr/bin/env python3
"""환경 특성 측정 (E075) — H007 판별용.

E074에서 같은 조작이 환경에 따라 정반대 효과를 냈다(env0 +0.0115 vs env2 -0.0219).
잡음인지 환경 특성과의 상호작용인지 가리려면 **환경을 정량화**해야 한다.
상관이 있으면 H007, 없으면 H007-null.

측정 항목(모두 학습 이전, 환경 초기 상태에서):
  - good/bad 먹이 개수와 비율
  - 에이전트 초기 위치에서 가장 가까운 good/bad 먹이까지 거리
  - good 먹이의 공간 분산(뭉쳐 있는가 흩어져 있는가)
  - 좌/우 반쪽의 good 먹이 개수 차이 (좌우 비대칭 — 반사와 상호작용할 수 있다)
"""
import sys, os, argparse, math, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_gym import ForagerGym, ForagerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--env-seed", type=int, required=True)
a = ap.parse_args()

random.seed(a.env_seed)
np.random.seed(a.env_seed)
env = ForagerGym(ForagerConfig())
env.reset()

foods = list(getattr(env, "foods", []))
W, H = env.config.width, env.config.height
ax, ay = env.agent_x, env.agent_y

good = [(x, y) for x, y, t in foods if t == 0]
bad = [(x, y) for x, y, t in foods if t == 1]

def dist(p):
    return math.hypot(p[0] - ax, p[1] - ay)

def spread(pts):
    if len(pts) < 2:
        return float("nan")
    arr = np.array(pts, dtype=float)
    c = arr.mean(axis=0)
    return float(np.mean(np.linalg.norm(arr - c, axis=1)))

left_good = sum(1 for x, y in good if x < W / 2)
right_good = len(good) - left_good

print("ENV seed=%d" % a.env_seed)
print("  n_good=%d n_bad=%d good_ratio=%.3f" % (len(good), len(bad),
      len(good) / max(len(foods), 1)))
print("  nearest_good=%.1f nearest_bad=%.1f" % (
      min((dist(p) for p in good), default=float("nan")),
      min((dist(p) for p in bad), default=float("nan"))))
print("  good_spread=%.1f" % spread(good))
print("  left_good=%d right_good=%d asym=%+d" % (left_good, right_good, left_good - right_good))
print("  agent_start=(%.0f,%.0f)" % (ax, ay))
