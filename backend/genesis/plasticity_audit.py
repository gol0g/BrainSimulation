#!/usr/bin/env python3
"""가소성 감사 (C30) — 이 뇌에서 학습이 일어날 수 있는 시냅스는 어디인가.

C25(수정판): 러너와 동일한 학습 호출을 넣고 보상 6776회를 줬는데도 학습경로 12개 시냅스가
|Δ|=0.0000. 남은 두 가능성:
  (a) 학습이 **다른 시냅스**에서 일어난다 → 전수 조사로 변화하는 곳을 찾는다
  (b) 그 12개가 **애초에 정적(static) 시냅스**다 → 이름만 학습경로

여기서는 전체 시냅스의 **weight update 모델**을 조회해 가소성 보유 여부를 판정하고,
훈련 전후 |Δ|를 **전 시냅스**에 대해 재서 실제로 움직이는 곳을 특정한다.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forager_brain import ForagerBrain, ForagerBrainConfig
from forager_gym import ForagerGym, ForagerConfig


def all_synapses(b):
    out = {}
    for name in dir(b):
        if name.startswith("_"):
            continue
        o = getattr(b, name, None)
        if o is None:
            continue
        try:
            if hasattr(o, "vars") and "g" in o.vars:
                out[name] = o
        except Exception:
            pass
    return out


def snap(syns):
    """SPARSE 시냅스는 **연결을 먼저 pull해야** values가 채워진다.
    이 호출을 빠뜨리면 빈 배열이 돌아오고, 빈 배열의 평균은 nan → 'nan > 1e-9'가 False라
    **조용히 '변화 없음'으로 오분류**된다(실제로 이 버그로 '기저핵 전체 동결' 오진 직전까지 감).
    로더 `_load_sparse_weights`는 이 호출을 하고 있었다."""
    out, empty = {}, []
    for name, syn in syns.items():
        try:
            try:
                syn.pull_connectivity_from_device()
            except Exception:
                pass          # DENSE는 연결 pull이 없음 — 정상
            syn.vars["g"].pull_from_device()
            v = syn.vars["g"].values
            if v is None or (hasattr(v, "size") and v.size == 0):
                v = syn.vars["g"].view
            arr = np.array(v, dtype=np.float64).copy()
            if arr.size == 0:
                empty.append(name)
                continue
            out[name] = arr
        except Exception:
            pass
    if empty:
        print("[주의] 값이 비어 판정 불가한 시냅스 %d개: %s" % (len(empty), ", ".join(empty)))
    return out


def wu_model(syn):
    for attr in ("wu_model", "w_update_model", "weight_update_model"):
        m = getattr(syn, attr, None)
        if m is not None:
            return type(m).__name__ if not isinstance(m, str) else m
    return type(syn).__name__


def main():
    b = ForagerBrain(ForagerBrainConfig())
    env = ForagerGym(ForagerConfig())
    syns = all_synapses(b)
    print("전체 시냅스 집단: %d개" % len(syns))

    before = snap(syns)
    obs = env.reset()
    eaten = 0
    for ep in range(15):
        obs = env.reset()
        for _ in range(1200):
            a, info = b.process(obs)
            obs, rew, done, inf = env.step((a,))
            if rew and rew > 0:
                b.release_dopamine(reward_magnitude=float(rew), primary_reward=True)
                eaten += 1
                cfg = b.config
                try:
                    if getattr(cfg, "perceptual_learning_enabled", False) and getattr(cfg, "it_enabled", False):
                        b.update_cortical_rstdp("good_food")
                    if getattr(cfg, "prediction_error_enabled", False):
                        b.update_prediction_error_rstdp("food")
                    b.learn_food_location(food_position=(obs["position_x"], obs["position_y"]))
                    b.add_experience(obs["position_x"], obs["position_y"], 0, getattr(env, "steps", 0), 25.0)
                except Exception:
                    pass
            if done:
                break
        try:
            b.replay_swr()
        except Exception:
            pass
    after = snap(syns)
    print("보상 %d회 후 전 시냅스 변화 조사\n" % eaten)

    moved, static = [], []
    for k in sorted(before):
        if k not in after or before[k].shape != after[k].shape:
            continue
        d = float(np.mean(np.abs(after[k] - before[k])))
        frac = float(np.mean(np.abs(after[k] - before[k]) > 1e-9))
        (moved if d > 1e-9 else static).append((k, d, frac, float(np.std(before[k])), float(np.std(after[k]))))

    print("### 변화한 시냅스: %d개 ###" % len(moved))
    print("%-34s %10s %9s %10s %10s" % ("이름", "|Δ|평균", "변화율", "std(전)", "std(후)"))
    for k, d, f, s0, s1 in sorted(moved, key=lambda x: -x[1]):
        print("%-34s %10.5f %8.1f%% %10.4f %10.4f" % (k, d, f * 100, s0, s1))

    print("\n### 변화 없는 시냅스: %d개 ###" % len(static))
    print(", ".join(k for k, _, _, _, _ in static))


if __name__ == "__main__":
    main()
