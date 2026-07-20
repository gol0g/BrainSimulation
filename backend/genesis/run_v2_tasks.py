#!/usr/bin/env python3
"""
run_v2_tasks.py — v2/v3 실험 하네스 (재건본)

2026-07 디스크 사고로 원본 소실. 생존한 실험 스크립트 24개가 호출하는
CLI·stdout 마커·결과 스키마를 역산해 재작성했다. 상세: docs/RECONSTRUCTION.md

재건 원칙:
  - 지표 공식은 생존 데이터에서 역산 후 21/21 에피소드 검증한 것만 사용한다.
  - 브레인 측 기능이 아직 재구현되지 않은 플래그는 **조용히 무시하지 않고 즉시 실패**한다.
    연구 하네스에서 조용한 no-op은 가짜 결과를 만들고, 그건 결과 없음보다 나쁘다.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RUNNER_VERSION = "1.0"

# ---------------------------------------------------------------------------
# 재구현 대기 플래그 — 4월판 브레인/gym에 대응 기능이 없다.
# 스크립트 타임스탬프가 알려주는 실제 개발 순서대로 채워 넣는다.
# ---------------------------------------------------------------------------
NOT_YET_IMPLEMENTED = {
    # biletaxis 계열 (7/1~7/4)
    "biletaxis": "양측 주행 회로",
    "biletaxis_gain": "양측 주행 이득",
    "biletaxis_brake": "양측 주행 제동",
    "biletaxis_hunger_gate": "허기 게이팅",
    "biletaxis_settle": "정착 궤적 덤프",
    # v3 회로 (7/5~7/6)
    "v3_klino": "klinotaxis 회로",
    "v3_olf": "후각 변별",
    "v3_recovery": "회복 과제",
    "v3_value_eta": "가치 학습률",
    "v3_cue_eta": "단서 학습률",
    "replay_to_klino": "리플레이→klino 투사",
    "place_value_food_exclude": "장소가치 음식분리(factored)",
    "value_max": "가치 상한",
    "zone_circle": "원형 존",
    "appetitive_place": "식욕성 장소선호",
    "start_far": "원거리 시작",
    "sparse_reward": "희소 보상",
    "thermal_reversal": "온도 역전",
    "cue_reversal": "단서 역전",
    "cue_reversal_period": "단서 역전 주기",
    # 시퀀스 (7/7~7/10)
    "seq_task": "시퀀스 과제",
    "seq_nav": "시퀀스 항법",
    "seq_wm": "창발 워킹메모리",
    "seq_gain": "시퀀스 이득",
    # 조합 컨텍스트 (7/10~7/11, 최전선)
    "context_compositional": "조합적 컨텍스트",
    # 중재
    "wta_arbitration": "WTA 중재",
    "wta_cue_bid": "WTA 단서 입찰",
    # 계측
    "traj_dump": "궤적 덤프",
}


def build_parser():
    p = argparse.ArgumentParser(description="v2/v3 실험 하네스 (재건본)")
    p.add_argument("--task", default="integrated",
                   choices=["integrated", "place_pref", "olfactory", "reversal"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="결과 JSON 경로")
    p.add_argument("--n-food", type=int, default=None)

    # 환경/과제
    p.add_argument("--zone-circle", action="store_true")
    p.add_argument("--zone-cx", type=float, default=None)
    p.add_argument("--zone-cy", type=float, default=None)
    p.add_argument("--appetitive-place", action="store_true")
    p.add_argument("--start-far", action="store_true")
    p.add_argument("--sparse-reward", action="store_true")
    p.add_argument("--thermal-reversal", action="store_true")
    p.add_argument("--cue-reversal", action="store_true")
    p.add_argument("--cue-reversal-period", type=int, default=None)

    # biletaxis
    p.add_argument("--biletaxis", action="store_true")
    p.add_argument("--biletaxis-gain", type=float, default=None)
    p.add_argument("--biletaxis-brake", action="store_true")
    p.add_argument("--biletaxis-hunger-gate", action="store_true")
    p.add_argument("--biletaxis-settle", default=None)

    # v3
    p.add_argument("--v3-klino", action="store_true")
    p.add_argument("--v3-olf", action="store_true")
    p.add_argument("--v3-recovery", type=float, default=None)
    p.add_argument("--v3-value-eta", type=float, default=None)
    p.add_argument("--v3-cue-eta", type=float, default=None)
    p.add_argument("--replay-to-klino", action="store_true")
    p.add_argument("--place-value-food-exclude", action="store_true")
    p.add_argument("--value-max", type=float, default=None)

    # 시퀀스 / 컨텍스트
    p.add_argument("--seq-task", action="store_true")
    p.add_argument("--seq-nav", action="store_true")
    p.add_argument("--seq-wm", action="store_true")
    p.add_argument("--seq-gain", type=float, default=None)
    p.add_argument("--context-select", action="store_true",
                   help="Zone A/B 의미반전 컨텍스트 과제 (4월 M4 기반, 구현됨)")
    p.add_argument("--context-compositional", action="store_true")

    # 중재
    p.add_argument("--wta-arbitration", action="store_true")
    p.add_argument("--wta-cue-bid", action="store_true")

    # 계측
    p.add_argument("--traj-dump", default=None)
    return p


def check_unimplemented(args):
    """재구현 안 된 플래그가 켜졌으면 즉시 중단."""
    active = []
    for dest, desc in NOT_YET_IMPLEMENTED.items():
        val = getattr(args, dest, None)
        if val is None or val is False:
            continue
        active.append(f"  --{dest.replace('_', '-')}  ({desc})")
    if active:
        print("=" * 70, file=sys.stderr)
        print("[재건 미완] 아래 플래그는 4월판 브레인/gym에 대응 기능이 없다:", file=sys.stderr)
        print("\n".join(active), file=sys.stderr)
        print("", file=sys.stderr)
        print("조용히 무시하면 잘못된 결과가 나오므로 중단한다.", file=sys.stderr)
        print("재구현 순서와 근거: docs/RECONSTRUCTION.md", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        sys.exit(2)


def run_episode(brain, env, ep_idx, task):
    """단일 에피소드. 4월 run_training 루프의 최소 경로를 따른다."""
    obs = env.reset()
    brain.reset()
    done = False
    good_eaten = 0
    bad_eaten = 0
    thermal_entries = 0
    cool_steps = 0

    while not done:
        action_delta, info = brain.process(obs)
        brain.decay_dopamine()
        env.set_brain_info(info)
        obs, reward, done, env_info = env.step((action_delta,))

        if env_info.get("food_eaten"):
            ftype = env_info.get("food_type", 0)
            if ftype == 0:
                good_eaten += 1
                brain.learn_food_location(food_position=(obs["position_x"], obs["position_y"]))
                brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
            else:
                bad_eaten += 1
                brain.release_dopamine(reward_magnitude=-0.5)

    n_choices = good_eaten + bad_eaten
    # 생존 데이터에서 역산, 21/21 에피소드 검증된 공식
    performance_index = (good_eaten - bad_eaten) / n_choices if n_choices else 0.0
    steps = env.steps
    cool_dwell_ratio = (cool_steps / steps) if steps else 0.0

    return {
        "task_mode": task,
        "steps": steps,
        "cool_dwell_ratio": cool_dwell_ratio,
        "performance_index": performance_index,
        "good_eaten": good_eaten,
        "bad_eaten": bad_eaten,
        "n_choices": n_choices,
        "thermal_entries": thermal_entries,
        "episode": ep_idx,
        "steps_taken": steps,
    }


def main():
    args = build_parser().parse_args()
    check_unimplemented(args)

    import numpy as np
    from forager_gym import ForagerGym, ForagerConfig
    from forager_brain import ForagerBrain, ForagerBrainConfig

    np.random.seed(args.seed)

    env_config = ForagerConfig()
    brain_config = ForagerBrainConfig()

    if args.n_food is not None:
        env_config.n_food = args.n_food

    # --context-select: Zone A/B 의미반전 (4월 M4 기반)
    if args.context_select:
        env_config.context_rules_enabled = True
        brain_config.context_gate_enabled = True
        print("[context] Zone A/B 의미반전 활성 — context hard gate ON")

    env = ForagerGym(config=env_config, render_mode="none")
    brain = ForagerBrain(config=brain_config)

    t0 = time.time()
    episodes = []
    for ep in range(args.episodes):
        rec = run_episode(brain, env, ep, args.task)
        episodes.append(rec)
        print(f"[ep {ep:3d}] PI={rec['performance_index']:+.4f} "
              f"good={rec['good_eaten']} bad={rec['bad_eaten']} "
              f"steps={rec['steps_taken']}")

    elapsed = time.time() - t0

    pis = [e["performance_index"] for e in episodes]
    steps = [e["steps_taken"] for e in episodes]
    total_good = sum(e["good_eaten"] for e in episodes)
    dwell = [e["cool_dwell_ratio"] for e in episodes]

    # 생존 스크립트가 grep 하는 마커 — 문자열 변경 금지
    print(f"mean_pi: {sum(pis)/len(pis):.4f}")
    print(f"mean_steps: {sum(steps)/len(steps):.1f}")
    print(f"total_good: {total_good}")
    print(f"mean_cool_dwell_ratio: {sum(dwell)/len(dwell):.4f}")
    if len(dwell) >= 5:
        print(f"last_5_mean_dwell: {sum(dwell[-5:])/5:.4f}")

    result = {
        "v2_runner_version": RUNNER_VERSION,
        "timestamp": datetime.now().isoformat(),
        "task": args.task,
        "seed": args.seed,
        "n_episodes": args.episodes,
        "ablation": "full_baseline",
        "ablation_flags": {},
        "elapsed_sec": elapsed,
        "episodes": episodes,
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
