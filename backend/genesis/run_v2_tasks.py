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
    # biletaxis/gain: D2. brake/settle: D4. hunger_gate: D5(satiety 게이팅 arbitration).
    # v3 회로 (7/5~7/6)
    # v3_klino: 시간적 gradient 변조로 재구현 (D7). replay_to_klino: 이미 replay_swr가
    #   value 지도 만들고 biletaxis가 읽음 = 구조적 충족(D7, 아래 수용).
    # v3_olf: 피질 R-STDP 변별학습 게이팅으로 재구현 (D8).
    "v3_recovery": "회복 과제",
    "v3_value_eta": "가치 학습률",
    "v3_cue_eta": "단서 학습률",
    # place_value_food_exclude: 구조적으로 이미 factored (D6, no-op). 아래 수용.
    "value_max": "가치 상한",
    # zone_circle/appetitive_place/start_far: place_pref 과제 레이어로 재구현됨 (D1)
    # sparse_reward: zone 진입→DA로 재구현됨 (D3)
    # biletaxis/biletaxis_gain: 학습 value 지도 양측 read-out으로 재구현됨 (D2)
    "thermal_reversal": "온도 역전",
    "cue_reversal": "단서 역전",
    "cue_reversal_period": "단서 역전 주기",
    # 시퀀스 (7/7~7/10): seq_task/seq_wm/seq_nav → SeqTask(D10). seq_gain → WM 되먹임 스케일(D10b).
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
    p.add_argument("--biletaxis-settle", action="store_true")

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
    p.add_argument("--seq-pattern-latch", action="store_true",
                   help="D16후속: 래치 판정을 rate 아닌 A-패턴 상관으로(희소 WM 패턴래치 readout).")
    p.add_argument("--seq-no-curiosity", action="store_true",
                   help="D18: curiosity 무작위탐색 OFF — biletaxis 방향조향만. order confound 격리.")
    p.add_argument("--seq-gain", type=float, default=None)
    p.add_argument("--inhib-wm", type=float, default=None,
                   help="D15: wm_inhibitory→WM 억제강도 (희소코딩). 기본 -5.0(포화). -200 권장(활성~23%).")
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


class PlacePrefTask:
    """v3 place_pref 과제 레이어 — 원본 소실, 증거 기반 재유도 (docs/research/DESIGN_RECOVERY.md §D1).

    4월판 gym에는 task_mode가 없고, 원본 run_v2_tasks는 gym 위의 별도 하네스였다.
    → gym을 건드리지 않고 러너에서 goal-zone 항법을 계측한다.

    좌표는 obs의 정규화(0~1) 위치를 쓴다. zone 중심/반경도 정규화.
    """

    def __init__(self, cx=0.5, cy=0.5, radius=0.12, appetitive=True,
                 start_far=False, sparse_reward=False):
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.appetitive = appetitive   # False = aversive(회피 과제)
        self.start_far = start_far
        self.sparse_reward = sparse_reward  # zone 진입 → DA (D3, place→value 학습 구동)
        self._reset_accum()

    def _reset_accum(self):
        self._dist_sum = 0.0
        self._in_zone = 0
        self._steps = 0
        self._was_in = False
        self._zone_rewards = 0   # 진단: zone 진입 보상 횟수 (0이면 학습 원천 없음)

    def on_reset(self, env):
        self._reset_accum()
        if self.start_far:
            # zone 반대편 코너에 배치 (정규화 → 픽셀)
            fx = 0.1 if self.cx > 0.5 else 0.9
            fy = 0.1 if self.cy > 0.5 else 0.9
            env.agent_x = fx * env.config.width
            env.agent_y = fy * env.config.height

    def on_step(self, env, action_delta, brain=None):
        px = env.agent_x / env.config.width
        py = env.agent_y / env.config.height
        dist = ((px - self.cx) ** 2 + (py - self.cy) ** 2) ** 0.5
        self._dist_sum += dist
        in_zone = dist < self.radius
        if in_zone:
            self._in_zone += 1
        self._steps += 1

        # D3 sparse-reward: zone 진입 순간 DA + 경험 버퍼링.
        # place→value는 replay_swr()에서 갱신된다(A2 디버깅으로 확정: 4월
        # run_training은 보상시 add_experience → 에피소드끝 replay_swr).
        # 1차 실험 실패 원인 = 이 경로 미호출. add_experience로 버퍼링하고
        # 에피소드 끝에서 run_episode가 replay_swr 호출.
        if self.sparse_reward and brain is not None and in_zone and not self._was_in:
            if self.appetitive:
                brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
                brain.add_experience(self.cx, self.cy, 0, env.steps, 25.0)
            else:
                brain.release_dopamine(reward_magnitude=-0.5)
                brain.add_experience(self.cx, self.cy, 1, env.steps, -5.0)
            # A2 재유도: zone=보상 랜드마크 → 전이 버퍼 기록.
            # 원본 transition_buffer.append는 "음식 가시성"에 게이팅(10918)돼
            # n_food 0에선 안 채워짐 → value 역backup 스킵 → 지도 평평.
            # v3 place_pref는 트리거를 zone 근접으로 대체. 여기서 복제.
            self._record_transition(brain)
            self._zone_rewards += 1
        self._was_in = in_zone

    @staticmethod
    def _record_transition(brain):
        import numpy as np
        prev = getattr(brain, "prev_place_activation", None)
        curr = getattr(brain, "last_active_place_cells", None)
        if prev is None or curr is None:
            return
        pa = np.where(np.asarray(prev) > 0.1)[0].tolist()
        ca = np.where(np.asarray(curr) > 0.1)[0].tolist()
        if pa and ca:
            brain.transition_buffer.append((pa, ca, 1.0))
            if len(brain.transition_buffer) > 100:
                brain.transition_buffer = brain.transition_buffer[-100:]

    def episode_metrics(self):
        n = self._steps or 1
        return {
            "goal_dist": self._dist_sum / n,
            "dwell_ratio": self._in_zone / n,
            "zone_rewards": self._zone_rewards,
        }


class SeqTask:
    """D10 시퀀스 과제 (A→B 순서) + WM 래치. 원본 소실, 4월판 WM 기질 위 재유도.

    존 2개(A/B). 정순 = A 먼저→B 나중. A 방문시 WM 래치가 "A완료" 보유(4월판
    working_memory 되먹임) → 목표를 B로 전환. 러너는 info["working_memory_rate"]로
    래치 상태를 읽어 현재 목표 존을 정한다(biletaxis가 place_to_value 읽던 것과 동형).

    지표:
      - 최종순서율: 에피소드에서 A→B 정순 완료 비율(후반 상승 = 순서 학습).
      - WM latch: A 방문 후 wm_rate가 baseline 대비 상승·지속하는가.
    """

    def __init__(self, ax=0.3, ay=0.3, bx=0.7, by=0.7, radius=0.12,
                 use_wm=False, pattern_latch=False):
        self.ax, self.ay = ax, ay       # 존 A
        self.bx, self.by = bx, by       # 존 B
        self.radius = radius
        self.use_wm = use_wm            # seq-wm: WM 래치로 목표 게이팅
        self.pattern_latch = pattern_latch  # D16후속: rate 아닌 A-패턴 상관으로 래치 판정
        self._reset()

    def _reset(self):
        self.visited_a = False
        self.correct_seq = 0            # A→B 정순 완료 횟수
        self.wrong_seq = 0              # B를 A보다 먼저(역순)
        self._wm_pre = []               # A 방문 전 wm_rate
        self._wm_post = []              # A 방문 후 wm_rate
        self._latched = False
        self._a_pattern = None          # A 방문 시점 WM 스파이크 패턴
        self._pat_corr = []             # A 이후 패턴 상관 (유지=높음)
        self._wm_roll = None            # 최근 WM 벡터 롤링 누적 (패턴 판정용)
        self._a_load = None             # A-적재 누적 (패턴 기준)
        self._a_load_n = 0

    @staticmethod
    def _wm_v(brain):
        """working_memory 뉴런별 스파이크카운트 벡터 (D14: 깨진 V 대신 신뢰 readout)."""
        import numpy as np
        ids = np.asarray(brain.working_memory.spike_recording_data[0][1], dtype=np.int64)
        n = brain.config.n_working_memory
        if ids.size == 0:
            return np.zeros(n, dtype=np.float32)
        return np.bincount(ids, minlength=n).astype(np.float32)

    @staticmethod
    def _corr(a, b):
        import numpy as np
        a = a - a.mean(); b = b - b.mean()
        d = (np.linalg.norm(a) * np.linalg.norm(b))
        return float((a @ b) / d) if d > 1e-9 else 0.0

    def on_reset(self, env):
        self._reset()

    def target(self):
        """현재 목표 존 (정규화 중심). WM 래치 걸리면 B, 아니면 A."""
        go_b = self.visited_a
        if self.use_wm:
            # 래치 상태(working_memory 지속)로 목표 전환 — 창발 WM 사용
            go_b = self._latched
        return (self.bx, self.by) if go_b else (self.ax, self.ay)

    def on_step(self, env, brain, wm_rate):
        px = env.agent_x / env.config.width
        py = env.agent_y / env.config.height
        in_a = ((px - self.ax) ** 2 + (py - self.ay) ** 2) ** 0.5 < self.radius
        in_b = ((px - self.bx) ** 2 + (py - self.by) ** 2) ** 0.5 < self.radius

        import numpy as _np
        # WM 래치 계측: A 방문 전/후 wm_rate 수집 (지표용)
        if not self.visited_a:
            self._wm_pre.append(wm_rate)
        else:
            self._wm_post.append(wm_rate)
            # rate 기반 래치 (pattern_latch OFF일 때만 — 기존 동작)
            if not self.pattern_latch and not self._latched and self._wm_pre:
                base = sum(self._wm_pre) / len(self._wm_pre)
                if wm_rate > base * 1.5 + 0.01:
                    self._latched = True

        # 패턴 기반 래치 (D16후속): A-적재 패턴을 기준으로, 현재 WM이 그 패턴과
        # 상관 높으면 latched. rate 무관 — 희소 WM은 발화율 아닌 특정뉴런에 A를 담음(D15).
        if self.visited_a:
            v = self._wm_v(brain)
            # A 적재 직후 몇 스텝 누적 = A 기준 패턴 (게이트 열려 적재된 상태)
            if self._a_load_n < 4:
                if self._a_load is None:
                    self._a_load = v.copy()
                else:
                    self._a_load += v
                self._a_load_n += 1
                if self._a_load_n == 4 and self._a_load.std() > 1e-4:
                    self._a_pattern = self._a_load
            elif self._a_pattern is not None:
                # 롤링 누적(최근 4스텝)으로 현재 WM 패턴 추정 → A기준과 상관
                if self._wm_roll is None:
                    self._wm_roll = v.copy()
                else:
                    self._wm_roll = 0.6 * self._wm_roll + v
                c = self._corr(self._wm_roll, self._a_pattern)
                self._pat_corr.append(c)
                if self.pattern_latch and self._wm_roll.std() > 1e-4 and c > 0.5:
                    self._latched = True

        if in_a and not self.visited_a:
            self.visited_a = True
            brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
            brain.add_experience(self.ax, self.ay, 0, env.steps, 25.0)
            self._restore_energy(env)   # 존=자원: 생존 → 탐색 시간
        elif in_b:
            if self.visited_a:
                self.correct_seq += 1
                brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
                brain.add_experience(self.bx, self.by, 0, env.steps, 25.0)
                self._restore_energy(env)
                self.visited_a = False   # 다음 사이클
                self._latched = False
            else:
                self.wrong_seq += 1      # 역순(B 먼저)

    @staticmethod
    def _restore_energy(env):
        # 존 도달이 에너지 회복 = 순서 완성이 생존과 직결(내재적 보상).
        cap = getattr(env.config, "max_energy", 100.0)
        env.energy = min(cap, env.energy + 40.0)

    def episode_metrics(self):
        total = self.correct_seq + self.wrong_seq
        order_rate = self.correct_seq / total if total else 0.0
        wm_pre = sum(self._wm_pre) / len(self._wm_pre) if self._wm_pre else 0.0
        wm_post = sum(self._wm_post) / len(self._wm_post) if self._wm_post else 0.0
        pat = (sum(self._pat_corr) / len(self._pat_corr)) if self._pat_corr else 0.0
        return {
            "order_rate": order_rate,
            "correct_seq": self.correct_seq,
            "wm_pre": wm_pre,
            "wm_post": wm_post,
            "wm_latch": wm_post - wm_pre,   # >0 = A후 WM 상승(래치)
            "wm_pattern_corr": pat,          # A패턴 유지 상관 (>0.5=진짜 래치)
        }


class BiletaxisSteering:
    """biletaxis 양측 조향 — 원본 소실, 증거 기반 재유도 (DESIGN_RECOVERY §D2).

    학습된 place→value 지도를 좌/우 heading으로 읽어 높은 쪽으로 조향.
    ground-truth 목표가 아니라 뇌가 학습한 value를 쓴다 — align은 학습 지도가
    목표를 실제로 가리키는지를 측정하므로 반드시 학습 지도여야 한다.

    러너 레벨(brain.place_to_value / brain.place_cell_centers 속성 read).
    원본 아키텍처(하네스가 gym/brain 위) 유지. value 지도는 에피소드 내 안정 →
    reset 때 1회 pull (매 스텝 GPU pull 회피).
    """

    def __init__(self, gain=0.5, look=0.10, delta=0.6, sigma=0.08,
                 brake=False, settle=False, hunger_gate=False, klino=False):
        self.gain = gain
        self.look = look      # 전방 샘플 거리(정규화)
        self.delta = delta    # 좌/우 각 오프셋(rad)
        self.sigma = sigma    # place-cell 가우시안 read 폭
        self.brake = brake    # D4: 고value 구역서 감속 → 정착 (#43/#49)
        self.settle = settle  # D4: 목표 근처 조향 감쇠 → orbit 방지
        self.hunger_gate = hunger_gate  # D5: 배부를때만 목표항법 (forage 우선, #61)
        self.klino = klino    # D7: 시간적 gradient — 멀어지면 재조향 강화, 가까워지면 직진
        self._vhere_prev = None  # klino: 직전 value_here (시간 비교)
        self._vmax = 1e-9     # 현재 지도 최대 value (정규화용)
        self._v_per_cell = None
        self._centers = None
        self._align_hit = 0
        self._align_tot = 0

    def on_reset(self, brain):
        import numpy as np
        self._align_hit = 0
        self._align_tot = 0
        self._vhere_prev = None   # klino 시간 히스토리 리셋
        # 학습된 value 지도 pull (place_to_value: place_cells → place_value)
        brain.place_to_value.vars["g"].pull_from_device()
        w = brain.place_to_value.vars["g"].view.copy()
        n_pc = brain.config.n_place_cells
        self._v_per_cell = w.reshape(n_pc, -1).mean(axis=1)   # 셀당 value
        self._centers = np.asarray(brain.place_cell_centers)   # (n_pc, 2) 정규화
        self._vmax = float(self._v_per_cell.max()) + 1e-9      # 정규화 기준
        # 진단: value 지도 분산 (0이면 평평 = 미학습, align 0의 원인)
        self.vmap_std = float(self._v_per_cell.std())

    def value_here_norm(self, env, target=None):
        """현재 위치의 학습 value를 [0,1]로 정규화. 1 = 지도 최댓값(=zone 중심).
        target 주면 그 목표 근방 마스킹 — seq A트랩 방지(현 목표서만 감속)."""
        ax = env.agent_x / env.config.width
        ay = env.agent_y / env.config.height
        return self._value_at(ax, ay, target=target) / self._vmax

    def satiety_gate(self, satiety):
        """D5 hunger-gate: 배부를때만 목표항법. 0(허기)→게이트닫힘, 1(포만)→열림.
        허기시 biletaxis 억제 → forage 반사 우선. hunger_gate OFF면 항상 1."""
        if not self.hunger_gate:
            return 1.0
        # satiety_rate(0~1) 소프트 임계: 0.2 미만 닫힘, 0.5+ 완전 열림
        return max(0.0, min(1.0, (satiety - 0.2) / 0.3))

    def brake_factor(self, env, satiety=1.0, target=None):
        """brake ON이면 고value 구역서 속도 배율 반환 (1=정상, →0.3 감속).
        hunger-gate시 허기땐 brake도 해제. target 주면 현 목표 근방서만 감속."""
        if not self.brake:
            return 1.0
        vn = max(0.0, min(1.0, self.value_here_norm(env, target=target)))
        g = self.satiety_gate(satiety)
        return 1.0 - 0.7 * vn * g   # 허기(g=0)면 감속 없음

    def _value_at(self, x, y, target=None):
        import numpy as np
        d2 = ((self._centers[:, 0] - x) ** 2 + (self._centers[:, 1] - y) ** 2)
        wr = np.exp(-d2 / (2 * self.sigma ** 2))
        vpc = self._v_per_cell
        if target is not None:
            # seq-nav: value를 목표 존 근방으로 마스킹 → 그 목표로만 climb
            tx, ty = target
            dt2 = ((self._centers[:, 0] - tx) ** 2 + (self._centers[:, 1] - ty) ** 2)
            mask = np.exp(-dt2 / (2 * (self.radius_mask ** 2)))
            vpc = vpc * mask
        s = wr.sum()
        return float((wr * vpc).sum() / s) if s > 1e-9 else 0.0

    radius_mask = 0.18   # seq-nav 목표 마스킹 반경

    def bias(self, env, place, satiety=1.0, target=None):
        """조향 보정량 반환 + align 누적. place: goal 방향 판정용.
        target: seq-nav용 목표 존(정규화 중심). 주면 value를 그 근방 마스킹해 조향.

        부호 규약 (A2b 디버깅으로 확정):
        gym은 `agent_angle += angle_delta`, 이동은 cos/sin(angle) →
        **angle_delta>0 = 반시계(CCW) = heading+δ 방향으로 회전.**
        따라서 CCW쪽(θ+δ) value가 높으면 +방향으로 틀어야 그쪽으로 간다.
        """
        import numpy as np
        ax = env.agent_x / env.config.width
        ay = env.agent_y / env.config.height
        th = env.agent_angle
        # CCW(왼쪽, angle_delta>0 방향) = θ+δ, CW(오른쪽) = θ-δ
        x_ccw = ax + self.look * np.cos(th + self.delta)
        y_ccw = ay + self.look * np.sin(th + self.delta)
        x_cw = ax + self.look * np.cos(th - self.delta)
        y_cw = ay + self.look * np.sin(th - self.delta)
        v_ccw = self._value_at(x_ccw, y_ccw, target=target)
        v_cw = self._value_at(x_cw, y_cw, target=target)
        d_turn = self.gain * (v_ccw - v_cw)   # CCW value 높으면 +(CCW로 조향)

        # D4 settle: 목표 근처(고value)서 조향 감쇠 → orbit 대신 정착.
        if self.settle:
            vn = max(0.0, min(1.0, self._value_at(ax, ay) / self._vmax))
            d_turn *= (1.0 - 0.8 * vn)   # 중심 근처일수록 조향 약화

        # D7 klinotaxis: 시간적 gradient. 현재 위치 value가 직전보다 떨어지면
        # (멀어지는 중) 재조향을 강화, 오르면(가까워지는 중) 현 조향 신뢰.
        # biletaxis(공간 좌우비교)의 시간축 보완. 방향은 biletaxis가, klino는 크기 변조.
        if self.klino:
            v_here = self._value_at(ax, ay)
            if self._vhere_prev is not None:
                trend = v_here - self._vhere_prev          # >0 가까워짐, <0 멀어짐
                tn = trend / (self._vmax + 1e-9)           # 정규화
                # 멀어질수록(tn<0) 조향 증폭, 가까워지면 1.0 유지
                d_turn *= (1.0 + 2.0 * max(0.0, -tn))
            self._vhere_prev = v_here

        # D5 hunger-gate: 허기시 목표항법 억제 → forage 반사 우선.
        d_turn *= self.satiety_gate(satiety)

        # align: 조향 부호가 실제 목표방향 부호와 일치하나
        if place is not None and abs(d_turn) > 1e-6:
            goal_ang = np.arctan2(place.cy - ay, place.cx - ax)
            rel = (goal_ang - th + np.pi) % (2 * np.pi) - np.pi  # [-π,π]
            # rel>0 = 목표가 CCW(왼쪽) → +조향(d_turn>0)이 정답
            correct = (d_turn > 0) == (rel > 0)
            self._align_hit += int(correct)
            self._align_tot += 1
        return d_turn

    def align_ratio(self):
        return self._align_hit / self._align_tot if self._align_tot else 0.0


def _discriminate(brain, reward_type):
    """D8 v3-olf: 먹이 먹은 순간 피질/예측오차 R-STDP 변별학습.
    run_training과 동일 (good_food/bad_food). config/pathway 없으면 조용히 skip.
    소리단서→좋음/나쁨 연합을 도파민 게이팅으로 학습 — 하드코딩 아닌 창발."""
    cfg = brain.config
    try:
        if getattr(cfg, "perceptual_learning_enabled", False) and getattr(cfg, "it_enabled", False):
            brain.update_cortical_rstdp(reward_type)
        if reward_type == "good_food" and getattr(cfg, "prediction_error_enabled", False):
            brain.update_prediction_error_rstdp("food")
    except Exception as e:
        print(f"  [warn] _discriminate({reward_type}) 실패: {e}", file=sys.stderr)


def run_episode(brain, env, ep_idx, task, place=None, biletaxis=None, olf=False, seq=None):
    """단일 에피소드. 4월 run_training 루프의 최소 경로를 따른다.

    place: PlacePrefTask 또는 None. 주어지면 goal-zone 항법 계측을 얹는다.
    """
    obs = env.reset()
    brain.reset()
    if place is not None:
        place.on_reset(env)
    if seq is not None:
        seq.on_reset(env)
    if biletaxis is not None:
        biletaxis.on_reset(brain)
    done = False
    good_eaten = 0
    bad_eaten = 0
    thermal_entries = 0
    cool_steps = 0

    satiety_sum = 0.0
    seq_wm_gated = seq is not None and getattr(seq, "use_wm", False)
    while not done:
        # PBWM: seq-wm이면 감각→WM 입력을 뇌 자신의 도파민으로 게이팅.
        # 도파민(보상시↑, 이후 감쇠) = 게이트 신호. 임계 튜닝 없이 연속값 사용.
        # 보상 없을땐 게이트 닫혀 현재위치가 WM 덮어쓰기 못함 → 되먹임이 상태 유지.
        if seq_wm_gated:
            brain.gate_wm_input(getattr(brain, "dopamine_level", 0.0))
        action_delta, info = brain.process(obs)
        brain.decay_dopamine()
        satiety = info.get("satiety_rate", 1.0)
        satiety_sum += satiety
        wm_rate = info.get("working_memory_rate", 0.0)
        # 탐색: 뇌의 curiosity_rate(신규성/불확실성 구동)로 분산 탐색.
        # 미방문 B 부트스트랩용. B로 조향(ground-truth) 아님 — 친숙한 곳 이탈(무방향).
        # 뇌가 언제 탐색할지 결정(curiosity), 러너는 움직임으로 번역(biletaxis와 동형).
        if seq is not None and not getattr(seq, "no_curiosity", False):
            cur = info.get("curiosity_rate", 0.0)
            if cur > 0.0:
                import numpy as _np
                action_delta = action_delta + _np.random.uniform(-1, 1) * cur * 1.5

        # biletaxis: 학습 value 지도 기반 조향 보정 (env.step 전에 heading 반영)
        if biletaxis is not None:
            seq_target = seq.target() if seq is not None else None
            action_delta = action_delta + biletaxis.bias(env, place, satiety=satiety,
                                                          target=seq_target)
        env.set_brain_info(info)
        # D4 brake: 고value 구역서 전진속도 축소 (env.step 동안만 config 변경 후 복원)
        _spd0 = None
        if biletaxis is not None and biletaxis.brake:
            _spd0 = env.config.agent_speed
            env.config.agent_speed = _spd0 * biletaxis.brake_factor(
                env, satiety=satiety, target=seq_target)
        obs, reward, done, env_info = env.step((action_delta,))
        if _spd0 is not None:
            env.config.agent_speed = _spd0

        if place is not None:
            place.on_step(env, action_delta, brain=brain)
        if seq is not None:
            seq.on_step(env, brain, wm_rate)

        if env_info.get("food_eaten"):
            ftype = env_info.get("food_type", 0)
            if ftype == 0:
                good_eaten += 1
                brain.learn_food_location(food_position=(obs["position_x"], obs["position_y"]))
                brain.release_dopamine(reward_magnitude=1.0, primary_reward=True)
                # D8 v3-olf: 피질 R-STDP 변별학습. 소리단서(good=고음)→접근 연합을
                # 도파민으로 학습(하드코딩 아님, 창발). run_training과 동일 호출.
                if olf:
                    _discriminate(brain, "good_food")
            else:
                bad_eaten += 1
                brain.release_dopamine(reward_magnitude=-0.5)
                if olf:
                    _discriminate(brain, "bad_food")

    # 에피소드 끝 SWR 리플레이 → place_to_value 갱신 (4월 run_training과 동일).
    # A2 디버깅: 이 호출 누락이 value 지도가 평평했던 근본 원인.
    # D19: seq 태스크도 A/B 존을 add_experience로 버퍼링하나 이 호출이 place에만
    # 게이팅돼 seq는 replay 미실행 → vmap_std=0(D18) → biletaxis 조향불능 → 순서실패.
    # seq에도 consolidation 배선(A트랙과 동일 기전, 하드코딩 아님).
    if (place is not None and place.sparse_reward) or (seq is not None):
        try:
            brain.replay_swr()
        except Exception as e:
            print(f"  [warn] replay_swr 실패: {e}", file=sys.stderr)

    n_choices = good_eaten + bad_eaten
    # 생존 데이터에서 역산, 21/21 에피소드 검증된 공식
    performance_index = (good_eaten - bad_eaten) / n_choices if n_choices else 0.0
    steps = env.steps

    rec = {
        "task_mode": task,
        "steps": steps,
        "mean_satiety": satiety_sum / steps if steps else 0.0,
        "performance_index": performance_index,
        "good_eaten": good_eaten,
        "bad_eaten": bad_eaten,
        "n_choices": n_choices,
        "thermal_entries": thermal_entries,
        "episode": ep_idx,
        "steps_taken": steps,
    }
    if place is not None:
        rec.update(place.episode_metrics())
        rec["cool_dwell_ratio"] = rec["dwell_ratio"]
    else:
        rec["cool_dwell_ratio"] = (cool_steps / steps) if steps else 0.0
    if biletaxis is not None:
        rec["biletaxis_align"] = biletaxis.align_ratio()
    if seq is not None:
        rec.update(seq.episode_metrics())
    return rec


def main():
    args = build_parser().parse_args()
    check_unimplemented(args)
    if args.episodes < 1:
        print("--episodes must be >= 1", file=sys.stderr)
        sys.exit(2)

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

    # D6 factored value: 이 러너는 place-value를 zone 보상만으로 학습(음식은
    # add_experience 대상 아님) → 구조적으로 이미 factored. 플래그는 no-op이나
    # 스크립트 호환 위해 수용. (원본은 음식이 place-value 오염 → 이 플래그로 제외)
    if args.place_value_food_exclude:
        print("[factored] place-value는 이미 zone-only 학습 (구조적 factored, no-op)")

    # D7 replay-to-klino: 이미 replay_swr가 value 지도 만들고 biletaxis가 그걸 읽어
    # 항법 → 구조적 충족. 플래그 수용(no-op).
    if args.replay_to_klino:
        print("[replay-to-klino] replay_swr→value지도→biletaxis read = 이미 충족 (no-op)")
    # D7 klino는 biletaxis 조향을 시간축으로 변조 → biletaxis 없으면 의미 없음.
    if args.v3_klino and not args.biletaxis:
        print("[klino] biletaxis 없이 단독 klino는 이 재구현서 미지원 (변조 대상 없음) — 무시",
              file=sys.stderr)

    # place_pref 과제 레이어 (v3, 증거 기반 재유도 — DESIGN_RECOVERY §D1/D3)
    place = None
    if args.task == "place_pref" or args.zone_circle:
        place = PlacePrefTask(
            cx=args.zone_cx if args.zone_cx is not None else 0.5,
            cy=args.zone_cy if args.zone_cy is not None else 0.5,
            appetitive=args.appetitive_place,
            start_far=args.start_far,
            sparse_reward=args.sparse_reward,
        )
        kind = "appetitive" if args.appetitive_place else "aversive"
        print(f"[place_pref] goal-zone ({place.cx},{place.cy}) r={place.radius} "
              f"{kind} start_far={args.start_far} sparse_reward={args.sparse_reward}")

    # biletaxis 양측 조향 (증거 기반 재유도 — DESIGN_RECOVERY §D2)
    biletaxis = None
    if args.biletaxis:
        biletaxis = BiletaxisSteering(
            gain=args.biletaxis_gain if args.biletaxis_gain is not None else 0.5,
            brake=args.biletaxis_brake,
            settle=args.biletaxis_settle,
            hunger_gate=args.biletaxis_hunger_gate,
            klino=args.v3_klino,
        )
        extras = []
        if biletaxis.brake:
            extras.append("brake")
        if biletaxis.settle:
            extras.append("settle")
        if biletaxis.hunger_gate:
            extras.append("hunger-gate")
        if biletaxis.klino:
            extras.append("klino")
        print(f"[biletaxis] 양측 조향 ON gain={biletaxis.gain}"
              f"{(' +' + '+'.join(extras)) if extras else ''} (학습 value 지도 read-out)")

    # D10b seq-gain: WM 되먹임/입력 균형 스케일 (bistable 래치 실험).
    # 근본원인(D10b): place_to_working_memory(10.0)가 현재위치로 WM 덮어써 유지 못함.
    # 되먹임만 키우면(seq-gain 스캔) place 드라이브에 짐 → place 드라이브도 낮춰 균형 이동.
    # 하드코딩 아님 — 생물 파라미터(감각 드라이브 vs 되먹임 유지)의 상대 강도.
    if args.seq_gain is not None:
        g = args.seq_gain
        rec0 = brain_config.working_memory_recurrent_weight
        plc0 = brain_config.place_to_working_memory_weight
        brain_config.working_memory_recurrent_weight = rec0 * g
        brain_config.place_to_working_memory_weight = plc0 / g   # 현재위치 덮어쓰기 완화
        print(f"[seq-gain ×{g}] WM 되먹임 {rec0}→{rec0*g}, place드라이브 {plc0}→{plc0/g}")

    # D10 seq-task: A→B 순서 과제 + WM 래치
    seq = None
    if args.seq_task:
        acx = args.zone_cx if args.zone_cx is not None else 0.3
        acy = args.zone_cy if args.zone_cy is not None else 0.3
        seq = SeqTask(ax=acx, ay=acy, bx=1.0 - acx, by=1.0 - acy,
                      use_wm=args.seq_wm, pattern_latch=args.seq_pattern_latch)
        seq.no_curiosity = args.seq_no_curiosity
        print(f"[seq] A({seq.ax},{seq.ay})→B({seq.bx},{seq.by}) "
              f"use_wm={args.seq_wm} seq_nav={args.seq_nav} "
              f"pattern_latch={args.seq_pattern_latch}")

    # D15: WM 희소코딩 — 포화 탈출. order_rate 아닌 희소성 목표로 보정된 값(용량 전제조건).
    if args.inhib_wm is not None:
        brain_config.inhibitory_to_wm_weight = args.inhib_wm
        print(f"[inhib-wm] wm_inhibitory→WM = {args.inhib_wm} (희소코딩, D15)")

    env = ForagerGym(config=env_config, render_mode="none")
    brain = ForagerBrain(config=brain_config)

    t0 = time.time()
    episodes = []
    for ep in range(args.episodes):
        rec = run_episode(brain, env, ep, args.task, place=place,
                          biletaxis=biletaxis, olf=args.v3_olf, seq=seq)
        episodes.append(rec)
        extra = ""
        if place is not None:
            extra += (f" goal-dist={rec['goal_dist']:.4f} dwell={rec['dwell_ratio']:.4f}"
                      f" zrew={rec.get('zone_rewards', 0)} sat={rec.get('mean_satiety', 0):.2f}")
        if biletaxis is not None:
            extra += f" align={rec['biletaxis_align']:.4f} vmap_std={biletaxis.vmap_std:.4f}"
        if seq is not None:
            extra += (f" 최종순서율={rec['order_rate']:.3f} correct={rec['correct_seq']}"
                      f" WM_latch={rec['wm_latch']:+.4f}")
        print(f"[ep {ep:3d}] PI={rec['performance_index']:+.4f} "
              f"good={rec['good_eaten']} bad={rec['bad_eaten']} "
              f"steps={rec['steps_taken']}{extra}")

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
    if place is not None:
        gd = [e["goal_dist"] for e in episodes]
        print(f"goal-dist: {sum(gd)/len(gd):.4f}")
    if biletaxis is not None:
        al = [e["biletaxis_align"] for e in episodes]
        print(f"biletaxis-align: {sum(al)/len(al):.4f}")
        print(f"last_5_align: {sum(al[-5:])/len(al[-5:]):.4f}")
    if seq is not None:
        orr = [e["order_rate"] for e in episodes]
        lat = [e["wm_latch"] for e in episodes]
        pc = [e.get("wm_pattern_corr", 0.0) for e in episodes]
        print(f"최종순서율: {sum(orr)/len(orr):.4f}")
        print(f"last_5_최종순서율: {sum(orr[-5:])/len(orr[-5:]):.4f}")
        print(f"WM latch: {sum(lat)/len(lat):+.4f}")
        print(f"WM pattern_corr: {sum(pc)/len(pc):.4f}")
        print(f"last_5_pattern_corr: {sum(pc[-5:])/len(pc[-5:]):.4f}")

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
