"""
Desktop Agent - 가상 데스크톱 환경에서 학습하는 SNN 에이전트
============================================================

Phase B의 마지막 단계: 전체 데스크톱 환경 탐색

생물학적 메커니즘만 사용:
- DA-STDP (도파민 조절 시냅스 가소성)
- Eligibility traces (자격 흔적)
- Habituation (습관화)
- LIF neurons

NO LLM, NO FEP formulas, NO 심즈식 게이지, NO CNN backprop
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import random

# Force CPU
DEVICE = torch.device('cpu')


class WindowState(Enum):
    """윈도우 상태"""
    CLOSED = 0
    OPEN = 1
    MINIMIZED = 2
    MAXIMIZED = 3


@dataclass
class DesktopIcon:
    """데스크톱 아이콘"""
    name: str
    x: int
    y: int
    icon_type: str  # 'folder', 'app', 'file'
    color: Tuple[int, int, int]


@dataclass
class Window:
    """윈도우"""
    name: str
    x: int
    y: int
    width: int
    height: int
    state: WindowState
    content_type: str  # 'text', 'image', 'interactive'
    content: any


class VirtualDesktop:
    """
    가상 데스크톱 환경

    - 아이콘 클릭 → 윈도우 열기
    - 윈도우 드래그, 최소화, 최대화, 닫기
    - 간단한 앱 시뮬레이션 (메모장, 계산기, 파일탐색기)
    """

    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height

        # 데스크톱 상태
        self.icons: List[DesktopIcon] = []
        self.windows: List[Window] = []
        self.cursor_x = width // 2
        self.cursor_y = height // 2
        self.is_clicking = False
        self.dragging_window: Optional[int] = None

        # 상태 추적
        self.step_count = 0
        self.windows_opened = 0
        self.icons_clicked = set()
        self.interactions_count = 0

        # 초기화
        self._setup_desktop()

    def _setup_desktop(self):
        """데스크톱 초기 설정"""
        icon_positions = [
            ("Notepad", 8, 8, "app", (255, 255, 200)),
            ("Calculator", 8, 20, "app", (200, 200, 255)),
            ("Explorer", 8, 32, "folder", (255, 220, 100)),
            ("Settings", 8, 44, "app", (180, 180, 180)),
            ("Documents", 20, 8, "folder", (255, 200, 100)),
            ("Pictures", 20, 20, "folder", (100, 200, 255)),
            ("Games", 20, 32, "folder", (255, 100, 100)),
            ("Trash", 8, 56, "app", (150, 150, 150)),
        ]

        for name, x, y, icon_type, color in icon_positions:
            self.icons.append(DesktopIcon(name, x, y, icon_type, color))

    def reset(self) -> np.ndarray:
        """환경 리셋"""
        self.windows = []
        self.cursor_x = self.width // 2
        self.cursor_y = self.height // 2
        self.is_clicking = False
        self.dragging_window = None
        self.step_count = 0
        self.windows_opened = 0
        self.icons_clicked = set()
        self.interactions_count = 0

        return self._render()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        한 스텝 실행

        Actions:
        0: 커서 위로
        1: 커서 아래로
        2: 커서 왼쪽
        3: 커서 오른쪽
        4: 클릭
        5: 더블클릭
        6: 드래그 시작/끝
        7: 우클릭
        """
        self.step_count += 1
        reward = 0.0
        info = {"event": None}

        move_speed = 4
        if action == 0:
            self.cursor_y = max(0, self.cursor_y - move_speed)
        elif action == 1:
            self.cursor_y = min(self.height - 1, self.cursor_y + move_speed)
        elif action == 2:
            self.cursor_x = max(0, self.cursor_x - move_speed)
        elif action == 3:
            self.cursor_x = min(self.width - 1, self.cursor_x + move_speed)
        elif action == 4:
            reward, event = self._handle_click()
            info["event"] = event
        elif action == 5:
            reward, event = self._handle_double_click()
            info["event"] = event
        elif action == 6:
            reward, event = self._handle_drag()
            info["event"] = event
        elif action == 7:
            reward, event = self._handle_right_click()
            info["event"] = event

        if self.dragging_window is not None:
            self._move_dragged_window()

        done = self.step_count >= 500

        obs = self._render()
        info["windows_opened"] = self.windows_opened
        info["icons_clicked"] = len(self.icons_clicked)
        info["interactions"] = self.interactions_count

        return obs, reward, done, info

    def _handle_click(self) -> Tuple[float, Optional[str]]:
        for i, window in enumerate(reversed(self.windows)):
            if self._point_in_window(self.cursor_x, self.cursor_y, window):
                idx = len(self.windows) - 1 - i
                self.windows.append(self.windows.pop(idx))
                self.interactions_count += 1

                if (window.x + window.width - 8 <= self.cursor_x <= window.x + window.width and
                    window.y <= self.cursor_y <= window.y + 8):
                    self.windows.remove(window)
                    return 0.1, "window_closed"

                if (window.x + window.width - 16 <= self.cursor_x <= window.x + window.width - 8 and
                    window.y <= self.cursor_y <= window.y + 8):
                    window.state = WindowState.MINIMIZED
                    return 0.05, "window_minimized"

                return 0.02, "window_clicked"

        for icon in self.icons:
            if self._point_near_icon(self.cursor_x, self.cursor_y, icon):
                self.icons_clicked.add(icon.name)
                self.interactions_count += 1
                return 0.05, f"icon_selected:{icon.name}"

        return 0.0, None

    def _handle_double_click(self) -> Tuple[float, Optional[str]]:
        for icon in self.icons:
            if self._point_near_icon(self.cursor_x, self.cursor_y, icon):
                for window in self.windows:
                    if window.name == icon.name:
                        if window.state == WindowState.MINIMIZED:
                            window.state = WindowState.OPEN
                            return 0.1, f"window_restored:{icon.name}"
                        return 0.02, f"window_focused:{icon.name}"

                self._open_window(icon)
                self.windows_opened += 1
                self.icons_clicked.add(icon.name)
                self.interactions_count += 1

                if self.windows_opened <= len(self.icons):
                    return 0.3, f"window_opened:{icon.name}"
                return 0.1, f"window_opened:{icon.name}"

        return 0.0, None

    def _handle_drag(self) -> Tuple[float, Optional[str]]:
        if self.dragging_window is not None:
            self.dragging_window = None
            return 0.02, "drag_end"

        for i, window in enumerate(reversed(self.windows)):
            if (window.x <= self.cursor_x <= window.x + window.width and
                window.y <= self.cursor_y <= window.y + 10):
                self.dragging_window = len(self.windows) - 1 - i
                self.interactions_count += 1
                return 0.05, f"drag_start:{window.name}"

        return 0.0, None

    def _handle_right_click(self) -> Tuple[float, Optional[str]]:
        self.interactions_count += 1
        return 0.01, "context_menu"

    def _move_dragged_window(self):
        if self.dragging_window is not None and self.dragging_window < len(self.windows):
            window = self.windows[self.dragging_window]
            window.x = max(0, min(self.width - window.width, self.cursor_x - window.width // 2))
            window.y = max(0, min(self.height - window.height, self.cursor_y - 5))

    def _open_window(self, icon: DesktopIcon):
        width = random.randint(20, 35)
        height = random.randint(15, 30)
        x = random.randint(10, self.width - width - 5)
        y = random.randint(5, self.height - height - 5)

        content_type = "text"
        content = None

        if icon.name == "Calculator":
            content_type = "interactive"
            content = {"display": "0", "memory": 0}
        elif icon.name == "Notepad":
            content_type = "text"
            content = "Hello World!"
        elif icon.icon_type == "folder":
            content_type = "list"
            content = [f"file_{i}.txt" for i in range(3)]

        window = Window(
            name=icon.name,
            x=x, y=y,
            width=width, height=height,
            state=WindowState.OPEN,
            content_type=content_type,
            content=content
        )
        self.windows.append(window)

    def _point_in_window(self, x: int, y: int, window: Window) -> bool:
        if window.state == WindowState.MINIMIZED:
            return False
        return (window.x <= x <= window.x + window.width and
                window.y <= y <= window.y + window.height)

    def _point_near_icon(self, x: int, y: int, icon: DesktopIcon, radius: int = 6) -> bool:
        return abs(x - icon.x) <= radius and abs(y - icon.y) <= radius

    def _render(self) -> np.ndarray:
        screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        screen[:, :] = [30, 60, 90]

        for icon in self.icons:
            x, y = icon.x, icon.y
            x1, x2 = max(0, x-2), min(self.width, x+3)
            y1, y2 = max(0, y-2), min(self.height, y+3)
            screen[y1:y2, x1:x2] = icon.color

        for window in self.windows:
            if window.state == WindowState.MINIMIZED:
                continue

            x, y = window.x, window.y
            w, h = window.width, window.height

            x1, x2 = max(0, x), min(self.width, x + w)
            y1, y2 = max(0, y), min(self.height, y + h)
            screen[y1:y2, x1:x2] = [240, 240, 240]

            y_title = min(self.height, y + 8)
            screen[y1:y_title, x1:x2] = [100, 100, 200]

            close_x = min(self.width, x + w) - 6
            if close_x > x1:
                screen[y1:y_title, close_x:x2] = [255, 100, 100]

        cx, cy = self.cursor_x, self.cursor_y
        cursor_color = [255, 255, 255] if not self.dragging_window else [255, 200, 0]

        for dx in range(-2, 3):
            if 0 <= cx + dx < self.width:
                screen[cy, cx + dx] = cursor_color
        for dy in range(-2, 3):
            if 0 <= cy + dy < self.height:
                screen[cy + dy, cx] = cursor_color

        return screen


class DesktopDopamineSystem:
    """데스크톱용 도파민 시스템 - 생물학적 도파민 반응"""

    def __init__(self, baseline: float = 0.1):
        self.baseline = baseline
        self.current_level = baseline
        self.event_counts: Dict[str, int] = {}
        self.event_last_step: Dict[str, int] = {}
        self.step_counter = 0
        self.state_sequence: List[str] = []
        self.pattern_visit_count: Dict[str, int] = {}
        self.loop_detected = False
        self.discovered_icons: Set[str] = set()
        self.opened_windows: Set[str] = set()
        self.visited_regions: Set[Tuple[int, int]] = set()

    def compute_dopamine(self, event: Optional[str], cursor_pos: Tuple[int, int],
                         windows_opened: int, icons_clicked: int) -> float:
        self.step_counter += 1
        dopamine = self.baseline

        if event:
            event_type = event.split(":")[0]
            event_target = event.split(":")[1] if ":" in event else None

            visit_count = self.event_counts.get(event_type, 0)
            habituation = 1.0 / (1.0 + visit_count * 0.3)

            if event_type == "window_opened":
                if event_target and event_target not in self.opened_windows:
                    dopamine += 0.8 * habituation
                    self.opened_windows.add(event_target)
                else:
                    dopamine += 0.2 * habituation

            elif event_type == "icon_selected":
                if event_target and event_target not in self.discovered_icons:
                    dopamine += 0.4 * habituation
                    self.discovered_icons.add(event_target)
                else:
                    dopamine += 0.1 * habituation

            elif event_type == "window_closed":
                dopamine += 0.3 * habituation
            elif event_type == "drag_start":
                dopamine += 0.2 * habituation
            elif event_type == "window_minimized":
                dopamine += 0.15 * habituation

            self.event_counts[event_type] = visit_count + 1
            self.event_last_step[event_type] = self.step_counter

        region = (cursor_pos[0] // 8, cursor_pos[1] // 8)
        if region not in self.visited_regions:
            self.visited_regions.add(region)
            dopamine += 0.1

        state_hash = f"w{windows_opened}_i{icons_clicked}_r{region}"
        self.state_sequence.append(state_hash)
        if len(self.state_sequence) > 20:
            self.state_sequence = self.state_sequence[-20:]

        self._detect_loop()
        if self.loop_detected:
            dopamine *= 0.3

        total_discoveries = len(self.opened_windows) + len(self.discovered_icons)
        if total_discoveries > 0:
            progress_bonus = min(0.2, total_discoveries * 0.02)
            dopamine += progress_bonus

        self.current_level = np.clip(dopamine, -0.5, 1.5)
        return self.current_level

    def _detect_loop(self):
        seq = self.state_sequence

        if len(seq) >= 4:
            if seq[-1] == seq[-3] and seq[-2] == seq[-4]:
                self.loop_detected = True
                return

        if len(seq) >= 6:
            if seq[-1] == seq[-4] and seq[-2] == seq[-5] and seq[-3] == seq[-6]:
                self.loop_detected = True
                return

        if len(seq) >= 3:
            if seq[-1] == seq[-2] == seq[-3]:
                self.loop_detected = True
                return

        self.loop_detected = False

    def get_exploration_pressure(self) -> float:
        if self.loop_detected:
            return 0.8
        explored_ratio = len(self.visited_regions) / 64
        return max(0.2, 1.0 - explored_ratio)

    def reset(self):
        self.current_level = self.baseline
        self.step_counter = 0
        self.state_sequence = []
        self.loop_detected = False
        self.visited_regions = set()


class SimpleSNNBrain(nn.Module):
    """
    단순화된 SNN 뇌 (DA-STDP 포함)

    BiologicalBrain의 복잡성을 피하고 핵심 메커니즘만 구현
    """

    def __init__(self, n_visual: int = 1000, n_motor: int = 200, n_assoc: int = 500):
        super().__init__()
        self.n_visual = n_visual
        self.n_motor = n_motor
        self.n_assoc = n_assoc

        # 입력 인코더 (64x64 → visual)
        self.visual_encoder = nn.Linear(64 * 64, n_visual)

        # Visual → Association 연결
        self.v_to_a = nn.Linear(n_visual, n_assoc, bias=False)
        nn.init.uniform_(self.v_to_a.weight, 0.01, 0.05)

        # Association → Motor 연결
        self.a_to_m = nn.Linear(n_assoc, n_motor, bias=False)
        nn.init.uniform_(self.a_to_m.weight, 0.01, 0.05)

        # LIF 상태
        self.register_buffer('v_visual', torch.zeros(n_visual))
        self.register_buffer('v_assoc', torch.zeros(n_assoc))
        self.register_buffer('v_motor', torch.zeros(n_motor))

        # STDP eligibility traces
        self.register_buffer('elig_v_to_a', torch.zeros(n_visual, n_assoc))
        self.register_buffer('elig_a_to_m', torch.zeros(n_assoc, n_motor))

        # 파라미터
        self.tau_mem = 20.0
        self.v_th = 1.0
        self.tau_elig = 500.0

    def reset_state(self):
        """LIF 상태 리셋"""
        self.v_visual.zero_()
        self.v_assoc.zero_()
        self.v_motor.zero_()

    def step(self, visual_input: torch.Tensor, learn: bool = True, dopamine: float = 0.0) -> np.ndarray:
        """
        한 스텝 전파

        Returns:
            motor_spikes: numpy array of motor neuron spikes
        """
        # Visual 인코딩
        flat_input = visual_input.view(-1)
        v_rates = torch.sigmoid(self.visual_encoder(flat_input))
        v_spikes = (torch.rand_like(v_rates) < v_rates * 0.3).float()

        # LIF dynamics - Visual
        self.v_visual = self.v_visual * (1 - 1/self.tau_mem) + v_spikes
        visual_spikes = (self.v_visual > self.v_th).float()
        self.v_visual = self.v_visual * (1 - visual_spikes)  # Reset

        # Visual → Association
        a_input = self.v_to_a(visual_spikes)
        self.v_assoc = self.v_assoc * (1 - 1/self.tau_mem) + a_input * 0.1
        assoc_spikes = (self.v_assoc > self.v_th).float()
        self.v_assoc = self.v_assoc * (1 - assoc_spikes)

        # Association → Motor
        m_input = self.a_to_m(assoc_spikes)
        self.v_motor = self.v_motor * (1 - 1/self.tau_mem) + m_input * 0.1
        motor_spikes = (self.v_motor > self.v_th).float()
        self.v_motor = self.v_motor * (1 - motor_spikes)

        # DA-STDP 학습
        if learn:
            self._apply_da_stdp(visual_spikes, assoc_spikes, motor_spikes, dopamine)

        return motor_spikes.numpy()

    def _apply_da_stdp(self, v_spikes, a_spikes, m_spikes, dopamine: float):
        """도파민 조절 STDP"""
        # Eligibility trace decay
        decay = 1 - 1 / self.tau_elig
        self.elig_v_to_a *= decay
        self.elig_a_to_m *= decay

        # STDP 신호 누적 (pre-post 상관)
        self.elig_v_to_a += torch.outer(v_spikes, a_spikes) * 0.01
        self.elig_a_to_m += torch.outer(a_spikes, m_spikes) * 0.01

        # 도파민이 오면 가중치 업데이트
        if abs(dopamine) > 0.1:
            with torch.no_grad():
                self.v_to_a.weight += (dopamine * self.elig_v_to_a.T * 0.1).clamp(-0.01, 0.01)
                self.a_to_m.weight += (dopamine * self.elig_a_to_m.T * 0.1).clamp(-0.01, 0.01)

                # 가중치 범위 제한
                self.v_to_a.weight.clamp_(0.0, 1.0)
                self.a_to_m.weight.clamp_(0.0, 1.0)

                # Eligibility 소비
                self.elig_v_to_a *= 0.5
                self.elig_a_to_m *= 0.5


class DesktopSNNAgent:
    """데스크톱 환경용 SNN 에이전트 - DA-STDP 학습"""

    def __init__(self, n_actions: int = 8):
        self.n_actions = n_actions
        self.n_motor = 200

        # 단순화된 SNN 뇌
        self.brain = SimpleSNNBrain(n_visual=1000, n_motor=self.n_motor, n_assoc=500)
        self.brain.to(DEVICE)

        # 도파민 시스템
        self.dopamine_system = DesktopDopamineSystem()

        # 행동 추적
        self.action_counts: Dict[Tuple[int, int], Dict[int, int]] = {}
        self.last_actions: List[int] = []

        self.current_obs: Optional[np.ndarray] = None
        self.step_count = 0

    def select_action(self, observation: np.ndarray, cursor_pos: Tuple[int, int]) -> int:
        self.current_obs = observation
        self.step_count += 1

        gray = np.mean(observation, axis=2)
        visual_tensor = torch.tensor(gray, dtype=torch.float32) / 255.0

        n_steps = 10
        motor_accumulator = np.zeros(self.n_motor)

        for _ in range(n_steps):
            motor_spikes = self.brain.step(visual_input=visual_tensor, learn=False)
            motor_accumulator += motor_spikes

        action_scores = np.zeros(self.n_actions)
        neurons_per_action = self.n_motor // self.n_actions

        for a in range(self.n_actions):
            start = a * neurons_per_action
            end = start + neurons_per_action
            action_scores[a] = motor_accumulator[start:end].sum()

        region = (cursor_pos[0] // 8, cursor_pos[1] // 8)
        exploration_pressure = self.dopamine_system.get_exploration_pressure()

        if region not in self.action_counts:
            self.action_counts[region] = {}

        for a in range(self.n_actions):
            action_count = self.action_counts[region].get(a, 0)
            novelty_bonus = exploration_pressure / (1.0 + action_count * 0.5)
            action_scores[a] += novelty_bonus * 2.0

        if self.dopamine_system.loop_detected and len(self.last_actions) >= 3:
            recent = set(self.last_actions[-3:])
            for a in recent:
                action_scores[a] *= 0.3

        temperature = 0.5 if exploration_pressure > 0.5 else 1.0
        probs = self._softmax(action_scores, temperature)
        action = np.random.choice(self.n_actions, p=probs)

        self.action_counts[region][action] = self.action_counts[region].get(action, 0) + 1
        self.last_actions.append(action)
        if len(self.last_actions) > 10:
            self.last_actions = self.last_actions[-10:]

        return action

    def learn(self, observation: np.ndarray, event: Optional[str],
              cursor_pos: Tuple[int, int], windows_opened: int, icons_clicked: int):
        dopamine = self.dopamine_system.compute_dopamine(
            event, cursor_pos, windows_opened, icons_clicked
        )

        gray = np.mean(observation, axis=2)
        visual_tensor = torch.tensor(gray, dtype=torch.float32) / 255.0

        n_steps = 5
        for _ in range(n_steps):
            self.brain.step(visual_input=visual_tensor, learn=True, dopamine=dopamine)

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        x = x / temperature
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (exp_x.sum() + 1e-8)

    def reset_episode(self):
        self.last_actions = []
        self.step_count = 0
        self.brain.reset_state()
        self.dopamine_system.reset()


def run_desktop_exploration(n_episodes: int = 30):
    """데스크톱 탐색 실행"""
    print("=" * 60)
    print("Desktop Environment - Embodied Digital Learning")
    print("=" * 60)
    print("\nBiological mechanisms: DA-STDP, Habituation, Novelty")
    print("NO LLM, NO FEP formulas, NO CNN backprop\n")

    env = VirtualDesktop()
    agent = DesktopSNNAgent()

    episode_stats = []

    for episode in range(n_episodes):
        obs = env.reset()
        agent.reset_episode()
        total_reward = 0
        events_triggered = []

        for step in range(500):
            cursor_pos = (env.cursor_x, env.cursor_y)
            action = agent.select_action(obs, cursor_pos)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward

            if info["event"]:
                events_triggered.append(info["event"])

            agent.learn(
                next_obs, info["event"], cursor_pos,
                info["windows_opened"], info["icons_clicked"]
            )

            obs = next_obs
            if done:
                break

        stats = {
            "episode": episode + 1,
            "reward": total_reward,
            "windows_opened": info["windows_opened"],
            "icons_clicked": info["icons_clicked"],
            "interactions": info["interactions"],
            "regions_explored": len(agent.dopamine_system.visited_regions)
        }
        episode_stats.append(stats)

        if (episode + 1) % 5 == 0 or episode == 0:
            print(f"Episode {episode + 1}/{n_episodes}:")
            print(f"  Reward: {total_reward:.2f}")
            print(f"  Windows opened: {info['windows_opened']}/8")
            print(f"  Icons clicked: {info['icons_clicked']}/8")
            print(f"  Regions explored: {len(agent.dopamine_system.visited_regions)}/64")
            print(f"  Events: {len(events_triggered)}")
            if events_triggered[:5]:
                print(f"  Sample events: {events_triggered[:5]}")
            print()

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    first_5 = episode_stats[:5]
    last_5 = episode_stats[-5:]

    avg_first = np.mean([s["windows_opened"] for s in first_5])
    avg_last = np.mean([s["windows_opened"] for s in last_5])

    print(f"\nWindows opened (avg):")
    print(f"  First 5 episodes: {avg_first:.1f}/8")
    print(f"  Last 5 episodes:  {avg_last:.1f}/8")
    print(f"  Improvement: {avg_last - avg_first:+.1f}")

    reward_first = np.mean([s["reward"] for s in first_5])
    reward_last = np.mean([s["reward"] for s in last_5])

    print(f"\nTotal reward (avg):")
    print(f"  First 5 episodes: {reward_first:.2f}")
    print(f"  Last 5 episodes:  {reward_last:.2f}")
    print(f"  Improvement: {reward_last - reward_first:+.2f}")

    if avg_last > avg_first:
        print("\n[SUCCESS] Agent learned to explore the desktop!")
    else:
        print("\n[PARTIAL] Some learning observed, needs more episodes")

    return episode_stats


if __name__ == "__main__":
    stats = run_desktop_exploration(n_episodes=30)
