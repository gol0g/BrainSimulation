"""
Desktop Environment for Genesis Brain

Phase 4: MiniGrid에서 검증된 하이브리드 아키텍처를 데스크탑으로 확장

안전 메커니즘:
1. Allowlist: 허용된 앱/창만 조작
2. ROI Restriction: 지정된 화면 영역만 사용
3. Action Rate Limit: 초당 행동 수 제한
4. Killswitch: ESC 키로 즉시 중단
5. Sandbox Mode: 실제 입력 없이 시뮬레이션
"""

import time
import numpy as np
from PIL import Image, ImageGrab
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import threading

# Optional imports for actual control
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # 마우스를 왼쪽 상단으로 이동하면 중단
    pyautogui.PAUSE = 0.1  # 행동 간 딜레이
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("[Warning] pyautogui not installed. Running in observation-only mode.")

try:
    import win32gui
    import win32process
    import psutil
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    print("[Warning] win32gui not installed. Window info limited.")


class ActionType(Enum):
    """가능한 행동 타입"""
    NONE = 0
    MOVE = 1        # 마우스 이동
    CLICK = 2       # 왼쪽 클릭
    RIGHT_CLICK = 3 # 오른쪽 클릭
    DOUBLE_CLICK = 4
    DRAG = 5        # 드래그
    TYPE = 6        # 텍스트 입력
    KEY = 7         # 특수 키


@dataclass
class DesktopAction:
    """데스크탑 행동 정의"""
    action_type: ActionType
    x: int = 0
    y: int = 0
    x2: int = 0  # 드래그 종료 좌표
    y2: int = 0
    text: str = ""
    key: str = ""


@dataclass
class SafetyConfig:
    """안전 설정"""
    allowed_apps: List[str]  # 허용된 앱 이름 목록
    allowed_region: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 허용 영역
    max_actions_per_second: float = 2.0
    sandbox_mode: bool = True  # True면 실제 입력 없이 시뮬레이션만
    killswitch_key: str = 'escape'


class SafetyGate:
    """안전 게이트 - 모든 행동을 검증"""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.last_action_time = 0.0
        self.action_count = 0
        self.blocked_count = 0
        self._killswitch_active = False

        # Killswitch 모니터링 스레드 (선택적)
        if PYAUTOGUI_AVAILABLE:
            self._start_killswitch_monitor()

    def _start_killswitch_monitor(self):
        """ESC 키 감지 스레드"""
        def monitor():
            try:
                import keyboard
                keyboard.on_press_key(self.config.killswitch_key,
                                      lambda _: self._activate_killswitch())
            except ImportError:
                pass  # keyboard 모듈 없으면 무시

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def _activate_killswitch(self):
        """킬스위치 활성화"""
        self._killswitch_active = True
        print("[KILLSWITCH] Emergency stop activated!")

    def is_safe(self, action: DesktopAction) -> Tuple[bool, str]:
        """행동이 안전한지 검증"""

        # 1. 킬스위치 확인
        if self._killswitch_active:
            return False, "Killswitch active"

        # 2. 샌드박스 모드면 항상 허용 (실행은 안 함)
        if self.config.sandbox_mode:
            return True, "Sandbox mode - simulated only"

        # 3. Rate limit 확인
        current_time = time.time()
        time_diff = current_time - self.last_action_time
        if time_diff < 1.0 / self.config.max_actions_per_second:
            return False, f"Rate limit: wait {1.0/self.config.max_actions_per_second - time_diff:.2f}s"

        # 4. 좌표 범위 확인
        x1, y1, x2, y2 = self.config.allowed_region
        if action.action_type in [ActionType.MOVE, ActionType.CLICK,
                                   ActionType.RIGHT_CLICK, ActionType.DOUBLE_CLICK]:
            if not (x1 <= action.x <= x2 and y1 <= action.y <= y2):
                self.blocked_count += 1
                return False, f"Position ({action.x}, {action.y}) outside allowed region"

        # 5. 드래그 종료 좌표 확인
        if action.action_type == ActionType.DRAG:
            if not (x1 <= action.x2 <= x2 and y1 <= action.y2 <= y2):
                self.blocked_count += 1
                return False, f"Drag end ({action.x2}, {action.y2}) outside allowed region"

        # 6. 현재 활성 창 확인 (Windows만)
        if WIN32_AVAILABLE and self.config.allowed_apps:
            active_app = self._get_active_app()
            if active_app and active_app.lower() not in [a.lower() for a in self.config.allowed_apps]:
                self.blocked_count += 1
                return False, f"App '{active_app}' not in allowed list"

        return True, "OK"

    def _get_active_app(self) -> Optional[str]:
        """현재 활성 창의 앱 이름 반환"""
        if not WIN32_AVAILABLE:
            return None

        try:
            hwnd = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)
            return process.name()
        except:
            return None

    def record_action(self):
        """행동 기록 (rate limit용)"""
        self.last_action_time = time.time()
        self.action_count += 1


class DesktopObserver:
    """화면 관측 모듈"""

    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size
        self.last_frame = None
        self.frame_count = 0

    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """화면 캡처 및 전처리"""
        # 전체 화면 또는 지정 영역 캡처
        if region:
            screenshot = ImageGrab.grab(bbox=region)
        else:
            screenshot = ImageGrab.grab()

        # 리사이즈
        img = screenshot.resize(self.target_size, Image.BILINEAR)

        # numpy 배열로 변환
        frame = np.array(img, dtype=np.float32) / 255.0

        self.last_frame = frame
        self.frame_count += 1

        return frame

    def get_frame_chw(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """CNN 입력 형식 (C, H, W)으로 반환"""
        frame = self.capture_screen(region)
        return np.transpose(frame, (2, 0, 1))

    def compute_frame_diff(self, frame: np.ndarray) -> float:
        """이전 프레임과의 차이 계산 (변화 감지)"""
        if self.last_frame is None:
            return 0.0

        diff = np.abs(frame - self.last_frame).mean()
        return diff


class DesktopExecutor:
    """행동 실행 모듈"""

    def __init__(self, safety_gate: SafetyGate):
        self.safety_gate = safety_gate
        self.execution_log: List[Dict] = []

    def execute(self, action: DesktopAction) -> Tuple[bool, str]:
        """행동 실행"""
        # 안전 검증
        is_safe, message = self.safety_gate.is_safe(action)

        log_entry = {
            'time': time.time(),
            'action': action,
            'safe': is_safe,
            'message': message,
            'executed': False
        }

        if not is_safe:
            self.execution_log.append(log_entry)
            return False, message

        # 샌드박스 모드면 실행 안 함
        if self.safety_gate.config.sandbox_mode:
            log_entry['executed'] = False
            log_entry['message'] = "Simulated (sandbox mode)"
            self.execution_log.append(log_entry)
            self.safety_gate.record_action()
            return True, "Simulated"

        # 실제 실행
        if not PYAUTOGUI_AVAILABLE:
            return False, "pyautogui not available"

        try:
            if action.action_type == ActionType.MOVE:
                pyautogui.moveTo(action.x, action.y)
            elif action.action_type == ActionType.CLICK:
                pyautogui.click(action.x, action.y)
            elif action.action_type == ActionType.RIGHT_CLICK:
                pyautogui.rightClick(action.x, action.y)
            elif action.action_type == ActionType.DOUBLE_CLICK:
                pyautogui.doubleClick(action.x, action.y)
            elif action.action_type == ActionType.DRAG:
                pyautogui.moveTo(action.x, action.y)
                pyautogui.drag(action.x2 - action.x, action.y2 - action.y)
            elif action.action_type == ActionType.TYPE:
                pyautogui.typewrite(action.text, interval=0.05)
            elif action.action_type == ActionType.KEY:
                pyautogui.press(action.key)

            log_entry['executed'] = True
            self.execution_log.append(log_entry)
            self.safety_gate.record_action()
            return True, "Executed"

        except Exception as e:
            log_entry['message'] = f"Execution error: {e}"
            self.execution_log.append(log_entry)
            return False, str(e)


class DesktopEnv:
    """
    데스크탑 환경 - MiniGrid와 유사한 인터페이스

    obs, reward, done, info = env.step(action)
    obs = env.reset()
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        # 기본 안전 설정 (메모장만 허용, 전체 화면)
        if config is None:
            config = SafetyConfig(
                allowed_apps=['notepad.exe', 'explorer.exe'],
                allowed_region=(0, 0, 1920, 1080),
                max_actions_per_second=2.0,
                sandbox_mode=True  # 기본은 샌드박스
            )

        self.config = config
        self.safety_gate = SafetyGate(config)
        self.observer = DesktopObserver(target_size=(256, 256))
        self.executor = DesktopExecutor(self.safety_gate)

        # 상태 추적
        self.step_count = 0
        self.episode_reward = 0.0
        self.done = False

        # 내재적 보상 계산용
        self.prev_frame = None
        self.novelty_history: List[float] = []

    def reset(self) -> np.ndarray:
        """환경 리셋"""
        self.step_count = 0
        self.episode_reward = 0.0
        self.done = False
        self.prev_frame = None
        self.novelty_history = []

        # 초기 관측
        obs = self.observer.get_frame_chw(self.config.allowed_region)
        self.prev_frame = obs.copy()

        return obs

    def step(self, action: DesktopAction) -> Tuple[np.ndarray, float, bool, Dict]:
        """한 스텝 실행"""
        self.step_count += 1

        # 행동 실행
        success, message = self.executor.execute(action)

        # 짧은 대기 (화면 변화 반영)
        time.sleep(0.1)

        # 새 관측
        obs = self.observer.get_frame_chw(self.config.allowed_region)

        # 내재적 보상 계산
        reward = self._compute_intrinsic_reward(obs, success)
        self.episode_reward += reward

        # 정보
        info = {
            'success': success,
            'message': message,
            'step': self.step_count,
            'blocked_count': self.safety_gate.blocked_count,
            'sandbox_mode': self.config.sandbox_mode
        }

        # 업데이트
        self.prev_frame = obs.copy()

        return obs, reward, self.done, info

    def _compute_intrinsic_reward(self, obs: np.ndarray, action_success: bool) -> float:
        """내재적 보상 계산 (FEP 기반)"""
        reward = 0.0

        # 1. 행동 성공 보너스
        if action_success:
            reward += 0.1

        # 2. 화면 변화 (novelty)
        if self.prev_frame is not None:
            diff = np.abs(obs - self.prev_frame).mean()

            # 적당한 변화에 보상 (너무 크거나 작으면 감점)
            if 0.01 < diff < 0.3:
                reward += 0.2 * diff

            self.novelty_history.append(diff)

        # 3. 다양한 행동 보너스 (탐색 장려)
        if len(self.executor.execution_log) > 1:
            recent_actions = [e['action'].action_type for e in self.executor.execution_log[-10:]]
            diversity = len(set(recent_actions)) / len(recent_actions)
            reward += 0.1 * diversity

        return reward

    def get_context_info(self) -> Dict:
        """현재 컨텍스트 정보 (하이브리드 에이전트용)"""
        info = {
            'step': self.step_count,
            'episode_reward': self.episode_reward,
            'action_count': self.safety_gate.action_count,
            'blocked_count': self.safety_gate.blocked_count,
        }

        # 마우스 위치
        if PYAUTOGUI_AVAILABLE:
            pos = pyautogui.position()
            info['mouse_x'] = pos[0] / 1920  # 정규화
            info['mouse_y'] = pos[1] / 1080

        # 활성 창 정보
        if WIN32_AVAILABLE:
            info['active_app'] = self.safety_gate._get_active_app()

        # 최근 novelty 평균
        if self.novelty_history:
            info['avg_novelty'] = np.mean(self.novelty_history[-10:])

        return info


def test_desktop_env():
    """데스크탑 환경 테스트 (샌드박스 모드)"""
    print("=" * 60)
    print("Desktop Environment Test (Sandbox Mode)")
    print("=" * 60)

    # 안전 설정 (샌드박스 모드)
    config = SafetyConfig(
        allowed_apps=['notepad.exe'],
        allowed_region=(100, 100, 800, 600),  # 제한된 영역
        max_actions_per_second=2.0,
        sandbox_mode=True
    )

    env = DesktopEnv(config)

    # 리셋
    print("\n[1] Reset environment...")
    obs = env.reset()
    print(f"    Observation shape: {obs.shape}")
    print(f"    Value range: [{obs.min():.3f}, {obs.max():.3f}]")

    # 몇 가지 행동 테스트
    test_actions = [
        DesktopAction(ActionType.MOVE, x=400, y=300),
        DesktopAction(ActionType.CLICK, x=400, y=300),
        DesktopAction(ActionType.MOVE, x=200, y=200),
        DesktopAction(ActionType.MOVE, x=1000, y=500),  # 허용 영역 밖 - 차단 예상
    ]

    print("\n[2] Testing actions...")
    for i, action in enumerate(test_actions):
        obs, reward, done, info = env.step(action)
        print(f"    Action {i+1} ({action.action_type.name} @ {action.x},{action.y}):")
        print(f"        Success: {info['success']}, Message: {info['message']}")
        print(f"        Reward: {reward:.3f}")

    # 컨텍스트 정보
    print("\n[3] Context info:")
    context = env.get_context_info()
    for key, value in context.items():
        print(f"    {key}: {value}")

    # 요약
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total steps: {env.step_count}")
    print(f"Episode reward: {env.episode_reward:.3f}")
    print(f"Blocked actions: {env.safety_gate.blocked_count}")
    print(f"Sandbox mode: {config.sandbox_mode}")


if __name__ == '__main__':
    test_desktop_env()
