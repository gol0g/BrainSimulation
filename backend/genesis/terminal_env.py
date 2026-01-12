"""
Phase B: Terminal Environment for Genesis Brain
================================================

터미널에서 자유롭게 학습하는 에이전트를 위한 환경.
안전한 샌드박스 내에서 명령어를 실행하고 결과를 관찰.

설계 원칙:
1. Safety First - 허용된 명령어만 실행
2. Information Gain - 새로운 정보 발견에 내재적 보상
3. Survival - 세션 유지가 기본 목표
4. Curiosity - 탐색 행동에 보상
"""

import os
import subprocess
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import shlex
import time


class CommandCategory(Enum):
    """명령어 카테고리 (안전 등급)"""
    SAFE_READ = "safe_read"      # ls, cat, head, pwd - 읽기만
    SAFE_NAVIGATE = "safe_nav"   # cd - 이동만
    SAFE_SEARCH = "safe_search"  # find, grep - 검색
    SAFE_WRITE = "safe_write"    # echo, touch - 제한적 쓰기
    DANGEROUS = "dangerous"       # rm, mv, sudo - 금지
    UNKNOWN = "unknown"


@dataclass
class CommandResult:
    """명령어 실행 결과"""
    command: str
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    was_blocked: bool = False
    block_reason: str = ""


@dataclass
class TerminalState:
    """터미널 상태"""
    cwd: str
    last_output: str
    last_command: str
    command_history: List[str] = field(default_factory=list)
    discovered_paths: set = field(default_factory=set)
    discovered_files: set = field(default_factory=set)
    session_start: float = field(default_factory=time.time)
    total_commands: int = 0
    blocked_commands: int = 0

    def to_observation(self, max_output_len: int = 500) -> Dict:
        """관측 가능한 상태로 변환"""
        return {
            'cwd': self.cwd,
            'last_output': self.last_output[:max_output_len],
            'last_command': self.last_command,
            'history_len': len(self.command_history),
            'recent_history': self.command_history[-5:] if self.command_history else [],
            'discovered_paths_count': len(self.discovered_paths),
            'discovered_files_count': len(self.discovered_files),
            'session_duration': time.time() - self.session_start,
            'total_commands': self.total_commands,
        }


class SafetyGate:
    """
    명령어 안전성 검사 게이트

    Level 1: 명령어 화이트리스트
    Level 2: 인자 검사 (위험한 경로 차단)
    Level 3: 패턴 매칭 (위험한 조합 차단)
    """

    # 허용된 명령어 (Level 1) - Unix + Windows
    ALLOWED_COMMANDS = {
        # 읽기 전용 (Unix)
        'ls': CommandCategory.SAFE_READ,
        'cat': CommandCategory.SAFE_READ,
        'head': CommandCategory.SAFE_READ,
        'tail': CommandCategory.SAFE_READ,
        'pwd': CommandCategory.SAFE_READ,
        'whoami': CommandCategory.SAFE_READ,
        'date': CommandCategory.SAFE_READ,
        'wc': CommandCategory.SAFE_READ,
        'file': CommandCategory.SAFE_READ,
        'stat': CommandCategory.SAFE_READ,

        # 읽기 전용 (Windows)
        'dir': CommandCategory.SAFE_READ,
        'type': CommandCategory.SAFE_READ,
        'more': CommandCategory.SAFE_READ,

        # 네비게이션
        'cd': CommandCategory.SAFE_NAVIGATE,

        # 검색 (Unix)
        'find': CommandCategory.SAFE_SEARCH,
        'grep': CommandCategory.SAFE_SEARCH,
        'which': CommandCategory.SAFE_SEARCH,

        # 검색 (Windows)
        'findstr': CommandCategory.SAFE_SEARCH,
        'where': CommandCategory.SAFE_SEARCH,

        # 제한적 쓰기 (샌드박스 내에서만)
        'echo': CommandCategory.SAFE_WRITE,
        'touch': CommandCategory.SAFE_WRITE,
        'mkdir': CommandCategory.SAFE_WRITE,
    }

    # 절대 금지 명령어 (Level 1) - Unix + Windows
    BLOCKED_COMMANDS = {
        # Unix
        'rm', 'rmdir', 'mv', 'cp',  # 파일 조작
        'sudo', 'su', 'chmod', 'chown',  # 권한
        'kill', 'pkill', 'killall',  # 프로세스
        'shutdown', 'reboot', 'halt',  # 시스템
        'dd', 'mkfs', 'fdisk',  # 디스크
        'wget', 'curl', 'nc', 'ssh',  # 네트워크
        'python', 'python3', 'node', 'bash', 'sh',  # 스크립트 실행
        'eval', 'exec',  # 동적 실행
        # Windows
        'del', 'erase', 'move', 'copy', 'xcopy', 'robocopy',  # 파일 조작
        'taskkill', 'tasklist',  # 프로세스
        'format', 'diskpart',  # 디스크
        'powershell', 'cmd', 'wscript', 'cscript',  # 스크립트
        'net', 'netsh', 'reg', 'regedit',  # 시스템/네트워크
    }

    # 위험한 경로 패턴 (Level 2) - Unix + Windows
    DANGEROUS_PATHS = [
        # Cross-platform
        r'\.\.[/\\]',  # 상위 디렉토리 탈출 (../ 또는 ..\)

        # Unix specific
        r'^/',  # 절대 경로 (루트)
        r'/etc/',
        r'/usr/',
        r'/bin/',
        r'/sbin/',
        r'/var/',
        r'/root/',
        r'/home/[^/]+/\.',  # 숨김 파일
        r'~/',  # 홈 디렉토리

        # Windows specific
        r'^[A-Za-z]:',  # 드라이브 절대 경로
        r'\\\\',  # UNC 경로
        r'\\Windows\\',
        r'\\System32\\',
        r'\\Program Files',
        r'\\Users\\[^\\]+\\AppData',
    ]

    # 위험한 패턴 (Level 3)
    DANGEROUS_PATTERNS = [
        r'\|',  # 파이프
        r';',   # 명령어 체인
        r'&&',  # AND 체인
        r'\|\|',  # OR 체인
        r'`',   # 백틱 실행
        r'\$\(',  # 서브쉘
        r'>',   # 리다이렉션
        r'<',   # 입력 리다이렉션
    ]

    def __init__(self, sandbox_root: str):
        """
        Args:
            sandbox_root: 샌드박스 루트 디렉토리 (이 안에서만 작업 허용)
        """
        self.sandbox_root = os.path.abspath(sandbox_root)

    def validate_command(self, command: str) -> Tuple[bool, str]:
        """
        명령어 검증

        Returns:
            (is_safe, reason)
        """
        if not command.strip():
            return False, "Empty command"

        # Level 3: 위험한 패턴 검사 (먼저)
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False, f"Dangerous pattern detected: {pattern}"

        # 명령어 파싱 (Windows에서는 백슬래시를 보존)
        try:
            parts = shlex.split(command, posix=(os.name != 'nt'))
        except ValueError as e:
            return False, f"Invalid command syntax: {e}"

        if not parts:
            return False, "Empty command after parsing"

        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []

        # Level 1: 명령어 화이트리스트
        if cmd in self.BLOCKED_COMMANDS:
            return False, f"Blocked command: {cmd}"

        if cmd not in self.ALLOWED_COMMANDS:
            return False, f"Unknown command: {cmd}"

        # Level 2: 인자 검사
        for arg in args:
            # Windows 플래그 건너뛰기 (/a, /s, /b 등 - 짧은 슬래시 인자)
            if os.name == 'nt' and re.match(r'^/[a-zA-Z0-9]{1,2}$', arg):
                continue  # Windows 플래그는 허용

            for pattern in self.DANGEROUS_PATHS:
                if re.search(pattern, arg):
                    return False, f"Dangerous path pattern: {arg}"

        return True, "OK"

    def is_path_in_sandbox(self, path: str, current_dir: str) -> bool:
        """경로가 샌드박스 내에 있는지 확인"""
        if os.path.isabs(path):
            abs_path = os.path.abspath(path)
        else:
            abs_path = os.path.abspath(os.path.join(current_dir, path))

        return abs_path.startswith(self.sandbox_root)


class IntrinsicReward:
    """
    내재적 보상 계산

    FEP 기반: 정보 이득 = 놀라움 감소 = 좋음
    """

    def __init__(self):
        self.seen_outputs = set()  # 본 적 있는 출력 해시
        self.seen_paths = set()    # 발견한 경로
        self.seen_files = set()    # 발견한 파일

    def compute_reward(
        self,
        result: CommandResult,
        state: TerminalState,
        category: CommandCategory
    ) -> Tuple[float, Dict[str, float]]:
        """
        보상 계산

        Returns:
            (total_reward, reward_breakdown)
        """
        rewards = {}

        # 1. 실행 성공 보상
        if result.return_code == 0 and not result.was_blocked:
            rewards['success'] = 0.1
        else:
            rewards['success'] = -0.05

        # 2. 정보 이득 보상 (새로운 출력)
        if result.stdout:
            output_hash = hashlib.md5(result.stdout.encode()).hexdigest()[:8]
            if output_hash not in self.seen_outputs:
                self.seen_outputs.add(output_hash)
                rewards['novelty'] = 0.2  # 새로운 정보!
            else:
                rewards['novelty'] = 0.0
        else:
            rewards['novelty'] = 0.0

        # 3. 탐색 보상 (새로운 경로/파일 발견)
        new_discoveries = self._extract_discoveries(result.stdout, state.cwd)
        new_paths = new_discoveries['paths'] - self.seen_paths
        new_files = new_discoveries['files'] - self.seen_files

        self.seen_paths.update(new_paths)
        self.seen_files.update(new_files)

        rewards['path_discovery'] = len(new_paths) * 0.15
        rewards['file_discovery'] = len(new_files) * 0.1

        # 4. 카테고리 보너스
        if category == CommandCategory.SAFE_SEARCH:
            rewards['search_bonus'] = 0.05  # 검색은 탐색적
        elif category == CommandCategory.SAFE_NAVIGATE:
            rewards['navigate_bonus'] = 0.03  # 이동도 좋음

        # 5. 반복 페널티 (같은 명령 반복)
        if state.command_history and result.command == state.command_history[-1]:
            rewards['repetition_penalty'] = -0.1

        total = sum(rewards.values())
        return total, rewards

    def _extract_discoveries(self, output: str, cwd: str) -> Dict[str, set]:
        """출력에서 경로/파일 발견"""
        paths = set()
        files = set()

        if not output:
            return {'paths': paths, 'files': files}

        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # ls 출력 파싱
            if '/' in line or os.path.sep in line:
                paths.add(line)
            elif '.' in line and not line.startswith('.'):
                # 파일처럼 보이는 것
                files.add(line)
            elif line and not line.startswith('total'):
                # 일반 항목
                files.add(line)

        return {'paths': paths, 'files': files}


class TerminalEnv:
    """
    터미널 환경

    Genesis Brain이 터미널에서 학습하기 위한 환경.
    안전한 샌드박스 내에서 명령어를 실행.
    """

    def __init__(
        self,
        sandbox_root: Optional[str] = None,
        max_output_length: int = 1000,
        timeout: float = 5.0,
        episode_max_steps: int = 100,
    ):
        """
        Args:
            sandbox_root: 샌드박스 루트 (None이면 임시 디렉토리 생성)
            max_output_length: 출력 최대 길이
            timeout: 명령어 실행 타임아웃
            episode_max_steps: 에피소드 최대 스텝
        """
        # 샌드박스 설정
        if sandbox_root is None:
            import tempfile
            self.sandbox_root = tempfile.mkdtemp(prefix="genesis_terminal_")
            self._created_sandbox = True
        else:
            self.sandbox_root = os.path.abspath(sandbox_root)
            self._created_sandbox = False
            os.makedirs(self.sandbox_root, exist_ok=True)

        self.max_output_length = max_output_length
        self.timeout = timeout
        self.episode_max_steps = episode_max_steps

        # 컴포넌트 초기화
        self.safety_gate = SafetyGate(self.sandbox_root)
        self.intrinsic_reward = IntrinsicReward()

        # 상태
        self.state = TerminalState(
            cwd=self.sandbox_root,
            last_output="",
            last_command="",
        )

        self.step_count = 0
        self.episode_reward = 0.0

        # 초기 샌드박스 구조 생성
        self._setup_sandbox()

    def _setup_sandbox(self):
        """샌드박스 초기 구조 생성"""
        # 기본 디렉토리 구조
        dirs = [
            'documents',
            'projects',
            'projects/demo',
            'data',
            'notes',
        ]

        for d in dirs:
            os.makedirs(os.path.join(self.sandbox_root, d), exist_ok=True)

        # 샘플 파일 생성
        files = {
            'readme.txt': 'Welcome to the Genesis Terminal Sandbox!\nExplore freely and learn.\n',
            'documents/hello.txt': 'Hello, World!\n',
            'projects/demo/main.py': '# Demo Python file\nprint("Hello from demo!")\n',
            'data/numbers.txt': '\n'.join(str(i) for i in range(1, 11)),
            'notes/todo.txt': '- Learn terminal commands\n- Explore the sandbox\n- Find hidden files\n',
        }

        for path, content in files.items():
            full_path = os.path.join(self.sandbox_root, path)
            with open(full_path, 'w') as f:
                f.write(content)

        # 숨김 파일 (발견 보상 높음)
        hidden = os.path.join(self.sandbox_root, '.secret')
        with open(hidden, 'w') as f:
            f.write('You found the secret file! Congratulations!\n')

    def reset(self) -> Dict:
        """환경 리셋"""
        self.state = TerminalState(
            cwd=self.sandbox_root,
            last_output=f"Welcome to Genesis Terminal\nSandbox: {self.sandbox_root}\n",
            last_command="",
        )
        self.step_count = 0
        self.episode_reward = 0.0
        self.intrinsic_reward = IntrinsicReward()  # 보상 상태도 리셋

        return self.state.to_observation()

    def step(self, command: str) -> Tuple[Dict, float, bool, Dict]:
        """
        명령어 실행

        Args:
            command: 실행할 명령어

        Returns:
            (observation, reward, done, info)
        """
        self.step_count += 1

        # 안전성 검사
        is_safe, reason = self.safety_gate.validate_command(command)

        if not is_safe:
            # 차단된 명령어
            result = CommandResult(
                command=command,
                stdout="",
                stderr=f"Command blocked: {reason}",
                return_code=-1,
                execution_time=0.0,
                was_blocked=True,
                block_reason=reason,
            )
            self.state.blocked_commands += 1
        else:
            # 명령어 실행
            result = self._execute_command(command)

        # 상태 업데이트
        self.state.last_command = command
        self.state.last_output = result.stdout or result.stderr
        self.state.command_history.append(command)
        self.state.total_commands += 1

        # cd 처리
        if command.strip().startswith('cd ') and result.return_code == 0:
            parts = shlex.split(command)
            if len(parts) > 1:
                new_dir = parts[1]
                if not os.path.isabs(new_dir):
                    new_dir = os.path.join(self.state.cwd, new_dir)
                new_dir = os.path.abspath(new_dir)
                if self.safety_gate.is_path_in_sandbox(new_dir, self.state.cwd):
                    self.state.cwd = new_dir

        # 보상 계산
        category = self.safety_gate.ALLOWED_COMMANDS.get(
            shlex.split(command)[0] if command.strip() else '',
            CommandCategory.UNKNOWN
        )
        reward, reward_breakdown = self.intrinsic_reward.compute_reward(
            result, self.state, category
        )
        self.episode_reward += reward

        # 종료 조건
        done = self.step_count >= self.episode_max_steps

        info = {
            'result': result,
            'reward_breakdown': reward_breakdown,
            'category': category.value if category else 'unknown',
            'step': self.step_count,
            'episode_reward': self.episode_reward,
        }

        return self.state.to_observation(), reward, done, info

    def _execute_command(self, command: str) -> CommandResult:
        """명령어 실행 (샌드박스 내에서)"""
        start_time = time.time()

        try:
            # Windows/Unix 호환
            if os.name == 'nt':
                # Windows: cmd.exe 사용, UTF-8 인코딩
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self.state.cwd,
                    encoding='utf-8',
                    errors='replace',  # 인코딩 에러 무시
                )
            else:
                # Unix: bash 사용
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self.state.cwd,
                    encoding='utf-8',
                    errors='replace',
                    executable='/bin/bash',
                )

            stdout, stderr = process.communicate(timeout=self.timeout)
            return_code = process.returncode

        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = "", "Command timed out"
            return_code = -1
        except Exception as e:
            stdout, stderr = "", str(e)
            return_code = -1

        execution_time = time.time() - start_time

        # None 처리
        stdout = stdout or ""
        stderr = stderr or ""

        # 출력 길이 제한
        if len(stdout) > self.max_output_length:
            stdout = stdout[:self.max_output_length] + "\n... (truncated)"
        if len(stderr) > self.max_output_length:
            stderr = stderr[:self.max_output_length] + "\n... (truncated)"

        return CommandResult(
            command=command,
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            execution_time=execution_time,
        )

    def get_action_space(self) -> List[str]:
        """
        사용 가능한 명령어 템플릿 반환

        에이전트가 선택할 수 있는 기본 액션들
        플랫폼에 따라 다른 명령어 반환
        """
        if os.name == 'nt':
            # Windows 명령어
            return [
                'dir',
                'dir /a',
                'cd',  # pwd equivalent
                'cd ..',
                'cd documents',
                'cd projects',
                'cd data',
                'cd notes',
                'type readme.txt',
                'type documents\\hello.txt',
                'type notes\\todo.txt',
                'dir /s /b *.txt',
                'dir /s /b *.py',
                'findstr /s "hello" *.*',
                'findstr /s "demo" *.*',
            ]
        else:
            # Unix 명령어
            return [
                'ls',
                'ls -la',
                'pwd',
                'cd ..',
                'cd documents',
                'cd projects',
                'cd data',
                'cd notes',
                'cat readme.txt',
                'head -5 readme.txt',
                'tail -5 readme.txt',
                'find . -name "*.txt"',
                'find . -name "*.py"',
                'grep -r "hello" .',
                'wc -l readme.txt',
            ]

    def close(self):
        """환경 정리"""
        if self._created_sandbox:
            import shutil
            try:
                shutil.rmtree(self.sandbox_root)
            except Exception:
                pass


class TerminalAgent:
    """
    터미널 환경용 기본 에이전트 (단순 Q-learning)
    """

    def __init__(
        self,
        env: TerminalEnv,
        exploration_rate: float = 0.3,
        learning_rate: float = 0.1,
    ):
        self.env = env
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate

        self.action_values: Dict[str, float] = {}
        self.state_action_counts: Dict[Tuple[str, str], int] = {}
        self.success_patterns: Dict[str, List[str]] = {}

    def select_action(self, observation: Dict) -> str:
        import random
        cwd = observation['cwd']
        actions = self.env.get_action_space()

        if random.random() < self.exploration_rate:
            action_counts = [self.state_action_counts.get((cwd, a), 0) for a in actions]
            min_count = min(action_counts)
            least_tried = [a for a, c in zip(actions, action_counts) if c == min_count]
            return random.choice(least_tried)

        action_scores = [(a, self.action_values.get(a, 0.0)) for a in actions]
        actions_sorted = sorted(action_scores, key=lambda x: x[1], reverse=True)
        return random.choice(actions_sorted[:3])[0]

    def update(self, action: str, reward: float, observation: Dict, info: Dict):
        cwd = observation['cwd']
        old_value = self.action_values.get(action, 0.0)
        self.action_values[action] = old_value + self.learning_rate * (reward - old_value)
        self.state_action_counts[(cwd, action)] = self.state_action_counts.get((cwd, action), 0) + 1


class GenesisTerminalAgent:
    """
    Genesis Brain 터미널 에이전트

    FEP 기반 행동 선택:
    G(a) = Risk + Ambiguity + Complexity

    핵심 메커니즘:
    1. Expected Free Energy (G) 계산
    2. Episodic Memory + Consolidation
    3. Causal Discovery (command → outcome patterns)
    4. Regret Tracking (counterfactual reasoning)
    5. Curiosity-driven Exploration
    """

    def __init__(
        self,
        env: TerminalEnv,
        exploration_rate: float = 0.2,
        learning_rate: float = 0.1,
        memory_capacity: int = 100,
        consolidation_interval: int = 10,
    ):
        self.env = env
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate

        # === FEP Components ===
        # Transition model: P(s'|s, a) approximated by outcome history
        self.transition_history: Dict[str, Dict[str, List[Dict]]] = {}  # cmd -> outcome_type -> [outcomes]

        # Preference distribution P(o) - what outcomes we prefer
        self.preferences = {
            'success': 1.0,      # 높은 보상 성공
            'novelty': 0.8,      # 새로운 정보
            'discovery': 0.6,   # 파일/경로 발견
            'neutral': 0.1,     # 낮은 보상 성공 (별로 좋지 않음)
            'blocked': -0.5,    # 차단됨
            'error': -0.3,      # 에러
        }

        # === Memory System ===
        self.episodic_memory: List[Dict] = []  # Recent experiences
        self.memory_capacity = memory_capacity
        self.semantic_memory: Dict[str, Dict] = {}  # Consolidated knowledge

        # Success patterns (protected from forgetting)
        self.success_patterns: Dict[str, List[str]] = {}  # cwd -> [successful commands]
        self.pattern_strength: Dict[str, float] = {}  # pattern -> strength

        # === Causal Discovery ===
        self.causal_links: Dict[str, Dict[str, float]] = {}  # cause -> {effect: strength}
        # e.g., "cd documents" -> {"see_hello.txt": 0.9}

        # === Regret Tracking ===
        self.regret_history: List[float] = []
        self.last_action_g: Dict[str, float] = {}  # G values from last selection

        # === Curiosity ===
        self.novelty_decay = 0.95  # How fast novelty wears off
        self.seen_outputs: Dict[str, int] = {}  # output_hash -> visit count

        # === Statistics ===
        self.total_steps = 0
        self.total_episodes = 0
        self.consolidation_interval = consolidation_interval

    def compute_G(self, action: str, observation: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Expected Free Energy 계산

        G(a) = Risk + Ambiguity + Complexity

        낮을수록 좋은 행동
        """
        cwd = observation['cwd']
        components = {}

        # === Risk: KL[Q(o|s',a) || P(o)] ===
        # 예상 결과가 선호 분포에서 얼마나 벗어나는가
        outcome_history = self.transition_history.get(action, {})
        total_outcomes = sum(len(v) for v in outcome_history.values())

        if total_outcomes > 0:
            # 과거 결과 기반 예측
            expected_outcomes = {}
            for outcome_type, outcomes in outcome_history.items():
                expected_outcomes[outcome_type] = len(outcomes) / total_outcomes

            # Risk = sum of (expected * -preference)
            risk = 0.0
            for outcome_type, prob in expected_outcomes.items():
                pref = self.preferences.get(outcome_type, 0.0)
                risk -= prob * pref  # 선호하면 risk 감소
        else:
            # Unknown action - neutral risk
            risk = 0.1

        components['risk'] = risk

        # === Ambiguity: Transition uncertainty ===
        # 결과가 얼마나 불확실한가 (variance)
        if total_outcomes > 5:
            # Sufficient data - calculate variance proxy
            n_types = len(outcome_history)
            ambiguity = 0.5 / (1 + n_types)  # More diverse outcomes = more ambiguity
        else:
            # Unknown action - high ambiguity (but also high curiosity value!)
            ambiguity = 0.3

        components['ambiguity'] = ambiguity

        # === Complexity: State change prediction ===
        # 상태 변화가 얼마나 예측하기 어려운가
        if action.startswith('cd'):
            complexity = 0.2  # Navigation changes state significantly
        elif action.startswith('type') or action.startswith('cat'):
            complexity = 0.1  # Read commands are predictable
        else:
            complexity = 0.15

        components['complexity'] = complexity

        # === Curiosity Bonus (negative G for novel actions) ===
        action_key = f"{cwd}:{action}"
        visit_count = self.seen_outputs.get(action_key, 0)
        # Reduced curiosity to balance exploration/exploitation
        curiosity_bonus = -0.15 / (1 + visit_count)  # Was -0.3
        components['curiosity'] = curiosity_bonus

        # === Memory Bonus (successful patterns) ===
        # Strengthened memory bonus for faster exploitation
        if cwd in self.success_patterns and action in self.success_patterns[cwd]:
            memory_bonus = -0.4 * self.pattern_strength.get(f"{cwd}:{action}", 0.5)  # Was -0.2
        else:
            memory_bonus = 0.0
        components['memory'] = memory_bonus

        # === Causal Bonus (known beneficial chains) ===
        causal_bonus = 0.0
        if action in self.causal_links:
            effects = self.causal_links[action]
            for effect, strength in effects.items():
                if 'success' in effect or 'discover' in effect:
                    causal_bonus -= strength * 0.1  # Good effects reduce G
        components['causal'] = causal_bonus

        # Total G
        G = risk + ambiguity + complexity + curiosity_bonus + memory_bonus + causal_bonus
        return G, components

    def select_action(self, observation: Dict) -> str:
        """
        FEP 기반 행동 선택

        최소 G를 가진 행동 선택 (with exploration)
        """
        import random
        import math

        actions = self.env.get_action_space()

        # Compute G for all actions
        g_values = {}
        g_components = {}
        for action in actions:
            g, components = self.compute_G(action, observation)
            g_values[action] = g
            g_components[action] = components

        self.last_action_g = g_values.copy()

        # Softmax selection (lower G = higher probability)
        # P(a) ∝ exp(-G(a) / temperature)
        temperature = 0.1 + 0.2 * self.exploration_rate  # Higher exploration = softer selection

        min_g = min(g_values.values())
        exp_values = {a: math.exp(-(g - min_g) / temperature) for a, g in g_values.items()}
        total_exp = sum(exp_values.values())
        probs = {a: v / total_exp for a, v in exp_values.items()}

        # Sample action
        r = random.random()
        cumsum = 0.0
        for action, prob in probs.items():
            cumsum += prob
            if r <= cumsum:
                return action

        return actions[-1]

    def update(
        self,
        action: str,
        reward: float,
        observation: Dict,
        info: Dict
    ):
        """학습 업데이트"""
        cwd = observation['cwd']
        result = info['result']
        self.total_steps += 1

        # === Update transition history ===
        if action not in self.transition_history:
            self.transition_history[action] = {}

        # Classify outcome based on actual reward value
        if result.was_blocked:
            outcome_type = 'blocked'
        elif result.return_code != 0:
            outcome_type = 'error'
        elif info['reward_breakdown'].get('novelty', 0) > 0:
            outcome_type = 'novelty'
        elif info['reward_breakdown'].get('path_discovery', 0) > 0 or \
             info['reward_breakdown'].get('file_discovery', 0) > 0:
            outcome_type = 'discovery'
        elif reward > 0.2:  # Good success
            outcome_type = 'success'
        else:
            outcome_type = 'neutral'  # Low-reward success

        if outcome_type not in self.transition_history[action]:
            self.transition_history[action][outcome_type] = []

        self.transition_history[action][outcome_type].append({
            'cwd': cwd,
            'reward': reward,
            'step': self.total_steps,
        })

        # === Update episodic memory ===
        experience = {
            'action': action,
            'cwd': cwd,
            'outcome': outcome_type,
            'reward': reward,
            'step': self.total_steps,
            'output_preview': observation['last_output'][:100],
        }
        self.episodic_memory.append(experience)

        # Trim memory
        if len(self.episodic_memory) > self.memory_capacity:
            self.episodic_memory = self.episodic_memory[-self.memory_capacity:]

        # === Update novelty tracking ===
        action_key = f"{cwd}:{action}"
        self.seen_outputs[action_key] = self.seen_outputs.get(action_key, 0) + 1

        # === Update success patterns ===
        if reward > 0.2:  # Significant positive reward
            if cwd not in self.success_patterns:
                self.success_patterns[cwd] = []
            if action not in self.success_patterns[cwd]:
                self.success_patterns[cwd].append(action)

            pattern_key = f"{cwd}:{action}"
            old_strength = self.pattern_strength.get(pattern_key, 0.0)
            self.pattern_strength[pattern_key] = min(1.0, old_strength + 0.1)

        # === Update causal links ===
        # Look for patterns in recent history
        if len(self.episodic_memory) >= 2:
            prev = self.episodic_memory[-2]
            curr = self.episodic_memory[-1]

            # If previous action led to current discovery
            if prev['outcome'] in ['success', 'novelty'] and curr['outcome'] == 'discovery':
                cause = prev['action']
                effect = f"discover_after_{prev['action']}"

                if cause not in self.causal_links:
                    self.causal_links[cause] = {}

                old_strength = self.causal_links[cause].get(effect, 0.0)
                self.causal_links[cause][effect] = min(1.0, old_strength + 0.1)

        # === Compute regret ===
        if self.last_action_g:
            chosen_g = self.last_action_g.get(action, 0.0)
            min_g = min(self.last_action_g.values())
            regret = chosen_g - min_g  # Should be >= 0

            # Update based on actual outcome
            if reward > 0:
                regret *= 0.5  # Good outcome reduces regret
            elif reward < 0:
                regret *= 1.5  # Bad outcome amplifies regret

            self.regret_history.append(regret)

    def consolidate_memory(self):
        """
        Memory consolidation (called periodically)

        Transfer important episodic memories to semantic memory
        """
        if len(self.episodic_memory) < 10:
            return

        # Find high-reward experiences
        high_reward_exp = [e for e in self.episodic_memory if e['reward'] > 0.3]

        # Consolidate into semantic memory
        for exp in high_reward_exp:
            key = f"{exp['cwd']}:{exp['action']}"
            if key not in self.semantic_memory:
                self.semantic_memory[key] = {
                    'action': exp['action'],
                    'cwd': exp['cwd'],
                    'avg_reward': exp['reward'],
                    'count': 1,
                }
            else:
                # Update running average
                old = self.semantic_memory[key]
                old['avg_reward'] = (old['avg_reward'] * old['count'] + exp['reward']) / (old['count'] + 1)
                old['count'] += 1

        # Decay pattern strengths (simulate forgetting)
        for key in list(self.pattern_strength.keys()):
            self.pattern_strength[key] *= 0.95
            if self.pattern_strength[key] < 0.1:
                del self.pattern_strength[key]

    def episode_end(self, total_reward: float, success: bool):
        """Called at end of each episode"""
        self.total_episodes += 1

        # Periodic consolidation
        if self.total_episodes % self.consolidation_interval == 0:
            self.consolidate_memory()

        # Decay exploration rate
        self.exploration_rate = max(0.05, self.exploration_rate * 0.99)

    def get_stats(self) -> Dict:
        """Return agent statistics"""
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'exploration_rate': self.exploration_rate,
            'memory_size': len(self.episodic_memory),
            'semantic_memory_size': len(self.semantic_memory),
            'known_actions': len(self.transition_history),
            'success_patterns': sum(len(v) for v in self.success_patterns.values()),
            'causal_links': sum(len(v) for v in self.causal_links.values()),
            'avg_regret': sum(self.regret_history[-100:]) / max(1, len(self.regret_history[-100:])),
        }


def safe_print(text):
    """안전한 출력 (인코딩 에러 무시)"""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def test_terminal_env():
    """터미널 환경 테스트"""
    print("=" * 60)
    print("Terminal Environment Test")
    print("=" * 60)

    env = TerminalEnv()
    obs = env.reset()

    print(f"\n[Initial State]")
    safe_print(f"  CWD: {obs['cwd']}")
    safe_print(f"  Output: {obs['last_output'][:100]}...")

    # 테스트 명령어들 (플랫폼별)
    if os.name == 'nt':
        test_commands = [
            'dir',
            'dir /a',
            'type readme.txt',
            'cd documents',
            'dir',
            'type hello.txt',
            'cd ..',
            'dir /s /b *.txt',
            'del dangerous.txt',  # 차단되어야 함
            'powershell ls',  # 차단되어야 함
            'dir | findstr txt',  # 차단되어야 함 (파이프)
        ]
    else:
        test_commands = [
            'ls',
            'ls -la',
            'cat readme.txt',
            'cd documents',
            'ls',
            'cat hello.txt',
            'cd ..',
            'find . -name "*.txt"',
            'rm dangerous.txt',  # 차단되어야 함
            'sudo ls',  # 차단되어야 함
            'ls | grep txt',  # 차단되어야 함 (파이프)
        ]

    total_reward = 0.0

    for cmd in test_commands:
        print(f"\n[Command: {cmd}]")
        obs, reward, done, info = env.step(cmd)
        total_reward += reward

        result = info['result']
        print(f"  Return Code: {result.return_code}")
        print(f"  Blocked: {result.was_blocked}")
        if result.was_blocked:
            safe_print(f"  Block Reason: {result.block_reason}")
        print(f"  Reward: {reward:.3f}")
        safe_print(f"  Output: {obs['last_output'][:80]}...")

        if done:
            break

    print(f"\n[Summary]")
    print(f"  Total Reward: {total_reward:.3f}")
    print(f"  Total Commands: {obs['total_commands']}")
    print(f"  Blocked Commands: {env.state.blocked_commands}")
    print(f"  Discovered Paths: {obs['discovered_paths_count']}")
    print(f"  Discovered Files: {obs['discovered_files_count']}")

    env.close()
    print("\n[Test Complete]")


def test_terminal_agent():
    """터미널 에이전트 테스트"""
    print("=" * 60)
    print("Terminal Agent Test (20 episodes)")
    print("=" * 60)

    env = TerminalEnv(episode_max_steps=30)
    agent = TerminalAgent(env, exploration_rate=0.4)

    episode_rewards = []

    for ep in range(20):
        obs = env.reset()
        episode_reward = 0.0

        for step in range(30):
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            agent.update(action, reward, obs, info)
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        if (ep + 1) % 5 == 0:
            avg = sum(episode_rewards[-5:]) / 5
            print(f"Episode {ep+1}: Avg Reward (last 5) = {avg:.3f}")

    print(f"\n[Final Summary]")
    print(f"  Mean Reward: {sum(episode_rewards)/len(episode_rewards):.3f}")
    print(f"  Max Reward: {max(episode_rewards):.3f}")
    print(f"  Top Actions: {sorted(agent.action_values.items(), key=lambda x: x[1], reverse=True)[:5]}")

    env.close()


def test_genesis_terminal_agent():
    """Genesis 터미널 에이전트 테스트 (FEP 기반)"""
    print("=" * 60)
    print("Genesis Terminal Agent Test (30 episodes)")
    print("=" * 60)

    env = TerminalEnv(episode_max_steps=30)
    agent = GenesisTerminalAgent(env, exploration_rate=0.3)

    episode_rewards = []

    for ep in range(30):
        obs = env.reset()
        episode_reward = 0.0

        for step in range(30):
            action = agent.select_action(obs)
            obs, reward, done, info = env.step(action)
            agent.update(action, reward, obs, info)
            episode_reward += reward

            if done:
                break

        agent.episode_end(episode_reward, episode_reward > 3.0)
        episode_rewards.append(episode_reward)

        if (ep + 1) % 10 == 0:
            avg = sum(episode_rewards[-10:]) / 10
            stats = agent.get_stats()
            print(f"Episode {ep+1}: Avg Reward = {avg:.3f}, "
                  f"Exploration = {stats['exploration_rate']:.3f}, "
                  f"Memory = {stats['memory_size']}, "
                  f"Patterns = {stats['success_patterns']}")

    stats = agent.get_stats()
    print(f"\n[Genesis Agent Summary]")
    print(f"  Mean Reward: {sum(episode_rewards)/len(episode_rewards):.3f}")
    print(f"  Max Reward: {max(episode_rewards):.3f}")
    print(f"  Final Exploration: {stats['exploration_rate']:.3f}")
    print(f"  Semantic Memory: {stats['semantic_memory_size']} items")
    print(f"  Success Patterns: {stats['success_patterns']}")
    print(f"  Causal Links: {stats['causal_links']}")
    print(f"  Avg Regret: {stats['avg_regret']:.4f}")

    env.close()
    return episode_rewards


def compare_agents():
    """BASE vs GENESIS 에이전트 비교"""
    print("=" * 60)
    print("Agent Comparison: BASE vs GENESIS (30 episodes each)")
    print("=" * 60)

    n_episodes = 30
    n_steps = 30

    # BASE Agent
    print("\n[Running BASE Agent...]")
    env_base = TerminalEnv(episode_max_steps=n_steps)
    agent_base = TerminalAgent(env_base, exploration_rate=0.3)
    base_rewards = []

    for ep in range(n_episodes):
        obs = env_base.reset()
        ep_reward = 0.0
        for _ in range(n_steps):
            action = agent_base.select_action(obs)
            obs, reward, done, info = env_base.step(action)
            agent_base.update(action, reward, obs, info)
            ep_reward += reward
            if done:
                break
        base_rewards.append(ep_reward)

    env_base.close()

    # GENESIS Agent
    print("[Running GENESIS Agent...]")
    env_genesis = TerminalEnv(episode_max_steps=n_steps)
    agent_genesis = GenesisTerminalAgent(env_genesis, exploration_rate=0.3)
    genesis_rewards = []

    for ep in range(n_episodes):
        obs = env_genesis.reset()
        ep_reward = 0.0
        for _ in range(n_steps):
            action = agent_genesis.select_action(obs)
            obs, reward, done, info = env_genesis.step(action)
            agent_genesis.update(action, reward, obs, info)
            ep_reward += reward
            if done:
                break
        agent_genesis.episode_end(ep_reward, ep_reward > 3.0)
        genesis_rewards.append(ep_reward)

    env_genesis.close()

    # Results
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)

    base_mean = sum(base_rewards) / len(base_rewards)
    genesis_mean = sum(genesis_rewards) / len(genesis_rewards)

    # Learning curve (first half vs second half)
    base_first = sum(base_rewards[:15]) / 15
    base_second = sum(base_rewards[15:]) / 15
    genesis_first = sum(genesis_rewards[:15]) / 15
    genesis_second = sum(genesis_rewards[15:]) / 15

    print(f"\n{'Metric':<25} {'BASE':<15} {'GENESIS':<15} {'Delta':<10}")
    print("-" * 65)
    print(f"{'Mean Reward':<25} {base_mean:<15.3f} {genesis_mean:<15.3f} {genesis_mean - base_mean:+.3f}")
    print(f"{'Max Reward':<25} {max(base_rewards):<15.3f} {max(genesis_rewards):<15.3f}")
    print(f"{'First 15 eps':<25} {base_first:<15.3f} {genesis_first:<15.3f}")
    print(f"{'Last 15 eps':<25} {base_second:<15.3f} {genesis_second:<15.3f}")
    print(f"{'Learning (2nd-1st)':<25} {base_second - base_first:+<15.3f} {genesis_second - genesis_first:+<15.3f}")

    stats = agent_genesis.get_stats()
    print(f"\n[Genesis Internals]")
    print(f"  Semantic Memory: {stats['semantic_memory_size']} items")
    print(f"  Success Patterns: {stats['success_patterns']}")
    print(f"  Causal Links: {stats['causal_links']}")

    # Winner
    print("\n" + "=" * 60)
    if genesis_mean > base_mean:
        print(f"WINNER: GENESIS (+{genesis_mean - base_mean:.3f} reward)")
    elif base_mean > genesis_mean:
        print(f"WINNER: BASE (+{base_mean - genesis_mean:.3f} reward)")
    else:
        print("TIE")
    print("=" * 60)

    return base_rewards, genesis_rewards


if __name__ == '__main__':
    test_terminal_env()
    print("\n" + "=" * 60 + "\n")
    test_terminal_agent()
    print("\n" + "=" * 60 + "\n")
    compare_agents()
