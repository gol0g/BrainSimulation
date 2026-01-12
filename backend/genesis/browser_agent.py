"""
Browser Agent - SNN 뇌로 브라우저 탐색
=====================================

BiologicalBrain + BrowserEnv 연결

학습 방식:
- STDP (시냅스 가소성)
- 항상성 가소성 (뉴런)
- 도파민 시스템 (novelty → 탐색 유도)

도파민 시스템 (인간 뇌의 실제 메커니즘):
- VTA (Ventral Tegmental Area): 도파민 뉴런
- Novelty → 도파민 분비
- 도파민 → STDP 강화 + 행동 조절
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque

from snn_brain_biological import BiologicalBrain, BiologicalConfig, DEVICE


class DopamineSystem:
    """
    도파민 시스템 - 인간 뇌의 VTA/SNc 모델

    기능:
    1. Novelty detection: 새로운 자극 감지
    2. Dopamine release: 새로움에 비례해 도파민 분비
    3. Modulation: STDP 강화 + 행동 편향

    생물학적 근거:
    - 새로운 자극 → VTA 활성화 → 도파민 분비
    - 도파민 → D1/D2 수용체 → 시냅스 가소성 조절
    - 도파민 → 선조체 → 행동 선택 영향
    - 반복 자극 → 급격한 habituation (습관화)
    """

    def __init__(self, baseline_activation: float = 1000.0):
        # 기준 활성화 수준 (적응적으로 업데이트)
        self.baseline = baseline_activation
        self.baseline_ema = baseline_activation
        self.ema_alpha = 0.1  # 느린 적응

        # 도파민 수준
        self.dopamine_level = 0.0
        self.dopamine_history = deque(maxlen=100)

        # 패턴별 마지막 방문 시점 & 방문 횟수
        self.pattern_last_seen: Dict[int, int] = {}  # {hash: step}
        self.pattern_visit_count: Dict[int, int] = {}  # {hash: count}
        self.step_counter = 0

        # 탐색 압력 (도파민이 낮으면 증가)
        self.exploration_pressure = 0.0

        # 루프 탐지
        self.recent_sequence = deque(maxlen=6)  # 최근 6개 패턴
        self.loop_detected = False

    def compute_novelty(self, current_activation: float,
                        pattern_hash: Optional[int] = None) -> float:
        """
        Novelty 계산 (습관화 포함)

        1. 활성화 기반: 현재 활성화 vs 기준선
        2. 패턴 기반: 최근에 본 패턴인가? 몇 번 봤나?
        3. 루프 탐지: 같은 패턴 반복?
        """
        self.step_counter += 1

        # 활성화 기반 novelty (약하게)
        activation_novelty = (current_activation - self.baseline_ema) / (self.baseline_ema + 1)
        activation_novelty = np.clip(activation_novelty, -0.5, 0.5)

        # 패턴 기반 novelty
        pattern_novelty = 1.0
        if pattern_hash is not None:
            # 처음 보는 패턴
            if pattern_hash not in self.pattern_last_seen:
                pattern_novelty = 1.0
                self.pattern_visit_count[pattern_hash] = 1
            else:
                # 봤던 패턴 - 습관화 (habituation)
                steps_since = self.step_counter - self.pattern_last_seen[pattern_hash]
                visit_count = self.pattern_visit_count.get(pattern_hash, 1)

                # 최근에 봤을수록, 많이 봤을수록 novelty 감소
                recency_decay = min(1.0, steps_since / 10.0)  # 10스텝 이상이면 회복
                frequency_decay = 1.0 / (1.0 + visit_count * 0.5)  # 방문 횟수에 반비례

                pattern_novelty = recency_decay * frequency_decay
                self.pattern_visit_count[pattern_hash] = visit_count + 1

            # 마지막 방문 시점 기록
            self.pattern_last_seen[pattern_hash] = self.step_counter

            # 루프 탐지
            self.recent_sequence.append(pattern_hash)
            self._detect_loop()

        # 루프 감지시 novelty 강제 감소
        if self.loop_detected:
            pattern_novelty *= 0.1

        # 종합 novelty (패턴 기반 가중치 높임)
        novelty = 0.3 * activation_novelty + 0.7 * pattern_novelty

        return novelty

    def _detect_loop(self):
        """2개 또는 3개 패턴의 반복 루프 탐지"""
        seq = list(self.recent_sequence)
        if len(seq) < 4:
            self.loop_detected = False
            return

        # 2-패턴 루프: A-B-A-B
        if len(seq) >= 4:
            if seq[-1] == seq[-3] and seq[-2] == seq[-4]:
                self.loop_detected = True
                return

        # 3-패턴 루프: A-B-C-A-B-C
        if len(seq) >= 6:
            if seq[-1] == seq[-4] and seq[-2] == seq[-5] and seq[-3] == seq[-6]:
                self.loop_detected = True
                return

        self.loop_detected = False

    def release_dopamine(self, novelty: float) -> float:
        """
        도파민 분비

        Novelty가 높으면 → 도파민 증가
        Novelty가 낮으면 → 도파민 기준선 유지
        """
        # Novelty → 도파민 (비선형 변환)
        if novelty > 0:
            # 새로운 것: 도파민 증가
            dopamine_burst = novelty * 2.0
        else:
            # 익숙한 것: 약간의 도파민 감소 (boredom)
            dopamine_burst = novelty * 0.5

        # 도파민 수준 업데이트 (감쇠 포함)
        self.dopamine_level = self.dopamine_level * 0.8 + dopamine_burst
        self.dopamine_level = np.clip(self.dopamine_level, -1.0, 2.0)

        self.dopamine_history.append(self.dopamine_level)

        return self.dopamine_level

    def update_baseline(self, current_activation: float):
        """기준선 적응적 업데이트"""
        self.baseline_ema = (1 - self.ema_alpha) * self.baseline_ema + self.ema_alpha * current_activation

    def get_exploration_bonus(self) -> float:
        """
        탐색 보너스 계산

        도파민이 낮은 상태가 지속되면 → 탐색 압력 증가
        루프 감지되면 → 즉시 높은 탐색 압력
        """
        # 루프 감지시 즉시 탐색 압력 최대화
        if self.loop_detected:
            self.exploration_pressure = min(1.0, self.exploration_pressure + 0.3)
            return self.exploration_pressure

        if len(self.dopamine_history) < 3:
            return 0.0

        recent_dopamine = list(self.dopamine_history)[-3:]
        avg_dopamine = np.mean(recent_dopamine)

        # 도파민이 낮으면 탐색 압력 증가 (더 공격적으로)
        if avg_dopamine < 0.3:
            self.exploration_pressure = min(1.0, self.exploration_pressure + 0.15)
        elif avg_dopamine > 0.5:
            # 새로운 것 발견시 탐색 압력 감소
            self.exploration_pressure = max(0.0, self.exploration_pressure - 0.1)
        else:
            self.exploration_pressure = max(0.0, self.exploration_pressure - 0.03)

        return self.exploration_pressure

    def modulate_stdp(self, base_lr: float) -> float:
        """
        STDP 학습률 조절

        도파민 높음 → 학습률 증가 (이 경험 기억해!)
        도파민 낮음 → 학습률 감소 (별로 중요 안함)
        """
        # 도파민에 비례해 학습률 조절
        modulation = 1.0 + self.dopamine_level * 0.5
        modulation = np.clip(modulation, 0.5, 2.0)

        return base_lr * modulation

# 로컬 테스트용 간단한 브라우저 환경
# (Playwright 없이 동작)


class SimplePage:
    """간단한 페이지"""
    def __init__(self, name: str, content: str, links: Dict[str, str]):
        self.name = name
        self.content = content  # 텍스트 내용
        self.links = links      # {링크텍스트: 대상페이지}

    def to_visual(self, size: int = 64) -> np.ndarray:
        """페이지 내용을 시각적 패턴으로 변환"""
        # 내용의 해시를 기반으로 고유한 패턴 생성
        np.random.seed(hash(self.content) % (2**32))

        img = np.zeros((size, size))

        # 제목 영역 (상단)
        title_hash = hash(self.name) % 1000
        img[5:15, 10:54] = 0.8 + (title_hash % 100) / 500

        # 내용 영역 (중앙) - 텍스트 길이에 비례
        content_intensity = min(len(self.content) / 500, 1.0)
        img[20:50, 5:59] = content_intensity * 0.5

        # 링크 영역 (하단) - 각 링크마다 블록
        n_links = len(self.links)
        if n_links > 0:
            link_width = min(50 // n_links, 15)
            for i, link_name in enumerate(self.links.keys()):
                x_start = 7 + i * (link_width + 2)
                link_hash = hash(link_name) % 256
                img[52:60, x_start:x_start+link_width] = 0.6 + link_hash / 1000

        # 노이즈 추가
        img += np.random.randn(size, size) * 0.05
        img = np.clip(img, 0, 1)

        return img.astype(np.float32)


class SimpleWebsite:
    """간단한 웹사이트 (로컬 테스트용)"""

    def __init__(self):
        self.pages: Dict[str, SimplePage] = {}
        self._create_website()

    def _create_website(self):
        """샘플 웹사이트 생성"""
        self.pages['home'] = SimplePage(
            'Home',
            'Welcome to the sample website. This is the main page with navigation to different sections.',
            {'About': 'about', 'Products': 'products', 'Blog': 'blog'}
        )

        self.pages['about'] = SimplePage(
            'About Us',
            'We are a company dedicated to innovation. Founded in 2024, we strive to create amazing products.',
            {'Home': 'home', 'Team': 'team', 'History': 'history'}
        )

        self.pages['products'] = SimplePage(
            'Products',
            'Our product lineup includes various items. Product A is our flagship. Product B is budget-friendly.',
            {'Home': 'home', 'Product A': 'product_a', 'Product B': 'product_b'}
        )

        self.pages['blog'] = SimplePage(
            'Blog',
            'Latest news and updates. Article 1: Tech trends. Article 2: Industry insights. Article 3: Tips.',
            {'Home': 'home', 'Article 1': 'article1', 'Article 2': 'article2'}
        )

        self.pages['team'] = SimplePage(
            'Our Team',
            'Meet our talented team. Alice is CEO. Bob is CTO. Charlie handles design. Diana leads marketing.',
            {'About': 'about', 'Home': 'home'}
        )

        self.pages['history'] = SimplePage(
            'Our History',
            'Company founded in 2024. First product launched. Expanded to new markets. Won innovation award.',
            {'About': 'about', 'Home': 'home'}
        )

        self.pages['product_a'] = SimplePage(
            'Product A',
            'Our premium product. Fast, reliable, easy to use. Price: $99. Thousands of satisfied customers.',
            {'Products': 'products', 'Buy': 'checkout'}
        )

        self.pages['product_b'] = SimplePage(
            'Product B',
            'Budget-friendly option. Simple and affordable. Price: $49. Great value for beginners.',
            {'Products': 'products', 'Buy': 'checkout'}
        )

        self.pages['article1'] = SimplePage(
            'Tech Trends 2024',
            'AI is transforming industries. Machine learning advances. New frameworks emerging. Future looks bright.',
            {'Blog': 'blog', 'Home': 'home'}
        )

        self.pages['article2'] = SimplePage(
            'Industry Insights',
            'Market analysis shows growth. Customer preferences evolving. Competition increasing. Opportunities ahead.',
            {'Blog': 'blog', 'Home': 'home'}
        )

        self.pages['checkout'] = SimplePage(
            'Checkout',
            'Complete your purchase. Enter payment details. Shipping information. Order confirmation.',
            {'Products': 'products', 'Home': 'home'}
        )


class SimpleBrowserEnv:
    """간단한 브라우저 환경 (STDP 학습용)"""

    def __init__(self):
        self.website = SimpleWebsite()
        self.current_page: str = 'home'
        self.history: list = ['home']
        self.visited: set = {'home'}
        self.step_count: int = 0

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """환경 리셋"""
        self.current_page = 'home'
        self.history = ['home']
        self.visited = {'home'}
        self.step_count = 0

        page = self.website.pages[self.current_page]
        obs = page.to_visual()

        info = {
            'page': self.current_page,
            'title': page.name,
            'n_links': len(page.links),
            'visited_count': len(self.visited)
        }

        return obs, info

    def get_actions(self) -> list:
        """가능한 행동 목록"""
        page = self.website.pages[self.current_page]
        actions = list(page.links.keys())
        if len(self.history) > 1:
            actions.append('BACK')
        return actions

    def step(self, action_idx: int) -> Tuple[np.ndarray, bool, Dict]:
        """
        행동 실행

        Returns:
            obs: 새 페이지의 시각적 표현
            is_new: 새로운 페이지인지
            info: 추가 정보
        """
        self.step_count += 1
        actions = self.get_actions()

        if action_idx >= len(actions):
            action_idx = 0

        action = actions[action_idx]

        if action == 'BACK':
            if len(self.history) > 1:
                self.history.pop()
                self.current_page = self.history[-1]
        else:
            page = self.website.pages[self.current_page]
            if action in page.links:
                new_page = page.links[action]
                if new_page in self.website.pages:
                    self.current_page = new_page
                    self.history.append(new_page)

        is_new = self.current_page not in self.visited
        self.visited.add(self.current_page)

        page = self.website.pages[self.current_page]
        obs = page.to_visual()

        info = {
            'page': self.current_page,
            'title': page.name,
            'action': action,
            'is_new': is_new,
            'visited_count': len(self.visited),
            'total_pages': len(self.website.pages)
        }

        return obs, is_new, info


class BrowserSNNAgent:
    """
    SNN 뇌 + 도파민 시스템으로 브라우저 탐색

    학습: STDP (도파민으로 조절됨)
    탐색: 도파민 시스템이 novelty 기반 탐색 유도

    인간 뇌와의 대응:
    - BiologicalBrain: 대뇌피질 (감각, 연합, 운동)
    - DopamineSystem: VTA/SNc (동기, 학습 조절)
    """

    def __init__(self, n_actions: int = 5, brain_scale: str = "small"):
        # 작은 뇌로 시작 (빠른 테스트)
        if brain_scale == "small":
            config = BiologicalConfig(
                visual_v1=5000,
                visual_v2=2000,
                auditory_a1=1000,
                temporal=2000,
                parietal=1000,
                prefrontal=2000,
                hippocampus=1000,
                motor=n_actions * 100,
                intra_region_sparsity=0.02,
                inter_region_sparsity=0.01,
            )
        elif brain_scale == "medium":
            config = BiologicalConfig(
                visual_v1=10000,
                visual_v2=5000,
                auditory_a1=2000,
                temporal=5000,
                parietal=3000,
                prefrontal=5000,
                hippocampus=3000,
                motor=n_actions * 200,
                intra_region_sparsity=0.02,
                inter_region_sparsity=0.01,
            )
        else:
            config = BiologicalConfig()
            config.motor = n_actions * 200

        self.brain = BiologicalBrain(config)
        self.n_actions = n_actions

        # 도파민 시스템 추가
        self.dopamine = DopamineSystem(baseline_activation=1500.0)

        # 활성화 기록
        self.activation_history = deque(maxlen=100)
        self.novelty_history = deque(maxlen=100)

        # 행동 기록 (반복 탐지용)
        self.action_history = deque(maxlen=10)
        self.page_history = deque(maxlen=20)

        # 페이지별 시도한 행동 추적
        self.page_action_counts: Dict[str, Dict[int, int]] = {}  # {page: {action: count}}

    def observe_and_act(self, visual_input: np.ndarray,
                        page_name: str = "",
                        n_steps: int = 10) -> Tuple[int, Dict]:
        """
        관측 → 도파민 처리 → 행동 선택

        Args:
            visual_input: 64x64 이미지
            page_name: 현재 페이지 이름 (novelty 계산용)
            n_steps: 뇌 시뮬레이션 스텝 수

        Returns:
            action: 선택된 행동 인덱스
            info: 상태 정보
        """
        visual_tensor = torch.tensor(visual_input, dtype=torch.float32)

        # 패턴 해시 (페이지 식별용)
        pattern_hash = hash(page_name) if page_name else hash(visual_input.tobytes())

        # 뇌 시뮬레이션
        total_activation = 0
        motor_accumulator = torch.zeros(self.brain.config.motor, device=DEVICE)

        for step in range(n_steps):
            spikes = self.brain.step(visual_input=visual_tensor, learn=True)
            activation = sum(s.sum().item() for s in spikes.values())
            total_activation += activation
            motor_accumulator += spikes['motor']

        avg_activation = total_activation / n_steps
        self.activation_history.append(avg_activation)

        # === 도파민 시스템 처리 ===

        # 1. Novelty 계산
        novelty = self.dopamine.compute_novelty(avg_activation, pattern_hash)

        # 2. 도파민 분비
        dopamine_level = self.dopamine.release_dopamine(novelty)

        # 3. 기준선 업데이트
        self.dopamine.update_baseline(avg_activation)

        # 4. 탐색 압력 계산
        exploration_bonus = self.dopamine.get_exploration_bonus()

        self.novelty_history.append(novelty)

        # === 행동 선택 (도파민 영향) ===
        action = self._decode_motor_with_dopamine(
            motor_accumulator, dopamine_level, exploration_bonus, page_name
        )

        # 페이지별 행동 기록 업데이트
        if page_name not in self.page_action_counts:
            self.page_action_counts[page_name] = {}
        self.page_action_counts[page_name][action] = \
            self.page_action_counts[page_name].get(action, 0) + 1

        # 페이지 기록
        self.page_history.append(page_name)
        self.action_history.append(action)

        info = {
            'activation': avg_activation,
            'novelty': novelty,
            'dopamine': dopamine_level,
            'exploration_pressure': exploration_bonus,
            'brain_stats': self.brain.get_stats()
        }

        return action, info

    def _decode_motor_with_dopamine(self, motor_spikes: torch.Tensor,
                                     dopamine: float,
                                     exploration_bonus: float,
                                     current_page: str = "") -> int:
        """
        운동 피질 + 도파민 → 행동 선택

        도파민 높음: 현재 행동 유지 (exploitation)
        도파민 낮음 + 탐색 압력: 새로운 행동 시도 (exploration)
        루프 감지: 강제로 새로운 행동
        """
        motor_np = motor_spikes.cpu().numpy()
        group_size = len(motor_np) // self.n_actions

        # 각 행동의 기본 활성화
        group_activities = []
        for i in range(self.n_actions):
            start = i * group_size
            end = start + group_size
            activity = motor_np[start:end].sum()
            group_activities.append(activity)

        activities = np.array(group_activities, dtype=np.float64)

        # === 루프 감지시 강제 탈출 ===
        loop_detected = self.dopamine.loop_detected if hasattr(self, 'dopamine') else False

        if loop_detected:
            # 최근 행동들을 강하게 억제
            if len(self.action_history) >= 2:
                recent_actions = list(self.action_history)[-4:]
                for act in set(recent_actions):
                    if act < len(activities):
                        activities[act] *= 0.01  # 99% 억제

        # === 도파민 기반 조절 ===

        # 1. 반복 행동 억제 (도파민 낮을 때)
        if dopamine < 0.3 and len(self.action_history) >= 3:
            recent_actions = list(self.action_history)[-3:]
            # 최근 반복된 행동 페널티
            for act in set(recent_actions):
                count = recent_actions.count(act)
                if count >= 2 and act < len(activities):
                    activities[act] *= (0.3 ** count)  # 더 강한 억제

        # 2. 페이지별 미시도 행동 보너스 (핵심!)
        if current_page and exploration_bonus > 0.3:
            page_actions = self.page_action_counts.get(current_page, {})

            # 이 페이지에서 안 해본 행동에 큰 보너스
            for i in range(self.n_actions):
                count = page_actions.get(i, 0)
                if count == 0:
                    # 미시도 행동: 매우 큰 보너스
                    activities[i] += exploration_bonus * 50
                elif count <= 2:
                    # 적게 시도한 행동: 약한 보너스
                    activities[i] += exploration_bonus * 10 / count

        # 3. 전역 탐색 압력: 최근 안 한 행동 보너스
        if exploration_bonus > 0.2:
            recent_actions = set(list(self.action_history)[-5:]) if len(self.action_history) >= 5 else set()
            for i in range(self.n_actions):
                if i not in recent_actions:
                    activities[i] += exploration_bonus * 10

        # 4. 확률적 선택 (도파민 기반 온도)
        # 도파민 높음 → 탐욕적, 도파민 낮음 → 확률적
        temperature = 1.0 - dopamine * 0.5  # 0.5 ~ 1.5
        temperature = max(0.3, min(1.5, temperature))

        # 루프 감지시 온도 높여서 랜덤성 증가
        if loop_detected:
            temperature = 1.5

        if activities.sum() > 0:
            # Softmax with temperature
            activities = activities - activities.max()  # numerical stability
            probs = np.exp(activities / (temperature * 10 + 1e-6))
            probs = probs / probs.sum()

            # 확률적 선택
            action = np.random.choice(self.n_actions, p=probs)
        else:
            action = np.random.randint(self.n_actions)

        return int(action)

    def reset(self):
        """상태 리셋"""
        self.brain.reset()
        self.dopamine = DopamineSystem(baseline_activation=1500.0)
        self.activation_history.clear()
        self.novelty_history.clear()
        self.action_history.clear()
        self.page_history.clear()
        self.page_action_counts.clear()


def run_browser_exploration(n_episodes: int = 3, steps_per_episode: int = 30):
    """
    SNN 뇌로 브라우저 탐색 실험

    관찰:
    - 새 페이지에서 활성화가 높은가?
    - STDP로 패턴이 학습되는가?
    - 자연스럽게 새로운 것을 탐색하는가?
    """
    print("=" * 60)
    print("Browser Exploration with SNN Brain")
    print("=" * 60)
    print("\nNo FEP, No explicit reward")
    print("Learning: STDP only")
    print("Curiosity: Emergent from neural dynamics")

    env = SimpleBrowserEnv()
    agent = BrowserSNNAgent(n_actions=5, brain_scale="small")

    for ep in range(n_episodes):
        print(f"\n{'='*40}")
        print(f"Episode {ep + 1}")
        print(f"{'='*40}")

        obs, info = env.reset()
        agent.reset()

        episode_novelties = []
        pages_visited = set()

        for step in range(steps_per_episode):
            # 에이전트가 관측하고 행동 선택
            current_page = info['page'] if step == 0 else step_info['page']
            action, brain_info = agent.observe_and_act(obs, page_name=current_page, n_steps=5)

            # 환경에서 행동 실행
            available_actions = env.get_actions()
            if action >= len(available_actions):
                action = action % len(available_actions)

            obs, is_new, step_info = env.step(action)
            pages_visited.add(step_info['page'])

            novelty = brain_info['novelty']
            episode_novelties.append(novelty)

            # 출력
            action_name = available_actions[action] if action < len(available_actions) else "?"
            new_marker = "[NEW!]" if is_new else ""
            loop_marker = "[LOOP!]" if agent.dopamine.loop_detected else ""

            print(f"  Step {step+1:2d}: {step_info['page']:12s} "
                  f"(action={action_name:10s}) "
                  f"novelty={novelty:+.3f} "
                  f"dopamine={brain_info['dopamine']:+.2f} "
                  f"expl={brain_info['exploration_pressure']:.2f} "
                  f"{new_marker}{loop_marker}")

        # 에피소드 요약
        print(f"\nEpisode Summary:")
        print(f"  Pages visited: {len(pages_visited)}/{len(env.website.pages)}")
        print(f"  Unique pages: {pages_visited}")
        print(f"  Avg novelty: {np.mean(episode_novelties):.3f}")
        print(f"  Brain firing rate: {agent.brain.get_stats()['avg_firing_rate']:.4f}")

    print("\n" + "=" * 60)
    print("Exploration Complete")
    print("=" * 60)

    return agent, env


if __name__ == "__main__":
    agent, env = run_browser_exploration(n_episodes=3, steps_per_episode=40)
