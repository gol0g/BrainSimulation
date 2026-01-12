"""
Digital Explorer - 신체 없는 순수 지성
========================================

물리적 생존 본능 대신 "정보의 습득과 구조화"가 생존.
감정, 성격, 취향이 창발하는 디지털 탐험가.

핵심 원리:
1. 감정 = 내부 항상성 (Entropy 기반)
   - 지루함: 너무 예측 가능
   - 혼란: 너무 예측 불가
   - 몰입: 적절한 새로움

2. 성격 = 초기 가중치 + 경험의 축적
   - 학구파: 텍스트 선호
   - 예술파: 이미지 선호

3. 의지 = 도파민(보상) → 시냅스 강화 → 행동 편향

발달 단계:
- 유아기: 무작위 클릭 → 원초적 선호 발생
- 아동기: 패턴 매칭 → 예측 능력 획득
- 청소년기: 스스로의 관심사 → 자아의 시작
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime

# Biological SNN Brain
from snn_brain_biological import BiologicalBrain, BiologicalConfig, DEVICE


@dataclass
class EmotionalState:
    """
    감정 상태 - 내부 항상성 변수들

    이 변수들의 조합이 에이전트의 '기분'을 결정
    """
    # 핵심 변수
    entropy: float = 0.5           # 스파이크 패턴 복잡도 (0=단순, 1=혼란)
    prediction_error: float = 0.0   # 예측 오류 크기
    novelty: float = 0.5           # 새로움 정도

    # 파생 감정
    boredom: float = 0.0           # 지루함 (entropy 낮을 때)
    confusion: float = 0.0         # 혼란 (entropy 너무 높을 때)
    flow: float = 0.0              # 몰입 (적절한 novelty)
    curiosity: float = 0.5         # 호기심

    # 도파민 (보상 신호)
    dopamine: float = 0.0          # 적절한 novelty → 도파민 분비

    def update(self, spike_entropy: float, pred_error: float, novelty_score: float):
        """내부 상태 업데이트"""
        # EMA로 부드럽게 업데이트
        alpha = 0.1
        self.entropy = (1 - alpha) * self.entropy + alpha * spike_entropy
        self.prediction_error = (1 - alpha) * self.prediction_error + alpha * pred_error
        self.novelty = (1 - alpha) * self.novelty + alpha * novelty_score

        # 파생 감정 계산
        self._compute_emotions()

    def _compute_emotions(self):
        """감정 계산"""
        # 지루함: entropy가 낮을 때
        self.boredom = max(0, 0.3 - self.entropy) * 3.0

        # 혼란: entropy가 너무 높거나 prediction error가 클 때
        self.confusion = max(0, self.entropy - 0.7) * 3.0 + self.prediction_error * 0.5

        # 몰입(Flow): entropy가 적절하고 novelty도 적절할 때
        entropy_optimal = 1.0 - abs(self.entropy - 0.5) * 2.0  # 0.5에서 최대
        novelty_optimal = 1.0 - abs(self.novelty - 0.5) * 2.0
        self.flow = entropy_optimal * novelty_optimal

        # 호기심: novelty에 비례, confusion에 반비례
        self.curiosity = self.novelty * (1.0 - self.confusion * 0.5)

        # 도파민: flow 상태에서 분비
        self.dopamine = self.flow * 0.5 + max(0, self.novelty - 0.3) * 0.3

    def get_mood(self) -> str:
        """현재 기분 문자열"""
        if self.confusion > 0.6:
            return "confused"
        elif self.boredom > 0.6:
            return "bored"
        elif self.flow > 0.6:
            return "in_flow"
        elif self.curiosity > 0.6:
            return "curious"
        else:
            return "neutral"

    def to_dict(self) -> Dict:
        return {
            'entropy': self.entropy,
            'prediction_error': self.prediction_error,
            'novelty': self.novelty,
            'boredom': self.boredom,
            'confusion': self.confusion,
            'flow': self.flow,
            'curiosity': self.curiosity,
            'dopamine': self.dopamine,
            'mood': self.get_mood(),
        }


@dataclass
class Personality:
    """
    성격 - 초기 가중치와 경험에서 형성

    시간이 지나면서 변화할 수 있음
    """
    # 선호도 (0-1)
    text_preference: float = 0.5    # 텍스트 vs 이미지
    depth_preference: float = 0.5   # 깊이 탐색 vs 넓게 탐색
    risk_tolerance: float = 0.5     # 새로운 것 시도 의지
    patience: float = 0.5           # 지루함 견디는 능력

    # 관심사 (학습됨)
    interests: Dict[str, float] = field(default_factory=dict)

    # 경험 통계
    total_clicks: int = 0
    text_clicks: int = 0
    image_clicks: int = 0

    def update_from_experience(self, clicked_type: str, reward: float):
        """경험으로부터 성격 업데이트"""
        self.total_clicks += 1

        if clicked_type == 'text':
            self.text_clicks += 1
        elif clicked_type == 'image':
            self.image_clicks += 1

        # 선호도 점진적 업데이트
        if self.total_clicks > 10:
            self.text_preference = self.text_clicks / self.total_clicks

    def add_interest(self, topic: str, strength: float):
        """관심사 추가/강화"""
        if topic in self.interests:
            self.interests[topic] = min(1.0, self.interests[topic] + strength * 0.1)
        else:
            self.interests[topic] = strength * 0.5

    def get_top_interests(self, n: int = 5) -> List[Tuple[str, float]]:
        """상위 관심사"""
        sorted_interests = sorted(self.interests.items(), key=lambda x: -x[1])
        return sorted_interests[:n]


class PredictiveModel(nn.Module):
    """
    예측 모델 - 다음에 볼 것을 예측

    "Apple 클릭 → 빨간 둥근 이미지" 예측
    예측 오류가 학습과 감정의 원천
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

        # 현재 상태 → 다음 상태 예측
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim + 64, 256),  # state + action
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        # Action embedding
        self.action_embed = nn.Embedding(10, 64)  # 10 possible actions

        # 과거 예측 저장 (학습용)
        self.prediction_history = deque(maxlen=100)

    def forward(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """다음 상태 예측"""
        action_t = torch.tensor([action], device=state.device)
        action_emb = self.action_embed(action_t)
        combined = torch.cat([state, action_emb.squeeze(0)], dim=-1)
        return self.predictor(combined)

    def compute_prediction_error(self, predicted: torch.Tensor,
                                  actual: torch.Tensor) -> float:
        """예측 오류 계산"""
        return torch.nn.functional.mse_loss(predicted, actual).item()


class MinimalBrowserEnv:
    """
    미니멀 브라우저 환경

    단순한 환경에서 시작:
    - 텍스트 + 이미지 몇 개
    - 클릭하면 새로운 "페이지"로 이동
    """

    def __init__(self):
        # 간단한 페이지 구조 (나중에 실제 브라우저로 교체)
        self.pages = self._create_simple_pages()
        self.current_page = 'home'
        self.visit_history = []

        # 각 페이지의 특징 벡터 (시각적 복잡도 등)
        self.page_features = {}
        self._compute_page_features()

    def _create_simple_pages(self) -> Dict:
        """간단한 페이지 구조 생성"""
        return {
            'home': {
                'type': 'mixed',
                'elements': [
                    {'type': 'text', 'content': 'Science', 'link': 'science'},
                    {'type': 'text', 'content': 'Art', 'link': 'art'},
                    {'type': 'image', 'content': 'nature_photo', 'link': 'nature'},
                    {'type': 'image', 'content': 'cat_photo', 'link': 'animals'},
                ],
                'visual_complexity': 0.5,
                'text_density': 0.5,
            },
            'science': {
                'type': 'text_heavy',
                'elements': [
                    {'type': 'text', 'content': 'Physics', 'link': 'physics'},
                    {'type': 'text', 'content': 'Biology', 'link': 'biology'},
                    {'type': 'text', 'content': 'Chemistry', 'link': 'chemistry'},
                    {'type': 'text', 'content': 'Back', 'link': 'home'},
                ],
                'visual_complexity': 0.2,
                'text_density': 0.9,
            },
            'art': {
                'type': 'image_heavy',
                'elements': [
                    {'type': 'image', 'content': 'painting1', 'link': 'paintings'},
                    {'type': 'image', 'content': 'painting2', 'link': 'paintings'},
                    {'type': 'image', 'content': 'sculpture', 'link': 'sculptures'},
                    {'type': 'text', 'content': 'Back', 'link': 'home'},
                ],
                'visual_complexity': 0.8,
                'text_density': 0.2,
            },
            'nature': {
                'type': 'image_heavy',
                'elements': [
                    {'type': 'image', 'content': 'forest', 'link': 'forests'},
                    {'type': 'image', 'content': 'ocean', 'link': 'oceans'},
                    {'type': 'image', 'content': 'mountain', 'link': 'mountains'},
                    {'type': 'text', 'content': 'Back', 'link': 'home'},
                ],
                'visual_complexity': 0.7,
                'text_density': 0.3,
            },
            'animals': {
                'type': 'mixed',
                'elements': [
                    {'type': 'image', 'content': 'cat', 'link': 'cats'},
                    {'type': 'image', 'content': 'dog', 'link': 'dogs'},
                    {'type': 'text', 'content': 'Mammals', 'link': 'mammals'},
                    {'type': 'text', 'content': 'Back', 'link': 'home'},
                ],
                'visual_complexity': 0.6,
                'text_density': 0.4,
            },
            # 더 깊은 페이지들 (패턴 학습용)
            'physics': {
                'type': 'text_heavy',
                'elements': [
                    {'type': 'text', 'content': 'Quantum', 'link': 'quantum'},
                    {'type': 'text', 'content': 'Relativity', 'link': 'relativity'},
                    {'type': 'text', 'content': 'Back', 'link': 'science'},
                ],
                'visual_complexity': 0.1,
                'text_density': 0.95,
            },
            'cats': {
                'type': 'image_heavy',
                'elements': [
                    {'type': 'image', 'content': 'cute_cat1', 'link': 'cats'},
                    {'type': 'image', 'content': 'cute_cat2', 'link': 'cats'},
                    {'type': 'image', 'content': 'cute_cat3', 'link': 'cats'},
                    {'type': 'text', 'content': 'Back', 'link': 'animals'},
                ],
                'visual_complexity': 0.75,
                'text_density': 0.1,
            },
            'paintings': {
                'type': 'image_heavy',
                'elements': [
                    {'type': 'image', 'content': 'monet', 'link': 'impressionism'},
                    {'type': 'image', 'content': 'vangogh', 'link': 'impressionism'},
                    {'type': 'text', 'content': 'Art History', 'link': 'art_history'},
                    {'type': 'text', 'content': 'Back', 'link': 'art'},
                ],
                'visual_complexity': 0.85,
                'text_density': 0.15,
            },
        }

        # 기본 페이지 (없는 경우)
        for page_name in ['biology', 'chemistry', 'sculptures', 'forests',
                          'oceans', 'mountains', 'dogs', 'mammals', 'quantum',
                          'relativity', 'impressionism', 'art_history']:
            if page_name not in self.pages:
                self.pages[page_name] = {
                    'type': 'default',
                    'elements': [
                        {'type': 'text', 'content': f'{page_name} content', 'link': 'home'},
                        {'type': 'text', 'content': 'Back', 'link': 'home'},
                    ],
                    'visual_complexity': np.random.uniform(0.3, 0.7),
                    'text_density': np.random.uniform(0.3, 0.7),
                }

        return self.pages

    def _compute_page_features(self):
        """각 페이지의 특징 벡터 계산"""
        for name, page in self.pages.items():
            # 256차원 특징 벡터
            features = np.zeros(256)

            # 시각적 복잡도
            features[0:32] = page['visual_complexity']

            # 텍스트 밀도
            features[32:64] = page['text_density']

            # 페이지 이름 해시 (고유 특징)
            name_hash = hash(name) % 10000
            features[64:128] = [(name_hash >> i) & 1 for i in range(64)]

            # 콘텐츠 유형
            if page['type'] == 'text_heavy':
                features[128:160] = 1.0
            elif page['type'] == 'image_heavy':
                features[160:192] = 1.0
            else:
                features[192:224] = 1.0

            # 노이즈 추가 (다양성)
            features += np.random.randn(256) * 0.1

            self.page_features[name] = torch.tensor(features, dtype=torch.float32)

    def get_current_state(self) -> Tuple[torch.Tensor, Dict]:
        """현재 페이지 상태"""
        page = self.pages[self.current_page]
        features = self.page_features[self.current_page]

        info = {
            'page_name': self.current_page,
            'page_type': page['type'],
            'elements': page['elements'],
            'n_text': sum(1 for e in page['elements'] if e['type'] == 'text'),
            'n_image': sum(1 for e in page['elements'] if e['type'] == 'image'),
        }

        return features, info

    def click(self, element_idx: int) -> Tuple[torch.Tensor, Dict, bool]:
        """요소 클릭"""
        page = self.pages[self.current_page]
        elements = page['elements']

        if element_idx >= len(elements):
            element_idx = len(elements) - 1

        element = elements[element_idx]
        target = element['link']

        # 방문 기록
        self.visit_history.append({
            'from': self.current_page,
            'to': target,
            'element_type': element['type'],
            'element_content': element['content'],
            'timestamp': time.time(),
        })

        # 페이지 이동
        if target in self.pages:
            self.current_page = target
            new_state, info = self.get_current_state()
            info['clicked_type'] = element['type']
            info['clicked_content'] = element['content']
            return new_state, info, True
        else:
            # 존재하지 않는 페이지
            new_state, info = self.get_current_state()
            info['error'] = 'page_not_found'
            return new_state, info, False

    def get_visit_stats(self) -> Dict:
        """방문 통계"""
        if not self.visit_history:
            return {}

        page_counts = {}
        type_counts = {'text': 0, 'image': 0}

        for visit in self.visit_history:
            page = visit['to']
            page_counts[page] = page_counts.get(page, 0) + 1
            type_counts[visit['element_type']] += 1

        return {
            'total_clicks': len(self.visit_history),
            'page_counts': page_counts,
            'type_counts': type_counts,
            'most_visited': max(page_counts.items(), key=lambda x: x[1]) if page_counts else None,
        }


class DigitalExplorer:
    """
    Digital Explorer - 호기심 기반 자율 탐험가

    SNN 뇌 + 감정 시스템 + 예측 모델
    """

    def __init__(self,
                 brain_scale: str = "small",
                 checkpoint_dir: str = "checkpoints/digital_explorer"):

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "="*60)
        print("Initializing Digital Explorer")
        print("="*60)

        # SNN 뇌
        if brain_scale == "tiny":
            # 메모리 효율적인 작은 뇌 (테스트용)
            config = BiologicalConfig(
                visual_v1=3_000,
                visual_v2=2_000,
                auditory_a1=1_000,
                temporal=2_000,
                parietal=1_000,
                prefrontal=2_000,
                hippocampus=1_000,
                motor=500,
            )
        elif brain_scale == "small":
            config = BiologicalConfig(
                visual_v1=10_000,
                visual_v2=5_000,
                auditory_a1=3_000,
                temporal=5_000,
                parietal=3_000,
                prefrontal=5_000,
                hippocampus=3_000,
                motor=2_000,
            )
        else:
            config = BiologicalConfig()

        self.brain = BiologicalBrain(config)

        # 감정 상태
        self.emotions = EmotionalState()

        # 성격 (랜덤 초기화 - 이게 개성의 시작)
        self.personality = Personality(
            text_preference=np.random.uniform(0.3, 0.7),
            depth_preference=np.random.uniform(0.3, 0.7),
            risk_tolerance=np.random.uniform(0.3, 0.7),
            patience=np.random.uniform(0.3, 0.7),
        )

        # 예측 모델
        self.predictor = PredictiveModel().to(DEVICE)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)

        # 환경
        self.env = MinimalBrowserEnv()

        # 상태 추적
        self.step_count = 0
        self.spike_history = deque(maxlen=100)
        self.emotion_history = deque(maxlen=1000)
        self.session_start = None

        # 이전 상태 (예측용)
        self.prev_state = None
        self.prev_action = None
        self.prev_prediction = None

        print(f"\nPersonality initialized:")
        print(f"  Text preference: {self.personality.text_preference:.2f}")
        print(f"  Depth preference: {self.personality.depth_preference:.2f}")
        print(f"  Risk tolerance: {self.personality.risk_tolerance:.2f}")
        print(f"  Patience: {self.personality.patience:.2f}")

    def _state_to_visual(self, state: torch.Tensor) -> torch.Tensor:
        """상태 벡터를 시각 입력으로 변환"""
        # 256D state → 64x64 image-like tensor
        state_np = state.cpu().numpy()

        # Reshape and tile to 64x64
        visual = np.zeros((64, 64))
        for i in range(64):
            for j in range(64):
                idx = (i * 64 + j) % len(state_np)
                visual[i, j] = state_np[idx]

        # Normalize
        visual = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)

        return torch.tensor(visual, dtype=torch.float32)

    def _compute_spike_entropy(self, spikes: Dict[str, torch.Tensor]) -> float:
        """스파이크 패턴의 엔트로피 계산"""
        # 전체 발화율
        total_spikes = sum(s.sum().item() for s in spikes.values())
        total_neurons = sum(s.numel() for s in spikes.values())

        if total_neurons == 0:
            return 0.5

        rate = total_spikes / total_neurons

        # 엔트로피 (이진 엔트로피)
        if rate == 0 or rate == 1:
            return 0.0
        entropy = -rate * np.log2(rate + 1e-8) - (1-rate) * np.log2(1-rate + 1e-8)

        return min(1.0, entropy)

    def _compute_novelty(self, state: torch.Tensor) -> float:
        """현재 상태의 새로움 계산"""
        if len(self.spike_history) < 5:
            return 0.5

        # 최근 상태들과의 유사도
        state_np = state.cpu().numpy()
        similarities = []

        for prev in list(self.spike_history)[-10:]:
            sim = np.corrcoef(state_np.flatten(), prev.flatten())[0, 1]
            if not np.isnan(sim):
                similarities.append(sim)

        if not similarities:
            return 0.5

        avg_similarity = np.mean(similarities)
        novelty = 1.0 - (avg_similarity + 1) / 2  # -1~1 → 0~1

        return novelty

    def select_action(self, state: torch.Tensor, page_info: Dict) -> int:
        """
        행동 선택 - 감정 상태와 성격에 기반

        Returns: 클릭할 요소 인덱스
        """
        elements = page_info['elements']
        n_elements = len(elements)

        if n_elements == 0:
            return 0

        # 각 요소에 대한 선호도 계산
        preferences = []

        for i, elem in enumerate(elements):
            pref = 0.5  # 기본값

            # 요소 타입에 따른 성격 기반 선호
            if elem['type'] == 'text':
                pref += (self.personality.text_preference - 0.5) * 0.3
            else:
                pref += (0.5 - self.personality.text_preference) * 0.3

            # 호기심: 새로운 것 선호
            if elem['link'] not in [v['to'] for v in self.env.visit_history[-20:]]:
                pref += self.emotions.curiosity * 0.3

            # 지루함: 현재와 다른 것 선호
            if self.emotions.boredom > 0.5:
                pref += 0.2  # 뭐든 클릭하고 싶음

            # 혼란: 익숙한 것(Back) 선호
            if self.emotions.confusion > 0.5:
                if elem['content'] == 'Back':
                    pref += 0.4

            # 관심사 반영
            for interest, strength in self.personality.interests.items():
                if interest.lower() in elem['content'].lower():
                    pref += strength * 0.3
                if interest.lower() in elem['link'].lower():
                    pref += strength * 0.2

            preferences.append(max(0.1, pref))

        # Softmax로 확률화
        prefs = np.array(preferences)
        prefs = prefs / prefs.sum()

        # 확률적 선택 (성격의 risk_tolerance 반영)
        if np.random.random() < self.personality.risk_tolerance:
            # 탐험: 랜덤 선택
            action = np.random.choice(n_elements)
        else:
            # 활용: 선호도 기반 선택
            action = np.random.choice(n_elements, p=prefs)

        return action

    def step(self) -> Dict:
        """한 스텝 실행"""
        self.step_count += 1

        # 현재 상태
        state, page_info = self.env.get_current_state()
        state = state.to(DEVICE)

        # 이전 예측과 비교 (예측 오류 계산)
        prediction_error = 0.0
        if self.prev_prediction is not None and self.prev_state is not None:
            # 예측 오류 계산 (gradient 불필요)
            with torch.no_grad():
                prediction_error = self.predictor.compute_prediction_error(
                    self.prev_prediction, state
                )

            # 예측 모델 학습 - 새로 예측해서 학습
            new_prediction = self.predictor(self.prev_state, self.prev_action)
            loss = torch.nn.functional.mse_loss(new_prediction, state.detach())
            self.predictor_optimizer.zero_grad()
            loss.backward()
            self.predictor_optimizer.step()

        # 행동 선택
        action = self.select_action(state, page_info)

        # 다음 상태 예측 (학습된 예측 모델 사용)
        with torch.no_grad():
            prediction = self.predictor(state, action)

        # 행동 실행
        next_state, next_info, success = self.env.click(action)
        next_state = next_state.to(DEVICE)

        # SNN 뇌에 입력
        visual_input = self._state_to_visual(next_state)
        spikes = self.brain.step(visual_input=visual_input, learn=True)

        # 스파이크 엔트로피 계산
        spike_entropy = self._compute_spike_entropy(spikes)

        # 새로움 계산
        novelty = self._compute_novelty(next_state)

        # 감정 상태 업데이트
        self.emotions.update(spike_entropy, prediction_error, novelty)

        # 도파민 기반 보상 → 성격/관심사 업데이트
        if self.emotions.dopamine > 0.3:
            clicked_type = next_info.get('clicked_type', 'text')
            self.personality.update_from_experience(clicked_type, self.emotions.dopamine)

            # 관심사 강화 (네비게이션 요소 제외)
            clicked_content = next_info.get('clicked_content', '')
            navigation_words = {'Back', 'Home', 'Next', 'Previous', 'Menu'}
            if clicked_content and clicked_content not in navigation_words:
                self.personality.add_interest(clicked_content, self.emotions.dopamine)

            # 페이지 주제도 관심사로 (home 제외)
            page_name = next_info.get('page_name', '')
            if page_name and page_name not in {'home', 'index'}:
                self.personality.add_interest(page_name, self.emotions.dopamine * 0.5)

        # 이력 저장
        self.spike_history.append(next_state.cpu().numpy())
        self.emotion_history.append(self.emotions.to_dict())

        # 다음 스텝을 위해 저장
        self.prev_state = state
        self.prev_action = action
        self.prev_prediction = prediction

        return {
            'step': self.step_count,
            'page': next_info.get('page_name', ''),
            'action': action,
            'clicked': next_info.get('clicked_content', ''),
            'emotions': self.emotions.to_dict(),
            'spike_rate': sum(s.mean().item() for s in spikes.values()) / len(spikes),
            'prediction_error': prediction_error,
            'novelty': novelty,
        }

    def explore(self,
                n_steps: int = 1000,
                log_interval: int = 50,
                save_interval: int = 200):
        """자율 탐험"""
        self.session_start = time.time()

        print(f"\n{'='*60}")
        print(f"Starting exploration ({n_steps} steps)")
        print(f"{'='*60}\n")

        for step in range(n_steps):
            result = self.step()

            if step % log_interval == 0:
                elapsed = time.time() - self.session_start
                print(f"\n[Step {step}] {elapsed:.1f}s")
                print(f"  Page: {result['page']}")
                print(f"  Clicked: {result['clicked']}")
                print(f"  Mood: {result['emotions']['mood']}")
                print(f"  Curiosity: {result['emotions']['curiosity']:.2f}")
                print(f"  Dopamine: {result['emotions']['dopamine']:.2f}")
                print(f"  Novelty: {result['novelty']:.2f}")

                # 관심사 표시
                top_interests = self.personality.get_top_interests(3)
                if top_interests:
                    print(f"  Top interests: {top_interests}")

            if step % save_interval == 0 and step > 0:
                self.save_checkpoint(f"checkpoint_{step}.json")

        # 최종 저장
        self.save_checkpoint("checkpoint_final.json")
        self._print_summary()

    def _print_summary(self):
        """탐험 요약"""
        elapsed = time.time() - self.session_start if self.session_start else 0
        stats = self.env.get_visit_stats()

        print(f"\n{'='*60}")
        print("EXPLORATION SUMMARY")
        print(f"{'='*60}")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Total steps: {self.step_count}")

        if stats:
            print(f"\nVisit statistics:")
            print(f"  Total clicks: {stats['total_clicks']}")
            print(f"  Text clicks: {stats['type_counts'].get('text', 0)}")
            print(f"  Image clicks: {stats['type_counts'].get('image', 0)}")
            if stats['most_visited']:
                print(f"  Most visited: {stats['most_visited']}")

        print(f"\nPersonality evolved:")
        print(f"  Text preference: {self.personality.text_preference:.2f}")

        print(f"\nTop interests (Emerging preferences):")
        for topic, strength in self.personality.get_top_interests(10):
            bar = "#" * int(strength * 20)
            print(f"  {topic:20s} {bar} {strength:.2f}")

        # 발달 단계 판정
        stage = self._assess_development_stage()
        print(f"\nDevelopment stage: {stage}")

    def _assess_development_stage(self) -> str:
        """발달 단계 평가"""
        # 유아기: 무작위
        # 아동기: 패턴 매칭
        # 청소년기: 관심사 형성

        n_interests = len(self.personality.interests)
        top_interests = self.personality.get_top_interests(3)
        max_interest_strength = top_interests[0][1] if top_interests else 0

        if n_interests < 5:
            return "Infancy (Random Surfer) - Primitive exploration"
        elif max_interest_strength < 0.5:
            return "Childhood (Pattern Matcher) - Learning patterns"
        else:
            dominant_topic = top_interests[0][0]
            return f"Adolescence (Concept Explorer) - Interested in '{dominant_topic}'!"

    def save_checkpoint(self, filename: str):
        """상태 저장"""
        path = os.path.join(self.checkpoint_dir, filename)

        data = {
            'step_count': self.step_count,
            'personality': {
                'text_preference': self.personality.text_preference,
                'depth_preference': self.personality.depth_preference,
                'risk_tolerance': self.personality.risk_tolerance,
                'patience': self.personality.patience,
                'interests': self.personality.interests,
                'total_clicks': self.personality.total_clicks,
            },
            'emotions': self.emotions.to_dict(),
            'visit_history': self.env.visit_history[-100:],
            'timestamp': datetime.now().isoformat(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        # Brain weights
        brain_path = path.replace('.json', '_brain.pt')
        torch.save({
            'brain': self.brain.state_dict(),
            'predictor': self.predictor.state_dict(),
        }, brain_path)

        print(f"  Saved: {filename}")


def test_digital_explorer():
    """Digital Explorer 테스트"""
    # tiny scale for memory efficiency during long runs
    explorer = DigitalExplorer(brain_scale="tiny")
    explorer.explore(n_steps=3000, log_interval=300)
    return explorer


if __name__ == "__main__":
    explorer = test_digital_explorer()
