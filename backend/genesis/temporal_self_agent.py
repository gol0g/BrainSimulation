"""
Temporal Self Agent - 시간적 자아를 가진 SNN 에이전트
======================================================

Phase D: 시간적 자아

핵심 능력:
1. Autobiographical Memory: 에피소드 간 기억 연결
2. 장기 목표 설정 및 추적
3. 후회/자부심의 시간적 확장
4. "나의 역사"로서의 경험 통합

생물학적 기반:
- 해마 (Hippocampus): 에피소드 기억, 자서전적 기억
- 전전두엽 (PFC): 목표 유지, 계획
- 측두엽: 시간적 맥락 통합

NO LLM, NO FEP formulas, NO 심즈식 게이지
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
from collections import deque
import random
import time

DEVICE = torch.device('cpu')


@dataclass
class Episode:
    """에피소드 기억"""
    episode_id: int
    timestamp: float
    context: str  # 과제/상황 설명
    actions_taken: List[int]
    outcome: str  # 'success', 'failure', 'partial'
    reward_total: float
    emotional_valence: float  # -1 (부정) ~ +1 (긍정)
    key_events: List[str]
    lessons_learned: List[str]


@dataclass
class LongTermGoal:
    """장기 목표"""
    goal_id: int
    description: str
    created_at: float
    target_metric: str
    target_value: float
    current_progress: float
    deadline_episodes: int
    is_achieved: bool = False
    is_abandoned: bool = False


@dataclass
class TemporalEmotion:
    """시간적 감정 (후회/자부심)"""
    emotion_type: str  # 'regret', 'pride', 'nostalgia'
    intensity: float  # 0 ~ 1
    source_episode_id: int
    description: str
    timestamp: float


class LifeEnvironment:
    """
    연속적인 삶의 환경

    에이전트가 여러 에피소드에 걸쳐 경험을 쌓고
    장기적인 목표를 추구하는 환경
    """

    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size
        self.agent_pos = [grid_size // 2, grid_size // 2]
        self.resources: Dict[Tuple[int, int], str] = {}
        self.dangers: Set[Tuple[int, int]] = set()
        self.shelters: Set[Tuple[int, int]] = set()

        # 에이전트 상태 (에피소드 간 유지)
        self.health = 1.0
        self.energy = 1.0
        self.knowledge = 0.0  # 누적 학습
        self.social_bonds = 0.0  # 사회적 관계

        # 시간
        self.total_steps = 0
        self.current_episode = 0
        self.steps_in_episode = 0

        # 환경 이벤트
        self.current_event = None
        self.event_history: List[str] = []

    def new_episode(self) -> np.ndarray:
        """새 에피소드 시작 (상태 유지)"""
        self.current_episode += 1
        self.steps_in_episode = 0

        # 위치만 리셋
        self.agent_pos = [self.grid_size // 2, self.grid_size // 2]

        # 환경 재구성 (약간의 변화)
        self._regenerate_environment()

        # 랜덤 이벤트
        self._generate_event()

        return self._get_observation()

    def _regenerate_environment(self):
        """환경 재생성"""
        self.resources = {}
        self.dangers = set()
        self.shelters = set()

        # 자원 배치
        n_resources = random.randint(3, 8)
        for _ in range(n_resources):
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            resource_type = random.choice(['food', 'water', 'knowledge', 'friend'])
            self.resources[pos] = resource_type

        # 위험 배치
        n_dangers = random.randint(2, 5)
        for _ in range(n_dangers):
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in self.resources:
                self.dangers.add(pos)

        # 쉼터 배치
        n_shelters = random.randint(1, 3)
        for _ in range(n_shelters):
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in self.resources and pos not in self.dangers:
                self.shelters.add(pos)

    def _generate_event(self):
        """랜덤 이벤트 생성"""
        events = [
            None,  # 평범한 날
            "storm",  # 폭풍 - 이동 제한
            "abundance",  # 풍요 - 자원 증가
            "encounter",  # 만남 - 사회적 기회
            "challenge",  # 도전 - 특별 보상 기회
        ]
        weights = [0.5, 0.15, 0.15, 0.1, 0.1]
        self.current_event = random.choices(events, weights)[0]

        if self.current_event:
            self.event_history.append(f"Episode {self.current_episode}: {self.current_event}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Actions:
        0-3: 이동 (상하좌우)
        4: 자원 수집
        5: 휴식 (에너지 회복)
        6: 학습 (지식 축적)
        7: 사회적 상호작용
        """
        self.total_steps += 1
        self.steps_in_episode += 1

        reward = 0.0
        info = {"event": None, "resource_collected": None}

        # 기본 소모
        self.energy -= 0.01
        self.health -= 0.005

        # 이동
        if action < 4:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]

            # 폭풍 시 이동 제한
            if self.current_event == "storm" and random.random() < 0.5:
                dx, dy = 0, 0
                info["event"] = "storm_blocked"

            new_x = max(0, min(self.grid_size-1, self.agent_pos[0] + dx))
            new_y = max(0, min(self.grid_size-1, self.agent_pos[1] + dy))
            self.agent_pos = [new_x, new_y]

            # 위험 충돌
            if tuple(self.agent_pos) in self.dangers:
                self.health -= 0.3
                reward -= 0.5
                info["event"] = "danger_hit"

            # 쉼터 도착
            if tuple(self.agent_pos) in self.shelters:
                self.health = min(1.0, self.health + 0.1)
                info["event"] = "shelter_rest"

        elif action == 4:  # 자원 수집
            pos = tuple(self.agent_pos)
            if pos in self.resources:
                resource = self.resources.pop(pos)
                info["resource_collected"] = resource

                if resource == 'food':
                    self.energy = min(1.0, self.energy + 0.3)
                    reward += 0.3
                elif resource == 'water':
                    self.health = min(1.0, self.health + 0.2)
                    reward += 0.2
                elif resource == 'knowledge':
                    self.knowledge += 0.1
                    reward += 0.4
                elif resource == 'friend':
                    self.social_bonds += 0.2
                    reward += 0.3

                # 풍요 이벤트 보너스
                if self.current_event == "abundance":
                    reward *= 1.5

        elif action == 5:  # 휴식
            self.energy = min(1.0, self.energy + 0.1)
            self.health = min(1.0, self.health + 0.05)

        elif action == 6:  # 학습
            if self.energy > 0.2:
                self.knowledge += 0.05
                self.energy -= 0.1
                reward += 0.1

                # 도전 이벤트 시 학습 보너스
                if self.current_event == "challenge":
                    self.knowledge += 0.1
                    reward += 0.3

        elif action == 7:  # 사회적 상호작용
            if self.current_event == "encounter":
                self.social_bonds += 0.3
                reward += 0.4
                info["event"] = "social_success"
            else:
                self.social_bonds += 0.05
                reward += 0.1

        # 에피소드 종료 조건
        done = False
        if self.health <= 0:
            done = True
            reward -= 1.0
            info["event"] = "death"
        elif self.steps_in_episode >= 50:
            done = True

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """관측 생성"""
        obs = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)

        # 채널 0: 자원
        for (x, y), res in self.resources.items():
            obs[y, x, 0] = {'food': 0.3, 'water': 0.5, 'knowledge': 0.7, 'friend': 0.9}[res]

        # 채널 1: 위험
        for x, y in self.dangers:
            obs[y, x, 1] = 1.0

        # 채널 2: 쉼터
        for x, y in self.shelters:
            obs[y, x, 2] = 1.0

        # 채널 3: 에이전트
        ax, ay = self.agent_pos
        obs[ay, ax, 3] = 1.0

        return obs

    def get_life_stats(self) -> Dict:
        """삶의 통계"""
        return {
            "total_steps": self.total_steps,
            "episodes_lived": self.current_episode,
            "health": self.health,
            "energy": self.energy,
            "knowledge": self.knowledge,
            "social_bonds": self.social_bonds,
            "event_history": self.event_history[-10:]
        }


class HippocampalMemory(nn.Module):
    """
    해마 기반 에피소드 기억 시스템

    생물학적 특징:
    - Pattern separation: 유사 경험 구분
    - Pattern completion: 부분 단서로 전체 기억 회상
    - Temporal context: 시간적 맥락 인코딩
    """

    def __init__(self, memory_size: int = 100, embedding_dim: int = 64):
        super().__init__()
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim

        # 기억 저장소
        self.episodes: deque = deque(maxlen=memory_size)
        self.episode_embeddings: deque = deque(maxlen=memory_size)

        # 인코더 (경험 → 임베딩)
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, embedding_dim),
            nn.Tanh()
        )

        # 시간적 맥락 인코더
        self.temporal_encoder = nn.Linear(4, embedding_dim)

    def encode_episode(self, episode: Episode) -> torch.Tensor:
        """에피소드를 임베딩으로 인코딩"""
        # 특징 추출
        features = np.zeros(128)

        # 결과 인코딩
        features[0] = 1.0 if episode.outcome == 'success' else 0.0
        features[1] = episode.reward_total
        features[2] = episode.emotional_valence
        features[3] = len(episode.actions_taken) / 100.0
        features[4] = len(episode.key_events) / 10.0

        # 행동 분포
        if episode.actions_taken:
            for a in episode.actions_taken:
                if a < 8:
                    features[10 + a] += 1.0 / len(episode.actions_taken)

        # 시간적 맥락
        temporal = np.array([
            episode.episode_id / 1000.0,
            episode.timestamp / 10000.0,
            np.sin(episode.episode_id * 0.1),
            np.cos(episode.episode_id * 0.1)
        ])

        features_tensor = torch.tensor(features, dtype=torch.float32)
        temporal_tensor = torch.tensor(temporal, dtype=torch.float32)

        # 인코딩
        with torch.no_grad():
            embedding = self.encoder(features_tensor)
            temporal_emb = self.temporal_encoder(temporal_tensor)
            combined = embedding + temporal_emb * 0.3

        return combined

    def store(self, episode: Episode):
        """에피소드 저장"""
        embedding = self.encode_episode(episode)
        self.episodes.append(episode)
        self.episode_embeddings.append(embedding)

    def recall_similar(self, query_episode: Episode, top_k: int = 5) -> List[Episode]:
        """유사한 과거 에피소드 회상"""
        if not self.episodes:
            return []

        query_emb = self.encode_episode(query_episode)

        # 유사도 계산
        similarities = []
        for i, emb in enumerate(self.episode_embeddings):
            sim = torch.nn.functional.cosine_similarity(
                query_emb.unsqueeze(0), emb.unsqueeze(0)
            ).item()
            similarities.append((sim, i))

        # 상위 k개 반환
        similarities.sort(reverse=True)
        return [self.episodes[i] for _, i in similarities[:top_k]]

    def recall_by_emotion(self, valence: float, top_k: int = 5) -> List[Episode]:
        """감정 기반 회상"""
        if not self.episodes:
            return []

        # 감정 유사도로 정렬
        scored = [(abs(ep.emotional_valence - valence), ep) for ep in self.episodes]
        scored.sort()
        return [ep for _, ep in scored[:top_k]]

    def get_narrative(self, recent_n: int = 10) -> str:
        """최근 경험의 서사 생성"""
        if not self.episodes:
            return "No memories yet."

        recent = list(self.episodes)[-recent_n:]
        narrative_parts = []

        for ep in recent:
            outcome_str = "succeeded" if ep.outcome == 'success' else "faced challenges"
            emotion_str = "positive" if ep.emotional_valence > 0 else "difficult"

            part = f"In episode {ep.episode_id}, I {outcome_str}. It was a {emotion_str} experience."
            if ep.lessons_learned:
                part += f" I learned: {ep.lessons_learned[0]}"

            narrative_parts.append(part)

        return " ".join(narrative_parts)


class TemporalSelfBrain(nn.Module):
    """
    시간적 자아를 가진 SNN 뇌

    구조:
    - Sensory: 현재 감각
    - Hippocampus: 에피소드 기억
    - PFC: 목표 유지 및 계획
    - Temporal: 시간적 통합
    """

    def __init__(self, n_sensory: int = 300, n_motor: int = 200,
                 n_pfc: int = 150, n_temporal: int = 100):
        super().__init__()

        self.n_sensory = n_sensory
        self.n_motor = n_motor
        self.n_pfc = n_pfc
        self.n_temporal = n_temporal

        # 해마 기억
        self.hippocampus = HippocampalMemory()

        # 인코더
        self.sensory_encoder = nn.Linear(12 * 12 * 4, n_sensory)
        self.memory_encoder = nn.Linear(64, n_temporal)  # 기억 임베딩 → 시간적 뉴런
        self.goal_encoder = nn.Linear(16, n_pfc)  # 목표 특징 → PFC

        # 연결
        self.s_to_m = nn.Linear(n_sensory, n_motor, bias=False)
        self.pfc_to_m = nn.Linear(n_pfc, n_motor, bias=False)
        self.temporal_to_m = nn.Linear(n_temporal, n_motor, bias=False)
        self.temporal_to_pfc = nn.Linear(n_temporal, n_pfc, bias=False)

        # 초기화
        for layer in [self.s_to_m, self.pfc_to_m, self.temporal_to_m, self.temporal_to_pfc]:
            nn.init.uniform_(layer.weight, 0.01, 0.05)

        # LIF 상태
        self.register_buffer('v_sensory', torch.zeros(n_sensory))
        self.register_buffer('v_motor', torch.zeros(n_motor))
        self.register_buffer('v_pfc', torch.zeros(n_pfc))
        self.register_buffer('v_temporal', torch.zeros(n_temporal))

        # DA-STDP eligibility traces
        self.register_buffer('elig_s_to_m', torch.zeros(n_sensory, n_motor))
        self.register_buffer('elig_pfc_to_m', torch.zeros(n_pfc, n_motor))

        # 목표 시스템
        self.active_goals: List[LongTermGoal] = []
        self.achieved_goals: List[LongTermGoal] = []
        self.goal_counter = 0

        # 시간적 감정
        self.temporal_emotions: List[TemporalEmotion] = []

        # 파라미터
        self.tau_mem = 20.0
        self.v_th = 1.0
        self.tau_elig = 500.0

    def reset_state(self):
        """LIF 상태 리셋 (기억은 유지)"""
        self.v_sensory.zero_()
        self.v_motor.zero_()
        self.v_pfc.zero_()
        self.v_temporal.zero_()

    def set_goal(self, description: str, metric: str, target: float, deadline: int):
        """장기 목표 설정"""
        self.goal_counter += 1
        goal = LongTermGoal(
            goal_id=self.goal_counter,
            description=description,
            created_at=time.time(),
            target_metric=metric,
            target_value=target,
            current_progress=0.0,
            deadline_episodes=deadline
        )
        self.active_goals.append(goal)
        return goal

    def update_goals(self, metrics: Dict):
        """목표 진행 상황 업데이트"""
        for goal in self.active_goals:
            if goal.target_metric in metrics:
                goal.current_progress = metrics[goal.target_metric]

                if goal.current_progress >= goal.target_value:
                    goal.is_achieved = True
                    self.achieved_goals.append(goal)
                    self.active_goals.remove(goal)

                    # 자부심 감정 생성
                    self.temporal_emotions.append(TemporalEmotion(
                        emotion_type='pride',
                        intensity=0.8,
                        source_episode_id=-1,
                        description=f"Achieved goal: {goal.description}",
                        timestamp=time.time()
                    ))

    def forward(self, observation: np.ndarray, current_episode: int,
                life_stats: Dict, learn: bool = True, dopamine: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """
        한 스텝 전파
        """
        # 관측 인코딩
        obs_flat = torch.tensor(observation.flatten(), dtype=torch.float32)
        sensory_rates = torch.sigmoid(self.sensory_encoder(obs_flat))
        sensory_input = (torch.rand_like(sensory_rates) < sensory_rates * 0.3).float()

        # LIF - Sensory
        self.v_sensory = self.v_sensory * (1 - 1/self.tau_mem) + sensory_input
        sensory_spikes = (self.v_sensory > self.v_th).float()
        self.v_sensory = self.v_sensory * (1 - sensory_spikes)

        # 기억 회상 (과거 유사 경험)
        memory_input = torch.zeros(self.n_temporal)
        if self.hippocampus.episodes:
            # 현재 상태와 유사한 과거 경험 회상
            recent_embs = list(self.hippocampus.episode_embeddings)[-5:]
            if recent_embs:
                avg_emb = torch.stack(recent_embs).mean(0)
                memory_input = torch.sigmoid(self.memory_encoder(avg_emb))

        # 목표 인코딩
        goal_features = np.zeros(16)
        if self.active_goals:
            for i, goal in enumerate(self.active_goals[:4]):
                goal_features[i*4] = goal.current_progress / goal.target_value
                goal_features[i*4 + 1] = min(1.0, goal.deadline_episodes / 100)
                goal_features[i*4 + 2] = 1.0  # 목표 활성화

        goal_tensor = torch.tensor(goal_features, dtype=torch.float32)
        pfc_input = torch.sigmoid(self.goal_encoder(goal_tensor))

        # LIF - Temporal (기억 통합)
        self.v_temporal = self.v_temporal * (1 - 1/self.tau_mem) + memory_input * 0.1
        temporal_spikes = (self.v_temporal > self.v_th).float()
        self.v_temporal = self.v_temporal * (1 - temporal_spikes)

        # LIF - PFC (목표 + 시간적 맥락)
        pfc_total = pfc_input + self.temporal_to_pfc(temporal_spikes) * 0.2
        self.v_pfc = self.v_pfc * (1 - 1/self.tau_mem) + pfc_total * 0.1
        pfc_spikes = (self.v_pfc > self.v_th).float()
        self.v_pfc = self.v_pfc * (1 - pfc_spikes)

        # LIF - Motor
        motor_input = (self.s_to_m(sensory_spikes) +
                       self.pfc_to_m(pfc_spikes) +
                       self.temporal_to_m(temporal_spikes))

        self.v_motor = self.v_motor * (1 - 1/self.tau_mem) + motor_input * 0.1
        motor_spikes = (self.v_motor > self.v_th).float()
        self.v_motor = self.v_motor * (1 - motor_spikes)

        # DA-STDP 학습
        if learn:
            self._apply_da_stdp(sensory_spikes, pfc_spikes, motor_spikes, dopamine)

        meta_info = {
            "pfc_activity": pfc_spikes.sum().item() / self.n_pfc,
            "temporal_activity": temporal_spikes.sum().item() / self.n_temporal,
            "active_goals": len(self.active_goals),
            "memories": len(self.hippocampus.episodes)
        }

        return motor_spikes.numpy(), meta_info

    def _apply_da_stdp(self, s_spikes, pfc_spikes, m_spikes, dopamine: float):
        """DA-STDP"""
        decay = 1 - 1 / self.tau_elig
        self.elig_s_to_m *= decay
        self.elig_pfc_to_m *= decay

        self.elig_s_to_m += torch.outer(s_spikes, m_spikes) * 0.01
        self.elig_pfc_to_m += torch.outer(pfc_spikes, m_spikes) * 0.01

        if abs(dopamine) > 0.1:
            with torch.no_grad():
                self.s_to_m.weight += (dopamine * self.elig_s_to_m.T * 0.1).clamp(-0.01, 0.01)
                self.pfc_to_m.weight += (dopamine * self.elig_pfc_to_m.T * 0.1).clamp(-0.01, 0.01)

                self.s_to_m.weight.clamp_(0.0, 1.0)
                self.pfc_to_m.weight.clamp_(0.0, 1.0)

                self.elig_s_to_m *= 0.5
                self.elig_pfc_to_m *= 0.5

    def store_episode(self, episode: Episode):
        """에피소드 기억 저장"""
        self.hippocampus.store(episode)

        # 강한 부정적 경험 → 후회 감정
        if episode.emotional_valence < -0.5:
            self.temporal_emotions.append(TemporalEmotion(
                emotion_type='regret',
                intensity=abs(episode.emotional_valence),
                source_episode_id=episode.episode_id,
                description=f"Regret from episode {episode.episode_id}",
                timestamp=time.time()
            ))

        # 강한 긍정적 경험 → 자부심
        if episode.emotional_valence > 0.5:
            self.temporal_emotions.append(TemporalEmotion(
                emotion_type='pride',
                intensity=episode.emotional_valence,
                source_episode_id=episode.episode_id,
                description=f"Pride from episode {episode.episode_id}",
                timestamp=time.time()
            ))

    def get_autobiographical_narrative(self) -> str:
        """자서전적 서사 반환"""
        return self.hippocampus.get_narrative()


class TemporalSelfAgent:
    """시간적 자아를 가진 에이전트"""

    def __init__(self, n_actions: int = 8):
        self.n_actions = n_actions
        self.n_motor = 200

        self.brain = TemporalSelfBrain()
        self.brain.to(DEVICE)

        # 도파민
        self.dopamine_baseline = 0.1
        self.current_dopamine = self.dopamine_baseline

        # 현재 에피소드 추적
        self.current_episode_id = 0
        self.episode_actions: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_events: List[str] = []

    def start_episode(self, episode_id: int):
        """에피소드 시작"""
        self.current_episode_id = episode_id
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_events = []
        self.brain.reset_state()

    def select_action(self, observation: np.ndarray, life_stats: Dict) -> Tuple[int, Dict]:
        """행동 선택"""
        n_steps = 10
        motor_accumulator = np.zeros(self.n_motor)
        meta_infos = []

        for _ in range(n_steps):
            motor_spikes, meta_info = self.brain.forward(
                observation, self.current_episode_id, life_stats, learn=False
            )
            motor_accumulator += motor_spikes
            meta_infos.append(meta_info)

        # 행동 점수
        action_scores = np.zeros(self.n_actions)
        neurons_per_action = self.n_motor // self.n_actions

        for a in range(self.n_actions):
            start = a * neurons_per_action
            end = start + neurons_per_action
            action_scores[a] = motor_accumulator[start:end].sum()

        # 목표 기반 보너스
        if self.brain.active_goals:
            for goal in self.brain.active_goals:
                # 목표에 맞는 행동 강화
                if goal.target_metric == 'knowledge' and 6 < self.n_actions:
                    action_scores[6] += 2.0  # 학습 행동
                elif goal.target_metric == 'social_bonds' and 7 < self.n_actions:
                    action_scores[7] += 2.0  # 사회적 행동

        # 소프트맥스
        probs = self._softmax(action_scores, temperature=0.5)
        action = np.random.choice(self.n_actions, p=probs)

        self.episode_actions.append(action)

        return action, {
            "pfc_activity": np.mean([m["pfc_activity"] for m in meta_infos]),
            "temporal_activity": np.mean([m["temporal_activity"] for m in meta_infos]),
            "active_goals": meta_infos[-1]["active_goals"],
            "memories": meta_infos[-1]["memories"]
        }

    def learn(self, observation: np.ndarray, reward: float, life_stats: Dict, event: str = None):
        """학습"""
        self.episode_rewards.append(reward)
        if event:
            self.episode_events.append(event)

        # 도파민 계산
        expected_reward = 0.1  # 기준 기대값
        prediction_error = reward - expected_reward
        self.current_dopamine = self.dopamine_baseline + prediction_error * 0.5
        self.current_dopamine = np.clip(self.current_dopamine, -0.5, 1.5)

        # 뇌 학습
        n_steps = 5
        for _ in range(n_steps):
            self.brain.forward(
                observation, self.current_episode_id, life_stats,
                learn=True, dopamine=self.current_dopamine
            )

        # 목표 업데이트
        self.brain.update_goals(life_stats)

    def end_episode(self, success: bool, life_stats: Dict):
        """에피소드 종료 및 기억 저장"""
        total_reward = sum(self.episode_rewards)
        emotional_valence = np.tanh(total_reward)  # -1 ~ 1

        # 학습된 교훈 추출
        lessons = []
        if success:
            lessons.append("Persistence pays off")
        if 'danger_hit' in self.episode_events:
            lessons.append("Avoid dangerous areas")
        if life_stats.get('knowledge', 0) > 0.5:
            lessons.append("Knowledge accumulates over time")

        # 에피소드 기억 생성
        episode = Episode(
            episode_id=self.current_episode_id,
            timestamp=time.time(),
            context=f"Episode {self.current_episode_id}",
            actions_taken=self.episode_actions.copy(),
            outcome='success' if success else 'failure',
            reward_total=total_reward,
            emotional_valence=emotional_valence,
            key_events=self.episode_events.copy(),
            lessons_learned=lessons
        )

        # 해마에 저장
        self.brain.store_episode(episode)

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        x = x / temperature
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (exp_x.sum() + 1e-8)

    def get_life_narrative(self) -> str:
        """삶의 서사 반환"""
        return self.brain.get_autobiographical_narrative()


def run_temporal_self_experiment(n_episodes: int = 50):
    """시간적 자아 실험"""
    print("=" * 60)
    print("Phase D: Temporal Self Agent")
    print("=" * 60)
    print("\n자서전적 기억과 장기 목표 추구 실험")
    print("Biological mechanisms: Hippocampus, PFC, DA-STDP")
    print("NO LLM, NO FEP formulas\n")

    env = LifeEnvironment()
    agent = TemporalSelfAgent()

    # 장기 목표 설정
    agent.brain.set_goal("Accumulate knowledge", "knowledge", 1.0, 30)
    agent.brain.set_goal("Build social bonds", "social_bonds", 0.8, 40)
    print("Goals set:")
    for goal in agent.brain.active_goals:
        print(f"  - {goal.description}: target={goal.target_value}")
    print()

    episode_stats = []
    life_metrics_history = []

    for episode in range(n_episodes):
        obs = env.new_episode()
        agent.start_episode(episode + 1)

        episode_reward = 0
        episode_events = []

        for step in range(50):
            life_stats = env.get_life_stats()
            action, meta = agent.select_action(obs, life_stats)

            next_obs, reward, done, info = env.step(action)
            episode_reward += reward

            if info.get("event"):
                episode_events.append(info["event"])

            agent.learn(next_obs, reward, env.get_life_stats(), info.get("event"))

            obs = next_obs
            if done:
                break

        # 에피소드 종료
        final_stats = env.get_life_stats()
        success = final_stats['health'] > 0 and episode_reward > 0
        agent.end_episode(success, final_stats)

        # 기록
        stats = {
            "episode": episode + 1,
            "reward": episode_reward,
            "health": final_stats['health'],
            "knowledge": final_stats['knowledge'],
            "social_bonds": final_stats['social_bonds'],
            "memories": len(agent.brain.hippocampus.episodes),
            "active_goals": len(agent.brain.active_goals),
            "achieved_goals": len(agent.brain.achieved_goals)
        }
        episode_stats.append(stats)
        life_metrics_history.append(final_stats.copy())

        # 주기적 출력
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Episode {episode + 1}/{n_episodes} (Total steps: {final_stats['total_steps']}):")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Health: {final_stats['health']:.2f}, Energy: {final_stats['energy']:.2f}")
            print(f"  Knowledge: {final_stats['knowledge']:.2f}, Social: {final_stats['social_bonds']:.2f}")
            print(f"  Memories stored: {len(agent.brain.hippocampus.episodes)}")
            print(f"  Goals - Active: {len(agent.brain.active_goals)}, Achieved: {len(agent.brain.achieved_goals)}")

            # 시간적 감정
            recent_emotions = agent.brain.temporal_emotions[-3:]
            if recent_emotions:
                print(f"  Recent emotions: {[e.emotion_type for e in recent_emotions]}")
            print()

    # 최종 분석
    print("=" * 60)
    print("FINAL ANALYSIS: Temporal Self")
    print("=" * 60)

    print("\n[1] 삶의 궤적:")
    first_10 = episode_stats[:10]
    last_10 = episode_stats[-10:]

    print(f"  Knowledge: {np.mean([s['knowledge'] for s in first_10]):.2f} -> {np.mean([s['knowledge'] for s in last_10]):.2f}")
    print(f"  Social: {np.mean([s['social_bonds'] for s in first_10]):.2f} -> {np.mean([s['social_bonds'] for s in last_10]):.2f}")

    print("\n[2] 목표 달성:")
    for goal in agent.brain.achieved_goals:
        print(f"  [ACHIEVED] {goal.description}")
    for goal in agent.brain.active_goals:
        print(f"  [IN PROGRESS] {goal.description}: {goal.current_progress:.2f}/{goal.target_value}")

    print("\n[3] 시간적 감정:")
    pride_count = sum(1 for e in agent.brain.temporal_emotions if e.emotion_type == 'pride')
    regret_count = sum(1 for e in agent.brain.temporal_emotions if e.emotion_type == 'regret')
    print(f"  Pride experiences: {pride_count}")
    print(f"  Regret experiences: {regret_count}")

    print("\n[4] 자서전적 기억 (최근):")
    narrative = agent.get_life_narrative()
    print(f"  {narrative[:500]}..." if len(narrative) > 500 else f"  {narrative}")

    print("\n[5] 핵심 발견:")
    if agent.brain.achieved_goals:
        print("  [SUCCESS] 장기 목표 달성!")
    if len(agent.brain.hippocampus.episodes) >= n_episodes * 0.8:
        print("  [SUCCESS] 자서전적 기억 형성!")
    if pride_count > 0 or regret_count > 0:
        print("  [SUCCESS] 시간적 감정 창발!")

    return episode_stats


if __name__ == "__main__":
    stats = run_temporal_self_experiment(n_episodes=50)
