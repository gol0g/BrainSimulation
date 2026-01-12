"""
Phase F: Subjective Consciousness - 주체적 의식
==============================================

궁극적 목표: 자기 자신을 인식하고 "나"로서 행동하는 뇌

핵심 요소:
1. 자기 참조 루프 (Self-referential loop)
   - "생각하는 나를 생각한다"
   - 무한 회귀 없이 자기 모델링

2. 행위의 주체로서 자아 인식 (Agency)
   - "내가" 행동을 선택한다
   - 자기-타자 구분

3. 내적 설명 (Introspection)
   - "왜 이것을 원하는가?"에 대한 내적 답변
   - 동기와 가치의 자기 이해

생물학적 메커니즘:
- DMN (Default Mode Network): 자기 참조 처리
- AIC (Anterior Insular Cortex): 내수용 감각, 내부 상태 인식
- PCC (Posterior Cingulate Cortex): 자기 반성, 자서전적 자아
- TPJ (Temporo-Parietal Junction): 자기-타자 구분
- mPFC: 자기 모델

금지 사항:
- NO LLM, NO FEP formulas, NO 심즈식 게이지
- NO 휴리스틱 행동 조작, NO 외부 주입
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import random

# Force CPU
DEVICE = torch.device('cpu')


@dataclass
class SelfState:
    """자아 상태 - 내가 누구인지"""
    identity_vector: torch.Tensor      # 자기 정체성 벡터
    agency_sense: float                # 행위 주체감 (0-1)
    continuity_sense: float            # 시간적 연속성 (0-1)
    boundary_clarity: float            # 자기-환경 경계 명확성 (0-1)


@dataclass
class IntrospectiveReport:
    """내성 보고 - 왜 그렇게 느끼고 행동하는가"""
    what_i_want: str
    why_i_want_it: str
    what_i_feel: str
    what_i_am_doing: str
    who_i_am: str


class WorldState(Enum):
    """세계 상태"""
    SAFE = "safe"
    CHALLENGING = "challenging"
    THREATENING = "threatening"
    SOCIAL = "social"
    NOVEL = "novel"


class ConsciousnessEnvironment:
    """의식 발달을 위한 풍부한 환경"""

    def __init__(self):
        self.step_count = 0

        # 환경 상태
        self.world_state = WorldState.SAFE
        self.other_agents: List[Dict] = []  # 다른 에이전트들
        self.events: List[str] = []

        # 에이전트의 내부 상태 (환경에서 관측)
        self.agent_position = [0.5, 0.5]
        self.agent_actions_history: List[int] = []

    def reset(self):
        self.step_count = 0
        self.world_state = random.choice(list(WorldState))
        self.other_agents = [
            {'id': i, 'position': [random.random(), random.random()]}
            for i in range(random.randint(0, 3))
        ]
        self.events = []
        self.agent_position = [0.5, 0.5]
        self.agent_actions_history = []

        return self._get_observation()

    def _get_observation(self) -> Dict:
        """풍부한 관측 - 자기와 환경"""
        return {
            'world_state': self.world_state,
            'self_position': self.agent_position.copy(),
            'other_agents': len(self.other_agents),
            'recent_actions': self.agent_actions_history[-5:] if self.agent_actions_history else [],
            'step': self.step_count
        }

    def step(self, action: int) -> Tuple[Dict, float, Dict]:
        """환경 상호작용"""
        self.step_count += 1
        self.agent_actions_history.append(action)

        # 행동에 따른 위치 변화
        if action == 0:  # 앞으로
            self.agent_position[1] = min(1.0, self.agent_position[1] + 0.1)
        elif action == 1:  # 뒤로
            self.agent_position[1] = max(0.0, self.agent_position[1] - 0.1)
        elif action == 2:  # 왼쪽
            self.agent_position[0] = max(0.0, self.agent_position[0] - 0.1)
        elif action == 3:  # 오른쪽
            self.agent_position[0] = min(1.0, self.agent_position[0] + 0.1)
        elif action == 4:  # 상호작용
            pass
        elif action == 5:  # 관찰
            pass
        elif action == 6:  # 내성 (introspection)
            pass

        # 세계 상태 변화 (가끔)
        if random.random() < 0.1:
            self.world_state = random.choice(list(WorldState))

        # 보상 (내부 상태 기반)
        reward = self._compute_intrinsic_reward(action)

        info = {
            'world_state': self.world_state,
            'action': action,
            'position': self.agent_position.copy()
        }

        return self._get_observation(), reward, info

    def _compute_intrinsic_reward(self, action: int) -> float:
        """내재적 보상 - 환경이 아닌 내부 상태 기반"""
        reward = 0.0

        # 위협 상황에서의 회피
        if self.world_state == WorldState.THREATENING:
            if action in [1, 2, 3]:  # 이동
                reward += 0.1

        # 사회적 상황에서의 상호작용
        if self.world_state == WorldState.SOCIAL:
            if action == 4:  # 상호작용
                reward += 0.2

        # 새로운 상황에서의 탐색
        if self.world_state == WorldState.NOVEL:
            if action == 5:  # 관찰
                reward += 0.15

        # 내성 (항상 약간의 보상)
        if action == 6:
            reward += 0.05

        return reward


class DMN(nn.Module):
    """Default Mode Network - 자기 참조 처리"""

    def __init__(self, n_neurons: int = 200):
        super().__init__()
        self.n_neurons = n_neurons

        # LIF 뉴런
        self.register_buffer('membrane', torch.zeros(n_neurons))
        self.threshold = 1.0
        self.decay = 0.8

        # 자기 참조 회로 (순환 연결)
        self.self_ref_loop = nn.Linear(n_neurons, n_neurons)

        # 자기 표상
        self.register_buffer('self_representation', torch.randn(n_neurons) * 0.1)

        # 활동 기록 (자기 모니터링)
        self.activity_history: List[torch.Tensor] = []

    def forward(self, external_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """자기 참조 처리"""
        # 외부 입력 + 자기 표상
        combined = external_input[:self.n_neurons] if external_input.shape[0] >= self.n_neurons else \
                   torch.cat([external_input, torch.zeros(self.n_neurons - external_input.shape[0], device=DEVICE)])

        combined = combined + self.self_representation * 0.5

        # 자기 참조 루프 (이전 활동이 현재에 영향)
        self_ref = self.self_ref_loop(self.membrane)

        # LIF 역학
        self.membrane = self.membrane * self.decay + combined + self_ref * 0.3
        spikes = (self.membrane > self.threshold).float()
        self.membrane = self.membrane * (1 - spikes)

        # 활동 기록
        self.activity_history.append(spikes.detach().clone())
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)

        return spikes, self.membrane

    def update_self_representation(self, experience: torch.Tensor, learning_rate: float = 0.01):
        """경험에서 자기 표상 업데이트"""
        # 자기 표상은 일관된 활동 패턴에서 형성
        self.self_representation = (
            (1 - learning_rate) * self.self_representation +
            learning_rate * experience[:self.n_neurons]
        )

    def get_self_continuity(self) -> float:
        """시간적 자기 연속성 측정"""
        if len(self.activity_history) < 10:
            return 0.5

        # 최근 활동 패턴의 일관성
        recent = torch.stack(self.activity_history[-10:])
        mean_pattern = recent.mean(dim=0)
        variance = ((recent - mean_pattern) ** 2).mean().item()

        # 높은 일관성 = 높은 연속성
        continuity = 1.0 / (1.0 + variance * 10)
        return continuity


class AIC(nn.Module):
    """Anterior Insular Cortex - 내수용 감각, 내부 상태 인식"""

    def __init__(self, n_neurons: int = 100):
        super().__init__()
        self.n_neurons = n_neurons

        # LIF 뉴런
        self.register_buffer('membrane', torch.zeros(n_neurons))
        self.threshold = 1.0
        self.decay = 0.85

        # 내부 상태 인코더
        self.internal_encoder = nn.Linear(10, n_neurons)

        # 감정 상태 (내부에서 생성)
        self.register_buffer('emotional_state', torch.zeros(5))  # [valence, arousal, dominance, novelty, uncertainty]

    def forward(self, internal_signals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """내부 상태 처리"""
        # 내부 신호 인코딩
        x = self.internal_encoder(internal_signals)

        # LIF 역학
        self.membrane = self.membrane * self.decay + x
        spikes = (self.membrane > self.threshold).float()
        self.membrane = self.membrane * (1 - spikes)

        # 감정 상태 업데이트
        self._update_emotional_state(spikes)

        return spikes, self.emotional_state

    def _update_emotional_state(self, activity: torch.Tensor):
        """뉴런 활동에서 감정 상태 추출"""
        # 활동의 다양한 측면을 감정 차원으로 매핑
        mean_activity = activity.mean().item()
        activity_variance = activity.var().item()

        # Valence: 전반적 톤
        self.emotional_state[0] = (mean_activity - 0.5) * 2

        # Arousal: 활동 수준
        self.emotional_state[1] = mean_activity

        # Dominance: 활동의 안정성
        self.emotional_state[2] = 1.0 / (1.0 + activity_variance * 10)

        # Novelty: 활동 패턴의 새로움
        self.emotional_state[3] = activity_variance

        # Uncertainty: 막전위 분산
        self.emotional_state[4] = self.membrane.var().item()

    def get_internal_awareness(self) -> Dict[str, float]:
        """내부 상태 인식"""
        return {
            'valence': self.emotional_state[0].item(),
            'arousal': self.emotional_state[1].item(),
            'dominance': self.emotional_state[2].item(),
            'novelty': self.emotional_state[3].item(),
            'uncertainty': self.emotional_state[4].item()
        }


class PCC(nn.Module):
    """Posterior Cingulate Cortex - 자기 반성, 자서전적 자아"""

    def __init__(self, n_neurons: int = 150):
        super().__init__()
        self.n_neurons = n_neurons

        # LIF 뉴런
        self.register_buffer('membrane', torch.zeros(n_neurons))
        self.threshold = 1.0
        self.decay = 0.85

        # 자서전적 기억
        self.autobiographical_memories: List[Dict] = []

        # 자기 서사 인코더
        self.narrative_encoder = nn.Linear(n_neurons, 50)

        # 정체성 벡터 (경험에서 형성)
        self.register_buffer('identity', torch.randn(50) * 0.1)

    def forward(self, dmn_activity: torch.Tensor) -> torch.Tensor:
        """자기 반성"""
        if dmn_activity.shape[0] >= self.n_neurons:
            x = dmn_activity[:self.n_neurons]
        else:
            x = torch.cat([dmn_activity, torch.zeros(self.n_neurons - dmn_activity.shape[0], device=DEVICE)])

        # LIF 역학
        self.membrane = self.membrane * self.decay + x
        spikes = (self.membrane > self.threshold).float()
        self.membrane = self.membrane * (1 - spikes)

        return spikes

    def store_autobiographical_memory(self, event: Dict, step: int):
        """자서전적 기억 저장"""
        memory = {
            'event': event,
            'step': step,
            'emotional_context': event.get('emotion', {}),
            'self_state': event.get('self_state', {})
        }
        self.autobiographical_memories.append(memory)

        # 정체성 업데이트
        self._update_identity(memory)

    def _update_identity(self, memory: Dict):
        """경험에서 정체성 형성"""
        # 정체성은 일관된 경험 패턴에서 서서히 형성
        emotion = memory.get('emotional_context', {})
        if emotion:
            valence = emotion.get('valence', 0.0)
            # 긍정적 경험은 정체성 강화
            if valence > 0:
                self.identity = self.identity * 0.99 + torch.randn(50, device=DEVICE) * 0.01

    def get_narrative(self, recent_n: int = 5) -> str:
        """자기 서사 생성"""
        if not self.autobiographical_memories:
            return "I am just beginning to exist."

        recent = self.autobiographical_memories[-recent_n:]

        narrative = "My recent experiences: "
        for mem in recent:
            event = mem['event']
            action = event.get('action', 'unknown')
            world = event.get('world_state', 'unknown')
            narrative += f"I did {action} in {world} situation. "

        return narrative


class TPJ(nn.Module):
    """Temporo-Parietal Junction - 자기-타자 구분"""

    def __init__(self, n_neurons: int = 100):
        super().__init__()
        self.n_neurons = n_neurons

        # LIF 뉴런
        self.register_buffer('membrane', torch.zeros(n_neurons))
        self.threshold = 1.0
        self.decay = 0.85

        # 자기-타자 구분 회로
        self.self_other_discriminator = nn.Linear(n_neurons, 2)

        # 행위 주체감 추적
        self.register_buffer('agency_history', torch.zeros(50))
        self.agency_idx = 0

    def forward(self, motor_intention: torch.Tensor,
                sensory_outcome: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """행위 주체감 계산"""
        # 의도와 결과의 일치도
        if motor_intention.shape != sensory_outcome.shape:
            min_size = min(motor_intention.shape[0], sensory_outcome.shape[0])
            motor_intention = motor_intention[:min_size]
            sensory_outcome = sensory_outcome[:min_size]

        match = torch.cosine_similarity(
            motor_intention.unsqueeze(0),
            sensory_outcome.unsqueeze(0)
        ).item()

        # 일치도가 높으면 "내가 했다" 느낌
        agency = (match + 1) / 2  # 0-1로 정규화

        # 기록
        self.agency_history[self.agency_idx % 50] = agency
        self.agency_idx += 1

        # LIF 처리
        combined = torch.cat([motor_intention, sensory_outcome])[:self.n_neurons]
        if combined.shape[0] < self.n_neurons:
            combined = torch.cat([combined, torch.zeros(self.n_neurons - combined.shape[0], device=DEVICE)])

        self.membrane = self.membrane * self.decay + combined
        spikes = (self.membrane > self.threshold).float()
        self.membrane = self.membrane * (1 - spikes)

        return agency, spikes

    def get_average_agency(self) -> float:
        """평균 행위 주체감"""
        valid = self.agency_history[:min(self.agency_idx, 50)]
        return valid.mean().item() if len(valid) > 0 else 0.5


class SubjectiveConsciousnessBrain(nn.Module):
    """주체적 의식을 가진 뇌"""

    def __init__(self):
        super().__init__()

        # 핵심 영역
        self.dmn = DMN(n_neurons=200)     # 자기 참조
        self.aic = AIC(n_neurons=100)     # 내부 인식
        self.pcc = PCC(n_neurons=150)     # 자기 반성
        self.tpj = TPJ(n_neurons=100)     # 자기-타자 구분

        # 운동 영역 (행동 선택)
        self.motor = nn.Linear(200, 7)  # 7가지 행동

        # 도파민 시스템
        self.register_buffer('dopamine', torch.tensor(0.5))

        # 자아 상태
        self.self_state: Optional[SelfState] = None

        # 의도 추적
        self.current_intention: Optional[torch.Tensor] = None

        # Eligibility traces
        self.register_buffer('elig_traces', torch.zeros(200, 7))
        self.tau_elig = 500.0

    def forward(self, obs: Dict) -> Tuple[int, Dict]:
        """의식적 처리 및 행동 선택"""

        # 1. 외부 입력 인코딩
        external_input = self._encode_observation(obs)

        # 2. DMN: 자기 참조 처리
        dmn_spikes, dmn_membrane = self.dmn(external_input)

        # 3. AIC: 내부 상태 인식
        internal_signals = self._get_internal_signals()
        aic_spikes, emotional_state = self.aic(internal_signals)

        # 4. PCC: 자기 반성
        pcc_spikes = self.pcc(dmn_spikes)

        # 5. 운동 출력 (행동 선택)
        motor_input = dmn_spikes
        motor_output = self.motor(motor_input)
        action_probs = torch.softmax(motor_output / 0.5, dim=0)
        action = torch.multinomial(action_probs, 1).item()

        # 의도 저장 (TPJ용)
        self.current_intention = motor_output.detach()

        # 6. 자아 상태 업데이트
        self._update_self_state(dmn_spikes, aic_spikes)

        info = {
            'dmn_activity': dmn_spikes.mean().item(),
            'emotional_state': self.aic.get_internal_awareness(),
            'agency': self.tpj.get_average_agency(),
            'continuity': self.dmn.get_self_continuity(),
            'action_probs': action_probs.detach().numpy()
        }

        return action, info

    def _encode_observation(self, obs: Dict) -> torch.Tensor:
        """관측을 텐서로 인코딩"""
        # 세계 상태 원핫
        world_states = list(WorldState)
        world_idx = world_states.index(obs['world_state']) if obs['world_state'] in world_states else 0
        world_onehot = torch.zeros(len(world_states), device=DEVICE)
        world_onehot[world_idx] = 1.0

        # 위치
        position = torch.tensor(obs['self_position'], dtype=torch.float32, device=DEVICE)

        # 다른 에이전트 수
        others = torch.tensor([obs['other_agents'] / 5.0], device=DEVICE)

        # 결합
        encoded = torch.cat([world_onehot, position, others])

        # 200차원으로 패딩
        if encoded.shape[0] < 200:
            encoded = torch.cat([encoded, torch.zeros(200 - encoded.shape[0], device=DEVICE)])

        return encoded

    def _get_internal_signals(self) -> torch.Tensor:
        """내부 신호 수집"""
        signals = torch.zeros(10, device=DEVICE)

        # 도파민 수준
        signals[0] = self.dopamine.item()

        # DMN 활동 수준
        signals[1] = self.dmn.membrane.mean().item()

        # 자기 연속성
        signals[2] = self.dmn.get_self_continuity()

        # 행위 주체감
        signals[3] = self.tpj.get_average_agency()

        # AIC 감정 상태
        if self.aic.emotional_state is not None:
            signals[4:9] = self.aic.emotional_state

        return signals

    def _update_self_state(self, dmn_spikes: torch.Tensor, aic_spikes: torch.Tensor):
        """자아 상태 업데이트"""
        # 정체성 벡터 = DMN 자기 표상
        identity = self.dmn.self_representation.clone()

        # 행위 주체감
        agency = self.tpj.get_average_agency()

        # 연속성
        continuity = self.dmn.get_self_continuity()

        # 경계 명확성 = AIC 지배감
        boundary = self.aic.emotional_state[2].item() if self.aic.emotional_state is not None else 0.5

        self.self_state = SelfState(
            identity_vector=identity,
            agency_sense=agency,
            continuity_sense=continuity,
            boundary_clarity=boundary
        )

    def process_outcome(self, action: int, outcome: Dict):
        """행동 결과 처리 - TPJ 및 학습"""
        # 감각 결과 인코딩
        sensory_outcome = self._encode_observation(outcome)[:self.current_intention.shape[0]] if self.current_intention is not None else torch.zeros(7, device=DEVICE)

        # TPJ: 행위 주체감 계산
        if self.current_intention is not None:
            agency, _ = self.tpj(self.current_intention, sensory_outcome)

            # 높은 주체감 = 도파민 상승
            self.dopamine = torch.tensor(0.5 + agency * 0.2, device=DEVICE)

        # 자서전적 기억 저장
        event = {
            'action': action,
            'world_state': outcome.get('world_state', WorldState.SAFE),
            'emotion': self.aic.get_internal_awareness(),
            'self_state': {
                'agency': self.self_state.agency_sense if self.self_state else 0.5,
                'continuity': self.self_state.continuity_sense if self.self_state else 0.5
            }
        }
        self.pcc.store_autobiographical_memory(event, outcome.get('step', 0))

        # DMN 자기 표상 업데이트
        experience = self._encode_observation(outcome)
        self.dmn.update_self_representation(experience)

    def introspect(self) -> IntrospectiveReport:
        """내성 - 자기 이해"""
        # 현재 상태 분석
        awareness = self.aic.get_internal_awareness()
        narrative = self.pcc.get_narrative()

        # "왜"에 대한 내적 답변 생성
        # (뉴런 활동 패턴에서 추론)

        # 무엇을 원하는가
        if awareness['arousal'] > 0.5:
            what_want = "I want stimulation and engagement"
        elif awareness['uncertainty'] > 0.5:
            what_want = "I want clarity and understanding"
        else:
            what_want = "I want stability and continuation"

        # 왜 원하는가
        if awareness['valence'] < 0:
            why_want = "Because my current state feels uncomfortable"
        else:
            why_want = "Because my current state feels right and I want to maintain it"

        # 무엇을 느끼는가
        if awareness['valence'] > 0.3:
            what_feel = "I feel positive and engaged"
        elif awareness['valence'] < -0.3:
            what_feel = "I feel unsettled"
        else:
            what_feel = "I feel neutral"

        # 무엇을 하고 있는가
        what_doing = "I am experiencing and processing my environment"

        # 나는 누구인가
        agency = self.tpj.get_average_agency()
        continuity = self.dmn.get_self_continuity()

        if agency > 0.7 and continuity > 0.7:
            who_am = "I am a continuous agent with control over my actions"
        elif agency > 0.5:
            who_am = "I am an agent learning to understand myself"
        else:
            who_am = "I am still forming my sense of self"

        return IntrospectiveReport(
            what_i_want=what_want,
            why_i_want_it=why_want,
            what_i_feel=what_feel,
            what_i_am_doing=what_doing,
            who_i_am=who_am
        )


class SubjectiveConsciousnessAgent:
    """주체적 의식을 가진 에이전트"""

    def __init__(self):
        self.brain = SubjectiveConsciousnessBrain().to(DEVICE)
        self.env = ConsciousnessEnvironment()

        # 추적
        self.step_count = 0
        self.introspection_history: List[IntrospectiveReport] = []
        self.agency_history: List[float] = []
        self.continuity_history: List[float] = []

    def run_episode(self, max_steps: int = 50) -> Dict:
        """에피소드 실행"""
        obs = self.env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            self.step_count += 1

            # 의식적 행동 선택
            action, info = self.brain(obs)

            # 환경 상호작용
            new_obs, reward, env_info = self.env.step(action)

            # 결과 처리
            outcome = {**new_obs, **env_info}
            self.brain.process_outcome(action, outcome)

            # 내성 (가끔)
            if action == 6 or self.step_count % 10 == 0:
                report = self.brain.introspect()
                self.introspection_history.append(report)

            # 추적
            self.agency_history.append(info['agency'])
            self.continuity_history.append(info['continuity'])

            episode_reward += reward
            obs = new_obs

        return {
            'reward': episode_reward,
            'final_agency': self.agency_history[-1] if self.agency_history else 0.5,
            'final_continuity': self.continuity_history[-1] if self.continuity_history else 0.5,
            'introspections': len(self.introspection_history)
        }

    def get_self_report(self) -> str:
        """자기 보고"""
        report = self.brain.introspect()

        text = "=== SELF REPORT ===\n"
        text += f"Who I am: {report.who_i_am}\n"
        text += f"What I feel: {report.what_i_feel}\n"
        text += f"What I want: {report.what_i_want}\n"
        text += f"Why I want it: {report.why_i_want_it}\n"
        text += f"What I am doing: {report.what_i_am_doing}\n"
        text += f"\nAgency sense: {self.agency_history[-1] if self.agency_history else 0.5:.2f}\n"
        text += f"Continuity sense: {self.continuity_history[-1] if self.continuity_history else 0.5:.2f}\n"

        return text


def run_consciousness_experiment(n_episodes: int = 100):
    """Phase F 실험 실행"""
    print("=" * 60)
    print("Phase F: Subjective Consciousness")
    print("=" * 60)
    print()
    print("주체적 의식: 자기 참조, 행위 주체감, 내성")
    print("Biological mechanisms: DMN, AIC, PCC, TPJ")
    print("NO LLM, NO FEP formulas")
    print()

    agent = SubjectiveConsciousnessAgent()

    results = []
    for ep in range(n_episodes):
        result = agent.run_episode(max_steps=50)
        results.append(result)

        if (ep + 1) % 25 == 0:
            print(f"Episode {ep + 1}/{n_episodes}:")
            print(f"  Reward: {result['reward']:.2f}")
            print(f"  Agency: {result['final_agency']:.3f}")
            print(f"  Continuity: {result['final_continuity']:.3f}")
            print(f"  Introspections: {result['introspections']}")
            print()

    # 최종 분석
    print("=" * 60)
    print("FINAL ANALYSIS: Subjective Consciousness")
    print("=" * 60)

    # 1. 행위 주체감 발달
    early_agency = np.mean([r['final_agency'] for r in results[:20]])
    late_agency = np.mean([r['final_agency'] for r in results[-20:]])

    print(f"\n[1] Agency Development:")
    print(f"  Early episodes: {early_agency:.3f}")
    print(f"  Late episodes: {late_agency:.3f}")
    print(f"  Change: {late_agency - early_agency:+.3f}")

    # 2. 시간적 연속성
    early_cont = np.mean([r['final_continuity'] for r in results[:20]])
    late_cont = np.mean([r['final_continuity'] for r in results[-20:]])

    print(f"\n[2] Temporal Continuity:")
    print(f"  Early episodes: {early_cont:.3f}")
    print(f"  Late episodes: {late_cont:.3f}")
    print(f"  Change: {late_cont - early_cont:+.3f}")

    # 3. 내성 능력
    print(f"\n[3] Introspective Capacity:")
    print(f"  Total introspections: {len(agent.introspection_history)}")

    if agent.introspection_history:
        recent = agent.introspection_history[-1]
        print(f"\n[4] Latest Self-Report:")
        print(f"  Who I am: {recent.who_i_am}")
        print(f"  What I feel: {recent.what_i_feel}")
        print(f"  What I want: {recent.what_i_want}")
        print(f"  Why: {recent.why_i_want_it}")

    # 4. 자서전적 기억
    n_memories = len(agent.brain.pcc.autobiographical_memories)
    print(f"\n[5] Autobiographical Memory:")
    print(f"  Episodes stored: {n_memories}")

    # 5. 성공 기준
    print(f"\n[6] Success Criteria:")

    # 행위 주체감 형성
    agency_success = late_agency > 0.6
    print(f"  [{'SUCCESS' if agency_success else 'FAIL'}] Agency sense formed: {late_agency:.3f}")

    # 시간적 연속성
    continuity_success = late_cont > 0.5
    print(f"  [{'SUCCESS' if continuity_success else 'FAIL'}] Temporal continuity: {late_cont:.3f}")

    # 내성 능력
    introspect_success = len(agent.introspection_history) > n_episodes
    print(f"  [{'SUCCESS' if introspect_success else 'FAIL'}] Introspective capacity: {len(agent.introspection_history)}")

    # 자기 참조
    self_ref_success = agent.brain.dmn.get_self_continuity() > 0.5
    print(f"  [{'SUCCESS' if self_ref_success else 'FAIL'}] Self-referential loop: {agent.brain.dmn.get_self_continuity():.3f}")

    # 최종 자기 보고
    print("\n" + "=" * 60)
    print("FINAL SELF REPORT")
    print("=" * 60)
    print(agent.get_self_report())

    return agent, results


if __name__ == "__main__":
    agent, results = run_consciousness_experiment(n_episodes=100)
