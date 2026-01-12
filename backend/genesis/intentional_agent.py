"""
Phase E: Intentional Agent - 의도와 목표 생성
=============================================

핵심 원리:
- 외부 보상 없이 내재적 목표 생성
- 가치 체계의 자율적 형성
- "나는 X를 탐구하고 싶다" 자발적 결정

생물학적 메커니즘:
- OFC (Orbitofrontal Cortex): 가치 비교
- vmPFC (Ventromedial Prefrontal Cortex): 내부 가치 표상
- ACC (Anterior Cingulate Cortex): 노력/비용 평가, 목표 선택
- Dopamine: 목표 추구 동기화

금지 사항:
- NO LLM, NO FEP formulas (G(a) = ...), NO 심즈식 게이지
- NO 외부에서 목표 설정, NO 휴리스틱 행동 조작
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import random

# Force CPU to avoid CUDA issues
DEVICE = torch.device('cpu')


class Domain(Enum):
    """탐색 가능한 도메인들"""
    PHYSICAL = "physical"      # 물리적 조작
    SOCIAL = "social"          # 사회적 상호작용
    ABSTRACT = "abstract"      # 추상적 패턴
    CREATIVE = "creative"      # 창의적 생성
    KNOWLEDGE = "knowledge"    # 지식 습득


@dataclass
class InternalGoal:
    """내부에서 생성된 목표"""
    domain: Domain
    description: str
    motivation_strength: float  # OFC에서 계산된 가치
    effort_estimate: float      # ACC에서 추정된 노력
    novelty: float              # 새로움 정도
    competence_match: float     # 현재 능력과의 적합도
    created_step: int
    pursuit_count: int = 0
    success_count: int = 0

    @property
    def intrinsic_value(self) -> float:
        """내재적 가치 = 동기 강도 * (새로움 + 능력 적합도) / 노력"""
        if self.effort_estimate < 0.1:
            self.effort_estimate = 0.1
        return self.motivation_strength * (self.novelty + self.competence_match) / self.effort_estimate


@dataclass
class ValueMemory:
    """경험에서 형성된 가치 기억"""
    domain: Domain
    outcome: float           # 결과의 좋고 나쁨
    effort_spent: float      # 투입한 노력
    surprise: float          # 예상과의 차이
    step: int


class OpenWorldEnvironment:
    """열린 세계 환경 - 다양한 도메인에서 자유로운 탐색"""

    def __init__(self):
        self.step_count = 0

        # 도메인별 숨겨진 복잡도 (에이전트는 모름)
        self.domain_complexity = {
            Domain.PHYSICAL: 0.3,    # 쉬움
            Domain.SOCIAL: 0.6,      # 중간
            Domain.ABSTRACT: 0.8,    # 어려움
            Domain.CREATIVE: 0.7,    # 중간-어려움
            Domain.KNOWLEDGE: 0.5    # 중간
        }

        # 도메인별 에이전트의 경험치
        self.domain_experience = {d: 0.0 for d in Domain}

        # 현재 상태
        self.current_domain: Optional[Domain] = None
        self.domain_state: Dict[Domain, float] = {d: 0.0 for d in Domain}

    def reset(self):
        self.step_count = 0
        self.current_domain = None
        self.domain_state = {d: 0.0 for d in Domain}
        return self._get_observation()

    def _get_observation(self) -> torch.Tensor:
        """관측: 각 도메인의 현재 상태 + 새로움 신호"""
        obs = []
        for domain in Domain:
            obs.append(self.domain_state[domain])
            obs.append(self.domain_experience[domain])
            # 새로움 = 경험이 적을수록 높음
            novelty = 1.0 / (1.0 + self.domain_experience[domain])
            obs.append(novelty)
        return torch.tensor(obs, dtype=torch.float32, device=DEVICE)

    def step(self, domain: Domain, effort: float) -> Tuple[torch.Tensor, float, Dict]:
        """특정 도메인에서 노력을 들여 행동"""
        self.step_count += 1
        self.current_domain = domain

        # 복잡도와 경험에 기반한 성공 확률
        complexity = self.domain_complexity[domain]
        experience = self.domain_experience[domain]
        skill = min(1.0, experience * 2)  # 경험이 쌓일수록 스킬 증가

        # 성공 확률 = 스킬 / (복잡도 + 0.5)
        success_prob = skill / (complexity + 0.5)
        success = random.random() < success_prob

        # 결과
        if success:
            progress = effort * (1.0 + skill * 0.5)
            self.domain_state[domain] = min(1.0, self.domain_state[domain] + progress * 0.1)
            self.domain_experience[domain] += effort * 0.2
        else:
            progress = effort * 0.1  # 실패해도 약간의 학습
            self.domain_experience[domain] += effort * 0.05

        # 예상과의 차이 (surprise)
        expected = skill * 0.5
        actual = progress if success else 0.0
        surprise = abs(actual - expected)

        info = {
            'domain': domain,
            'success': success,
            'progress': progress,
            'surprise': surprise,
            'complexity': complexity,
            'skill': skill
        }

        return self._get_observation(), progress, info


class OFC(nn.Module):
    """Orbitofrontal Cortex - 가치 비교 및 계산"""

    def __init__(self, n_neurons: int = 100):
        super().__init__()
        self.n_neurons = n_neurons

        # LIF 뉴런 상태
        self.register_buffer('membrane', torch.zeros(n_neurons))
        self.threshold = 1.0
        self.decay = 0.9

        # 도메인별 가치 표상
        self.domain_values = nn.Parameter(torch.zeros(len(Domain)))

        # 가치 통합 가중치
        self.value_weights = nn.Linear(len(Domain) * 3, n_neurons)
        self.value_output = nn.Linear(n_neurons, len(Domain))

        # Eligibility traces for DA-STDP
        self.register_buffer('elig_traces', torch.zeros(len(Domain) * 3, n_neurons))
        self.tau_elig = 500.0

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """관측에서 각 도메인의 가치 계산"""
        # 뉴런 입력
        x = self.value_weights(obs)

        # LIF dynamics
        self.membrane = self.membrane * self.decay + x
        spikes = (self.membrane > self.threshold).float()
        self.membrane = self.membrane * (1 - spikes)

        # 가치 출력
        values = self.value_output(spikes)

        return values, spikes

    def compute_intrinsic_value(self, domain_idx: int, novelty: float,
                                competence: float, effort: float) -> float:
        """내재적 가치 계산 (뉴런 활동 기반)"""
        base_value = self.domain_values[domain_idx].item()

        # 내재적 동기: 새로움 + 적절한 도전
        intrinsic = novelty * 0.5 + competence * 0.5

        # 노력 대비 가치
        if effort < 0.1:
            effort = 0.1
        value = (base_value + intrinsic) / effort

        return value

    def apply_da_stdp(self, input_spikes: torch.Tensor, dopamine: float):
        """도파민 기반 STDP - 가치 학습"""
        # Eligibility trace 업데이트
        self.elig_traces = self.elig_traces * (1 - 1/self.tau_elig)

        if input_spikes.dim() == 1:
            outer = torch.outer(input_spikes, self.membrane[:input_spikes.shape[0]])
            if outer.shape == self.elig_traces.shape:
                self.elig_traces += outer * 0.1

        # 도파민이 있으면 가중치 업데이트
        if abs(dopamine) > 0.1:
            with torch.no_grad():
                delta = dopamine * self.elig_traces.T * 0.05
                delta = delta.clamp(-0.01, 0.01)
                if delta.shape == self.value_weights.weight.shape:
                    self.value_weights.weight += delta


class vmPFC(nn.Module):
    """Ventromedial Prefrontal Cortex - 내부 가치 표상 및 목표 생성"""

    def __init__(self, n_neurons: int = 150):
        super().__init__()
        self.n_neurons = n_neurons

        # LIF 뉴런
        self.register_buffer('membrane', torch.zeros(n_neurons))
        self.threshold = 1.0
        self.decay = 0.85

        # 가치 기억 저장
        self.value_memories: List[ValueMemory] = []

        # 목표 생성 회로
        self.goal_generator = nn.Linear(len(Domain) * 2, n_neurons)
        self.goal_selector = nn.Linear(n_neurons, len(Domain))

        # 가치 체계 (경험에서 형성)
        self.register_buffer('value_system', torch.zeros(len(Domain)))

        # Eligibility traces
        self.register_buffer('elig_traces', torch.zeros(len(Domain) * 2, n_neurons))
        self.tau_elig = 500.0

    def store_value_memory(self, memory: ValueMemory):
        """가치 기억 저장"""
        self.value_memories.append(memory)

        # 가치 체계 업데이트 (경험에서 자율적으로 형성)
        domain_idx = list(Domain).index(memory.domain)

        # 좋은 결과 + 낮은 노력 + 높은 surprise = 가치 있는 도메인
        value_update = (memory.outcome - memory.effort_spent * 0.5 + memory.surprise * 0.3)

        # 지수 이동 평균으로 가치 체계 업데이트
        alpha = 0.1
        self.value_system[domain_idx] = (
            (1 - alpha) * self.value_system[domain_idx] +
            alpha * value_update
        )

    def generate_goals(self, ofc_values: torch.Tensor,
                       current_obs: torch.Tensor) -> torch.Tensor:
        """자율적 목표 생성"""
        # OFC 가치 + 현재 관측에서 목표 도메인 선택
        # 관측에서 도메인 상태와 새로움 추출
        domain_states = current_obs[::3][:len(Domain)]
        novelty_signals = current_obs[2::3][:len(Domain)]

        # 목표 생성 입력
        goal_input = torch.cat([domain_states, novelty_signals])

        # LIF 뉴런 처리
        x = self.goal_generator(goal_input)
        self.membrane = self.membrane * self.decay + x
        spikes = (self.membrane > self.threshold).float()
        self.membrane = self.membrane * (1 - spikes)

        # 목표 선호도 = 뉴런 출력 + 가치 체계 + OFC 가치
        goal_prefs = self.goal_selector(spikes) + self.value_system + ofc_values

        return goal_prefs

    def select_goal(self, goal_prefs: torch.Tensor,
                    competence: Dict[Domain, float]) -> Tuple[Domain, float]:
        """목표 선택 - 내부 가치와 능력 적합도 기반"""
        # 능력 적합도 가중치 (너무 쉬우면 지루, 너무 어려우면 좌절)
        # 최적: 능력이 0.3~0.7인 도메인
        competence_weights = []
        for d in Domain:
            c = competence.get(d, 0.5)
            # 역U자 곡선: 0.5에서 최대
            weight = 1.0 - abs(c - 0.5) * 2
            competence_weights.append(weight)

        competence_tensor = torch.tensor(competence_weights, device=DEVICE)

        # 최종 선호도 = 목표 선호도 * 능력 적합도
        final_prefs = goal_prefs * competence_tensor

        # 소프트맥스 선택 (약간의 탐색)
        probs = torch.softmax(final_prefs / 0.5, dim=0)
        selected_idx = torch.multinomial(probs, 1).item()

        selected_domain = list(Domain)[selected_idx]
        motivation = final_prefs[selected_idx].item()

        return selected_domain, motivation

    def apply_da_stdp(self, input_pattern: torch.Tensor, dopamine: float):
        """도파민 기반 학습"""
        self.elig_traces = self.elig_traces * (1 - 1/self.tau_elig)

        if input_pattern.dim() == 1 and input_pattern.shape[0] == len(Domain) * 2:
            outer = torch.outer(input_pattern, self.membrane[:input_pattern.shape[0]])
            if outer.shape == self.elig_traces.shape:
                self.elig_traces += outer * 0.1

        if abs(dopamine) > 0.1:
            with torch.no_grad():
                delta = dopamine * self.elig_traces.T * 0.05
                delta = delta.clamp(-0.01, 0.01)
                if delta.shape == self.goal_generator.weight.shape:
                    self.goal_generator.weight += delta


class ACC_GoalSelection(nn.Module):
    """Anterior Cingulate Cortex - 노력/비용 평가, 갈등 해결"""

    def __init__(self, n_neurons: int = 100):
        super().__init__()
        self.n_neurons = n_neurons

        # LIF 뉴런
        self.register_buffer('membrane', torch.zeros(n_neurons))
        self.threshold = 1.0
        self.decay = 0.9

        # 노력 예측
        self.effort_predictor = nn.Linear(len(Domain), n_neurons)
        self.effort_output = nn.Linear(n_neurons, len(Domain))

        # 갈등 탐지
        self.register_buffer('conflict_history', torch.zeros(100))
        self.conflict_idx = 0

    def estimate_effort(self, competence: Dict[Domain, float]) -> torch.Tensor:
        """각 도메인에 대한 노력 추정"""
        # 능력이 낮으면 노력 높음
        comp_tensor = torch.tensor([competence.get(d, 0.5) for d in Domain], device=DEVICE)
        effort_estimate = 1.0 - comp_tensor  # 역상관

        # 뉴런 처리
        x = self.effort_predictor(comp_tensor)
        self.membrane = self.membrane * self.decay + x
        spikes = (self.membrane > self.threshold).float()
        self.membrane = self.membrane * (1 - spikes)

        # 학습된 노력 예측 + 기본 추정
        learned_effort = self.effort_output(spikes)

        return (effort_estimate + learned_effort) / 2

    def detect_conflict(self, goal_prefs: torch.Tensor) -> float:
        """목표 간 갈등 탐지"""
        # 상위 2개 목표의 선호도 차이가 작으면 갈등
        sorted_prefs, _ = torch.sort(goal_prefs, descending=True)
        if len(sorted_prefs) >= 2:
            conflict = 1.0 - (sorted_prefs[0] - sorted_prefs[1]).abs().item()
        else:
            conflict = 0.0

        # 갈등 기록
        self.conflict_history[self.conflict_idx % 100] = conflict
        self.conflict_idx += 1

        return conflict

    def resolve_conflict(self, goal_prefs: torch.Tensor,
                        effort: torch.Tensor) -> torch.Tensor:
        """갈등 해결 - 노력 대비 가치가 높은 목표 선호"""
        # 가치/노력 비율
        effort_safe = effort.clamp(min=0.1)
        value_per_effort = goal_prefs / effort_safe

        return value_per_effort


class IntentionalBrain(nn.Module):
    """의도와 목표를 자율적으로 생성하는 뇌"""

    def __init__(self):
        super().__init__()

        # 핵심 영역
        self.ofc = OFC(n_neurons=100)      # 가치 계산
        self.vmpfc = vmPFC(n_neurons=150)  # 목표 생성
        self.acc = ACC_GoalSelection(n_neurons=100)  # 노력/갈등

        # 도파민 시스템
        self.register_buffer('dopamine_level', torch.tensor(0.5))
        self.dopamine_baseline = 0.5

        # 내부 상태
        self.competence: Dict[Domain, float] = {d: 0.3 for d in Domain}
        self.active_goals: List[InternalGoal] = []
        self.goal_history: List[InternalGoal] = []

        # 자율적 가치 체계 (경험에서 형성)
        self.autonomous_values: Dict[Domain, float] = {d: 0.0 for d in Domain}

    def compute_dopamine(self, surprise: float, success: bool) -> float:
        """도파민 신호 계산"""
        # 예상치 못한 좋은 결과 → 도파민 상승
        # 예상한 결과 → 변화 없음
        # 예상치 못한 나쁜 결과 → 도파민 하락

        if success:
            dopamine = self.dopamine_baseline + surprise * 0.5
        else:
            dopamine = self.dopamine_baseline - surprise * 0.3

        self.dopamine_level = torch.tensor(dopamine, device=DEVICE)
        return dopamine

    def generate_intrinsic_goal(self, obs: torch.Tensor, step: int) -> Optional[InternalGoal]:
        """내재적 목표 자율 생성 - 핵심 기능"""

        # 1. OFC: 각 도메인의 가치 계산
        ofc_values, ofc_spikes = self.ofc(obs)

        # 2. vmPFC: 목표 선호도 생성
        goal_prefs = self.vmpfc.generate_goals(ofc_values, obs)

        # 3. ACC: 노력 추정 및 갈등 탐지
        effort_estimates = self.acc.estimate_effort(self.competence)
        conflict = self.acc.detect_conflict(goal_prefs)

        # 4. 갈등 해결
        if conflict > 0.7:
            goal_prefs = self.acc.resolve_conflict(goal_prefs, effort_estimates)

        # 5. 목표 선택
        selected_domain, motivation = self.vmpfc.select_goal(goal_prefs, self.competence)

        # 6. 새로움 및 능력 적합도 계산
        domain_idx = list(Domain).index(selected_domain)
        novelty = obs[domain_idx * 3 + 2].item()  # 새로움 신호

        competence = self.competence.get(selected_domain, 0.5)
        # 최적 도전: 능력 0.3~0.7
        competence_match = 1.0 - abs(competence - 0.5) * 2

        # 7. 목표 생성
        goal = InternalGoal(
            domain=selected_domain,
            description=f"Explore {selected_domain.value} domain",
            motivation_strength=motivation,
            effort_estimate=effort_estimates[domain_idx].item(),
            novelty=novelty,
            competence_match=competence_match,
            created_step=step
        )

        return goal

    def update_from_experience(self, domain: Domain, outcome: float,
                              effort: float, surprise: float, step: int):
        """경험에서 가치 체계 업데이트"""

        # 가치 기억 저장
        memory = ValueMemory(
            domain=domain,
            outcome=outcome,
            effort_spent=effort,
            surprise=surprise,
            step=step
        )
        self.vmpfc.store_value_memory(memory)

        # 능력 업데이트
        domain_idx = list(Domain).index(domain)
        if outcome > 0:
            self.competence[domain] = min(1.0, self.competence[domain] + 0.05)
        else:
            self.competence[domain] = max(0.0, self.competence[domain] - 0.02)

        # 자율적 가치 체계 업데이트
        value_change = outcome - effort * 0.5 + surprise * 0.3
        alpha = 0.1
        self.autonomous_values[domain] = (
            (1 - alpha) * self.autonomous_values[domain] +
            alpha * value_change
        )

        # 도파민 계산 및 DA-STDP
        dopamine = self.compute_dopamine(surprise, outcome > 0)

        # 각 영역에 DA-STDP 적용
        obs_like = torch.zeros(len(Domain) * 3, device=DEVICE)
        obs_like[domain_idx * 3] = outcome
        obs_like[domain_idx * 3 + 1] = effort
        obs_like[domain_idx * 3 + 2] = surprise

        self.ofc.apply_da_stdp(obs_like, dopamine - self.dopamine_baseline)

        goal_input = torch.zeros(len(Domain) * 2, device=DEVICE)
        goal_input[domain_idx] = outcome
        goal_input[len(Domain) + domain_idx] = surprise
        self.vmpfc.apply_da_stdp(goal_input, dopamine - self.dopamine_baseline)

    def decide_action(self, obs: torch.Tensor, step: int) -> Tuple[Domain, float]:
        """행동 결정 - 목표 추구 또는 새 목표 생성"""

        # 활성 목표가 없거나 오래되었으면 새 목표 생성
        should_generate = (
            len(self.active_goals) == 0 or
            step - self.active_goals[-1].created_step > 50 or
            random.random() < 0.1  # 10% 확률로 새 목표 탐색
        )

        if should_generate:
            new_goal = self.generate_intrinsic_goal(obs, step)
            if new_goal and new_goal.intrinsic_value > 0:
                self.active_goals.append(new_goal)

        if not self.active_goals:
            # 기본: 랜덤 탐색
            domain = random.choice(list(Domain))
            effort = 0.5
        else:
            # 가장 가치 있는 목표 추구
            best_goal = max(self.active_goals, key=lambda g: g.intrinsic_value)
            domain = best_goal.domain
            effort = min(1.0, best_goal.motivation_strength)
            best_goal.pursuit_count += 1

        return domain, effort


class IntentionalAgent:
    """의도를 가진 에이전트"""

    def __init__(self):
        self.brain = IntentionalBrain().to(DEVICE)
        self.env = OpenWorldEnvironment()

        # 추적
        self.step_count = 0
        self.generated_goals: List[InternalGoal] = []
        self.value_evolution: List[Dict[Domain, float]] = []

    def run_episode(self, max_steps: int = 100) -> Dict:
        """에피소드 실행"""
        obs = self.env.reset()
        episode_reward = 0.0
        goals_generated = 0

        for _ in range(max_steps):
            self.step_count += 1

            # 행동 결정
            domain, effort = self.brain.decide_action(obs, self.step_count)

            # 환경 상호작용
            new_obs, progress, info = self.env.step(domain, effort)

            # 경험에서 학습
            self.brain.update_from_experience(
                domain=domain,
                outcome=progress,
                effort=effort,
                surprise=info['surprise'],
                step=self.step_count
            )

            episode_reward += progress
            obs = new_obs

            # 새 목표 생성 확인
            if len(self.brain.active_goals) > len(self.generated_goals):
                new_goals = self.brain.active_goals[len(self.generated_goals):]
                self.generated_goals.extend(new_goals)
                goals_generated += len(new_goals)

        # 가치 체계 기록
        self.value_evolution.append(dict(self.brain.autonomous_values))

        return {
            'reward': episode_reward,
            'goals_generated': goals_generated,
            'competence': dict(self.brain.competence),
            'values': dict(self.brain.autonomous_values),
            'domain_experience': dict(self.env.domain_experience)
        }

    def get_autonomous_intent(self) -> str:
        """현재 의도를 자연어로 표현"""
        if not self.brain.active_goals:
            return "I am exploring without a specific goal."

        best_goal = max(self.brain.active_goals, key=lambda g: g.intrinsic_value)

        # 가치 체계에서 가장 높은 도메인
        top_domain = max(self.brain.autonomous_values.items(), key=lambda x: x[1])

        intent = f"I want to {best_goal.description}. "
        intent += f"I value {top_domain[0].value} most (value={top_domain[1]:.2f}). "
        intent += f"My motivation is {best_goal.motivation_strength:.2f}."

        return intent


def run_intentional_experiment(n_episodes: int = 100):
    """Phase E 실험 실행"""
    print("=" * 60)
    print("Phase E: Intentional Agent")
    print("=" * 60)
    print()
    print("자율적 목표 생성, 가치 체계 형성")
    print("Biological mechanisms: OFC, vmPFC, ACC, DA-STDP")
    print("NO LLM, NO FEP formulas, NO external goal setting")
    print()

    agent = IntentionalAgent()

    results = []
    for ep in range(n_episodes):
        result = agent.run_episode(max_steps=50)
        results.append(result)

        if (ep + 1) % 20 == 0:
            print(f"Episode {ep + 1}/{n_episodes}:")
            print(f"  Reward: {result['reward']:.2f}")
            print(f"  Goals generated: {result['goals_generated']}")
            print(f"  Intent: {agent.get_autonomous_intent()}")
            print(f"  Value system: {result['values']}")
            print()

    # 최종 분석
    print("=" * 60)
    print("FINAL ANALYSIS: Intentional Agent")
    print("=" * 60)

    # 1. 목표 생성 분석
    total_goals = len(agent.generated_goals)
    domain_goals = {}
    for goal in agent.generated_goals:
        domain_goals[goal.domain] = domain_goals.get(goal.domain, 0) + 1

    print(f"\n[1] Goal Generation:")
    print(f"  Total goals generated: {total_goals}")
    print(f"  Goals by domain:")
    for domain, count in sorted(domain_goals.items(), key=lambda x: -x[1]):
        print(f"    {domain.value}: {count} ({count/total_goals*100:.1f}%)")

    # 2. 가치 체계 진화
    print(f"\n[2] Value System Evolution:")
    initial_values = agent.value_evolution[0] if agent.value_evolution else {}
    final_values = agent.value_evolution[-1] if agent.value_evolution else {}

    for domain in Domain:
        init_v = initial_values.get(domain, 0.0)
        final_v = final_values.get(domain, 0.0)
        change = final_v - init_v
        direction = "+" if change >= 0 else ""
        print(f"  {domain.value}: {init_v:.2f} -> {final_v:.2f} ({direction}{change:.2f})")

    # 3. 능력 발달
    print(f"\n[3] Competence Development:")
    for domain in Domain:
        competence = agent.brain.competence.get(domain, 0.0)
        experience = agent.env.domain_experience.get(domain, 0.0)
        print(f"  {domain.value}: competence={competence:.2f}, experience={experience:.2f}")

    # 4. 최종 의도
    print(f"\n[4] Final Autonomous Intent:")
    print(f"  {agent.get_autonomous_intent()}")

    # 5. 핵심 지표
    print(f"\n[5] Key Metrics:")
    avg_reward = np.mean([r['reward'] for r in results[-20:]])
    print(f"  Average reward (last 20): {avg_reward:.2f}")

    # 가치 분화 정도 (높을수록 선호가 명확함)
    final_vals = list(final_values.values())
    value_variance = np.var(final_vals) if final_vals else 0
    print(f"  Value differentiation (variance): {value_variance:.3f}")

    # 성공 검증
    print(f"\n[6] Success Criteria:")

    # 목표가 자율적으로 생성되었는가?
    goal_success = total_goals > n_episodes * 0.5
    print(f"  [{'SUCCESS' if goal_success else 'FAIL'}] Goals autonomously generated: {total_goals}")

    # 가치 체계가 경험에서 형성되었는가?
    value_formed = value_variance > 0.01
    print(f"  [{'SUCCESS' if value_formed else 'FAIL'}] Value system formed: variance={value_variance:.3f}")

    # 특정 도메인에 대한 선호가 발달했는가?
    top_domain = max(final_values.items(), key=lambda x: x[1])
    preference_formed = top_domain[1] > 0.1
    print(f"  [{'SUCCESS' if preference_formed else 'FAIL'}] Domain preference: {top_domain[0].value}={top_domain[1]:.2f}")

    return agent, results


if __name__ == "__main__":
    agent, results = run_intentional_experiment(n_episodes=100)
