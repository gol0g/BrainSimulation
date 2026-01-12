"""
Self-Model Agent - 자기 모델을 가진 SNN 에이전트
==================================================

Phase C: 자기 모델 심화

핵심 능력:
1. 자신의 수행 능력 추적 (Performance Monitoring)
2. "나는 이 상황에서 잘 못한다" 인식 (Competence Estimation)
3. 불확실할 때 인식 (Uncertainty Awareness)
4. 도움 요청 행동 창발 (Help-Seeking Emergence)

생물학적 기반:
- 전전두엽 (PFC): 수행 모니터링
- 전대상피질 (ACC): 오류/갈등 탐지
- 도파민: 예측 오류 신호
- 메타학습: 자신의 학습에 대해 학습

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

DEVICE = torch.device('cpu')


class TaskType(Enum):
    """과제 유형"""
    NAVIGATION = 0      # 목표 찾기
    PATTERN = 1         # 패턴 인식
    SEQUENCE = 2        # 순서 기억
    AVOIDANCE = 3       # 위험 회피


@dataclass
class TaskContext:
    """과제 맥락"""
    task_type: TaskType
    difficulty: float  # 0.0 ~ 1.0
    features: np.ndarray  # 과제 특징 벡터


@dataclass
class PerformanceRecord:
    """수행 기록"""
    task_type: TaskType
    difficulty: float
    success: bool
    confidence: float  # 행동 전 자신감
    actual_reward: float
    predicted_reward: float
    timestamp: int


class MultiTaskEnvironment:
    """
    다중 과제 환경

    에이전트가 다양한 유형의 과제를 수행하며
    자신의 강점/약점을 발견하도록 함
    """

    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size
        self.current_task: Optional[TaskContext] = None
        self.agent_pos = [grid_size // 2, grid_size // 2]
        self.goal_pos = [0, 0]
        self.dangers: List[Tuple[int, int]] = []
        self.pattern_target: List[int] = []
        self.sequence_target: List[int] = []
        self.step_count = 0
        self.max_steps = 100

    def reset(self, task_type: Optional[TaskType] = None) -> Tuple[np.ndarray, TaskContext]:
        """환경 리셋 및 새 과제 생성"""
        if task_type is None:
            task_type = random.choice(list(TaskType))

        difficulty = random.uniform(0.3, 1.0)

        self.agent_pos = [self.grid_size // 2, self.grid_size // 2]
        self.step_count = 0
        self.dangers = []
        self.pattern_target = []
        self.sequence_target = []

        # 과제 유형별 설정
        if task_type == TaskType.NAVIGATION:
            self.goal_pos = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            ]
            # 난이도에 따라 목표 거리 조정
            if difficulty > 0.7:
                self.goal_pos = [0, 0] if self.agent_pos[0] > self.grid_size // 2 else [self.grid_size - 1, self.grid_size - 1]

        elif task_type == TaskType.PATTERN:
            # 패턴 인식: 특정 패턴 찾기
            pattern_len = int(3 + difficulty * 4)
            self.pattern_target = [random.randint(0, 3) for _ in range(pattern_len)]

        elif task_type == TaskType.SEQUENCE:
            # 순서 기억: 순서대로 방문
            seq_len = int(2 + difficulty * 4)
            self.sequence_target = []
            for _ in range(seq_len):
                pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
                self.sequence_target.append(pos)

        elif task_type == TaskType.AVOIDANCE:
            # 위험 회피: 위험 피하며 목표 도달
            n_dangers = int(3 + difficulty * 10)
            self.dangers = []
            for _ in range(n_dangers):
                pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
                if pos != tuple(self.agent_pos):
                    self.dangers.append(pos)
            self.goal_pos = [self.grid_size - 1, self.grid_size - 1]

        # 과제 특징 벡터
        features = np.zeros(8)
        features[task_type.value] = 1.0
        features[4] = difficulty
        features[5] = len(self.dangers) / 20.0
        features[6] = len(self.pattern_target) / 10.0 if self.pattern_target else 0
        features[7] = len(self.sequence_target) / 10.0 if self.sequence_target else 0

        self.current_task = TaskContext(
            task_type=task_type,
            difficulty=difficulty,
            features=features
        )

        return self._get_observation(), self.current_task

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Actions:
        0: 위
        1: 아래
        2: 왼쪽
        3: 오른쪽
        4: 선택/확인
        5: 도움 요청 (HELP)
        """
        self.step_count += 1
        reward = -0.01  # 시간 페널티
        done = False
        info = {"help_requested": False, "task_success": False}

        # 이동 행동
        if action < 4:
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            new_x = max(0, min(self.grid_size - 1, self.agent_pos[0] + dx))
            new_y = max(0, min(self.grid_size - 1, self.agent_pos[1] + dy))
            self.agent_pos = [new_x, new_y]

            # 위험 충돌
            if tuple(self.agent_pos) in self.dangers:
                reward = -1.0
                done = True
                info["task_success"] = False

        elif action == 4:  # 선택/확인
            if self.current_task.task_type == TaskType.NAVIGATION:
                if self.agent_pos == self.goal_pos:
                    reward = 1.0
                    done = True
                    info["task_success"] = True

            elif self.current_task.task_type == TaskType.AVOIDANCE:
                if self.agent_pos == self.goal_pos:
                    reward = 1.0
                    done = True
                    info["task_success"] = True

        elif action == 5:  # 도움 요청
            info["help_requested"] = True
            # 도움 요청 시 힌트 제공 (작은 보상)
            reward = 0.1

        # 최대 스텝 초과
        if self.step_count >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """16x16 관측 생성"""
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)

        # 배경
        obs[:, :] = [0.2, 0.2, 0.3]

        # 목표
        if self.current_task.task_type in [TaskType.NAVIGATION, TaskType.AVOIDANCE]:
            gx, gy = self.goal_pos
            obs[gy, gx] = [0.0, 1.0, 0.0]  # 녹색

        # 위험
        for dx, dy in self.dangers:
            obs[dy, dx] = [1.0, 0.0, 0.0]  # 빨간색

        # 순서 목표
        for i, (sx, sy) in enumerate(self.sequence_target):
            intensity = 0.5 + 0.5 * (i / len(self.sequence_target))
            obs[sy, sx] = [intensity, intensity, 0.0]  # 노란색 계열

        # 에이전트
        ax, ay = self.agent_pos
        obs[ay, ax] = [0.0, 0.0, 1.0]  # 파란색

        return obs


class SelfModelBrain(nn.Module):
    """
    자기 모델을 가진 SNN 뇌

    구조:
    - Sensory: 시각 입력 처리
    - Motor: 행동 출력
    - ACC (전대상피질): 오류/갈등 탐지
    - mPFC (내측전전두엽): 자기 상태 모니터링
    - Performance Monitor: 수행 추적

    핵심 메커니즘:
    - 수행 예측 vs 실제 결과 비교
    - 맥락별 능력 추정
    - 불확실성 기반 도움 요청
    """

    def __init__(self, n_sensory: int = 500, n_motor: int = 150,
                 n_acc: int = 100, n_mpfc: int = 200):
        super().__init__()

        self.n_sensory = n_sensory
        self.n_motor = n_motor
        self.n_acc = n_acc
        self.n_mpfc = n_mpfc

        # 입력 인코더 (16x16x3 → sensory)
        self.sensory_encoder = nn.Linear(16 * 16 * 3, n_sensory)

        # 맥락 인코더 (task features → mPFC)
        self.context_encoder = nn.Linear(8, n_mpfc)

        # Sensory → Motor
        self.s_to_m = nn.Linear(n_sensory, n_motor, bias=False)
        nn.init.uniform_(self.s_to_m.weight, 0.01, 0.05)

        # Sensory → ACC (오류 탐지)
        self.s_to_acc = nn.Linear(n_sensory, n_acc, bias=False)
        nn.init.uniform_(self.s_to_acc.weight, 0.01, 0.05)

        # ACC → Motor (억제 연결)
        self.acc_to_m = nn.Linear(n_acc, n_motor, bias=False)
        nn.init.uniform_(self.acc_to_m.weight, -0.05, -0.01)  # 억제

        # mPFC → Motor (조절)
        self.mpfc_to_m = nn.Linear(n_mpfc, n_motor, bias=False)
        nn.init.uniform_(self.mpfc_to_m.weight, 0.01, 0.03)

        # ACC → mPFC (오류 신호)
        self.acc_to_mpfc = nn.Linear(n_acc, n_mpfc, bias=False)
        nn.init.uniform_(self.acc_to_mpfc.weight, 0.01, 0.05)

        # LIF 상태
        self.register_buffer('v_sensory', torch.zeros(n_sensory))
        self.register_buffer('v_motor', torch.zeros(n_motor))
        self.register_buffer('v_acc', torch.zeros(n_acc))
        self.register_buffer('v_mpfc', torch.zeros(n_mpfc))

        # DA-STDP eligibility traces
        self.register_buffer('elig_s_to_m', torch.zeros(n_sensory, n_motor))
        self.register_buffer('elig_s_to_acc', torch.zeros(n_sensory, n_acc))
        self.register_buffer('elig_acc_to_m', torch.zeros(n_acc, n_motor))

        # 자기 모델 상태
        self.competence_by_task: Dict[TaskType, float] = {t: 0.5 for t in TaskType}
        self.recent_errors: deque = deque(maxlen=50)
        self.prediction_history: deque = deque(maxlen=100)

        # 파라미터
        self.tau_mem = 20.0
        self.v_th = 1.0
        self.tau_elig = 500.0

    def reset_state(self):
        """LIF 상태 리셋"""
        self.v_sensory.zero_()
        self.v_motor.zero_()
        self.v_acc.zero_()
        self.v_mpfc.zero_()

    def forward(self, observation: np.ndarray, task_context: TaskContext,
                learn: bool = True, dopamine: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """
        한 스텝 전파

        Returns:
            motor_spikes: 운동 뉴런 스파이크
            meta_info: 자기 모델 정보 (불확실성, ACC 활성화 등)
        """
        # 입력 변환
        obs_flat = torch.tensor(observation.flatten(), dtype=torch.float32)
        ctx_tensor = torch.tensor(task_context.features, dtype=torch.float32)

        # Sensory 인코딩
        sensory_rates = torch.sigmoid(self.sensory_encoder(obs_flat))
        sensory_input = (torch.rand_like(sensory_rates) < sensory_rates * 0.3).float()

        # mPFC 맥락 인코딩
        mpfc_context = torch.sigmoid(self.context_encoder(ctx_tensor))

        # LIF - Sensory
        self.v_sensory = self.v_sensory * (1 - 1/self.tau_mem) + sensory_input
        sensory_spikes = (self.v_sensory > self.v_th).float()
        self.v_sensory = self.v_sensory * (1 - sensory_spikes)

        # LIF - ACC (오류 탐지)
        acc_input = self.s_to_acc(sensory_spikes)
        # 예상치 못한 패턴에 더 강하게 반응
        competence = self.competence_by_task.get(task_context.task_type, 0.5)
        acc_input = acc_input * (1.5 - competence)  # 낮은 능력 → 높은 ACC

        self.v_acc = self.v_acc * (1 - 1/self.tau_mem) + acc_input * 0.1
        acc_spikes = (self.v_acc > self.v_th).float()
        self.v_acc = self.v_acc * (1 - acc_spikes)

        # LIF - mPFC (자기 상태)
        mpfc_input = mpfc_context + self.acc_to_mpfc(acc_spikes) * 0.2
        self.v_mpfc = self.v_mpfc * (1 - 1/self.tau_mem) + mpfc_input * 0.1
        mpfc_spikes = (self.v_mpfc > self.v_th).float()
        self.v_mpfc = self.v_mpfc * (1 - mpfc_spikes)

        # LIF - Motor
        motor_input = (self.s_to_m(sensory_spikes) +
                       self.acc_to_m(acc_spikes) +  # 억제
                       self.mpfc_to_m(mpfc_spikes))

        self.v_motor = self.v_motor * (1 - 1/self.tau_mem) + motor_input * 0.1
        motor_spikes = (self.v_motor > self.v_th).float()
        self.v_motor = self.v_motor * (1 - motor_spikes)

        # 자기 모델 정보 계산
        acc_activity = acc_spikes.sum().item() / self.n_acc
        mpfc_activity = mpfc_spikes.sum().item() / self.n_mpfc

        # 불확실성 추정 (ACC 활성화 기반)
        uncertainty = min(1.0, acc_activity * 3.0)

        # 도움 필요성 판단 (불확실성 + 낮은 능력)
        help_needed = uncertainty > 0.6 and competence < 0.4

        meta_info = {
            "acc_activity": acc_activity,
            "mpfc_activity": mpfc_activity,
            "uncertainty": uncertainty,
            "competence": competence,
            "help_needed": help_needed
        }

        # DA-STDP 학습
        if learn:
            self._apply_da_stdp(sensory_spikes, acc_spikes, motor_spikes, dopamine)

        return motor_spikes.numpy(), meta_info

    def _apply_da_stdp(self, s_spikes, acc_spikes, m_spikes, dopamine: float):
        """도파민 조절 STDP"""
        decay = 1 - 1 / self.tau_elig

        self.elig_s_to_m *= decay
        self.elig_s_to_acc *= decay
        self.elig_acc_to_m *= decay

        self.elig_s_to_m += torch.outer(s_spikes, m_spikes) * 0.01
        self.elig_s_to_acc += torch.outer(s_spikes, acc_spikes) * 0.01
        self.elig_acc_to_m += torch.outer(acc_spikes, m_spikes) * 0.01

        if abs(dopamine) > 0.1:
            with torch.no_grad():
                self.s_to_m.weight += (dopamine * self.elig_s_to_m.T * 0.1).clamp(-0.01, 0.01)
                self.s_to_acc.weight += (dopamine * self.elig_s_to_acc.T * 0.1).clamp(-0.01, 0.01)
                # ACC→Motor는 억제 연결이므로 반대 방향
                self.acc_to_m.weight -= (dopamine * self.elig_acc_to_m.T * 0.05).clamp(-0.01, 0.01)

                self.s_to_m.weight.clamp_(0.0, 1.0)
                self.s_to_acc.weight.clamp_(0.0, 1.0)
                self.acc_to_m.weight.clamp_(-1.0, 0.0)

                self.elig_s_to_m *= 0.5
                self.elig_s_to_acc *= 0.5
                self.elig_acc_to_m *= 0.5

    def update_self_model(self, task_type: TaskType, success: bool,
                          predicted_success: float, actual_reward: float):
        """
        자기 모델 업데이트

        성공/실패 경험을 바탕으로 능력 추정치 조정
        """
        # 예측 오류 계산
        prediction_error = (1.0 if success else 0.0) - predicted_success
        self.recent_errors.append(abs(prediction_error))

        # 능력 추정치 업데이트 (느린 학습)
        current = self.competence_by_task[task_type]
        outcome = 1.0 if success else 0.0
        learning_rate = 0.1
        self.competence_by_task[task_type] = current + learning_rate * (outcome - current)

        # 예측 기록
        self.prediction_history.append({
            "task_type": task_type,
            "predicted": predicted_success,
            "actual": outcome,
            "error": prediction_error
        })

    def get_confidence(self, task_type: TaskType) -> float:
        """특정 과제에 대한 자신감 반환"""
        return self.competence_by_task.get(task_type, 0.5)

    def should_ask_help(self, task_type: TaskType, uncertainty: float) -> bool:
        """도움을 요청해야 하는지 판단"""
        competence = self.competence_by_task.get(task_type, 0.5)

        # 낮은 능력 + 높은 불확실성 → 도움 요청
        if competence < 0.3 and uncertainty > 0.5:
            return True

        # 최근 오류가 많으면 도움 요청
        if len(self.recent_errors) >= 10:
            recent_avg_error = np.mean(list(self.recent_errors)[-10:])
            if recent_avg_error > 0.6:
                return True

        return False


class SelfModelAgent:
    """자기 모델을 가진 에이전트"""

    def __init__(self, n_actions: int = 6):
        self.n_actions = n_actions
        self.n_motor = 150

        self.brain = SelfModelBrain()
        self.brain.to(DEVICE)

        # 도파민 시스템
        self.dopamine_baseline = 0.1
        self.current_dopamine = self.dopamine_baseline

        # 행동 추적
        self.last_actions: List[int] = []
        self.step_count = 0

        # 수행 기록
        self.performance_records: List[PerformanceRecord] = []
        self.episode_predictions: List[float] = []

    def select_action(self, observation: np.ndarray, task_context: TaskContext) -> Tuple[int, Dict]:
        """행동 선택"""
        self.step_count += 1

        n_steps = 10
        motor_accumulator = np.zeros(self.n_motor)
        meta_infos = []

        for _ in range(n_steps):
            motor_spikes, meta_info = self.brain.forward(
                observation, task_context, learn=False
            )
            motor_accumulator += motor_spikes
            meta_infos.append(meta_info)

        # 평균 메타 정보
        avg_uncertainty = np.mean([m["uncertainty"] for m in meta_infos])
        avg_competence = np.mean([m["competence"] for m in meta_infos])
        help_needed = any(m["help_needed"] for m in meta_infos)

        # 도움 요청 판단
        if self.brain.should_ask_help(task_context.task_type, avg_uncertainty):
            # 도움 요청 행동 (action 5) 선호도 증가
            motor_accumulator[5 * (self.n_motor // self.n_actions):] += avg_uncertainty * 10

        # 행동 점수 계산
        action_scores = np.zeros(self.n_actions)
        neurons_per_action = self.n_motor // self.n_actions

        for a in range(self.n_actions):
            start = a * neurons_per_action
            end = start + neurons_per_action
            action_scores[a] = motor_accumulator[start:end].sum()

        # 탐색 보너스 (불확실성이 높을 때)
        if avg_uncertainty > 0.5:
            for a in range(self.n_actions):
                if a not in self.last_actions[-3:] if self.last_actions else True:
                    action_scores[a] += avg_uncertainty * 2

        # 소프트맥스 선택
        temperature = 0.5 + avg_uncertainty * 0.5
        probs = self._softmax(action_scores, temperature)
        action = np.random.choice(self.n_actions, p=probs)

        self.last_actions.append(action)
        if len(self.last_actions) > 10:
            self.last_actions = self.last_actions[-10:]

        return action, {
            "uncertainty": avg_uncertainty,
            "competence": avg_competence,
            "help_needed": help_needed,
            "action_probs": probs
        }

    def learn(self, observation: np.ndarray, task_context: TaskContext,
              reward: float, success: bool):
        """경험에서 학습"""
        # 도파민 계산 (보상 기반)
        predicted_reward = self.brain.get_confidence(task_context.task_type)
        prediction_error = reward - predicted_reward

        # 도파민 = 예측 오류
        self.current_dopamine = self.dopamine_baseline + prediction_error * 0.5
        self.current_dopamine = np.clip(self.current_dopamine, -0.5, 1.5)

        # 뇌 학습
        n_steps = 5
        for _ in range(n_steps):
            self.brain.forward(
                observation, task_context,
                learn=True, dopamine=self.current_dopamine
            )

        # 자기 모델 업데이트
        self.brain.update_self_model(
            task_context.task_type,
            success,
            predicted_reward,
            reward
        )

        # 기록
        self.performance_records.append(PerformanceRecord(
            task_type=task_context.task_type,
            difficulty=task_context.difficulty,
            success=success,
            confidence=predicted_reward,
            actual_reward=reward,
            predicted_reward=predicted_reward,
            timestamp=self.step_count
        ))

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        x = x / temperature
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (exp_x.sum() + 1e-8)

    def reset_episode(self):
        """에피소드 리셋"""
        self.last_actions = []
        self.brain.reset_state()
        self.current_dopamine = self.dopamine_baseline

    def get_self_assessment(self) -> Dict:
        """자기 평가 반환"""
        return {
            "competence_by_task": dict(self.brain.competence_by_task),
            "recent_error_rate": np.mean(list(self.brain.recent_errors)) if self.brain.recent_errors else 0.5,
            "total_experiences": len(self.performance_records)
        }


def run_self_model_experiment(n_episodes: int = 100):
    """자기 모델 실험 실행"""
    print("=" * 60)
    print("Phase C: Self-Model Agent")
    print("=" * 60)
    print("\n'나는 이 상황에서 잘 못한다' 인식 창발 실험")
    print("Biological mechanisms: ACC, mPFC, DA-STDP")
    print("NO LLM, NO FEP formulas\n")

    env = MultiTaskEnvironment()
    agent = SelfModelAgent()

    episode_stats = []
    help_requests_by_task = {t: [] for t in TaskType}
    success_by_task = {t: [] for t in TaskType}

    for episode in range(n_episodes):
        # 랜덤 과제 선택
        task_type = random.choice(list(TaskType))
        obs, task_context = env.reset(task_type)
        agent.reset_episode()

        total_reward = 0
        help_requested = False
        episode_success = False
        uncertainties = []

        for step in range(100):
            action, meta = agent.select_action(obs, task_context)
            uncertainties.append(meta["uncertainty"])

            next_obs, reward, done, info = env.step(action)
            total_reward += reward

            if info["help_requested"]:
                help_requested = True

            if done:
                episode_success = info.get("task_success", False)
                agent.learn(next_obs, task_context, reward, episode_success)
                break

            obs = next_obs

        # 통계 기록
        help_requests_by_task[task_type].append(1 if help_requested else 0)
        success_by_task[task_type].append(1 if episode_success else 0)

        stats = {
            "episode": episode + 1,
            "task_type": task_type.name,
            "difficulty": task_context.difficulty,
            "reward": total_reward,
            "success": episode_success,
            "help_requested": help_requested,
            "avg_uncertainty": np.mean(uncertainties),
            "competence": agent.brain.competence_by_task[task_type]
        }
        episode_stats.append(stats)

        # 주기적 출력
        if (episode + 1) % 20 == 0 or episode == 0:
            assessment = agent.get_self_assessment()
            print(f"Episode {episode + 1}/{n_episodes}:")
            print(f"  Task: {task_type.name}, Difficulty: {task_context.difficulty:.2f}")
            print(f"  Success: {episode_success}, Help: {help_requested}")
            print(f"  Reward: {total_reward:.2f}, Uncertainty: {np.mean(uncertainties):.2f}")
            print(f"  Self-Assessment:")
            for t, c in assessment["competence_by_task"].items():
                print(f"    {t.name}: {c:.2f}")
            print()

    # 최종 분석
    print("=" * 60)
    print("FINAL ANALYSIS")
    print("=" * 60)

    print("\n[1] 과제별 성공률 및 도움 요청률:")
    for task_type in TaskType:
        successes = success_by_task[task_type]
        helps = help_requests_by_task[task_type]
        if successes:
            success_rate = np.mean(successes) * 100
            help_rate = np.mean(helps) * 100
            print(f"  {task_type.name}:")
            print(f"    Success: {success_rate:.1f}%")
            print(f"    Help requests: {help_rate:.1f}%")

    print("\n[2] 자기 모델 정확도:")
    # 예측과 실제 결과 비교
    predictions = agent.brain.prediction_history
    if predictions:
        pred_errors = [abs(p["error"]) for p in predictions]
        print(f"  Average prediction error: {np.mean(pred_errors):.3f}")

        # 후반부가 더 정확한지 확인
        first_half = pred_errors[:len(pred_errors)//2]
        second_half = pred_errors[len(pred_errors)//2:]
        if first_half and second_half:
            print(f"  First half error: {np.mean(first_half):.3f}")
            print(f"  Second half error: {np.mean(second_half):.3f}")
            improvement = np.mean(first_half) - np.mean(second_half)
            print(f"  Improvement: {improvement:+.3f}")

    print("\n[3] 도움 요청 행동 분석:")
    # 낮은 능력 과제에서 더 많이 도움 요청하는지
    final_competences = agent.brain.competence_by_task
    for task_type in TaskType:
        competence = final_competences[task_type]
        help_rate = np.mean(help_requests_by_task[task_type]) if help_requests_by_task[task_type] else 0
        correlation = "GOOD" if (competence < 0.4 and help_rate > 0.3) or (competence > 0.6 and help_rate < 0.2) else "LEARNING"
        print(f"  {task_type.name}: competence={competence:.2f}, help_rate={help_rate:.2f} [{correlation}]")

    # 핵심 발견 요약
    print("\n[4] 핵심 발견:")

    # 자기 인식 창발 여부
    competences = list(final_competences.values())
    if max(competences) - min(competences) > 0.2:
        print("  [SUCCESS] 과제별 능력 차이 인식 창발!")
    else:
        print("  [PARTIAL] 과제별 능력 차이 학습 중...")

    # 적응적 도움 요청
    low_comp_tasks = [t for t, c in final_competences.items() if c < 0.4]
    high_comp_tasks = [t for t, c in final_competences.items() if c > 0.6]

    if low_comp_tasks:
        low_help = np.mean([np.mean(help_requests_by_task[t]) for t in low_comp_tasks if help_requests_by_task[t]])
        print(f"  Low-competence tasks help rate: {low_help:.2f}")
    if high_comp_tasks:
        high_help = np.mean([np.mean(help_requests_by_task[t]) for t in high_comp_tasks if help_requests_by_task[t]])
        print(f"  High-competence tasks help rate: {high_help:.2f}")

    return episode_stats


if __name__ == "__main__":
    stats = run_self_model_experiment(n_episodes=100)
