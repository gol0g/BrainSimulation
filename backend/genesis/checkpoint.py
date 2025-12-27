"""
v3.6: Checkpoint System - Save/Load Brain State
v3.7: Reproducibility - Seed Management

저장 대상:
1. transition_model (delta_mean, delta_std, count)
2. precision_learner 상태
3. hierarchy_controller 상태 (context beliefs, expectations, transitions)
4. preference_learner 상태
5. 메타 정보 (version, step_count, timestamp, seed)

용도:
- 학습된 뇌 상태 보존
- 재현성 보장
- 대회/챌린지 제출
"""

import json
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class CheckpointMetadata:
    """체크포인트 메타데이터"""
    version: str = "3.6"
    timestamp: str = ""
    step_count: int = 0
    episode_count: int = 0
    total_food: int = 0
    total_deaths: int = 0
    seed: Optional[int] = None
    description: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class NumpyEncoder(json.JSONEncoder):
    """NumPy 배열을 JSON으로 직렬화"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {'__numpy__': True, 'data': obj.tolist(), 'dtype': str(obj.dtype)}
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def numpy_decoder(obj):
    """JSON에서 NumPy 배열 복원"""
    if '__numpy__' in obj:
        return np.array(obj['data'], dtype=obj['dtype'])
    return obj


class BrainCheckpoint:
    """
    Genesis Brain 체크포인트 관리자

    사용법:
        # 저장
        checkpoint = BrainCheckpoint()
        checkpoint.save(action_selector, world, "my_brain.json")

        # 로드
        checkpoint.load(action_selector, world, "my_brain.json")
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self,
             action_selector,
             world,
             filename: str,
             description: str = "") -> str:
        """
        전체 뇌 상태 저장

        Args:
            action_selector: ActionSelector 인스턴스
            world: GenesisWorld 인스턴스
            filename: 저장할 파일명
            description: 체크포인트 설명

        Returns:
            저장된 파일 경로
        """
        checkpoint = {
            'metadata': asdict(CheckpointMetadata(
                step_count=world.step_count,
                total_food=world.total_food,
                total_deaths=world.total_deaths,
                description=description
            )),
            'transition_model': self._save_transition_model(action_selector),
            'precision': self._save_precision(action_selector),
            'hierarchy': self._save_hierarchy(action_selector),
            'preference_learning': self._save_preference_learning(action_selector),
            'world': self._save_world(world)
        }

        filepath = self.checkpoint_dir / filename
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, cls=NumpyEncoder, indent=2)

        return str(filepath)

    def load(self,
             action_selector,
             world,
             filename: str) -> CheckpointMetadata:
        """
        체크포인트에서 뇌 상태 복원

        Args:
            action_selector: ActionSelector 인스턴스
            world: GenesisWorld 인스턴스
            filename: 로드할 파일명

        Returns:
            체크포인트 메타데이터
        """
        filepath = self.checkpoint_dir / filename
        with open(filepath, 'r') as f:
            checkpoint = json.load(f, object_hook=numpy_decoder)

        self._load_transition_model(action_selector, checkpoint['transition_model'])
        self._load_precision(action_selector, checkpoint['precision'])
        self._load_hierarchy(action_selector, checkpoint.get('hierarchy'))
        self._load_preference_learning(action_selector, checkpoint.get('preference_learning'))
        self._load_world(world, checkpoint['world'])

        return CheckpointMetadata(**checkpoint['metadata'])

    def list_checkpoints(self) -> list:
        """사용 가능한 체크포인트 목록"""
        checkpoints = []
        for f in self.checkpoint_dir.glob("*.json"):
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    meta = data.get('metadata', {})
                    checkpoints.append({
                        'filename': f.name,
                        'timestamp': meta.get('timestamp'),
                        'step_count': meta.get('step_count'),
                        'description': meta.get('description', '')
                    })
            except:
                pass
        return sorted(checkpoints, key=lambda x: x.get('timestamp', ''), reverse=True)

    # === Private Methods ===

    def _save_transition_model(self, action_selector) -> Dict:
        """전이 모델 저장"""
        return {
            'delta_mean': action_selector.transition_model['delta_mean'],
            'delta_std': action_selector.transition_model['delta_std'],
            'count': action_selector.transition_model['count']
        }

    def _load_transition_model(self, action_selector, data: Dict):
        """전이 모델 복원"""
        action_selector.transition_model['delta_mean'] = data['delta_mean']
        action_selector.transition_model['delta_std'] = data['delta_std']
        action_selector.transition_model['count'] = data['count']

    def _save_precision(self, action_selector) -> Dict:
        """Precision learner 저장"""
        pl = action_selector.precision_learner
        return {
            'sensory_precision': pl.sensory_precision,
            'transition_precision': pl.transition_precision,
            'goal_precision': pl.goal_precision,
            'ema_error': pl.ema_error,
            'volatility': pl.volatility
        }

    def _load_precision(self, action_selector, data: Dict):
        """Precision learner 복원"""
        pl = action_selector.precision_learner
        pl.sensory_precision = data['sensory_precision']
        pl.transition_precision = data['transition_precision']
        pl.goal_precision = data['goal_precision']
        pl.ema_error = data.get('ema_error', pl.ema_error)
        pl.volatility = data['volatility']

    def _save_hierarchy(self, action_selector) -> Optional[Dict]:
        """Hierarchy controller 저장"""
        hc = action_selector.hierarchy_controller
        if hc is None:
            return None

        state = hc.get_state()
        if state is None:
            return {'enabled': True, 'initialized': False}

        return {
            'enabled': True,
            'initialized': True,
            'K': hc.K,
            'update_interval': hc.update_interval,
            'Q_context': state.Q_context,
            'obs_stats': state.obs_stats,
            'step_count': state.step_count,
            'context_expectations': hc.inference.context_expectations,
            'context_transition_delta': hc.inference.context_transition_delta,
            'context_transition_error': hc.inference.context_transition_error
        }

    def _load_hierarchy(self, action_selector, data: Optional[Dict]):
        """Hierarchy controller 복원"""
        if data is None or not data.get('enabled'):
            action_selector.hierarchy_controller = None
            return

        if not data.get('initialized'):
            action_selector.enable_hierarchy(K=data.get('K', 4))
            return

        # Enable with saved params
        action_selector.enable_hierarchy(
            K=data['K'],
            update_interval=data['update_interval']
        )

        # Restore state
        hc = action_selector.hierarchy_controller
        hc.state.Q_context = data['Q_context']
        hc.state.obs_stats = data['obs_stats']
        hc.state.step_count = data['step_count']
        hc.inference.context_expectations = data['context_expectations']
        hc.inference.context_transition_delta = data['context_transition_delta']
        hc.inference.context_transition_error = data['context_transition_error']

    def _save_preference_learning(self, action_selector) -> Optional[Dict]:
        """Preference learner 저장"""
        pl = action_selector.preference_learner
        if pl is None:
            return None

        return {
            'enabled': action_selector.preference_learning_enabled,
            'energy_mode': pl.energy_pref.mode,
            'energy_concentration': pl.energy_pref.concentration,
            'pain_mode': pl.pain_pref.mode,
            'pain_concentration': pl.pain_pref.concentration,
            'G_baseline': pl._G_baseline,
            'update_count': pl._update_count,
            'mode_lr': pl.mode_lr,
            'concentration_lr': pl.concentration_lr
        }

    def _load_preference_learning(self, action_selector, data: Optional[Dict]):
        """Preference learner 복원"""
        if data is None:
            action_selector.preference_learner = None
            action_selector.preference_learning_enabled = False
            return

        # Enable and restore
        action_selector.enable_preference_learning(
            mode_lr=data.get('mode_lr', 0.02),
            concentration_lr=data.get('concentration_lr', 0.01)
        )

        pl = action_selector.preference_learner
        pl.energy_pref.mode = data['energy_mode']
        pl.energy_pref.concentration = data['energy_concentration']
        pl.pain_pref.mode = data['pain_mode']
        pl.pain_pref.concentration = data['pain_concentration']
        pl._G_baseline = data['G_baseline']
        pl._update_count = data['update_count']

        action_selector.preference_learning_enabled = data['enabled']
        action_selector._apply_learned_preferences()

    def _save_world(self, world) -> Dict:
        """World 상태 저장 (v4.5: drift 포함)"""
        data = {
            'agent_pos': list(world.agent_pos),
            'food_pos': list(world.food_pos),
            'danger_pos': list(world.danger_pos),
            'energy': world.energy,
            'step_count': world.step_count,
            'total_food': world.total_food,
            'total_deaths': world.total_deaths
        }
        # v4.5: Drift 상태 저장
        if hasattr(world, 'drift_enabled'):
            data['drift'] = {
                'enabled': world.drift_enabled,
                'type': world.drift_type,
                'delay_counter': world._drift_delay_counter,
                'delay_threshold': world._drift_delay_threshold,
                'probabilistic_ratio': world._probabilistic_ratio,
                'energy_decay_multiplier': getattr(world, '_energy_decay_multiplier', 1.0)
            }
        return data

    def _load_world(self, world, data: Dict):
        """World 상태 복원 (v4.5: drift 포함)"""
        world.agent_pos = data['agent_pos']
        world.food_pos = data['food_pos']
        world.danger_pos = data['danger_pos']
        world.energy = data['energy']
        world.step_count = data['step_count']
        world.total_food = data['total_food']
        world.total_deaths = data['total_deaths']
        # v4.5: Drift 상태 복원
        if 'drift' in data and hasattr(world, 'drift_enabled'):
            drift = data['drift']
            world.drift_enabled = drift['enabled']
            world.drift_type = drift['type']
            world._drift_delay_counter = drift['delay_counter']
            world._drift_delay_threshold = drift['delay_threshold']
            world._probabilistic_ratio = drift['probabilistic_ratio']
            world._energy_decay_multiplier = drift.get('energy_decay_multiplier', 1.0)


# === Headless Evaluation Runner ===

@dataclass
class EpisodeResult:
    """에피소드 결과"""
    episode: int
    steps: int
    food_eaten: int
    died: bool
    final_energy: float
    avg_G: float
    action_distribution: Dict[int, int]


@dataclass
class EvaluationResult:
    """평가 결과"""
    n_episodes: int
    total_steps: int
    total_food: int
    total_deaths: int
    avg_steps_per_episode: float
    avg_food_per_episode: float
    survival_rate: float
    episodes: list  # List[EpisodeResult]
    seed: Optional[int] = None  # v3.7: 재현성을 위한 시드

    def to_dict(self) -> Dict:
        return {
            'seed': self.seed,  # v3.7
            'n_episodes': self.n_episodes,
            'total_steps': self.total_steps,
            'total_food': self.total_food,
            'total_deaths': self.total_deaths,
            'avg_steps_per_episode': self.avg_steps_per_episode,
            'avg_food_per_episode': self.avg_food_per_episode,
            'survival_rate': self.survival_rate,
            'episodes': [asdict(ep) for ep in self.episodes]
        }


class HeadlessRunner:
    """
    헤드리스 평가 러너

    UI 없이 N 에피소드 자동 평가 + 결과 JSON 출력

    사용법:
        runner = HeadlessRunner(agent, world, action_selector)
        result = runner.run(n_episodes=100, max_steps_per_episode=500)
        runner.save_result(result, "evaluation_result.json")
    """

    def __init__(self, agent, world, action_selector, seed: Optional[int] = None):
        self.agent = agent
        self.world = world
        self.action_selector = action_selector
        self.seed = seed

    def _set_seed(self, seed: int):
        """모든 랜덤 소스의 시드 고정 (재현성)"""
        np.random.seed(seed)
        random.seed(seed)

    def run(self,
            n_episodes: int = 100,
            max_steps_per_episode: int = 500,
            verbose: bool = True,
            seed: Optional[int] = None) -> EvaluationResult:
        """
        N 에피소드 평가 실행

        Args:
            n_episodes: 에피소드 수
            max_steps_per_episode: 에피소드당 최대 스텝
            verbose: 진행 상황 출력
            seed: 재현성을 위한 시드 (None이면 self.seed 사용)

        Returns:
            EvaluationResult
        """
        # 시드 설정 (재현성)
        effective_seed = seed if seed is not None else self.seed
        if effective_seed is not None:
            self._set_seed(effective_seed)

        episodes = []
        total_steps = 0
        total_food = 0
        total_deaths = 0

        for ep in range(n_episodes):
            result = self._run_episode(ep, max_steps_per_episode)
            episodes.append(result)

            total_steps += result.steps
            total_food += result.food_eaten
            if result.died:
                total_deaths += 1

            if verbose and (ep + 1) % 10 == 0:
                print(f"Episode {ep + 1}/{n_episodes}: "
                      f"food={result.food_eaten}, steps={result.steps}, "
                      f"died={result.died}")

        return EvaluationResult(
            n_episodes=n_episodes,
            total_steps=total_steps,
            total_food=total_food,
            total_deaths=total_deaths,
            avg_steps_per_episode=total_steps / n_episodes,
            avg_food_per_episode=total_food / n_episodes,
            survival_rate=(n_episodes - total_deaths) / n_episodes,
            episodes=episodes,
            seed=effective_seed  # v3.7: 재현성
        )

    def _run_episode(self, episode: int, max_steps: int) -> EpisodeResult:
        """단일 에피소드 실행"""
        self.world.reset()
        self.agent.reset()

        action_counts = {}
        food_eaten = 0
        G_values = []
        died = False

        last_action = 0

        for step in range(max_steps):
            obs = self.world.get_observation()

            if step == 0:
                state = self.agent.step(obs)
            else:
                state = self.agent.step_with_action(obs, last_action)

            action = int(state.action)
            action_counts[action] = action_counts.get(action, 0) + 1

            # Record G
            if action in state.G_decomposition:
                G_values.append(state.G_decomposition[action].G)

            # Execute
            outcome = self.world.execute_action(action)

            if outcome.get('ate_food'):
                food_eaten += 1

            if outcome.get('died'):
                died = True
                break

            # Learning
            obs_after = self.world.get_observation()
            self.action_selector.update_transition_model(action, obs, obs_after)
            self.action_selector.update_precision(obs_after, action)

            last_action = action

        return EpisodeResult(
            episode=episode,
            steps=step + 1,
            food_eaten=food_eaten,
            died=died,
            final_energy=self.world.energy,
            avg_G=np.mean(G_values) if G_values else 0.0,
            action_distribution=action_counts
        )

    def save_result(self, result: EvaluationResult, filename: str):
        """결과를 JSON으로 저장"""
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        return str(filepath)
