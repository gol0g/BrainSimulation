"""
Genesis Brain - Main Server

This runs the Free Energy minimizing agent in a simple environment.

The agent:
- Observes the world
- Infers hidden states (perception)
- Selects actions to minimize Expected Free Energy
- Learns from experience

Everything else (emotions, goals, behavior patterns) EMERGES.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import numpy as np
import threading
import time

from genesis.agent import GenesisAgent, AgentState
from genesis.free_energy import FreeEnergyState
from genesis.scenarios import ScenarioManager, ScenarioType
from genesis.checkpoint import BrainCheckpoint, HeadlessRunner
from genesis.reproducibility import (
    set_global_seed, get_global_seed, get_seed_manager,
    run_reproducibility_test, SimulationFingerprint
)



# === SIMULATION CLOCK ===
class SimulationClock:
    """
    Time-based throttling for browser interference prevention.
    
    - min_step_interval_ms: Minimum time between steps (e.g., 50ms = max 20 FPS)
    - If called faster than this, returns cached response
    - Learning only happens every learning_interval ticks
    """
    def __init__(self, min_step_interval_ms: int = 50, learning_interval: int = 5):
        self.tick_id = 0
        self.cached_response = None
        self.min_step_interval = min_step_interval_ms / 1000.0
        self.learning_interval = learning_interval
        self.lock = threading.Lock()
        self.last_step_time = 0.0

    def should_advance(self):
        """Only advance if enough time has passed since last step."""
        now = time.time()
        return (now - self.last_step_time) >= self.min_step_interval

    def should_learn(self):
        return self.tick_id % self.learning_interval == 0

    def advance(self):
        self.tick_id += 1
        self.last_step_time = time.time()

    def get_status(self):
        return {
            "tick_id": self.tick_id,
            "learning_interval": self.learning_interval,
            "min_step_interval_ms": int(self.min_step_interval * 1000)
        }


sim_clock = SimulationClock(min_step_interval_ms=50, learning_interval=5)  # 50ms = max 20 FPS

app = FastAPI(title="Genesis Brain - Free Energy Principle")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === WORLD CONFIGURATION ===
GRID_SIZE = 10
N_STATES = GRID_SIZE * GRID_SIZE  # Position states
N_OBSERVATIONS = 8  # [food_prox, danger_prox, food_dx, food_dy, danger_dx, danger_dy, energy, pain]
N_ACTIONS = 6  # stay, up, down, left, right, THINK (v3.4)

# === WORLD STATE ===
class World:
    def __init__(self, size: int):
        self.size = size
        self.agent_pos = [size // 2, size // 2]
        self.food_pos = self._nearby_pos()  # Start food nearby
        self.danger_pos = self._far_pos()   # Start danger far
        self.step_count = 0

        # === INTERNAL HOMEOSTATIC VARIABLES (v2.5) ===
        # These are the "true" states that matter for survival
        # The agent will learn that external objects affect these
        self.energy = 1.0  # Start full (homeostatic setpoint ~0.7)
        self.pain = 0.0    # Start with no pain (homeostatic setpoint = 0)

        # Infant phase: protected environment for learning
        self.infant_steps = 500  # First 500 steps are protected

        # === STATISTICS TRACKING (v3.6) ===
        self.total_food = 0      # Total food eaten
        self.total_deaths = 0    # Total deaths

        # === SERVER-SIDE DRIFT (v4.5) ===
        # Drift changes action-to-movement mapping
        self.drift_enabled = False
        self.drift_type = 'rotate'  # rotate, flip_x, flip_y, reverse, partial, delayed, probabilistic
        self._drift_delay_counter = 0
        self._drift_delay_threshold = 10  # for delayed drift
        self._probabilistic_ratio = 0.7   # for probabilistic drift
        self._energy_decay_multiplier = 1.0  # for partial drift (2.0 = 2x faster decay)

    def _random_pos(self) -> List[int]:
        return [np.random.randint(0, self.size), np.random.randint(0, self.size)]

    def _nearby_pos(self) -> List[int]:
        """Spawn food within 2 cells of agent."""
        ax, ay = self.size // 2, self.size // 2
        dx = np.random.randint(-2, 3)
        dy = np.random.randint(-2, 3)
        if dx == 0 and dy == 0:
            dx = 1
        x = max(0, min(self.size - 1, ax + dx))
        y = max(0, min(self.size - 1, ay + dy))
        return [x, y]

    def _far_pos(self) -> List[int]:
        """Spawn danger far from agent."""
        ax, ay = self.size // 2, self.size // 2
        while True:
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            dist = abs(x - ax) + abs(y - ay)
            if dist >= 5:  # At least 5 cells away
                return [x, y]

    def _spawn_food_nearby(self) -> List[int]:
        """Spawn food within 3 cells of agent's current position."""
        ax, ay = self.agent_pos
        for _ in range(20):  # Try 20 times
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            if dx == 0 and dy == 0:
                continue
            x = max(0, min(self.size - 1, ax + dx))
            y = max(0, min(self.size - 1, ay + dy))
            # Don't spawn on danger
            if [x, y] != self.danger_pos:
                return [x, y]
        return self._random_pos()  # Fallback

    def apply_drift(self, action: int) -> int:
        """
        Apply drift to action (v4.5).

        Drift types:
        - rotate: UP→RIGHT→DOWN→LEFT→UP (방향 90도 회전)
        - flip_x: LEFT↔RIGHT
        - flip_y: UP↔DOWN
        - reverse: UP↔DOWN, LEFT↔RIGHT
        - partial: 내부 상태(energy)에만 영향 (action은 그대로)
        - delayed: N스텝 후부터 reverse 적용
        - probabilistic: 70% 확률로 reverse 적용

        Returns:
            Modified action (or same action if no drift)
        """
        if not self.drift_enabled:
            return action

        # STAY(0)와 THINK(5)는 영향 없음
        if action in (0, 5):
            return action

        if self.drift_type == 'rotate':
            # UP(1)→RIGHT(4)→DOWN(2)→LEFT(3)→UP(1)
            mapping = {1: 4, 4: 2, 2: 3, 3: 1}
            return mapping.get(action, action)

        elif self.drift_type == 'flip_x':
            # LEFT(3)↔RIGHT(4)
            mapping = {3: 4, 4: 3}
            return mapping.get(action, action)

        elif self.drift_type == 'flip_y':
            # UP(1)↔DOWN(2)
            mapping = {1: 2, 2: 1}
            return mapping.get(action, action)

        elif self.drift_type == 'reverse':
            # UP(1)↔DOWN(2), LEFT(3)↔RIGHT(4)
            mapping = {1: 2, 2: 1, 3: 4, 4: 3}
            return mapping.get(action, action)

        elif self.drift_type == 'partial':
            # 외부 행동은 그대로, 내부 dynamics만 변경
            # energy_decay_multiplier가 execute_action에서 적용됨
            # 에이전트가 "세상 규칙이 바뀜"을 transition_std로 감지해야 함
            return action

        elif self.drift_type == 'delayed':
            self._drift_delay_counter += 1
            if self._drift_delay_counter < self._drift_delay_threshold:
                return action
            # threshold 이후 reverse 적용
            mapping = {1: 2, 2: 1, 3: 4, 4: 3}
            return mapping.get(action, action)

        elif self.drift_type == 'probabilistic':
            if np.random.random() < self._probabilistic_ratio:
                mapping = {1: 2, 2: 1, 3: 4, 4: 3}
                return mapping.get(action, action)
            return action

        return action

    def enable_drift(self, drift_type: str = 'rotate', energy_decay_mult: float = 2.0):
        """Enable drift with specified type.

        Args:
            drift_type: Type of drift
            energy_decay_mult: For partial drift, multiplier for energy decay (2.0 = 2x faster)
        """
        self.drift_enabled = True
        self.drift_type = drift_type
        self._drift_delay_counter = 0
        # partial drift: set energy decay multiplier
        if drift_type == 'partial':
            self._energy_decay_multiplier = energy_decay_mult
        else:
            self._energy_decay_multiplier = 1.0

    def disable_drift(self):
        """Disable drift."""
        self.drift_enabled = False
        self._drift_delay_counter = 0
        self._energy_decay_multiplier = 1.0  # Reset to normal

    def get_drift_status(self) -> Dict:
        """Get current drift status."""
        status = {
            'enabled': self.drift_enabled,
            'type': self.drift_type if self.drift_enabled else None,
        }
        if self.drift_enabled:
            if self.drift_type == 'delayed':
                status['delay_counter'] = self._drift_delay_counter
                status['delay_threshold'] = self._drift_delay_threshold
            elif self.drift_type == 'probabilistic':
                status['probabilistic_ratio'] = self._probabilistic_ratio
            elif self.drift_type == 'partial':
                status['energy_decay_multiplier'] = self._energy_decay_multiplier
        return status

    def get_observation(self) -> np.ndarray:
        """
        Get observation vector (v2.5: includes internal states).

        8 dimensions:
        - [0]: food_proximity (1 = on food, 0 = far) - EXTEROCEPTION
        - [1]: danger_proximity (1 = on danger, 0 = far) - EXTEROCEPTION
        - [2]: food_dx (normalized: -1 = left, +1 = right) - EXTEROCEPTION
        - [3]: food_dy (normalized: -1 = up, +1 = down) - EXTEROCEPTION
        - [4]: danger_dx - EXTEROCEPTION
        - [5]: danger_dy - EXTEROCEPTION
        - [6]: energy (0-1, internal state) - INTEROCEPTION
        - [7]: pain (0-1, internal state) - INTEROCEPTION
        """
        obs = np.zeros(N_OBSERVATIONS)

        # === EXTEROCEPTION (external world) ===
        # Food proximity and direction
        dx_food = self.food_pos[0] - self.agent_pos[0]
        dy_food = self.food_pos[1] - self.agent_pos[1]
        dist_food = abs(dx_food) + abs(dy_food)
        obs[0] = max(0, 1 - dist_food / self.size)  # food_proximity
        obs[2] = np.sign(dx_food) if dx_food != 0 else 0  # food_dx (-1, 0, or 1)
        obs[3] = np.sign(dy_food) if dy_food != 0 else 0  # food_dy (-1, 0, or 1)

        # Danger proximity and direction
        dx_danger = self.danger_pos[0] - self.agent_pos[0]
        dy_danger = self.danger_pos[1] - self.agent_pos[1]
        dist_danger = abs(dx_danger) + abs(dy_danger)
        obs[1] = max(0, 1 - dist_danger / 5.0)  # danger_proximity
        obs[4] = np.sign(dx_danger) if dx_danger != 0 else 0  # danger_dx
        obs[5] = np.sign(dy_danger) if dy_danger != 0 else 0  # danger_dy

        # === INTEROCEPTION (internal states) ===
        obs[6] = self.energy  # 0-1
        obs[7] = self.pain    # 0-1

        return obs

    def execute_action(self, action: int) -> Dict:
        """
        Execute action in world.

        Actions: 0=stay, 1=up, 2=down, 3=left, 4=right, 5=THINK (v3.4)

        v3.4: THINK action은 물리적 이동 없이 시간만 흐름.
        시간이 흐르면서 energy 감소, danger 이동 가능성 존재.
        이것이 "생각의 자연스러운 비용".

        v4.5: Drift가 활성화되면 action-to-movement 매핑이 변경됨.

        Returns dict with outcome info.
        """
        old_pos = self.agent_pos.copy()
        original_action = action
        is_think = (action == 5)  # v3.4: THINK action

        # === v4.5: Apply drift before movement ===
        if self.drift_enabled and not is_think:
            action = self.apply_drift(action)

        # Move (THINK는 이동하지 않음)
        if not is_think:
            if action == 1 and self.agent_pos[1] > 0:
                self.agent_pos[1] -= 1
            elif action == 2 and self.agent_pos[1] < self.size - 1:
                self.agent_pos[1] += 1
            elif action == 3 and self.agent_pos[0] > 0:
                self.agent_pos[0] -= 1
            elif action == 4 and self.agent_pos[0] < self.size - 1:
                self.agent_pos[0] += 1

        hit_wall = (self.agent_pos == old_pos and action != 0 and not is_think)

        # Infant phase check
        is_infant = self.step_count < self.infant_steps

        # Check food
        ate_food = (self.agent_pos == self.food_pos)
        if ate_food:
            self.energy = min(1.0, self.energy + 0.3)
            self.total_food += 1  # v3.6: track total food eaten
            # Infant: respawn food nearby, Adult: random
            if is_infant:
                self.food_pos = self._spawn_food_nearby()
            else:
                self.food_pos = self._random_pos()

        # Check danger
        hit_danger = (self.agent_pos == self.danger_pos)
        if hit_danger:
            # Infant: less damage, Adult: full damage
            damage = 0.15 if is_infant else 0.3
            self.energy = max(0.0, self.energy - damage)
            # Pain increases on danger contact (v2.5)
            self.pain = min(1.0, self.pain + 0.5)

        # === HOMEOSTATIC DYNAMICS (v2.5) ===
        # Energy decay (slower during infant phase)
        # v4.5: partial drift applies energy_decay_multiplier
        base_decay = 0.001 if is_infant else 0.003
        decay = base_decay * self._energy_decay_multiplier
        self.energy = max(0.0, self.energy - decay)

        # Pain naturally decays over time (healing)
        pain_decay = 0.02
        self.pain = max(0.0, self.pain - pain_decay)

        # Move danger randomly (slower during infant phase)
        danger_move_chance = 0.1 if is_infant else 0.3
        if np.random.random() < danger_move_chance:
            dx = np.random.choice([-1, 0, 1])
            dy = np.random.choice([-1, 0, 1])
            new_x = max(0, min(self.size - 1, self.danger_pos[0] + dx))
            new_y = max(0, min(self.size - 1, self.danger_pos[1] + dy))
            self.danger_pos = [new_x, new_y]

        self.step_count += 1

        # Check death
        died = self.energy <= 0
        if died:
            self.total_deaths += 1  # v3.6: track total deaths

        return {
            'ate_food': ate_food,
            'hit_danger': hit_danger,
            'hit_wall': hit_wall,
            'energy': self.energy,
            'pain': self.pain,  # v2.5
            'died': died,
            'is_think': is_think,  # v3.4: THINK action 여부
            # v4.5: Drift tracking for reproducibility and learning
            'drift_active': self.drift_enabled,
            'intended_action': original_action,  # 에이전트가 의도한 행동
            'applied_action': action,             # 실제 적용된 행동 (drift 후)
            'action_modified': action != original_action,
        }

    def reset(self):
        """Reset world for new episode. Keeps total_food and total_deaths."""
        self.agent_pos = [self.size // 2, self.size // 2]
        self.food_pos = self._nearby_pos()  # Start food nearby
        self.danger_pos = self._far_pos()   # Start danger far
        self.energy = 1.0
        self.pain = 0.0  # v2.5
        self.step_count = 0
        # Note: total_food and total_deaths persist across episodes

    def reset_statistics(self):
        """Reset cumulative statistics (v3.6)."""
        self.total_food = 0
        self.total_deaths = 0


# === INITIALIZE ===
world = World(GRID_SIZE)

# Create agent with preferences (v2.5: 8 dimensions)
# Observation: [food_prox, danger_prox, food_dx, food_dy, danger_dx, danger_dy, energy, pain]
#
# Phase 0: External preferences still active, internal states observed but not yet preferred
# Phase 1: Will mix P_external and P_internal with lambda
# Phase 2: Only internal preferences (energy ~0.7, pain = 0)
#
preferred_obs = np.array([1.0,   # [0] food_proximity = 1.0 (want to be ON food) - EXTERNAL
                          0.0,   # [1] danger_proximity = 0.0 (want to be FAR from danger) - EXTERNAL
                          0.0,   # [2] food_dx = 0 (no preference for direction)
                          0.0,   # [3] food_dy = 0 (no preference for direction)
                          0.0,   # [4] danger_dx = 0 (no preference for direction)
                          0.0,   # [5] danger_dy = 0 (no preference for direction)
                          0.5,   # [6] energy = 0.5 (neutral for now, Phase 1 will prefer ~0.7) - INTERNAL
                          0.0])  # [7] pain = 0.0 (neutral/prefer no pain) - INTERNAL
agent = GenesisAgent(N_STATES, N_OBSERVATIONS, N_ACTIONS, preferred_obs)

# === SCENARIO MANAGER ===
scenario_manager = ScenarioManager(world, agent)

# Track last action
last_action = 0
last_state: Optional[AgentState] = None


# === API MODELS ===
class StepParams(BaseModel):
    tick_id: Optional[int] = None


def to_python(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {to_python(k): to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    return obj


@app.post("/step")
async def step(params: StepParams):
    """
    Run one step of the simulation.
    Uses SimulationClock for idempotent calls and learning downsampling.
    """
    global last_action, last_state

    # === FAST PATH: Check BEFORE lock to avoid queue backup ===
    # If we have a fresh cached response, return it immediately
    cached = sim_clock.cached_response
    if cached is not None and not sim_clock.should_advance():
        return cached

    with sim_clock.lock:
        # Double-check inside lock (another thread may have just updated)
        if not sim_clock.should_advance():
            if sim_clock.cached_response is not None:
                return sim_clock.cached_response

        # Get observation BEFORE action
        obs_before = world.get_observation()

        # === SCENARIO: 관측 수정 (센서 노이즈, 부분 관측 등) ===
        obs_modified = scenario_manager.modify_observation(obs_before)

        # === MEMORY RECALL (v4.0) ===
        # G 계산 전에 기억 회상 → memory_bias 설정
        agent.action_selector.recall_from_memory(obs_modified)

        # Agent step (수정된 관측 사용)
        if last_state is None:
            state = agent.step(obs_modified)
        else:
            state = agent.step_with_action(obs_modified, last_action)

        # === v4.6: CONTINUOUS DRIFT (phase별 drift 전환) ===
        apply_continuous_drift_for_step()

        # === SCENARIO: DRIFT 체크 (G1 Gate) ===
        drift_just_activated = scenario_manager.check_and_activate_drift()

        # === SCENARIO: 행동 수정 (미끄러짐, DRIFT 등) ===
        intended_action = int(state.action)
        action = scenario_manager.modify_action(intended_action)

        # === MEMORY PREPARE (v4.0) ===
        # 에피소드 저장 준비 (행동 후 결과와 함께 저장됨)
        G_before = min(g.G for g in state.G_decomposition.values()) if state.G_decomposition else 0.0
        agent.action_selector.prepare_episode(
            t=sim_clock.tick_id,
            obs_before=obs_modified,
            action=action,
            G_before=G_before
        )

        # Execute action in world
        outcome = world.execute_action(action)

        # Get observation AFTER action
        obs_after = world.get_observation()

        # === COUNTERFACTUAL + REGRET (v4.4) ===
        # Counterfactual 계산: 선택한 행동이 대안보다 더 큰 G를 초래했는가?
        # regret 신호는 memory_gate_boost, lr_boost, THINK benefit에 연결됨
        # v4.5: Use intended_action for regret (what the agent chose, not what was applied)
        # This way regret reflects "did I make the right choice?" even if drift modified the action
        if agent.action_selector.regret_enabled:
            agent.action_selector.compute_counterfactual(
                chosen_action=intended_action,
                obs_before=obs_before,
                obs_after=obs_after
            )

        # === TRANSITION MODEL LEARNING (downsampled) ===
        # v4.2: Use intended_action (not drift-modified action) for transition learning
        # This ensures drift creates prediction errors (agent learns what it CHOSE, not what was EXECUTED)
        if sim_clock.should_learn():
            agent.action_selector.update_transition_model(intended_action, obs_before, obs_after)

        # === PRECISION LEARNING ===
        # 예측과 실제 관측 비교하여 precision 업데이트
        agent.action_selector.update_precision(obs_after, action)

        # === HIERARCHICAL MODELS (v3.2) ===
        # Slow layer 업데이트 (if enabled)
        # v3.2: action 전달로 context별 전이 모델 학습
        agent.action_selector.update_hierarchy(
            pred_error=state.prediction_error,
            ambiguity=state.ambiguity,
            complexity=state.complexity,
            observation=obs_after,
            action=action
        )

        # === PREFERENCE LEARNING (v3.5) ===
        # 선호 분포 업데이트 (if enabled)
        # G 값과 예측 오차를 기반으로 내부 선호(energy, pain) 학습
        selected_G = state.G_decomposition.get(action)
        if selected_G:
            agent.action_selector.update_preference_learning(
                current_obs=obs_after,
                G_value=selected_G.G,
                prediction_error=state.prediction_error
            )

        # === UNCERTAINTY TRACKING (v4.3) ===
        # 불확실성 업데이트 (if enabled)
        # 4가지 소스: belief entropy, action entropy, transition std, prediction error
        avg_transition_std = np.mean(agent.action_selector.transition_model['delta_std'][:, :2])
        agent.action_selector.update_uncertainty(
            prediction_error=state.prediction_error,
            transition_std=avg_transition_std
        )

        # === DRIFT-AWARE SUPPRESSION (v4.6) ===
        # 예측 오차 급증 시 recall 억제
        # v4.6.2: Regret spike도 함께 고려
        regret_spike_for_suppression = False
        if agent.action_selector.regret_enabled:
            regret_status = agent.action_selector.get_regret_modulation()
            regret_spike_for_suppression = regret_status.get('is_spike', False)

        agent.action_selector.update_drift_suppression(
            prediction_error=state.prediction_error,
            regret_spike=regret_spike_for_suppression
        )

        # === MEMORY STORE (v4.0) ===
        # 에피소드 저장 시도 (memory_gate로 확률 결정)
        G_after = min(g.G for g in state.G_decomposition.values()) if state.G_decomposition else 0.0
        store_result = agent.action_selector.store_episode(obs_after, G_after)

        # === CONSOLIDATION TRIGGER UPDATE (v4.1) ===
        # Sleep 트리거 상태 업데이트 및 자동 통합 체크
        merged = store_result.get('merged', False) if store_result else False
        context_id = 0
        hc = agent.action_selector.hierarchy_controller
        if hc is not None and hasattr(hc, 'slow_layer') and hc.slow_layer is not None:
            context_id = int(np.argmax(hc.slow_layer.Q_context))

        agent.action_selector.update_consolidation_trigger(
            surprise=state.prediction_error,
            merged=merged,
            context_id=context_id
        )

        # Auto-consolidation check (if enabled)
        consolidation_result = agent.action_selector.check_and_consolidate()
        if consolidation_result is not None:
            print(f"[SLEEP] Consolidation completed: {consolidation_result.episodes_replayed} episodes, "
                  f"std {consolidation_result.transition_std_before:.3f} -> {consolidation_result.transition_std_after:.3f}")

        # Handle death
        if outcome['died']:
            world.reset()
            agent.reset()

        # === SCENARIO: 로깅 ===
        # v4.3: G1 Gate용 확장 로깅 (DRIFT 시나리오)
        # v4.6: intended_action, transition_error, regret_spike 추가
        if (scenario_manager.current_scenario is not None and
            scenario_manager.current_scenario.type == ScenarioType.DRIFT):
            # Get volatility info from transition model
            trans_error = agent.action_selector._last_transition_error
            volatility_ratio = trans_error.get('volatility_ratio', 0.0) if trans_error else 0.0
            std_change_pct = trans_error.get('std_change_pct', 0.0) if trans_error else 0.0
            transition_error = trans_error.get('mean_error', 0.0) if trans_error else 0.0
            transition_std = float(np.mean(agent.action_selector.transition_model['delta_std'][:, :2]))
            G_value = float(state.G_decomposition[action].G) if action in state.G_decomposition else 0.0

            # v4.6: Get regret spike status
            regret_spike = False
            if agent.action_selector.regret_enabled:
                regret_status = agent.action_selector.get_regret_modulation()
                regret_spike = regret_status.get('is_spike', False)

            scenario_manager.log_step_g1(
                state, action, outcome, transition_std, G_value,
                volatility_ratio=volatility_ratio,
                std_change_pct=std_change_pct,
                intended_action=intended_action,
                transition_error=transition_error,
                regret_spike=regret_spike
            )
        else:
            scenario_manager.log_step(state, action, outcome)

        # Get explanation
        explanation = agent.get_explanation(state)

        # Store for next step
        last_action = action
        last_state = state

        response = to_python({
            'world': {
            'agent_pos': world.agent_pos,
            'food_pos': world.food_pos,
            'danger_pos': world.danger_pos,
            'energy': world.energy,
            'step': world.step_count,
            'phase': 'infant' if world.step_count < world.infant_steps else 'adult',
            'phase_progress': min(1.0, world.step_count / world.infant_steps),
            'total_food': world.total_food,       # v3.6
            'total_deaths': world.total_deaths,   # v3.6
        },

            'free_energy': {
            'F': state.F,
            'dF_dt': state.dF_dt,
            'prediction_error': state.prediction_error,
            'F_history': state.F_history[-20:]
        },

            'action': {
            'selected': action,
            'G': {str(a): float(g.G) for a, g in state.G_decomposition.items()},
            'risk': {str(a): float(g.risk) for a, g in state.G_decomposition.items()},
            'ambiguity': {str(a): float(g.ambiguity) for a, g in state.G_decomposition.items()},
            'complexity': {str(a): float(g.complexity) for a, g in state.G_decomposition.items()},
            'selected_risk': state.risk,
            'selected_ambiguity': state.ambiguity,
            'selected_complexity': state.complexity,
            'dominant_factor': state.dominant_factor
        },

            'derived': {
            'information_rate': state.information_rate,
            'preference_divergence': state.preference_divergence,
            'belief_update': state.belief_update
        },

            'interpretation': explanation,

            'outcome': outcome,

            'clock': {
            'tick_id': sim_clock.tick_id,
            'learning_this_tick': sim_clock.should_learn()
        },

            'scenario': {
            'active': scenario_manager.current_scenario is not None,
            'type': scenario_manager.current_scenario.type.value if scenario_manager.current_scenario else None,
            'progress': len(scenario_manager._step_logs) if scenario_manager.current_scenario else 0,
            'duration': scenario_manager.current_scenario.duration if scenario_manager.current_scenario else 0,
            'action_modified': action != intended_action,
            'drift_active': scenario_manager._drift_active,
            'drift_just_activated': drift_just_activated,
        },

            'precision': {
            'sensory': agent.action_selector.get_precision_state().sensory_precision.tolist(),
            'transition': agent.action_selector.get_precision_state().transition_precision.tolist(),
            'goal': agent.action_selector.get_precision_state().goal_precision,
            'volatility': agent.action_selector.get_precision_state().volatility,
            'confidence': agent.action_selector.get_precision_state().confidence,
            'attention': agent.action_selector.precision_learner.get_attention_map()
        },

            'temporal': {
            **agent.action_selector.get_temporal_config(),
            'rollout_info': agent.action_selector.get_rollout_info()
        },

            'hierarchy': agent.action_selector.get_hierarchy_status(),

            'think': agent.action_selector.get_think_status(),  # v3.4

            'preference_learning': agent.action_selector.get_preference_learning_status(),  # v3.5

            'uncertainty': agent.action_selector.get_uncertainty_status(),  # v4.3

            'memory': agent.action_selector.get_memory_status(),  # v4.0

            'consolidation': agent.action_selector.get_consolidation_status(),  # v4.1

            'regret': agent.action_selector.get_regret_status(),  # v4.4

            # v4.2: Transition model learning info for G1 Gate debugging
            'transition_learning': {
                'avg_std': float(np.mean(agent.action_selector.transition_model['delta_std'][:, :2])),
                'std_per_action': [float(np.mean(agent.action_selector.transition_model['delta_std'][a, :2]))
                                   for a in range(5)],
                'counts': agent.action_selector.transition_model['count'][:5].tolist(),
                'last_error': agent.action_selector._last_transition_error,
            }
        })
        sim_clock.advance()
        sim_clock.cached_response = response
        return response


@app.post("/reset")
async def reset():
    """Reset world and agent."""
    global last_action, last_state
    with sim_clock.lock:
        world.reset()
        agent.reset()
        # v4.6.2 fix: drift도 리셋
        world.drift_enabled = False
        world.drift_type = None
        last_action = 0
        last_state = None
        sim_clock.tick_id = 0
        sim_clock.cached_response = None
    return {'status': 'reset'}


@app.get("/clock")
async def get_clock():
    """Get simulation clock status."""
    return sim_clock.get_status()


# === SCENARIO API ===

@app.get("/scenarios")
async def list_scenarios():
    """사용 가능한 시나리오 목록"""
    return {
        "scenarios": [
            {
                "id": "conflict",
                "name": "갈등 (Conflict)",
                "description": "음식과 위험이 같은 방향. 음식을 얻으려면 위험을 감수해야 함."
            },
            {
                "id": "deadend",
                "name": "막다른 길 (Dead End)",
                "description": "코너에서 시작. 탈출 경로에 위험이 있음."
            },
            {
                "id": "temptation",
                "name": "유혹-위협 (Temptation)",
                "description": "위험이 음식 바로 옆에. 음식을 먹으면 위험에 노출."
            },
            {
                "id": "sensor_noise",
                "name": "센서 노이즈 (Sensor Noise)",
                "description": "방향 정보가 15% 확률로 잘못됨."
            },
            {
                "id": "partial_obs",
                "name": "부분 관측 (Partial Observation)",
                "description": "위험 방향 정보가 숨겨짐."
            },
            {
                "id": "slip",
                "name": "미끄러짐 (Slip)",
                "description": "10% 확률로 의도와 다른 방향으로 이동."
            }
        ]
    }


@app.post("/scenario/start/{scenario_id}")
async def start_scenario(
    scenario_id: str,
    duration: int = 200,
    drift_after: int = 100,
    drift_type: str = "rotate"
):
    """
    시나리오 시작

    Args:
        scenario_id: 시나리오 ID (conflict, deadend, temptation, sensor_noise, partial_obs, slip, drift)
        duration: 총 스텝 수
        drift_after: DRIFT 시나리오용 - 몇 스텝 후 drift 시작
        drift_type: DRIFT 시나리오용 - rotate, flip_x, flip_y, reverse
    """
    global last_action, last_state

    scenario_map = {
        "conflict": ScenarioType.CONFLICT,
        "deadend": ScenarioType.DEADEND,
        "temptation": ScenarioType.TEMPTATION,
        "sensor_noise": ScenarioType.SENSOR_NOISE,
        "partial_obs": ScenarioType.PARTIAL_OBS,
        "slip": ScenarioType.SLIP,
        "drift": ScenarioType.DRIFT,  # G1 Gate용
    }

    if scenario_id not in scenario_map:
        return {"error": f"Unknown scenario: {scenario_id}"}

    with sim_clock.lock:
        last_action = 0
        last_state = None
        sim_clock.tick_id = 0
        sim_clock.cached_response = None

        # DRIFT 시나리오는 추가 파라미터 필요
        if scenario_id == "drift":
            result = scenario_manager.start_scenario(
                scenario_map[scenario_id],
                duration=duration,
                drift_after=drift_after,
                drift_type=drift_type
            )
        else:
            result = scenario_manager.start_scenario(
                scenario_map[scenario_id],
                duration=duration
            )

    return result


@app.post("/scenario/stop")
async def stop_scenario():
    """시나리오 중지 및 결과 반환"""
    result = scenario_manager.end_scenario()
    return result


@app.get("/scenario/status")
async def scenario_status():
    """현재 시나리오 상태"""
    if scenario_manager.current_scenario is None:
        return {"active": False}

    return {
        "active": True,
        "type": scenario_manager.current_scenario.type.value,
        "progress": len(scenario_manager._step_logs),
        "duration": scenario_manager.current_scenario.duration,
        "complete": scenario_manager.is_scenario_complete()
    }


@app.get("/scenario/results")
async def scenario_results():
    """모든 시나리오 결과 조회"""
    results = []
    for r in scenario_manager.results:
        results.append({
            "scenario": r.scenario_type,
            "steps": r.total_steps,
            "food_eaten": r.food_eaten,
            "danger_hits": r.danger_hits,
            "deaths": r.deaths,
            "avg_F": round(r.avg_F, 3),
            "avg_risk": round(r.avg_risk, 3),
            "avg_ambiguity": round(r.avg_ambiguity, 3),
            "avg_complexity": round(r.avg_complexity, 3),
            "dominant_factors": r.dominant_factor_counts,
            "oscillation_count": r.oscillation_count,
            "adaptation_score": round(r.adaptation_score, 3),
        })
    return {"results": results}


# === SERVER-SIDE DRIFT (v4.5) ===

@app.post("/drift/enable")
async def enable_drift(
    drift_type: str = 'rotate',
    delay_threshold: int = 10,
    probabilistic_ratio: float = 0.7,
    energy_decay_mult: float = 2.0
):
    """
    환경 drift 활성화

    Drift는 action-to-movement 매핑을 변경하여 환경 dynamics를 바꿈.
    에이전트의 학습된 모델이 무효화되어 적응이 필요해짐.

    Args:
        drift_type: rotate, flip_x, flip_y, reverse, partial, delayed, probabilistic
        delay_threshold: delayed drift일 때 몇 스텝 후 적용할지
        probabilistic_ratio: probabilistic drift일 때 변경 확률 (0-1)
        energy_decay_mult: partial drift일 때 energy decay 배율 (2.0 = 2배 빠른 감소)

    Returns:
        현재 drift 상태
    """
    world._drift_delay_threshold = delay_threshold
    world._probabilistic_ratio = probabilistic_ratio
    world.enable_drift(drift_type, energy_decay_mult=energy_decay_mult)
    return {
        "status": "enabled",
        "drift_type": drift_type,
        "message": f"Drift '{drift_type}' activated. Agent's learned model is now invalid."
    }


@app.post("/drift/disable")
async def disable_drift():
    """환경 drift 비활성화"""
    world.disable_drift()
    return {"status": "disabled", "message": "Drift deactivated. Normal dynamics restored."}


@app.get("/drift/status")
async def drift_status():
    """현재 drift 상태 조회"""
    return world.get_drift_status()


@app.get("/scenario/g1_gate")
async def g1_gate_result():
    """
    G1 Gate (Generalization Test) 결과 조회 - v4.3 Enhanced

    DRIFT 시나리오 실행 후 호출하여 결과 확인

    Returns:
        v4.3 Enhanced Metrics:
        - std_auc_shock: shock window에서 std 면적 (급변 감지 강도)
        - peak_std_ratio: max(std) / pre_std (스파이크 강도)
        - volatility_auc: shock window에서 volatility 면적
        - time_to_recovery: 회복까지 걸린 스텝
        - food_rate_*: 각 phase별 food rate
    """
    result = scenario_manager.get_g1_gate_result()
    if result is None:
        return {
            "error": "DRIFT 시나리오가 실행되지 않았거나 충분한 데이터가 없습니다",
            "hint": "POST /scenario/start/drift?duration=100&drift_after=50&drift_type=reverse 로 시나리오 시작"
        }

    return {
        "drift_type": result.drift_type,
        "drift_after": result.drift_after,

        # Pre-drift
        "pre_drift": {
            "food": result.pre_drift_food,
            "danger": result.pre_drift_danger,
            "avg_G": result.pre_drift_avg_G,
            "transition_std": result.pre_drift_transition_std,
            "food_rate": result.food_rate_pre,
        },

        # Post-drift
        "post_drift": {
            "food": result.post_drift_food,
            "danger": result.post_drift_danger,
            "avg_G": result.post_drift_avg_G,
            "transition_std": result.post_drift_transition_std,
        },

        # v4.3 Enhanced Metrics
        "enhanced_metrics": {
            "std_auc_shock": result.std_auc_shock,
            "peak_std_ratio": result.peak_std_ratio,
            "volatility_auc": result.volatility_auc,
            "time_to_recovery": result.time_to_recovery,
            "food_rate_shock": result.food_rate_shock,
            "food_rate_adapt": result.food_rate_adapt,
        },

        # Original metrics
        "std_increase_ratio": result.std_increase_ratio,
        "adaptation_steps": result.adaptation_steps,
        "recovery_ratio": result.recovery_ratio,

        # Gate result
        "passed": result.passed,
        "reasons": result.reasons,

        # v4.6 Drift Adaptation Metrics
        "drift_adaptation": {
            "policy_mismatch_rate": result.policy_mismatch_rate,
            "intended_outcome_error": result.intended_outcome_error,
            "regret_spike_rate": result.regret_spike_rate,
        },
    }


# === v4.6 DRIFT ADAPTATION REPORT API ===

from genesis.scenarios import (
    DriftAdaptationReport, AblationMatrix, AblationMatrixCell,
    create_drift_adaptation_report, STANDARD_DRIFT_TYPES, STANDARD_FEATURE_CONFIGS,
    ContinuousDriftConfig
)


def _get_current_feature_config() -> Dict[str, bool]:
    """현재 활성화된 기능 조회"""
    return {
        'memory': agent.action_selector.memory_enabled,
        'sleep': agent.action_selector.consolidation_enabled,
        'think': agent.action_selector.think_enabled,
        'hierarchy': agent.action_selector.hierarchy_controller is not None,
        'regret': agent.action_selector.regret_enabled,
    }


@app.get("/scenario/drift_report")
async def get_drift_adaptation_report():
    """
    v4.6 표준화된 Drift 적응 리포트

    G1 Gate 결과를 원인-결과 분석 형태로 재구성:
    1. Intervention (원인): drift가 실제로 개입한 정도
    2. Detection (신호): 에이전트가 drift를 "느낀" 정도
    3. Learning Response (반응): 학습 자원 재배치 여부
    4. Adaptation (결과): 최종 적응 성과

    Returns:
        DriftAdaptationReport as JSON with causal chain analysis
    """
    g1_result = scenario_manager.get_g1_gate_result()
    if g1_result is None:
        return {
            "error": "DRIFT 시나리오 결과가 없습니다",
            "hint": "POST /scenario/start/drift?duration=100&drift_after=50 로 시나리오 실행"
        }

    feature_config = _get_current_feature_config()
    total_steps = len(scenario_manager._step_logs)

    report = create_drift_adaptation_report(g1_result, feature_config, total_steps)
    return report.to_dict()


@app.get("/scenario/drift_report/summary")
async def get_drift_report_summary():
    """v4.6 Drift 리포트 1줄 요약 (ablation 매트릭스용)"""
    g1_result = scenario_manager.get_g1_gate_result()
    if g1_result is None:
        return {"error": "No drift scenario result", "summary": None}

    feature_config = _get_current_feature_config()
    total_steps = len(scenario_manager._step_logs)
    report = create_drift_adaptation_report(g1_result, feature_config, total_steps)

    return {
        "summary": report.get_summary_line(),
        "verdict": report.final_verdict,
        "score": AblationMatrixCell(
            drift_type=report.drift_type,
            config_name="current",
            report=report
        ).get_score()
    }


# === v4.6 ABLATION MATRIX API ===

# In-memory ablation matrix (populated by running tests)
_ablation_matrix: Dict[str, Dict[str, AblationMatrixCell]] = {}


@app.post("/ablation/matrix/record")
async def record_ablation_result(config_name: str = "baseline"):
    """
    현재 drift 테스트 결과를 ablation 매트릭스에 기록

    Usage:
    1. POST /ablation/apply?memory=true 로 config 설정
    2. POST /scenario/start/drift?drift_type=rotate 로 테스트 시작
    3. (run steps)
    4. POST /ablation/matrix/record?config_name=+memory 로 결과 기록
    5. 다른 drift_type/config로 반복
    """
    global _ablation_matrix

    g1_result = scenario_manager.get_g1_gate_result()
    if g1_result is None:
        return {"error": "No G1 Gate result to record"}

    drift_type = g1_result.drift_type
    feature_config = _get_current_feature_config()
    total_steps = len(scenario_manager._step_logs)

    report = create_drift_adaptation_report(g1_result, feature_config, total_steps)
    cell = AblationMatrixCell(drift_type=drift_type, config_name=config_name, report=report)

    # Initialize drift_type dict if needed
    if drift_type not in _ablation_matrix:
        _ablation_matrix[drift_type] = {}

    _ablation_matrix[drift_type][config_name] = cell

    return {
        "status": "recorded",
        "drift_type": drift_type,
        "config_name": config_name,
        "score": cell.get_score(),
        "verdict": report.final_verdict,
        "summary": report.get_summary_line()
    }


@app.get("/ablation/matrix")
async def get_ablation_matrix():
    """
    v4.6 Ablation 매트릭스 조회

    행: drift types (rotate, flip_x, delayed, probabilistic)
    열: feature configs (baseline, +memory, +regret, etc.)

    Returns:
        - table: Markdown 형식 테이블
        - best_configs: 각 drift별 최고 config
        - contributions: 각 feature의 평균 기여도
    """
    if not _ablation_matrix:
        return {
            "error": "Ablation matrix is empty",
            "hint": "Use POST /ablation/matrix/record after each test"
        }

    # Collect all drift types and config names
    drift_types = list(_ablation_matrix.keys())
    config_names = set()
    for configs in _ablation_matrix.values():
        config_names.update(configs.keys())
    config_names = sorted(list(config_names))

    matrix = AblationMatrix(
        drift_types=drift_types,
        config_names=config_names,
        cells=_ablation_matrix
    )

    # Build detailed results
    detailed = {}
    for drift in drift_types:
        detailed[drift] = {}
        for config in config_names:
            cell = matrix.get_cell(drift, config)
            if cell and cell.report:
                detailed[drift][config] = {
                    "score": cell.get_score(),
                    "verdict": cell.report.final_verdict,
                    "survival": cell.report.survival,
                    "recovery_steps": cell.report.time_to_recovery,
                    "retention": cell.report.performance_retention,
                }
            else:
                detailed[drift][config] = None

    return {
        "table_markdown": matrix.to_markdown_table(),
        "best_configs": matrix.get_best_config_per_drift(),
        "feature_contributions": matrix.get_feature_contribution(),
        "detailed_results": detailed,
        "drift_types": drift_types,
        "config_names": config_names
    }


@app.post("/ablation/matrix/reset")
async def reset_ablation_matrix():
    """Ablation 매트릭스 초기화"""
    global _ablation_matrix
    _ablation_matrix = {}
    return {"status": "reset", "message": "Ablation matrix cleared"}


@app.get("/ablation/standard_configs")
async def get_standard_configs():
    """표준 ablation 설정 목록 조회"""
    return {
        "drift_types": STANDARD_DRIFT_TYPES,
        "feature_configs": STANDARD_FEATURE_CONFIGS,
        "usage": {
            "1": "POST /ablation/apply with desired config",
            "2": "POST /scenario/start/drift?drift_type=X",
            "3": "Run steps",
            "4": "POST /ablation/matrix/record?config_name=Y",
            "5": "Repeat for all combinations",
            "6": "GET /ablation/matrix to see results"
        }
    }


# === v4.6 CONTINUOUS DRIFT API ===

# In-memory continuous drift state
_continuous_drift_config: Optional[ContinuousDriftConfig] = None
_continuous_drift_step: int = 0


@app.post("/drift/continuous/start")
async def start_continuous_drift(
    sequence: str = "normal,rotate,flip_x,normal",
    phase_duration: int = 50,
    overlap: int = 0
):
    """
    v4.6 연속/중첩 Drift 시작

    여러 drift 타입이 시간에 따라 순차적으로 적용됨.
    에이전트의 연속적 환경 변화 적응 능력 테스트.

    Args:
        sequence: 콤마로 구분된 drift 시퀀스 (예: "normal,rotate,flip_x,normal")
                  'normal' = drift 없음
        phase_duration: 각 phase 길이 (스텝)
        overlap: 전환 시 overlap 스텝 (점진적 전환)

    Example:
        sequence="normal,rotate,flip_x,probabilistic,normal"
        phase_duration=50
        → 0-50: normal, 50-100: rotate, 100-150: flip_x, 150-200: probabilistic, 200-250: normal
    """
    global _continuous_drift_config, _continuous_drift_step

    drift_sequence = [s.strip() for s in sequence.split(',')]

    # Validate drift types
    valid_types = {'normal', 'rotate', 'flip_x', 'flip_y', 'reverse', 'delayed', 'probabilistic', 'partial'}
    for dt in drift_sequence:
        if dt not in valid_types:
            return {"error": f"Invalid drift type: {dt}", "valid_types": list(valid_types)}

    _continuous_drift_config = ContinuousDriftConfig.from_sequence(
        drift_sequence, phase_duration, overlap
    )
    _continuous_drift_step = 0

    # Reset world
    world.reset()
    agent.reset()

    # Start scenario
    scenario_manager.start_scenario(ScenarioType.DRIFT, duration=_continuous_drift_config.get_total_steps())
    scenario_manager.current_scenario.params['drift_type'] = 'continuous'
    scenario_manager.current_scenario.params['continuous_config'] = {
        'sequence': drift_sequence,
        'phase_duration': phase_duration,
        'overlap': overlap
    }

    return {
        "status": "started",
        "sequence": drift_sequence,
        "phase_duration": phase_duration,
        "total_steps": _continuous_drift_config.get_total_steps(),
        "phases": _continuous_drift_config.phases
    }


@app.get("/drift/continuous/status")
async def get_continuous_drift_status():
    """현재 continuous drift 상태 조회"""
    global _continuous_drift_config, _continuous_drift_step

    if _continuous_drift_config is None:
        return {"active": False, "message": "No continuous drift configured"}

    current_drift = _continuous_drift_config.get_drift_at_step(_continuous_drift_step)
    current_phase = None
    for i, phase in enumerate(_continuous_drift_config.phases):
        if phase['start'] <= _continuous_drift_step < phase['end']:
            current_phase = i
            break

    return {
        "active": True,
        "current_step": _continuous_drift_step,
        "total_steps": _continuous_drift_config.get_total_steps(),
        "current_drift": current_drift,
        "current_phase": current_phase,
        "phases": _continuous_drift_config.phases
    }


@app.post("/drift/continuous/stop")
async def stop_continuous_drift():
    """Continuous drift 중단"""
    global _continuous_drift_config, _continuous_drift_step

    _continuous_drift_config = None
    _continuous_drift_step = 0
    world.disable_drift()

    return {"status": "stopped", "message": "Continuous drift stopped"}


def apply_continuous_drift_for_step():
    """
    매 스텝에서 continuous drift 적용 (internal use)
    main step 함수에서 호출
    """
    global _continuous_drift_config, _continuous_drift_step

    if _continuous_drift_config is None:
        return

    current_drift = _continuous_drift_config.get_drift_at_step(_continuous_drift_step)
    _continuous_drift_step += 1

    if current_drift is None:
        world.disable_drift()
    else:
        world.enable_drift(current_drift)


# === TEMPORAL DEPTH API ===

@app.post("/temporal/enable")
async def enable_temporal(
    horizon: int = 3,
    discount: float = 0.9,
    n_samples: int = 3,
    complexity_decay: float = 0.7
):
    """
    Multi-step rollout 활성화 (v2.4.1: Monte Carlo + Complexity Decay)

    Args:
        horizon: 몇 스텝 앞까지 볼 것인가 (1-5 권장)
        discount: 미래 가치 할인율 (0.8-0.99)
        n_samples: Monte Carlo 샘플 수 (2-5 권장)
        complexity_decay: 미래 complexity 감쇠율 (0.5-0.9)
    """
    agent.action_selector.enable_rollout(
        horizon=horizon,
        discount=discount,
        n_samples=n_samples,
        complexity_decay=complexity_decay
    )
    return {
        "status": "enabled",
        "horizon": agent.action_selector.rollout_horizon,
        "discount": agent.action_selector.rollout_discount,
        "n_samples": agent.action_selector.rollout_n_samples,
        "complexity_decay": agent.action_selector.rollout_complexity_decay
    }


@app.post("/temporal/disable")
async def disable_temporal():
    """1-step 모드로 복귀"""
    agent.action_selector.disable_rollout()
    return {"status": "disabled"}


@app.post("/temporal/adaptive")
async def enable_adaptive_temporal(
    quantile: float = 0.3,
    horizon: int = 3,
    discount: float = 0.9,
    n_samples: int = 3,
    complexity_decay: float = 0.7
):
    """
    Adaptive rollout 활성화 (v2.4.3)

    decision_entropy 기반 + 분위수 threshold + ambiguity 보완.
    "결정이 애매하거나 전이가 불확실할 때만 깊이 생각"

    Args:
        quantile: rollout 비율 (0.0-1.0)
            - 0.3 = 상위 30% 불확실한 상황에서만 rollout
            - 0.0 = 항상 rollout
            - 1.0 = 절대 안함
        horizon, discount, n_samples, complexity_decay: rollout 파라미터
    """
    agent.action_selector.enable_adaptive_rollout(
        quantile=quantile,
        horizon=horizon,
        discount=discount,
        n_samples=n_samples,
        complexity_decay=complexity_decay
    )
    return {
        "status": "adaptive",
        "quantile": agent.action_selector.rollout_quantile,
        "horizon": agent.action_selector.rollout_horizon,
        "discount": agent.action_selector.rollout_discount,
        "n_samples": agent.action_selector.rollout_n_samples,
        "complexity_decay": agent.action_selector.rollout_complexity_decay
    }


@app.get("/temporal/status")
async def temporal_status():
    """현재 temporal depth 설정 조회"""
    config = agent.action_selector.get_temporal_config()
    rollout_info = agent.action_selector.get_rollout_info()
    return {
        **config,
        "last_rollout": rollout_info
    }


# === PREFERENCE CONTROL API (v2.5) ===

@app.post("/preference/internal_weight")
async def set_internal_weight(weight: float = 0.0):
    """
    내부/외부 선호 혼합 비율 설정 (v2.5 Phase 1)

    Args:
        weight: 내부 선호 가중치 (0.0-1.0)
            - 0.0 = 외부 선호만 (Phase 0) - "음식 가까이, 위험 멀리"
            - 0.5 = 혼합 (Phase 1) - 외부 + 내부
            - 1.0 = 내부 선호만 (Phase 2) - "에너지 유지, 통증 없음"

    철학적 의미:
        - 0.0: 의미가 외부에서 주어짐 (프로그래머가 정의)
        - 1.0: 의미를 스스로 발견해야 함 (음식→에너지 연결 학습)
    """
    weight = max(0.0, min(1.0, weight))
    agent.action_selector.preferences.set_internal_weight(weight)
    return {
        "status": "updated",
        "internal_weight": agent.action_selector.preferences.internal_pref_weight,
        "phase": "Phase 0" if weight == 0 else ("Phase 2" if weight == 1.0 else "Phase 1")
    }


@app.get("/preference/status")
async def preference_status():
    """현재 선호 설정 조회"""
    prefs = agent.action_selector.preferences
    return {
        "internal_weight": prefs.internal_pref_weight,
        "phase": "Phase 0" if prefs.internal_pref_weight == 0 else (
            "Phase 2" if prefs.internal_pref_weight == 1.0 else "Phase 1"
        ),
        "external_prefs": {
            "food_proximity": f"Beta({prefs.food_prox_pref.alpha}, {prefs.food_prox_pref.beta})",
            "danger_proximity": f"Beta({prefs.danger_prox_pref.alpha}, {prefs.danger_prox_pref.beta})"
        },
        "internal_prefs": {
            "energy": f"Beta({prefs.energy_pref.alpha}, {prefs.energy_pref.beta})",
            "pain": f"Beta({prefs.pain_pref.alpha}, {prefs.pain_pref.beta})"
        }
    }


# === HIERARCHICAL MODELS API (v3.0) ===

@app.post("/hierarchy/enable")
async def enable_hierarchy(
    K: int = 4,
    update_interval: int = 10,
    transition_self: float = 0.95,
    ema_alpha: float = 0.1
):
    """
    계층적 처리 활성화 (v3.0)

    Slow layer가 fast layer의 precision을 조절.

    Args:
        K: Context 수 (추천: 4)
        update_interval: Slow layer 업데이트 주기 (step)
        transition_self: P(c_t = c_{t-1}) - 자기 유지 확률
        ema_alpha: 관측 통계 EMA alpha
    """
    agent.action_selector.enable_hierarchy(
        K=K,
        update_interval=update_interval,
        transition_self=transition_self,
        ema_alpha=ema_alpha
    )
    return {
        "status": "enabled",
        "K": K,
        "update_interval": update_interval,
        "transition_self": transition_self,
        "ema_alpha": ema_alpha
    }


@app.post("/hierarchy/disable")
async def disable_hierarchy():
    """계층적 처리 비활성화"""
    agent.action_selector.disable_hierarchy()
    return {"status": "disabled"}


@app.get("/hierarchy/status")
async def hierarchy_status():
    """현재 계층 상태 조회"""
    return agent.action_selector.get_hierarchy_status()


@app.post("/hierarchy/alpha")
async def set_context_alpha(
    alpha_ext: float = 0.2,
    alpha_int: float = 0.1,
    clamp: float = 0.05,
    use_confidence: bool = True
):
    """
    Context-weighted 전이 모델 혼합 비율 설정 (v3.3.1)

    Args:
        alpha_ext: 외부 상태 (food/danger prox) 혼합 비율 (0.0-0.5, 기본 0.2)
        alpha_int: 내부 상태 (energy/pain) 혼합 비율 (0.0-0.3, 기본 0.1)
        clamp: delta_ctx 최대 변화량 (0.01-0.2, 기본 0.05)
        use_confidence: 신뢰도 기반 alpha 자동 조절 (기본 True)

    v3.3.1 개선:
        - 내부/외부 alpha 분리 (internal은 더 보수적)
        - delta_ctx clamp로 스케일 폭주 방지
        - 신뢰도 기반: context가 불확실하면 physics 의존
    """
    alpha_ext = max(0.0, min(0.5, alpha_ext))
    alpha_int = max(0.0, min(0.3, alpha_int))
    clamp = max(0.01, min(0.2, clamp))

    agent.action_selector.context_transition_alpha_external = alpha_ext
    agent.action_selector.context_transition_alpha_internal = alpha_int
    agent.action_selector.delta_ctx_clamp = clamp
    agent.action_selector.use_confidence_alpha = use_confidence

    return {
        "status": "updated",
        "alpha_external": alpha_ext,
        "alpha_internal": alpha_int,
        "delta_ctx_clamp": clamp,
        "use_confidence_alpha": use_confidence
    }


# === THINK ACTION API (v3.4) ===

@app.post("/think/enable")
async def enable_think(energy_cost: float = 0.003):
    """
    THINK action 활성화 (v3.4 Metacognition)

    THINK가 활성화되면 에이전트가 "생각할지 말지"를 G로 선택.
    THINK 선택 시 rollout 실행 → 더 나은 행동 발견.

    Args:
        energy_cost: THINK 시 energy 감소량 (기본 0.003)
    """
    agent.action_selector.enable_think(energy_cost=energy_cost)
    return {
        "status": "enabled",
        "energy_cost": energy_cost
    }


@app.post("/think/disable")
async def disable_think():
    """THINK action 비활성화"""
    agent.action_selector.disable_think()
    return {"status": "disabled"}


@app.get("/think/status")
async def think_status():
    """현재 THINK 상태 조회"""
    return agent.action_selector.get_think_status()


# === PREFERENCE LEARNING API (v3.5) ===

@app.post("/preference/learning/enable")
async def enable_preference_learning(mode_lr: float = 0.02, concentration_lr: float = 0.01):
    """
    온라인 선호 학습 활성화 (v3.5)

    내부 선호(energy, pain)의 Beta 파라미터를 경험에서 학습.

    핵심 원리:
    - G가 낮았으면 (좋은 결과) → 현재 내부 상태를 더 선호하도록 mode 이동
    - G가 높았으면 (나쁜 결과) → 현재 내부 상태에서 멀어지도록 mode 이동
    - 일관된 경험 → concentration 증가 (확신)
    - 불일치 경험 → concentration 감소 (불확실)

    Args:
        mode_lr: mode 학습률 (기본 0.02)
        concentration_lr: concentration 학습률 (기본 0.01)
    """
    agent.action_selector.enable_preference_learning(
        mode_lr=mode_lr,
        concentration_lr=concentration_lr
    )
    return {
        "status": "enabled",
        "mode_lr": mode_lr,
        "concentration_lr": concentration_lr
    }


@app.post("/preference/learning/disable")
async def disable_preference_learning():
    """온라인 선호 학습 비활성화"""
    agent.action_selector.disable_preference_learning()
    return {"status": "disabled"}


@app.post("/preference/learning/reset")
async def reset_preference_learning():
    """선호 학습 리셋 (초기값으로)"""
    agent.action_selector.reset_preference_learning()
    return {"status": "reset"}


@app.get("/preference/learning/status")
async def preference_learning_status():
    """현재 선호 학습 상태 조회"""
    return agent.action_selector.get_preference_learning_status()


# === UNCERTAINTY/CONFIDENCE API (v4.3) ===

@app.post("/uncertainty/enable")
async def enable_uncertainty(
    belief_weight: float = 0.25,
    action_weight: float = 0.30,
    model_weight: float = 0.20,
    surprise_weight: float = 0.25,
    sensitivity: float = 1.0
):
    """
    불확실성 추적 활성화 (v4.3)

    불확실성이 "자동으로" 행동을 바꾸게 만듦:
    - THINK 선택 확률/비용 연동 (불확실 → THINK 유리)
    - Precision 메타-조절 (불확실 → precision 낮춤)
    - 탐색/회피 균형 (불확실 → 탐색 보너스)
    - 기억 저장 게이트 (v4.0 준비)

    4가지 불확실성 소스:
    1) Belief (context entropy): "어떤 상황인지 모름"
    2) Action (decision entropy): "뭘 해야 할지 모름"
    3) Model (transition std): "행동 결과를 모름"
    4) Surprise (prediction error): "예측이 틀림"

    Args:
        belief_weight: context belief 엔트로피 가중치
        action_weight: action 선택 엔트로피 가중치
        model_weight: 전이 모델 불확실성 가중치
        surprise_weight: 예측 오차 가중치
        sensitivity: 조절 민감도 (0.0~2.0)
    """
    agent.action_selector.enable_uncertainty(
        belief_weight=belief_weight,
        action_weight=action_weight,
        model_weight=model_weight,
        surprise_weight=surprise_weight,
        sensitivity=sensitivity
    )
    return {
        "status": "enabled",
        "weights": {
            "belief": belief_weight,
            "action": action_weight,
            "model": model_weight,
            "surprise": surprise_weight
        },
        "sensitivity": sensitivity
    }


@app.post("/uncertainty/disable")
async def disable_uncertainty():
    """불확실성 추적 비활성화"""
    agent.action_selector.disable_uncertainty()
    return {"status": "disabled"}


@app.post("/uncertainty/reset")
async def reset_uncertainty():
    """불확실성 상태 리셋"""
    agent.action_selector.reset_uncertainty()
    return {"status": "reset"}


@app.get("/uncertainty/status")
async def uncertainty_status():
    """
    현재 불확실성 상태 조회

    Returns:
        - state: global_uncertainty, global_confidence, components
        - modulation: think_bias, precision_mult, exploration_bonus, memory_gate
        - top_factor: 현재 불확실성의 주 원인
        - top_factor_distribution: 최근 top factor 분포
    """
    return agent.action_selector.get_uncertainty_status()


@app.get("/uncertainty/memory_gate")
async def get_memory_gate():
    """
    기억 저장 게이트 값 반환 (v4.0 Memory 준비용)

    Returns:
        memory_gate: 0.0~1.0
            - 0.0 = 저장 안 함 (안정적, 반복적)
            - 1.0 = 강하게 저장 (놀라움, 불확실)
    """
    return {
        "memory_gate": agent.action_selector.get_memory_gate()
    }


# === LONG-TERM MEMORY API (v4.0) ===

@app.post("/memory/enable")
async def enable_memory(
    max_episodes: int = 1000,
    store_threshold: float = 0.5,
    store_sharpness: float = 5.0,
    similarity_threshold: float = 0.95,
    recall_top_k: int = 5,
    store_enabled: bool = True,
    recall_enabled: bool = True
):
    """
    장기 기억 활성화 (v4.0)

    기억 = 미래 F/G를 줄이는 압축 모델
    - 저장: memory_gate > threshold일 때 확률적 저장
    - 회상: 유사 상황 검색 → G bias (행동 직접 지시 X)
    - 불확실할 때 기억에 더 의존

    Args:
        max_episodes: 최대 저장 에피소드 수
        store_threshold: 저장 임계값 (0.5 = memory_gate 50%에서 50% 확률)
        store_sharpness: sigmoid 기울기
        similarity_threshold: 중복 억제 유사도 (0.95 = 95% 유사하면 병합)
        recall_top_k: 회상 시 상위 k개 에피소드 사용
        store_enabled: v4.6 분해 실험 - 저장 허용
        recall_enabled: v4.6 분해 실험 - 회상 bias 적용
    """
    agent.action_selector.enable_memory(
        max_episodes=max_episodes,
        store_threshold=store_threshold,
        store_sharpness=store_sharpness,
        similarity_threshold=similarity_threshold,
        recall_top_k=recall_top_k,
        store_enabled=store_enabled,
        recall_enabled=recall_enabled
    )
    return {
        "status": "enabled",
        "max_episodes": max_episodes,
        "store_threshold": store_threshold,
        "similarity_threshold": similarity_threshold,
        "recall_top_k": recall_top_k,
        "store_enabled": store_enabled,
        "recall_enabled": recall_enabled
    }


@app.post("/memory/mode")
async def set_memory_mode(
    store_enabled: bool = True,
    recall_enabled: bool = True
):
    """
    v4.6: 메모리 분해 실험 모드 설정

    분해 실험:
    - store_only: store_enabled=True, recall_enabled=False
    - recall_only: store_enabled=False, recall_enabled=True
    - full: store_enabled=True, recall_enabled=True

    Args:
        store_enabled: 새 에피소드 저장 허용
        recall_enabled: 회상 bias 적용 허용
    """
    agent.action_selector.set_memory_mode(
        store_enabled=store_enabled,
        recall_enabled=recall_enabled
    )
    return {
        "status": "mode_set",
        "store_enabled": store_enabled,
        "recall_enabled": recall_enabled
    }


@app.post("/memory/drift_suppression/enable")
async def enable_drift_suppression(
    spike_threshold: float = 2.0,
    recovery_rate: float = 0.05,
    use_regret: bool = True
):
    """
    v4.6: Drift-aware Recall Suppression 활성화

    예측 오차가 급증하면 (drift 감지) recall weight를 자동으로 억제.
    적응 후 점진적으로 회복.

    v4.6.2: Regret spike 결합
    - transition error spike + regret spike → 더 강한 억제
    - regret은 "회상 강화"가 아니라 "억제 보조 신호"로 사용

    Args:
        spike_threshold: baseline 대비 몇 배면 spike로 간주 (기본 2.0)
        recovery_rate: 매 step 회복률 (기본 0.05)
        use_regret: regret spike를 억제 트리거로 사용할지 (기본 True)
    """
    agent.action_selector.enable_drift_suppression(
        spike_threshold=spike_threshold,
        recovery_rate=recovery_rate,
        use_regret=use_regret
    )
    return {
        "status": "enabled",
        "spike_threshold": spike_threshold,
        "recovery_rate": recovery_rate,
        "use_regret": use_regret
    }


@app.post("/memory/drift_suppression/disable")
async def disable_drift_suppression():
    """Drift-aware suppression 비활성화"""
    agent.action_selector.disable_drift_suppression()
    return {"status": "disabled"}


@app.get("/memory/drift_suppression/status")
async def get_drift_suppression_status():
    """Drift suppression 상태 조회"""
    return agent.action_selector.get_drift_suppression_status()


@app.post("/memory/disable")
async def disable_memory():
    """장기 기억 비활성화 (저장소는 유지)"""
    agent.action_selector.disable_memory()
    return {"status": "disabled"}


@app.post("/memory/reset")
async def reset_memory():
    """장기 기억 저장소 초기화"""
    agent.action_selector.reset_memory()
    return {"status": "reset"}


@app.get("/memory/status")
async def memory_status():
    """
    현재 기억 상태 조회

    Returns:
        - enabled: 활성화 여부
        - stats: 저장/회상 통계
        - last_recall: 마지막 회상 결과
        - last_store: 마지막 저장 결과
    """
    return agent.action_selector.get_memory_status()


@app.get("/memory/episodes")
async def get_episodes(limit: int = 10):
    """최근 저장된 에피소드 목록"""
    return {
        "episodes": agent.action_selector.get_recent_episodes(limit)
    }


# === CONSOLIDATION API (v4.1) ===

@app.post("/consolidation/enable")
async def enable_consolidation(
    similarity_threshold: float = 0.9,
    min_cluster_size: int = 2,
    replay_batch_size: int = 20,
    auto_trigger: bool = True
):
    """
    v4.1 Memory Consolidation 활성화

    Args:
        similarity_threshold: 프로토타입 생성 유사도 기준 (0.9)
        min_cluster_size: 클러스터링 최소 에피소드 수 (2)
        replay_batch_size: replay 시 처리할 에피소드 수 (20)
        auto_trigger: 자동 sleep 트리거 여부 (True)

    Returns:
        status: enabled, configuration
    """
    agent.action_selector.enable_consolidation(
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
        replay_batch_size=replay_batch_size,
        auto_trigger=auto_trigger
    )
    return {
        "status": "enabled",
        "similarity_threshold": similarity_threshold,
        "min_cluster_size": min_cluster_size,
        "replay_batch_size": replay_batch_size,
        "auto_trigger": auto_trigger
    }


@app.post("/consolidation/disable")
async def disable_consolidation():
    """Memory Consolidation 비활성화"""
    agent.action_selector.disable_consolidation()
    return {"status": "disabled"}


@app.post("/consolidation/trigger")
async def trigger_consolidation():
    """
    수동 Sleep 트리거

    자동 트리거 조건과 무관하게 즉시 통합 실행.

    Returns:
        result: ConsolidationResult (episodes_replayed, transition_std before/after, etc.)
    """
    result = agent.action_selector.force_consolidate()
    if result is None:
        return {"status": "failed", "reason": "consolidation not enabled or no memory"}

    return {
        "status": "completed",
        "episodes_replayed": result.episodes_replayed,
        "transition_updates": result.transition_updates,
        "prototypes_created": result.prototypes_created,
        "context_updates": result.context_updates,
        "transition_std_before": result.transition_std_before,
        "transition_std_after": result.transition_std_after,
        "compression_ratio": result.compression_ratio,
        "avg_cluster_size": result.avg_cluster_size
    }


@app.get("/consolidation/status")
async def consolidation_status():
    """
    현재 통합 시스템 상태 조회

    Returns:
        - enabled: 활성화 여부
        - should_sleep: 현재 sleep 트리거 조건 충족 여부
        - trigger_signals: 각 트리거 신호 값
        - stats: 총 sleeps, 프로토타입 수 등
        - last_result: 마지막 통합 결과
        - current_transition_std: 현재 평균 transition_std
    """
    return agent.action_selector.get_consolidation_status()


@app.post("/consolidation/reset")
async def reset_consolidation():
    """통합 시스템 초기화"""
    agent.action_selector.reset_consolidation()
    return {"status": "reset"}


# === CHECKPOINT API (v3.6) ===

checkpoint_manager = BrainCheckpoint()


@app.post("/checkpoint/save")
async def save_checkpoint(filename: str = "brain_checkpoint.json", description: str = ""):
    """
    현재 뇌 상태를 체크포인트로 저장

    저장 대상:
    - transition_model (전이 모델)
    - precision_learner (정밀도 학습기)
    - hierarchy_controller (계층 컨트롤러)
    - preference_learner (선호 학습기)
    - world 상태
    """
    filepath = checkpoint_manager.save(
        action_selector=agent.action_selector,
        world=world,
        filename=filename,
        description=description
    )
    return {"status": "saved", "filepath": filepath}


@app.post("/checkpoint/load")
async def load_checkpoint(filename: str = "brain_checkpoint.json"):
    """체크포인트에서 뇌 상태 복원"""
    try:
        metadata = checkpoint_manager.load(
            action_selector=agent.action_selector,
            world=world,
            filename=filename
        )
        return {
            "status": "loaded",
            "metadata": {
                "version": metadata.version,
                "timestamp": metadata.timestamp,
                "step_count": metadata.step_count,
                "description": metadata.description
            }
        }
    except FileNotFoundError:
        return {"status": "error", "message": f"Checkpoint '{filename}' not found"}


@app.get("/checkpoint/list")
async def list_checkpoints():
    """사용 가능한 체크포인트 목록"""
    return checkpoint_manager.list_checkpoints()


# === HEADLESS EVALUATION API (v3.6) ===

@app.post("/evaluate")
async def run_evaluation(
    n_episodes: int = 10,
    max_steps: int = 500,
    seed: int = None
):
    """
    헤드리스 평가 실행

    UI 없이 N 에피소드 자동 평가.

    Args:
        n_episodes: 에피소드 수
        max_steps: 에피소드당 최대 스텝
        seed: 랜덤 시드 (재현성)
    """
    runner = HeadlessRunner(agent, world, agent.action_selector, seed=seed)
    result = runner.run(n_episodes=n_episodes, max_steps_per_episode=max_steps, verbose=False)
    return result.to_dict()


@app.post("/evaluate/save")
async def run_and_save_evaluation(
    n_episodes: int = 10,
    max_steps: int = 500,
    seed: int = None,
    filename: str = "evaluation_result.json"
):
    """평가 실행 후 결과를 JSON으로 저장"""
    runner = HeadlessRunner(agent, world, agent.action_selector, seed=seed)
    result = runner.run(n_episodes=n_episodes, max_steps_per_episode=max_steps, verbose=False)
    filepath = runner.save_result(result, filename)
    return {"status": "saved", "filepath": filepath, "summary": {
        "n_episodes": result.n_episodes,
        "avg_food": result.avg_food_per_episode,
        "survival_rate": result.survival_rate
    }}


# === SEED & REPRODUCIBILITY API (v3.7) ===

@app.post("/seed")
async def set_seed(seed: int):
    """
    글로벌 시드 설정 (재현성)

    모든 랜덤 소스(np.random, random)의 시드를 고정.
    동일 시드 → 동일 결과 보장.
    """
    set_global_seed(seed)
    return {"status": "seed_set", "seed": seed}


@app.get("/seed")
async def get_seed():
    """현재 시드 상태 조회"""
    return get_seed_manager().get_status()


@app.post("/reproducibility/test")
async def test_reproducibility(
    seed: int = 42,
    n_steps: int = 100,
    n_runs: int = 3
):
    """
    재현성 테스트 실행

    동일 seed로 n_runs번 시뮬레이션을 돌려서
    모든 결과가 동일한지 검증.

    합격 기준: 모든 실행의 fingerprint가 동일
    """
    result = run_reproducibility_test(
        agent=agent,
        world=world,
        action_selector=agent.action_selector,
        seed=seed,
        n_steps=n_steps,
        n_runs=n_runs
    )
    return {
        "passed": result.passed,
        "seed": result.seed,
        "n_runs": result.n_runs,
        "fingerprints": result.fingerprints,
        "message": result.message,
        "details": result.details
    }


# === ABLATION STUDY API (v4.3) ===

from genesis.ablation import AblationRunner, AblationConfig, STANDARD_ABLATIONS, compute_ablation_contribution

# Note: AblationRunner needs actual step execution which is complex
# For now, provide config listing and manual ablation support

@app.get("/ablation/configs")
async def ablation_configs():
    """사용 가능한 ablation 설정 목록"""
    return {
        "configs": [
            {
                "name": c.name,
                "memory": c.memory_enabled,
                "sleep": c.sleep_enabled,
                "think": c.think_enabled,
                "hierarchy": c.hierarchy_enabled,
            }
            for c in STANDARD_ABLATIONS
        ],
        "description": "각 config에 대해 G1 Gate 실행 후 /scenario/g1_gate로 결과 비교"
    }


@app.post("/ablation/apply")
async def apply_ablation_config(
    memory: bool = False,
    sleep: bool = False,
    think: bool = False,
    hierarchy: bool = False,
    regret: bool = False
):
    """
    Ablation 설정 적용

    Args:
        memory: LTM 활성화 여부
        sleep: Sleep/Consolidation 활성화 여부
        think: THINK action 활성화 여부
        hierarchy: Hierarchical context 활성화 여부
        regret: Counterfactual + Regret 활성화 여부 (v4.4)

    Usage:
        1. POST /ablation/apply?memory=true&hierarchy=true&regret=true
        2. POST /scenario/start/drift?duration=100&drift_after=50&drift_type=reverse
        3. (run steps)
        4. GET /scenario/g1_gate
        5. 다른 config로 반복하여 비교
    """
    # Memory
    if memory:
        agent.action_selector.enable_memory(store_threshold=0.5)
    else:
        agent.action_selector.disable_memory()

    # Sleep
    if sleep:
        agent.action_selector.enable_consolidation(auto_trigger=True)
    else:
        agent.action_selector.disable_consolidation()

    # THINK
    if think:
        agent.action_selector.enable_think()
    else:
        agent.action_selector.disable_think()

    # Hierarchy
    if hierarchy:
        agent.action_selector.enable_hierarchy(K=4, update_interval=10)
    else:
        agent.action_selector.disable_hierarchy()

    # Regret (v4.4)
    if regret:
        agent.action_selector.enable_regret()
    else:
        agent.action_selector.disable_regret()

    return {
        "applied": {
            "memory": memory,
            "sleep": sleep,
            "think": think,
            "hierarchy": hierarchy,
            "regret": regret
        },
        "status": "ok",
        "next_step": "POST /scenario/start/drift 로 G1 Gate 테스트 시작"
    }


# === REGRET API (v4.4) ===

@app.post("/regret/enable")
async def enable_regret(modulation_enabled: bool = True):
    """
    Counterfactual + Regret 활성화 (v4.4)

    후회(regret)는 "감정 변수"가 아니라,
    선택한 행동이 대안 행동보다 얼마나 더 큰 G를 초래했는지에 대한 '사후 EFE 차이'.

    연결 방식 (FEP스럽게):
    - 정책 직접 변경 X
    - memory_gate, lr_boost, THINK 비용/편익 조정 O
    → "후회가 '학습/추론 자원 배분'을 바꾸는 구조"

    Args:
        modulation_enabled: v4.6 분해 실험 - modulation 적용 여부
    """
    agent.action_selector.enable_regret(modulation_enabled=modulation_enabled)
    return {
        "status": "enabled",
        "modulation_enabled": modulation_enabled,
        "message": "Counterfactual + Regret system activated"
    }


@app.post("/regret/mode")
async def set_regret_mode(modulation_enabled: bool = True):
    """
    v4.6: Regret 분해 실험 모드 설정

    분해 실험:
    - calc_only: modulation_enabled=False (계산만, 적용 안함)
    - full: modulation_enabled=True (계산 + 적용)

    Args:
        modulation_enabled: memory_gate_boost, lr_boost, think_benefit 적용 여부
    """
    agent.action_selector.set_regret_mode(modulation_enabled=modulation_enabled)
    return {
        "status": "mode_set",
        "modulation_enabled": modulation_enabled
    }


@app.post("/regret/disable")
async def disable_regret():
    """Counterfactual + Regret 비활성화"""
    agent.action_selector.disable_regret()
    return {"status": "disabled"}


@app.post("/regret/reset")
async def reset_regret():
    """Regret 상태 초기화 (엔진은 활성 상태 유지)"""
    agent.action_selector.reset_regret()
    return {"status": "reset"}


@app.get("/regret/status")
async def regret_status():
    """
    현재 Regret 상태 조회

    Returns:
        - enabled: 활성화 여부
        - engine_status: CounterfactualEngine 상태
            - step_counter: 총 counterfactual 계산 횟수
            - regret_state: 누적 regret, 최적 선택 비율 등
            - last_result: 마지막 counterfactual 결과
            - modulation: memory_gate_boost, lr_boost_factor, think_benefit_boost
    """
    return agent.action_selector.get_regret_status()


@app.get("/regret/modulation")
async def regret_modulation():
    """
    Regret 기반 조절 파라미터 반환

    Returns:
        - memory_gate_boost: 기억 저장 게이트 증가량 (0~0.3)
        - lr_boost_factor: 학습률 승수 (1.0~1.5)
        - think_benefit_boost: THINK 효과 증가량 (0~0.2)
        - regret_real: 실제 후회 값
        - regret_pred: 예측 기반 후회 값
        - is_spike: 후회 급증 여부
    """
    return agent.action_selector.get_regret_modulation()


@app.get("/info")
async def info():
    """Get system info."""
    return {
        'name': 'Genesis Brain',
        'version': 'v4.0 - Long-Term Memory',
        'principle': 'Free Energy Minimization',
        'equations': {
            'F': 'Prediction Error + Complexity',
            'G(a)': 'Risk + Ambiguity + Complexity',
            'Risk': 'KL[Q(o|a) || P(o)] - preference violation',
            'Ambiguity': 'E[H[P(o|s\')]] - transition uncertainty',
            'Complexity': 'KL[Q(s\'|a) || P(s\')] - belief divergence from preferred states'
        },
        'behavior_explanation': (
            'All behavior emerges from minimizing G(a). '
            'Risk explains avoidance (looks like "fear"). '
            'Ambiguity reduction explains exploration (looks like "curiosity"). '
            'Complexity avoidance explains cognitive conservatism (looks like "habit"). '
            'The agent has no emotion labels - it just minimizes G.'
        ),
        'grid_size': GRID_SIZE,
        'n_states': N_STATES,
        'n_observations': N_OBSERVATIONS,
        'n_actions': N_ACTIONS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
