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

        Returns dict with outcome info.
        """
        old_pos = self.agent_pos.copy()
        is_think = (action == 5)  # v3.4: THINK action

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
        decay = 0.001 if is_infant else 0.003
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

        return {
            'ate_food': ate_food,
            'hit_danger': hit_danger,
            'hit_wall': hit_wall,
            'energy': self.energy,
            'pain': self.pain,  # v2.5
            'died': self.energy <= 0,
            'is_think': is_think,  # v3.4: THINK action 여부
        }

    def reset(self):
        self.agent_pos = [self.size // 2, self.size // 2]
        self.food_pos = self._nearby_pos()  # Start food nearby
        self.danger_pos = self._far_pos()   # Start danger far
        self.energy = 1.0
        self.pain = 0.0  # v2.5
        self.step_count = 0


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
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
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

        # Agent step (수정된 관측 사용)
        if last_state is None:
            state = agent.step(obs_modified)
        else:
            state = agent.step_with_action(obs_modified, last_action)

        # === SCENARIO: 행동 수정 (미끄러짐 등) ===
        intended_action = int(state.action)
        action = scenario_manager.modify_action(intended_action)

        # Execute action in world
        outcome = world.execute_action(action)

        # Get observation AFTER action
        obs_after = world.get_observation()

        # === TRANSITION MODEL LEARNING (downsampled) ===
        if sim_clock.should_learn():
            agent.action_selector.update_transition_model(action, obs_before, obs_after)

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

        # Handle death
        if outcome['died']:
            world.reset()
            agent.reset()

        # === SCENARIO: 로깅 ===
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
            'phase_progress': min(1.0, world.step_count / world.infant_steps)
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
            'action_modified': action != intended_action
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

            'think': agent.action_selector.get_think_status()  # v3.4
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
async def start_scenario(scenario_id: str, duration: int = 200):
    """시나리오 시작"""
    global last_action, last_state

    scenario_map = {
        "conflict": ScenarioType.CONFLICT,
        "deadend": ScenarioType.DEADEND,
        "temptation": ScenarioType.TEMPTATION,
        "sensor_noise": ScenarioType.SENSOR_NOISE,
        "partial_obs": ScenarioType.PARTIAL_OBS,
        "slip": ScenarioType.SLIP,
    }

    if scenario_id not in scenario_map:
        return {"error": f"Unknown scenario: {scenario_id}"}

    with sim_clock.lock:
        last_action = 0
        last_state = None
        sim_clock.tick_id = 0
        sim_clock.cached_response = None

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


@app.get("/info")
async def info():
    """Get system info."""
    return {
        'name': 'Genesis Brain',
        'version': 'v3.4 - THINK Action (Metacognition)',
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
