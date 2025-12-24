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
N_OBSERVATIONS = 6  # [food_prox, danger_prox, food_dx, food_dy, danger_dx, danger_dy]
N_ACTIONS = 5  # stay, up, down, left, right

# === WORLD STATE ===
class World:
    def __init__(self, size: int):
        self.size = size
        self.agent_pos = [size // 2, size // 2]
        self.food_pos = self._nearby_pos()  # Start food nearby
        self.danger_pos = self._far_pos()   # Start danger far
        self.step_count = 0

        # Energy represents homeostatic variable
        self.energy = 1.0  # Start full

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
        Get observation vector.

        6 dimensions:
        - [0]: food_proximity (1 = on food, 0 = far)
        - [1]: danger_proximity (1 = on danger, 0 = far)
        - [2]: food_dx (normalized: -1 = food is left, +1 = food is right, 0 = same column)
        - [3]: food_dy (normalized: -1 = food is up, +1 = food is down, 0 = same row)
        - [4]: danger_dx (normalized direction to danger)
        - [5]: danger_dy (normalized direction to danger)
        """
        obs = np.zeros(N_OBSERVATIONS)

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

        return obs

    def execute_action(self, action: int) -> Dict:
        """
        Execute action in world.

        Actions: 0=stay, 1=up, 2=down, 3=left, 4=right

        Returns dict with outcome info.
        """
        old_pos = self.agent_pos.copy()

        # Move
        if action == 1 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 2 and self.agent_pos[1] < self.size - 1:
            self.agent_pos[1] += 1
        elif action == 3 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 4 and self.agent_pos[0] < self.size - 1:
            self.agent_pos[0] += 1

        hit_wall = (self.agent_pos == old_pos and action != 0)

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

        # Energy decay (slower during infant phase)
        decay = 0.001 if is_infant else 0.003
        self.energy = max(0.0, self.energy - decay)

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
            'died': self.energy <= 0
        }

    def reset(self):
        self.agent_pos = [self.size // 2, self.size // 2]
        self.food_pos = self._nearby_pos()  # Start food nearby
        self.danger_pos = self._far_pos()   # Start danger far
        self.energy = 1.0
        self.step_count = 0


# === INITIALIZE ===
world = World(GRID_SIZE)

# Create agent with preferences
# Observation: [food_prox, danger_prox, food_dx, food_dy, danger_dx, danger_dy]
# Prefer: HIGH food proximity, LOW danger proximity
# Direction preferences: 0 (neutral - direction is informational, not preferred)
preferred_obs = np.array([1.0,   # food_proximity = 1.0 (want to be ON food)
                          0.0,   # danger_proximity = 0.0 (want to be FAR from danger)
                          0.0,   # food_dx = 0 (no preference for direction)
                          0.0,   # food_dy = 0 (no preference for direction)
                          0.0,   # danger_dx = 0 (no preference for direction)
                          0.0])  # danger_dy = 0 (no preference for direction)
agent = GenesisAgent(N_STATES, N_OBSERVATIONS, N_ACTIONS, preferred_obs)

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

        # Agent step
        if last_state is None:
            state = agent.step(obs_before)
        else:
            state = agent.step_with_action(obs_before, last_action)

        # Execute action in world
        action = int(state.action)
        outcome = world.execute_action(action)

        # Get observation AFTER action
        obs_after = world.get_observation()

        # === TRANSITION MODEL LEARNING (downsampled) ===
        if sim_clock.should_learn():
            agent.action_selector.update_transition_model(action, obs_before, obs_after)

        # Handle death
        if outcome['died']:
            world.reset()
            agent.reset()

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
            'selected_risk': state.risk,
            'selected_ambiguity': state.ambiguity,
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
        last_action = 0
        last_state = None
        sim_clock.tick_id = 0
        sim_clock.cached_response = None
    return {'status': 'reset'}


@app.get("/clock")
async def get_clock():
    """Get simulation clock status."""
    return sim_clock.get_status()


@app.get("/info")
async def info():
    """Get system info."""
    return {
        'name': 'Genesis Brain',
        'principle': 'Free Energy Minimization',
        'equation': 'F = prediction_error + complexity',
        'behavior_explanation': (
            'All behavior emerges from minimizing F. '
            'There are no explicit emotion labels, goal variables, or reward signals. '
            'What observers might call "fear" is high expected F. '
            'What observers might call "curiosity" is high epistemic value. '
            'The agent just minimizes F.'
        ),
        'grid_size': GRID_SIZE,
        'n_states': N_STATES,
        'n_observations': N_OBSERVATIONS,
        'n_actions': N_ACTIONS
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
