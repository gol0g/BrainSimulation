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

from genesis.agent import GenesisAgent, AgentState
from genesis.free_energy import FreeEnergyState

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
N_OBSERVATIONS = 8  # 4 directions for food + 4 directions for danger
N_ACTIONS = 5  # stay, up, down, left, right

# === WORLD STATE ===
class World:
    def __init__(self, size: int):
        self.size = size
        self.agent_pos = [size // 2, size // 2]
        self.food_pos = self._random_pos()
        self.danger_pos = self._random_pos()
        self.step_count = 0

        # Energy represents homeostatic variable
        self.energy = 0.8

    def _random_pos(self) -> List[int]:
        return [np.random.randint(0, self.size), np.random.randint(0, self.size)]

    def get_observation(self) -> np.ndarray:
        """
        Get observation vector.

        8 dimensions:
        - [0-3]: Food direction signals (up, down, left, right)
        - [4-7]: Danger direction signals (up, down, left, right)

        Signal strength decreases with distance.
        """
        obs = np.zeros(N_OBSERVATIONS)

        # Food signals
        dx_food = self.food_pos[0] - self.agent_pos[0]
        dy_food = self.food_pos[1] - self.agent_pos[1]
        dist_food = abs(dx_food) + abs(dy_food)
        food_strength = max(0, 1 - dist_food / self.size)

        if dy_food < 0: obs[0] = food_strength  # up
        if dy_food > 0: obs[1] = food_strength  # down
        if dx_food < 0: obs[2] = food_strength  # left
        if dx_food > 0: obs[3] = food_strength  # right

        # Danger signals
        dx_danger = self.danger_pos[0] - self.agent_pos[0]
        dy_danger = self.danger_pos[1] - self.agent_pos[1]
        dist_danger = abs(dx_danger) + abs(dy_danger)
        danger_strength = max(0, 1 - dist_danger / (self.size / 2))

        if dy_danger < 0: obs[4] = danger_strength  # up
        if dy_danger > 0: obs[5] = danger_strength  # down
        if dx_danger < 0: obs[6] = danger_strength  # left
        if dx_danger > 0: obs[7] = danger_strength  # right

        # Normalize
        if obs.sum() > 0:
            obs = obs / obs.max()

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

        # Check food
        ate_food = (self.agent_pos == self.food_pos)
        if ate_food:
            self.energy = min(1.0, self.energy + 0.3)
            self.food_pos = self._random_pos()

        # Check danger
        hit_danger = (self.agent_pos == self.danger_pos)
        if hit_danger:
            self.energy = max(0.0, self.energy - 0.3)

        # Energy decay
        self.energy = max(0.0, self.energy - 0.01)

        # Move danger randomly
        if np.random.random() < 0.3:
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
        self.food_pos = self._random_pos()
        self.danger_pos = self._random_pos()
        self.energy = 0.8
        self.step_count = 0


# === INITIALIZE ===
world = World(GRID_SIZE)

# Create agent with preferences
# Prefer: high food signal, low danger signal, medium energy
preferred_obs = np.array([0.3, 0.3, 0.3, 0.3,   # Some food signal is good
                          0.0, 0.0, 0.0, 0.0])   # No danger is best
agent = GenesisAgent(N_STATES, N_OBSERVATIONS, N_ACTIONS, preferred_obs)

# Track last action
last_action = 0
last_state: Optional[AgentState] = None


# === API MODELS ===
class StepParams(BaseModel):
    pass


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
    """
    global last_action, last_state

    # Get observation
    observation = world.get_observation()

    # Agent step
    if last_state is None:
        state = agent.step(observation)
    else:
        state = agent.step_with_action(observation, last_action)

    # Execute action in world
    action = int(state.action)  # Convert to Python int
    outcome = world.execute_action(action)

    # Handle death
    if outcome['died']:
        world.reset()
        agent.reset()

    # Get explanation
    explanation = agent.get_explanation(state)

    # Store for next step
    last_action = action
    last_state = state

    return to_python({
        # World state
        'world': {
            'agent_pos': world.agent_pos,
            'food_pos': world.food_pos,
            'danger_pos': world.danger_pos,
            'energy': world.energy,
            'step': world.step_count
        },

        # Core quantities (the ONLY internal states)
        'free_energy': {
            'F': state.F,
            'dF_dt': state.dF_dt,
            'prediction_error': state.prediction_error,
            'F_history': state.F_history[-20:]
        },

        # Action info
        'action': {
            'selected': action,
            'G': {str(k): float(v) for k, v in state.G.items()},
            'epistemic': {str(k): float(v) for k, v in state.epistemic.items()},
            'pragmatic': {str(k): float(v) for k, v in state.pragmatic.items()}
        },

        # Derived (for observers)
        'derived': {
            'information_rate': state.information_rate,
            'preference_divergence': state.preference_divergence,
            'belief_update': state.belief_update
        },

        # Human interpretation (NOT used by agent)
        'interpretation': explanation,

        # Outcome
        'outcome': outcome
    })


@app.post("/reset")
async def reset():
    """Reset world and agent."""
    global last_action, last_state
    world.reset()
    agent.reset()
    last_action = 0
    last_state = None
    return {'status': 'reset'}


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
