"""
Genesis Brain Simulation v2 - Unified Principles Architecture

This version replaces the fragmented module approach with:
- ONE cost function explaining all behavior
- Emergent emotions (not labels)
- Fast/slow pathways
- Viability instead of HP

Key changes from v1:
- homeostasis → viability (absorbing state concept)
- emotion labels → emergent states from prediction errors
- separate goals → unified cost minimization
- hardcoded fear → learned alarm patterns
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import random

# === CORE NEW ARCHITECTURE ===
from core import (
    ViabilitySystem,
    WorldModel,
    UnifiedValue,
    AlarmSystem,
    UnifiedPolicy,
    ActionDecision
)

# === KEEP USEFUL COMPONENTS ===
from network import Network
from environment import GridWorld
from agency import AgencyDetector
from working_memory import WorkingMemorySystem
from attention import AttentionSystem
from memory_ltm import LongTermMemory
from narrative import NarrativeSelf

app = FastAPI(title="Genesis Brain v2 - Unified Principles")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CORE SYSTEMS (NEW) ===
viability = ViabilitySystem()
world_model = WorldModel()
unified_value = UnifiedValue()
alarm = AlarmSystem()

# Policy integrates all core systems
policy = UnifiedPolicy(viability, world_model, alarm, unified_value)

# === SUPPORTING SYSTEMS (KEPT) ===
network = Network()
world = GridWorld(15, 15)
agency_detector = AgencyDetector()
working_memory = WorkingMemorySystem(decay=0.92, gain=0.4, threshold=0.15, soft_inhibition=0.05)
attention = AttentionSystem(base_gain=1.5, suppression=0.3, shift_threshold=0.4, focus_decay=0.95)
long_term_memory = LongTermMemory(max_episodes=100)
narrative_self = NarrativeSelf(history_window=200)

# === STATE ===
action_history = []
last_decision: Optional[ActionDecision] = None

# === NETWORK SETUP (simplified for v2) ===
for s_id in ["s_up", "s_down", "s_left", "s_right"]:
    network.add_neuron(s_id, neuron_type="RS")
for p_id in ["p_up", "p_down", "p_left", "p_right"]:
    network.add_neuron(p_id, neuron_type="RS")
for h_id in ["h_up", "h_down", "h_left", "h_right"]:
    network.add_neuron(h_id, neuron_type="RS")
for a_id in ["a_up", "a_down", "a_left", "a_right"]:
    network.add_neuron(a_id, neuron_type="RS")
network.add_neuron("gaba", neuron_type="FS")

# Basic connectivity (will be learned via STDP)
W_INIT = 40.0
for direction in ["up", "down", "left", "right"]:
    network.connect(f"s_{direction}", f"h_{direction}", weight=W_INIT)
    network.connect(f"h_{direction}", f"a_{direction}", weight=W_INIT)
    network.connect(f"p_{direction}", f"h_{direction}", weight=40.0, is_inhibitory=True)

# === API MODELS ===
class SimulationParams(BaseModel):
    currents: Optional[Dict[str, float]] = None
    noise_level: float = 2.0


@app.post("/network/step")
async def step_network(params: SimulationParams):
    """
    Main simulation step using unified principles architecture.
    """
    global last_decision

    # === GET SENSORS ===
    sensors = world.get_sensory_input()
    predator_info = world.get_predator_info()

    # === UPDATE VIABILITY (replaces homeostasis) ===
    viability_update = viability.update(
        ate_food=False,  # Will be updated after action
        took_damage=False,
        near_threat=predator_info.get('distance', 999) < 5 if predator_info else False,
        resting=False
    )

    # === WORLD MODEL PREDICTION ===
    # Before action: predict what will happen
    viability_metrics = viability.get_visualization_data()
    snn_preferences = _get_snn_preferences(sensors)

    # === SELECT ACTION (unified policy) ===
    predator_context = None
    if predator_info:
        predator_context = {
            'position': predator_info.get('position'),
            'agent_pos': world.agent_pos,
            'distance': predator_info.get('distance', 999)
        }

    decision = policy.select_action(
        sensory=sensors,
        predator_info=predator_context,
        snn_preferences=snn_preferences
    )
    last_decision = decision

    # === EXECUTE ACTION ===
    action_code = decision.action_code
    reward, was_reset, collision_info = world.move_agent(action_code)

    # === UPDATE SYSTEMS BASED ON OUTCOME ===
    ate_food = collision_info.get('ate_food')
    took_damage = collision_info.get('predator_damage', False)
    hit_wall = collision_info.get('hit_wall', False)

    # Update viability with actual outcome
    if ate_food:
        viability.on_food()
    if took_damage:
        viability.on_damage(0.25)
        alarm.on_damage('predator')

    # Update world model with actual outcome
    prediction_error = world_model.compute_error(
        actual_sensory=world.get_sensory_input(),
        actual_viability=viability.get_visualization_data(),
        actual_reward=reward,
        hit_wall=hit_wall,
        took_damage=took_damage
    )

    # === ATTENTION based on alarm ===
    alarm_data = alarm.get_visualization_data()
    if alarm_data['is_active']:
        # Narrow attention to threat
        attention.on_stimulus('threat', alarm_data['level'])

    # === MEMORY RECORDING ===
    if ate_food or took_damage or was_reset:
        # Record significant events
        narrative_self.record_event(
            'significant_event',
            1.0 if ate_food else -1.0,
            {'step': world.total_lifetime_steps}
        )

    # === PREPARE RESPONSE ===
    emergent_state = world_model.get_emergent_state(viability_metrics)

    response = {
        'trajectories': _get_trajectories(),
        'synapses': _get_synapses(),
        'world': _get_world_state(reward, was_reset),

        # === NEW V2 DATA ===
        'viability': viability.get_visualization_data(),
        'emergent_state': emergent_state,  # Replaces emotion labels
        'alarm': alarm.get_visualization_data(),
        'unified_value': unified_value.get_visualization_data(
            viability.get_visualization_data(),
            world_model.get_prediction_error_summary(),
            emergent_state
        ),
        'policy': policy.get_visualization_data(decision),
        'world_model': world_model.get_visualization_data(),

        # === KEPT FROM V1 ===
        'agency': agency_detector.to_dict(),
        'attention': attention.get_visualization_data(),
        'narrative': narrative_self.get_visualization_data(),

        # Action info
        'action': {
            'selected': decision.action,
            'source': decision.source,
            'explanation': decision.explanation,
            'confidence': decision.confidence,
            'scores': decision.scores
        }
    }

    return response


def _get_snn_preferences(sensors: Dict) -> Dict[str, float]:
    """Get SNN's learned preferences for each direction."""
    # Placeholder - in full integration, this would run SNN
    preferences = {}
    for direction in ['up', 'down', 'left', 'right']:
        # Higher sensor value = stronger preference
        preferences[direction] = sensors.get(f'food_{direction}', 0)
    return preferences


def _get_trajectories() -> Dict:
    """Get neuron trajectory data."""
    return {nid: [] for nid in network.neurons}


def _get_synapses() -> list:
    """Get synapse data."""
    return [
        {'pre': syn.pre_neuron_id, 'post': syn.post_neuron_id, 'weight': syn.weight}
        for syn in network.synapses
    ][:15]


def _get_world_state(reward: float, died: bool) -> Dict:
    """Get world state for frontend."""
    return {
        'agent_pos': world.agent_pos,
        'food_pos': world.food_pos,
        'predator': world.predator.pos if world.predator else None,
        'energy': viability.state.energy * 100,  # Convert to percentage
        'reward': reward,
        'died': died,
        'step': world.total_lifetime_steps
    }


@app.post("/network/reset")
async def reset_network():
    """Reset all systems."""
    global last_decision

    viability.reset()
    world_model.reset()
    unified_value.reset()
    alarm.reset()
    alarm.clear_learning()
    world.reset()
    working_memory.reset()
    attention.reset()
    long_term_memory.clear()
    narrative_self.clear()
    last_decision = None

    return {"status": "reset", "message": "All systems reset to initial state"}


@app.get("/architecture/info")
async def get_architecture_info():
    """Return information about the v2 architecture."""
    return {
        'version': '2.0',
        'name': 'Unified Principles Architecture',
        'core_systems': [
            {
                'name': 'Viability',
                'replaces': 'Homeostasis + HP',
                'concept': 'Distance from absorbing state (death = loss of agency)'
            },
            {
                'name': 'World Model',
                'replaces': 'Separate emotion labels',
                'concept': 'Prediction + error → emergent emotions'
            },
            {
                'name': 'Unified Value',
                'replaces': 'Separate goals (SAFE/FEED/REST)',
                'concept': 'One cost function explains all behavior'
            },
            {
                'name': 'Alarm',
                'replaces': 'Hardcoded fear',
                'concept': 'Fast pathway - learned threat patterns → instant response'
            },
            {
                'name': 'Policy',
                'replaces': 'Goal-based action selection',
                'concept': 'Cost minimization with fast/slow integration'
            }
        ],
        'philosophy': {
            'death': 'Absorbing state where future agency = 0',
            'fear': 'Emerges from high p_absorb → policy change',
            'curiosity': 'Emerges from high uncertainty + low threat',
            'satisfaction': 'Emerges from decreasing prediction errors'
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port for v2
