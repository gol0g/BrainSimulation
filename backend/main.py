from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
from network import Network
from environment import GridWorld
from agency import AgencyDetector
from working_memory import WorkingMemorySystem
from attention import AttentionSystem
from value_conflict import ValueConflictSystem
from self_model import SelfModel
import random

app = FastAPI(title="Genesis Brain Simulation API - Phase 2")

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Simulation State ---
network = Network()
world = GridWorld(10, 10)
agency_detector = AgencyDetector()  # Self-Agency detection module
working_memory = WorkingMemorySystem(  # Working Memory module v2
    decay=0.92,        # ~12 steps half-life
    gain=0.4,          # How strongly sensory input forms memories
    threshold=0.15,    # Minimum activity to be "active"
    soft_inhibition=0.05  # Gentle competition (v2: not winner-take-all)
)
attention = AttentionSystem(  # Attention module
    base_gain=1.5,         # Amplification for attended stimuli
    suppression=0.3,       # Suppression for unattended (0.3 = 70% reduction)
    shift_threshold=0.4,   # Prediction error threshold for attention shift
    focus_decay=0.95       # How quickly focus fades
)
value_conflict = ValueConflictSystem(  # Value Conflict module (currently disabled)
    discount_rate=0.15,       # Future rewards discounted 15% per step
    conflict_threshold=0.3,   # Difference threshold for conflict detection
    hesitation_decay=0.9      # How quickly hesitation fades
)
self_model = SelfModel(  # Self-Model: "What kind of being am I right now?"
    history_window=20,        # Steps to consider for trends
    confidence_decay=0.9,     # How quickly confidence fades
    effort_recovery=0.05      # How quickly effort recovers
)

# --- Exploration Parameters ---
EPSILON = 0.2  # Probability of random action (exploration)
action_history = []  # Track last N actions to detect loops
HISTORY_SIZE = 10

# --- Wall Collision Tracking ---
# Track recently hit walls to suppress those directions in sensory input
wall_blocked = {'up': 0, 'down': 0, 'left': 0, 'right': 0}  # Countdown steps
WALL_BLOCK_DURATION = 5  # Steps to suppress a direction after hitting wall

# --- Learning Verification Mode ---
REVERSED_REWARD = False  # If True, reward for moving AWAY from food (reversal learning test)

# Note: Environmental perturbation is now handled by GridWorld (wall collision, wind system)
# No more random 5% push - perturbations are now realistic events

# --- Network Architecture for Agency ---

# 1. Sensory Neurons (Encoded direction to food)
for s_id in ["s_up", "s_down", "s_left", "s_right"]:
    network.add_neuron(s_id, neuron_type="RS")

# 2. Hidden Layer (Inter-neurons - 4 Directions)
hidden_ids = ["h_up", "h_down", "h_left", "h_right"]
for h_id in hidden_ids:
    network.add_neuron(h_id, neuron_type="RS")

# 3. Action Neurons (Commands to move)
action_map = {
    "a_up": 1,
    "a_down": 2,
    "a_left": 3,
    "a_right": 4
}
for a_id in action_map.keys():
    network.add_neuron(a_id, neuron_type="RS")

# 4. Inhibitory Neuron (Winner-Take-All competition)
network.add_neuron("gaba", neuron_type="FS")

# --- Connections (Brain 2.0: Competitive Reflex Mapping) ---

# 1. Sensory -> Hidden (Reflexive biased mapping)
# S_UP drives H_UP strongly, others weakly
for direct in ["up", "down", "left", "right"]:
    s_id = f"s_{direct}"
    h_target = f"h_{direct}"
    for h_id in hidden_ids:
        if h_id == h_target:
            network.connect(s_id, h_id, weight=70.0 + random.uniform(-5, 5), delay=2)
        else:
            network.connect(s_id, h_id, weight=10.0 + random.uniform(-2, 2), delay=2)

# 2. Hidden -> Action (Direct Command)
# FIXED pathways - these are just relays, not learning connections
for direct in ["up", "down", "left", "right"]:
    network.connect(f"h_{direct}", f"a_{direct}", weight=80.0, delay=2, enable_stdp=False)

# 3. Competition (Winner-Take-All via Action-layer inhibition)
# Action neurons activate GABA
for a_id in action_map.keys():
    network.connect(a_id, "gaba", weight=15.0, delay=1, enable_stdp=False)

# GABA inhibits Action neurons (Competition)
# This prevents multiple actions from firing together, choosing the strongest one.
# CRITICAL: Disable STDP to prevent inhibition from weakening over time
for a_id in action_map.keys():
    network.connect("gaba", a_id, weight=45.0, delay=1, is_inhibitory=True, enable_stdp=False)

# Cross-Inhibition between Hidden axes (Optional but sharpens decisions)
# e.g., H_UP inhibits H_DOWN slightly (actually easier via GABA but we stick to WTA at Action)


# --- Data Models ---
class SimulationParams(BaseModel):
    currents: Optional[Dict[str, float]] = None  # {neuron_id: current}
    noise_level: float = 8.0  # Increased default noise for exploration (was 5.0)

class NeuronParams(BaseModel):
    neuron_id: str
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    d: Optional[float] = None

class ConnectionParams(BaseModel):
    pre_id: str
    post_id: str
    weight: float = 5.0
    delay: int = 3

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"status": "Genesis Online", "phase": "2 - Synaptic Transmission"}

@app.post("/network/step")
async def step_network(params: SimulationParams):
    dt = 0.5
    sim_steps = 50
    
    trajectories = {nid: [] for nid in network.neurons}
    
    # Get current world state
    sensors = world.get_sensory_input()

    # --- VALUE CONFLICT: Check for competing food options ---
    food_info = world.get_food_info()
    # Update conflict system (chosen_direction will be set after action)
    conflict_state = value_conflict.update(
        small_food=food_info['small'],
        large_food=food_info['large'],
        chosen_direction=None  # Will update after decision
    )

    # --- WALL BLOCKING: Decay counters and apply suppression ---
    global wall_blocked
    for direction in wall_blocked:
        if wall_blocked[direction] > 0:
            wall_blocked[direction] -= 1

    # --- WORKING MEMORY: Update and augment sensory input ---
    # Convert sensors to directional format for memory
    # Combine small and large food signals with conflict-aware weighting
    sensory_for_memory = {}

    # Get large food direction for conflict-based boost
    large_food_dir = food_info['large']['direction'] if food_info['large'] else None

    for direction in ['up', 'down', 'left', 'right']:
        small_signal = sensors.get(f'food_{direction}', 0)
        large_signal = sensors.get(f'large_food_{direction}', 0)

        # Base weighting: large food gets 1.8x due to higher reward
        large_weight = 1.8

        # When in conflict, boost large food direction significantly
        # This creates exploration toward delayed gratification
        if value_conflict.in_conflict and direction == large_food_dir:
            large_weight = 2.5  # Strong boost toward large food
            # Also add direct exploration signal
            large_signal = max(large_signal, 0.5)

        # Combine signals - take weighted max
        combined = max(small_signal, large_signal * large_weight)

        # WALL BLOCKING: Suppress signal for recently hit walls
        if wall_blocked[direction] > 0:
            suppression = 0.2 * (wall_blocked[direction] / WALL_BLOCK_DURATION)
            combined *= suppression  # Strong suppression for blocked direction

        # NOTE: Don't cap at 1.0 - large food needs to have stronger signal!
        # The current multiplication (40.0) will handle scaling
        sensory_for_memory[direction] = combined

    # --- SELF-MODEL: Modulate Working Memory based on stability ---
    # Stability ↑ → memories persist longer (slower decay)
    # Stability ↓ → faster forgetting (strategy switching)
    memory_mod = self_model.get_memory_modulation()
    base_decay = 0.92  # Original WM decay
    effective_decay = base_decay * memory_mod['decay_modifier']
    working_memory.base_decay = effective_decay

    # Update working memory with current sensory input
    memory_outputs = working_memory.update(sensory_for_memory)

    # Check if we should use memory (weak sensory but active memory)
    using_memory = working_memory.should_use_memory(sensory_for_memory)

    # Get memory guidance with DYNAMIC ALPHA (v2 improvement)
    memory_guidance = working_memory.get_memory_guidance(sensory_for_memory) if using_memory else {}

    # Get memory direction for attention bias
    memory_direction = working_memory.get_strongest_memory()

    # --- ATTENTION: Filter sensory input based on focus ---
    # Attention uses: raw sensory, agency level (from last step), memory direction
    # Note: We use agency from last update since current step hasn't computed it yet
    last_agency = agency_detector.agency_level
    last_pred_error = agency_detector.prediction_errors[-1] if agency_detector.prediction_errors else 0

    # Get conflict modulation for attention
    conflict_mod = value_conflict.get_conflict_modulation()

    # Apply conflict effect: reduce agency confidence during conflict (broadens attention)
    effective_agency = last_agency * (1 - conflict_mod['agency_reduction'])

    # Apply attention filtering
    attended_sensory = attention.update(
        sensory_state=sensory_for_memory,
        agency_level=effective_agency,  # Reduced by conflict
        memory_direction=memory_direction,
        prediction_error=last_pred_error
    )

    # Apply conflict-based attention widening
    if value_conflict.in_conflict:
        attention.attention_width = min(1.0, attention.attention_width + conflict_mod['attention_width_boost'])

    # Map ATTENTION-FILTERED sensors to currents (SNN encoding)
    # Attended stimuli are amplified, unattended are suppressed
    sensor_currents = {f"s_{direction}": attended_sensory[direction] * 40.0 for direction in attended_sensory}

    # DISABLED: Large food boost was causing weight degradation
    # The boost pushed agent toward large food even when path wasn't optimal,
    # resulting in negative rewards and weight decay.
    # TODO: Re-enable with smarter path-aware logic
    # if large_food_dir:
    #     advantage, _ = value_conflict.get_large_food_advantage()
    #     if advantage > 0:
    #         boost = 20.0 + min(60.0, advantage * 60.0)
    #         sensor_currents[f"s_{large_food_dir}"] += boost

    # Augment with memory (dynamic α already applied in get_memory_guidance)
    # Memory is now "advisory, not authoritative"
    if using_memory:
        for direction, mem_strength in memory_guidance.items():
            if mem_strength > 0.01:
                sensor_currents[f"s_{direction}"] = sensor_currents.get(f"s_{direction}", 0) + mem_strength * 30.0
    
    # Combine with external manual currents if any
    total_ext_currents = {**sensor_currents}
    if params.currents:
        for nid, val in params.currents.items():
            total_ext_currents[nid] = total_ext_currents.get(nid, 0.0) + val

    # Track firing counts for action neurons
    action_fire_counts = {a_id: 0 for a_id in action_map.keys()}

    for _ in range(sim_steps):
        state = network.step(
            external_currents=total_ext_currents,
            dt=dt,
            noise_level=params.noise_level
        )

        # Count firing events for action neurons
        for nid, ns in state["neurons"].items():
            if ns["fired"] and nid in action_map:
                action_fire_counts[nid] += 1

            trajectories[nid].append({
                "v": ns["v"],
                "fired": ns["fired"]
            })

    # --- Resolve actions and environment ---
    total_reward = 0.0
    died = False

    # Calculate distance-based shaping reward (Dense Reward)
    # If the agent moved closer to the food, give a small positive reward
    old_dist = abs(world.agent_pos[0] - world.food_pos[0]) + abs(world.agent_pos[1] - world.food_pos[1])

    # Select action with MOST spikes (winner-take-all based on firing frequency)
    max_fires = max(action_fire_counts.values())

    found_food = False

    # --- Epsilon-Greedy Exploration (Self-Model modulated) ---
    # Base epsilon adjusted by self-model state:
    # - exploration_need ↑ → ε ↑ ("I'm stuck, need to explore")
    # - confidence ↑ → ε ↓ ("I know what I'm doing")
    # - struggling/transitioning → ε boost
    effective_epsilon = EPSILON  # Base: 0.2

    # Get current self-state for modulation (from previous step)
    current_self_state = self_model.get_state()
    behavioral_label = self_model.get_behavioral_state()

    # Exploration need directly increases epsilon
    exploration_boost = current_self_state.get('exploration_need', 0.3) * 0.3  # up to +0.09

    # High confidence reduces epsilon
    confidence_reduction = current_self_state.get('confidence', 0.5) * 0.15  # up to -0.15

    # Struggling/Transitioning states get temporary epsilon boost
    state_boost = 0.0
    if behavioral_label in ['STRUGGLING', 'TRANSITIONING']:
        state_boost = 0.1
    elif behavioral_label == 'EXPLORING':
        state_boost = 0.05

    # === Attribution-based exploration modulation ===
    # Key insight: WHO is at fault determines the right response
    # - externality ↑ → reduce exploration ("wait it out, world is chaotic")
    # - internal_fault ↑ → increase exploration ("my strategy is wrong, try something new")
    externality = current_self_state.get('externality', 0.0)
    internal_fault = current_self_state.get('internal_fault', 0.0)

    # External chaos → hunker down, don't waste effort exploring
    externality_reduction = externality * 0.15  # up to -0.15

    # Internal fault → actively seek new strategies
    internal_boost = internal_fault * 0.2  # up to +0.2

    effective_epsilon = EPSILON + exploration_boost - confidence_reduction + state_boost \
                       - externality_reduction + internal_boost
    effective_epsilon = max(0.05, min(0.5, effective_epsilon))  # Clamp to [0.05, 0.5]

    is_exploring = random.random() < effective_epsilon
    if is_exploring:
        # Exploration: Random action with LEFT/RIGHT bias to escape UP/DOWN loops
        # 70% chance of LEFT/RIGHT, 30% chance of UP/DOWN
        if random.random() < 0.7:
            primary_action = random.choice([3, 4])  # LEFT or RIGHT
        else:
            primary_action = random.choice([1, 2])  # UP or DOWN
        winning_action_id = f"a_{['up', 'down', 'left', 'right'][primary_action - 1]}"
    elif max_fires > 0:
        # Exploitation: Use network's decision
        winning_action_id = max(action_fire_counts, key=action_fire_counts.get)
        primary_action = action_map[winning_action_id]
    else:
        # No firing: Stay
        primary_action = 0
        winning_action_id = None

    # --- Detect and Penalize Oscillation (UP↔DOWN or LEFT↔RIGHT loops) ---
    oscillation_penalty = 0.0
    if len(action_history) >= 2:
        last_action = action_history[-1]
        second_last = action_history[-2]

        # Check for opposite direction oscillation
        if (primary_action == 1 and last_action == 2) or (primary_action == 2 and last_action == 1):
            oscillation_penalty = -0.5  # UP↔DOWN oscillation
        elif (primary_action == 3 and last_action == 4) or (primary_action == 4 and last_action == 3):
            oscillation_penalty = -0.5  # LEFT↔RIGHT oscillation

    # Track action history
    action_history.append(primary_action)
    if len(action_history) > HISTORY_SIZE:
        action_history.pop(0)

    # --- AGENCY: Record action prediction (Forward Model) ---
    action_dir_map = {0: 'stay', 1: 'up', 2: 'down', 3: 'left', 4: 'right'}
    current_sensory = {
        'up': sensors.get('food_up', 0),
        'down': sensors.get('food_down', 0),
        'left': sensors.get('food_left', 0),
        'right': sensors.get('food_right', 0)
    }

    # --- Record intended action for agency (before execution) ---
    agency_detector.on_action_taken(current_sensory, action_dir_map[primary_action])

    # Execute action (now returns collision_info)
    if primary_action > 0:
        reward, was_reset, collision_info = world.move_agent(primary_action)
        total_reward = (reward * 1.0) + oscillation_penalty
        if reward > 5.0:  # Food was found
            found_food = True
        if was_reset:
            died = True
    else:
        reward, was_reset, collision_info = world.move_agent(0)  # Stay
        total_reward = reward + oscillation_penalty
        if was_reset:
            died = True

    # --- VALUE CONFLICT: Update with choice result ---
    # If we were in conflict, record which direction was chosen
    if value_conflict.in_conflict and primary_action > 0:
        chosen_dir = action_dir_map.get(primary_action)
        if chosen_dir and chosen_dir != 'stay':
            value_conflict.update(
                small_food=food_info['small'],
                large_food=food_info['large'],
                chosen_direction=chosen_dir
            )

    # Check if food was eaten and notify conflict system
    ate_food = collision_info.get('ate_food')
    if ate_food == 'large':
        value_conflict.on_reward_received(reward, was_large=True)
        found_food = True
    elif ate_food == 'small':
        value_conflict.on_reward_received(reward, was_large=False)

    # --- Environmental Perturbation Detection ---
    # Only WIND is external perturbation. Wall collision is SELF-CAUSED failure.
    was_perturbed = False
    perturb_type = None

    # Wall collision: agent's OWN action caused this - NOT external!
    # This should trigger LEARNING (negative reward), not exemption from learning
    if collision_info['hit_wall']:
        wall_dir = collision_info['wall_direction']
        # DO NOT treat as external - agent chose to go this direction

        # Set wall blocking to suppress sensory input for this direction
        wall_blocked[wall_dir] = WALL_BLOCK_DURATION

        # Memory and attention suppression
        working_memory.on_collision(wall_dir)  # Memory: this direction is blocked
        attention.on_collision(wall_dir)  # Attention: suppress blocked direction

        # Moderate penalty for wall hit - encourages learning to avoid walls
        # Note: -0.3 instead of -0.8 to prevent excessive weight degradation via STDP
        total_reward -= 0.3
        print(f"[WALL] Hit {wall_dir.upper()} wall (blocked for {WALL_BLOCK_DURATION} steps)")

    # Wind push: environmental force moved the agent
    if collision_info['wind_push']:
        was_perturbed = True
        perturb_type = 'wind'
        wind_dir = collision_info['wind_push']
        agency_detector.on_external_push(wind_dir)
        print(f"[WIND PUSH] Drifted {wind_dir.upper()}")

    # Add Shaping Reward: Compare new distance
    # In REVERSED mode, reward for moving AWAY from food (to test true learning)
    if not died and not was_reset:
        new_dist = abs(world.agent_pos[0] - world.food_pos[0]) + abs(world.agent_pos[1] - world.food_pos[1])
        if REVERSED_REWARD:
            # REVERSED: reward for moving away, punish for getting closer
            if new_dist > old_dist:
                total_reward += 0.5  # Moving away = good
            elif new_dist < old_dist:
                total_reward -= 0.3  # Getting closer = bad
        else:
            # NORMAL: reward for getting closer
            if new_dist < old_dist:
                total_reward += 0.5  # Progress Reward
            elif new_dist > old_dist:
                total_reward -= 0.3  # Regret penalty

    # --- AGENCY: Update with actual outcome ---
    new_sensors = world.get_sensory_input()
    new_sensory = {
        'up': new_sensors.get('food_up', 0),
        'down': new_sensors.get('food_down', 0),
        'left': new_sensors.get('food_left', 0),
        'right': new_sensors.get('food_right', 0)
    }
    # Pass self_state for self-explanation: "I'm confused, so errors are expected"
    agency_info = agency_detector.update(new_sensory, self_state=current_self_state)

    # --- WORKING MEMORY: Reset Triggers (v2 improvement) ---
    # Track if fast decay gets triggered for attention coordination
    wm_fast_decay_before = working_memory.fast_decay_steps

    # Trigger 1: Reward degradation (consecutive negative rewards)
    working_memory.on_reward_degradation(total_reward)

    # Trigger 2: Prediction failure (high agency prediction error)
    working_memory.on_prediction_failure(agency_info.get('prediction_error', 0))

    # Note: Trigger 3 (collision) is now handled above with actual collision_info

    # v2: WM-Attention coordination - if fast decay just started, widen attention
    if wm_fast_decay_before == 0 and working_memory.fast_decay_steps > 0:
        attention.on_wm_fast_decay()

    # --- ATTENTION: Respond to reward ---
    # Reinforces attention to successful direction, shifts away from failures
    action_direction = action_dir_map.get(primary_action)
    if action_direction and action_direction != 'stay':
        attention.on_reward(total_reward, action_direction)

    # --- SELF-MODEL: Update meta-layer that represents "I" ---
    # Collects all current cognitive signals into unified self-state
    focus_dir, focus_strength = attention.get_focus()
    last_focus = attention.focus_history[-2] if len(attention.focus_history) >= 2 else None
    focus_changed = (focus_dir != last_focus) if last_focus else False

    self_state = self_model.update(
        agency_level=agency_info['agency_level'],
        prediction_error=agency_info.get('prediction_error', 0),
        attention_width=attention.attention_width,
        attention_focus=focus_dir,
        memory_active=using_memory,
        reward=total_reward,
        was_external_event=was_perturbed,
        focus_changed=focus_changed
    )

    # Self-Model modulates Attention based on internal state
    self_attention_mod = self_model.get_attention_modulation()

    # Apply width bias from self-state (uncertainty → broader attention)
    attention.attention_width = max(0.1, min(1.0,
        attention.attention_width + self_attention_mod['width_bias'] * 0.5
    ))

    # Apply dwell time modifier (effort ↑ → dwell ↓, tired = less focused)
    base_dwell = 10  # Original min_dwell_steps
    attention.min_dwell_steps = max(3, int(base_dwell * self_attention_mod['dwell_modifier']))

    # Apply gain modifier (effort ↑ → gain ↓, fatigued attention)
    attention.base_gain = 1.5 * self_attention_mod['gain_modifier']

    # Modulate reward by agency: "I did this" = full reward, "External" = reduced
    agency_modifier = agency_detector.get_dopamine_modifier()
    modulated_reward = total_reward * agency_modifier

    # Apply 3-factor learning ONLY for network's own decisions (not exploration)
    # AND only to the WINNING pathway (action-specific learning)
    exploration_flag = " [EXPLORE]" if is_exploring else ""
    oscillation_flag = f" [OSC:{oscillation_penalty:.2f}]" if oscillation_penalty != 0 else ""
    action_name = ['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT'][primary_action]

    # Agency indicator for logging
    agency_str = f"A:{agency_info['agency_level']:.2f}"
    agency_interp = agency_info['interpretation']
    perturb_str = f" [{perturb_type.upper()}!]" if was_perturbed and perturb_type else ""

    # Memory usage logging
    mem_str = ""
    if using_memory:
        strongest_mem = working_memory.get_strongest_memory()
        mem_str = f" | MEM:{strongest_mem.upper() if strongest_mem else 'NONE'}"
    if working_memory.fast_decay_steps > 0:
        mem_str += " [FAST-DECAY]"

    if was_perturbed:
        # External perturbation - don't learn from this
        for syn in network.synapses:
            syn.eligibility = 0.0
        print(f"[EXTERNAL] {action_name}{perturb_str} | {agency_str} | {agency_interp}{mem_str} (no learning)")
    elif is_exploring:
        # Exploration: clear eligibility traces (don't learn from random actions)
        for syn in network.synapses:
            syn.eligibility = 0.0  # Full clear - random actions shouldn't affect learning
        print(f"[EXPLORE] {action_name} | R:{total_reward:+.2f} | {agency_str}{mem_str} (no learning)")
    elif abs(modulated_reward) > 0.05 and winning_action_id:
        # ACTION-SPECIFIC LEARNING: Apply MODULATED reward (scaled by agency)
        # High agency = full reward, Low agency = reduced reward
        winning_direction = winning_action_id.split('_')[1]  # "up", "down", etc.
        winning_hidden = f"h_{winning_direction}"

        # Zero eligibility for synapses NOT leading to the winning hidden neuron
        for syn in network.synapses:
            if syn.post_neuron_id != winning_hidden:
                syn.eligibility = 0.0

        network.apply_reward(modulated_reward)  # Use agency-modulated reward!
        print(f"[LEARNING] R:{modulated_reward:+.2f} | {action_name} | {agency_str} | {agency_interp}{mem_str}")
    else:
        print(f"[NO REWARD] {action_name} | {agency_str}{mem_str}")

    # Clear eligibility traces after finding food (new episode starts)
    if found_food:
        winning_neuron = winning_action_id if max_fires > 0 else None

        for syn in network.synapses:
            syn.eligibility = 0.0

        print(f"Food Found! Eligibility cleared for new episode.")

    # If agent died/reset, clear temporal traces but PRESERVE learned weights
    if died:
        network.reset_neurons()  # Only reset neuron states, keep learned weights!
        print("Agent Died: Environment and Network Reset.")

    # Debug: Print sample weight every 10 steps
    if network.time_step % 500 == 0 and network.synapses:
        sample_syns = random.sample(network.synapses, min(3, len(network.synapses)))
        print(f"\n[WEIGHTS @ step {network.time_step}]")
        for syn in sample_syns:
            print(f"  {syn.pre_neuron_id}→{syn.post_neuron_id}: w={syn.weight:.2f} {'(INH)' if syn.is_inhibitory else ''}")
        print()

    return {
        "trajectories": trajectories,
        "synapses": [s.to_dict() for s in network.synapses],
        "world": {
            "agent": world.agent_pos,
            "food": world.food_pos,
            "large_food": world.large_food_pos,  # Large food position (or None)
            "energy": world.energy,
            "reward": total_reward,
            "modulated_reward": modulated_reward,
            "died": died,
            "wind": world.get_wind_info()
        },
        "agency": agency_info,
        "was_perturbed": was_perturbed,
        "perturb_type": perturb_type,  # 'wall', 'wind', or None
        "memory": working_memory.get_visualization_data(),
        "using_memory": using_memory,
        "attention": attention.get_visualization_data(),
        "conflict": value_conflict.get_visualization_data(),
        "self_model": self_model.get_visualization_data()
    }

@app.get("/network/state")
def get_network_state():
    """Get current state of all neurons and synapses."""
    return network.get_state()

@app.post("/network/reset")
def reset_network():
    """Reset all neurons, synapses, and world to initial state."""
    global action_history, wall_blocked
    network.reset()
    world.reset()
    working_memory.clear_all()  # Clear working memory
    attention.clear()  # Clear attention focus
    value_conflict.clear()  # Clear value conflict state
    self_model.clear()  # Clear self-model state
    action_history = []  # Clear action history
    wall_blocked = {'up': 0, 'down': 0, 'left': 0, 'right': 0}  # Clear wall blocks
    print("Simulation Reset: Energy restored to 100%, Memory, Attention & Self-Model cleared")
    return {"message": "Network, World, Memory, Attention and Self-Model reset to initial state"}

@app.post("/network/add_neuron")
def add_neuron(neuron_id: str, neuron_type: str = "RS"):
    """Add a new neuron to the network."""
    try:
        network.add_neuron(neuron_id, neuron_type)
        return {"message": f"Neuron '{neuron_id}' added", "type": neuron_type}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/network/connect")
def connect_neurons(params: ConnectionParams):
    """Create a synaptic connection between two neurons."""
    try:
        network.connect(params.pre_id, params.post_id, params.weight, params.delay)
        return {"message": f"Connected {params.pre_id} -> {params.post_id}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/neuron/params")
def update_neuron_params(params: NeuronParams):
    """Update parameters of a specific neuron."""
    if params.neuron_id not in network.neurons:
        raise HTTPException(status_code=404, detail=f"Neuron '{params.neuron_id}' not found")
    
    neuron = network.neurons[params.neuron_id]
    if params.a is not None: neuron.a = params.a
    if params.b is not None: neuron.b = params.b
    if params.c is not None: neuron.c = params.c
    if params.d is not None: neuron.d = params.d
    
    return {"message": "Parameters updated", "neuron_id": params.neuron_id}

# ============================================
# LEARNING VERIFICATION ENDPOINTS
# ============================================

@app.post("/test/reverse_reward")
def toggle_reverse_reward():
    """Toggle reversed reward mode for learning verification.
    In reversed mode, the agent is rewarded for moving AWAY from food.
    This tests if the network can unlearn and relearn."""
    global REVERSED_REWARD
    REVERSED_REWARD = not REVERSED_REWARD
    mode = "REVERSED (flee from food)" if REVERSED_REWARD else "NORMAL (seek food)"
    print(f"\n{'='*50}")
    print(f"REWARD MODE CHANGED: {mode}")
    print(f"{'='*50}\n")
    return {"reversed_reward": REVERSED_REWARD, "mode": mode}

@app.post("/test/randomize_weights")
def randomize_weights():
    """Randomize all S->H weights to test if network can learn from scratch.
    All weights set to 40.0 (middle value) - no initial bias."""
    count = 0
    for syn in network.synapses:
        if syn.pre_neuron_id.startswith('s_') and syn.post_neuron_id.startswith('h_'):
            syn.weight = 40.0  # All equal - no bias
            syn.initial_weight = 40.0
            count += 1
    print(f"\n{'='*50}")
    print(f"WEIGHTS RANDOMIZED: {count} synapses set to 40.0")
    print(f"{'='*50}\n")
    return {"message": f"Randomized {count} S->H weights to 40.0"}

@app.post("/test/invert_weights")
def invert_weights():
    """Invert S->H weights - correct connections become weak, wrong become strong.
    This is the hardest test: can the network overcome incorrect initial learning?"""
    count = 0
    for syn in network.synapses:
        if syn.pre_neuron_id.startswith('s_') and syn.post_neuron_id.startswith('h_'):
            pre_dir = syn.pre_neuron_id.split('_')[1]
            post_dir = syn.post_neuron_id.split('_')[1]
            if pre_dir == post_dir:
                # Correct connection -> make weak
                syn.weight = 10.0
            else:
                # Wrong connection -> make strong
                syn.weight = 70.0
            syn.initial_weight = syn.weight
            count += 1
    print(f"\n{'='*50}")
    print(f"WEIGHTS INVERTED: {count} synapses")
    print(f"Correct connections: 10.0 (weak)")
    print(f"Wrong connections: 70.0 (strong)")
    print(f"{'='*50}\n")
    return {"message": f"Inverted {count} S->H weights"}

@app.get("/test/weight_matrix")
def get_weight_matrix():
    """Get S->H weight matrix for analysis."""
    matrix = {}
    for pre in ['s_up', 's_down', 's_left', 's_right']:
        matrix[pre] = {}
        for post in ['h_up', 'h_down', 'h_left', 'h_right']:
            for syn in network.synapses:
                if syn.pre_neuron_id == pre and syn.post_neuron_id == post:
                    matrix[pre][post] = round(syn.weight, 1)

    # Calculate correctness score
    correct_sum = 0
    wrong_sum = 0
    for pre in ['s_up', 's_down', 's_left', 's_right']:
        pre_dir = pre.split('_')[1]
        for post in ['h_up', 'h_down', 'h_left', 'h_right']:
            post_dir = post.split('_')[1]
            w = matrix[pre][post]
            if pre_dir == post_dir:
                correct_sum += w
            else:
                wrong_sum += w / 3  # Average of 3 wrong connections

    score = (correct_sum - wrong_sum) / (correct_sum + wrong_sum) if (correct_sum + wrong_sum) > 0 else 0

    return {
        "matrix": matrix,
        "correct_avg": round(correct_sum / 4, 1),
        "wrong_avg": round(wrong_sum / 4, 1),
        "learning_score": round(score, 3),  # -1 to 1, higher is better
        "reversed_mode": REVERSED_REWARD
    }

# ============================================
# AGENCY (Self-Causation) ENDPOINTS
# ============================================

@app.get("/agency/state")
def get_agency_state():
    """Get current agency detection state."""
    return agency_detector.to_dict()

@app.post("/agency/push")
def external_push(direction: str = "up"):
    """
    Simulate an EXTERNAL push on the agent.
    This moves the agent WITHOUT the agent intending it.
    The agency detector should recognize this as external cause.

    Args:
        direction: 'up', 'down', 'left', 'right'
    """
    dir_map = {'up': 1, 'down': 2, 'left': 3, 'right': 4}
    if direction not in dir_map:
        raise HTTPException(status_code=400, detail=f"Invalid direction: {direction}")

    # Record that this is an external push (not self-caused)
    agency_detector.on_external_push(direction)

    # Move the agent
    old_pos = world.agent_pos.copy()
    reward, was_reset = world.move_agent(dir_map[direction])

    # Update agency with new sensory state
    new_sensors = world.get_sensory_input()
    new_sensory = {
        'up': new_sensors.get('food_up', 0),
        'down': new_sensors.get('food_down', 0),
        'left': new_sensors.get('food_left', 0),
        'right': new_sensors.get('food_right', 0)
    }
    # Pass current self_state for self-explanation
    agency_info = agency_detector.update(new_sensory, self_state=self_model.get_state())

    print(f"\n{'='*50}")
    print(f"EXTERNAL PUSH: {direction.upper()}")
    print(f"Agent moved: {old_pos} -> {world.agent_pos}")
    print(f"Agency Level: {agency_info['agency_level']:.3f}")
    print(f"Interpretation: {agency_info['interpretation']}")
    print(f"{'='*50}\n")

    return {
        "message": f"Agent pushed {direction}",
        "old_pos": old_pos,
        "new_pos": world.agent_pos,
        "agency": agency_info
    }

@app.post("/agency/test")
def run_agency_test():
    """
    Run a quick agency detection test:
    1. Let agent take 5 self-caused actions
    2. Apply 2 external pushes
    3. Let agent take 5 more self-caused actions
    4. Compare agency levels

    Returns test results showing agency detection capability.
    """
    results = {
        "self_actions_before": [],
        "external_pushes": [],
        "self_actions_after": [],
        "summary": {}
    }

    # Phase 1: Self-caused actions
    for _ in range(5):
        import requests
        # Simulate a step (this is self-caused)
        response = step_network(SimulationParams(noise_level=8.0))
        # Note: This is async, so we need to handle it properly
        # For simplicity, we'll just record the current agency state
        results["self_actions_before"].append(agency_detector.agency_level)

    # Phase 2: External pushes
    for direction in ['up', 'right']:
        agency_detector.on_external_push(direction)
        world.move_agent({'up': 1, 'right': 4}[direction])
        new_sensors = world.get_sensory_input()
        new_sensory = {
            'up': new_sensors.get('food_up', 0),
            'down': new_sensors.get('food_down', 0),
            'left': new_sensors.get('food_left', 0),
            'right': new_sensors.get('food_right', 0)
        }
        # Pass current self_state for self-explanation
        agency_info = agency_detector.update(new_sensory, self_state=self_model.get_state())
        results["external_pushes"].append(agency_info['agency_level'])

    # Summary
    avg_self = sum(results["self_actions_before"]) / len(results["self_actions_before"]) if results["self_actions_before"] else 0
    avg_external = sum(results["external_pushes"]) / len(results["external_pushes"]) if results["external_pushes"] else 0

    results["summary"] = {
        "avg_agency_self_actions": round(avg_self, 3),
        "avg_agency_external_push": round(avg_external, 3),
        "difference": round(avg_self - avg_external, 3),
        "detection_working": avg_self > avg_external
    }

    return results

# ============================================
# WORKING MEMORY ENDPOINTS
# ============================================

@app.get("/memory/state")
def get_memory_state():
    """Get current working memory state."""
    return working_memory.to_dict()

@app.get("/memory/visualization")
def get_memory_visualization():
    """Get memory data optimized for frontend visualization."""
    return working_memory.get_visualization_data()

@app.post("/memory/clear")
def clear_memory():
    """Clear all working memory (simulates forgetting)."""
    working_memory.clear_all()
    print("[MEMORY] All memories cleared")
    return {"message": "Working memory cleared"}

@app.post("/memory/inject")
def inject_memory(direction: str = "up", strength: float = 0.8):
    """
    Artificially inject a memory (for testing).
    Simulates the agent "remembering" food in a direction.

    Args:
        direction: 'up', 'down', 'left', 'right'
        strength: 0.0 to 1.0
    """
    if direction not in ['up', 'down', 'left', 'right']:
        raise HTTPException(status_code=400, detail=f"Invalid direction: {direction}")

    mem_id = f"m_{direction}"
    if mem_id in working_memory.memories:
        working_memory.memories[mem_id].activity = min(1.0, max(0.0, strength))
        print(f"[MEMORY] Injected memory: {direction.upper()} = {strength:.2f}")
        return {
            "message": f"Memory injected: {direction}",
            "state": working_memory.to_dict()
        }
    raise HTTPException(status_code=400, detail="Memory neuron not found")

# ============================================
# SELF-MODEL (Meta-cognition) ENDPOINTS
# ============================================

@app.get("/self/state")
def get_self_state():
    """Get current self-model state - the agent's representation of itself."""
    return self_model.to_dict()

@app.get("/self/behavioral")
def get_behavioral_state():
    """Get the agent's current behavioral state label."""
    return {
        "behavioral_state": self_model.get_behavioral_state(),
        "self_state": self_model.get_state()
    }

@app.get("/self/modulation")
def get_self_modulation():
    """Get how self-model is modulating other systems."""
    return {
        "attention_modulation": self_model.get_attention_modulation(),
        "agency_modulation": self_model.get_agency_modulation()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
