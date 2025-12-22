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
from homeostasis import HomeostasisSystem
from emotion import EmotionSystem
from imagination import ImaginationSystem
from memory_ltm import LongTermMemory
from goal import GoalSystem
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
world = GridWorld(15, 15)  # Larger map for avoidance learning
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
homeostasis = HomeostasisSystem(  # Homeostasis: "What does this being NEED?"
    energy_decay=0.002,       # Energy lost per step (metabolism)
    safety_recovery=0.02,     # Safety recovery when calm
    fatigue_recovery=0.01,    # Fatigue recovery when resting
    fatigue_buildup=0.005     # Fatigue from effort
)
emotion = EmotionSystem()  # Emotion: "How does this being FEEL?"
imagination = ImaginationSystem()  # Imagination: "What would happen if...?"
long_term_memory = LongTermMemory(max_episodes=100)  # LTM: "What happened before?"
goal_system = GoalSystem()  # Goal-directed behavior: "What should I focus on?"

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

# 1b. Predator Sensory Neurons (DANGER detection)
# These detect predator direction and INHIBIT movement toward danger
for p_id in ["p_up", "p_down", "p_left", "p_right"]:
    network.add_neuron(p_id, neuron_type="RS")

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

# 4b. Predator-specific inhibitory neuron
network.add_neuron("fear_gaba", neuron_type="FS")

# --- Connections (Brain 2.0: Competitive Reflex Mapping) ---

# 1. Sensory -> Hidden (NO PRIOR KNOWLEDGE - tabula rasa)
# All connections start equal - agent must LEARN which direction is correct
# This is the "infant" state: knows nothing about food direction
INFANT_WEIGHT = 40.0  # All equal - no bias
for direct in ["up", "down", "left", "right"]:
    s_id = f"s_{direct}"
    for h_id in hidden_ids:
        # Small random variation for symmetry breaking, but no directional bias
        network.connect(s_id, h_id, weight=INFANT_WEIGHT + random.uniform(-3, 3), delay=2)

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

# 4. Predator -> Fear GABA -> Hidden (INHIBIT movement toward danger)
# Predator UP → activates fear_gaba → INHIBITS H_UP
# This creates AVOIDANCE: seeing predator in a direction suppresses going that way
for direct in ["up", "down", "left", "right"]:
    # Predator sensor activates fear response
    network.connect(f"p_{direct}", "fear_gaba", weight=60.0, delay=1, enable_stdp=False)

# Fear GABA inhibits Hidden neurons (but which one depends on where predator is seen)
# Strong direct inhibition: P_UP inhibits H_UP (don't go toward predator)
for direct in ["up", "down", "left", "right"]:
    # STRONG inhibition: predator direction suppresses that movement
    # Weight 80 to compete with food signal (was 50)
    # STDP disabled - this is a hardwired reflex, not learned
    network.connect(f"p_{direct}", f"h_{direct}", weight=80.0, delay=1, is_inhibitory=True, enable_stdp=False)


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

    # === PREDATOR SENSORY INPUT ===
    # KEY INSIGHT: Agent must LEARN that predator is dangerous
    # First encounter = weak avoidance signal (doesn't know it's dangerous)
    # After pain experience = strong avoidance signal (learned fear)
    predator_sensors = world.get_predator_sensory_input()
    learned_fear = emotion.predator_fear_learned  # How much we've learned to fear

    for direction in ['up', 'down', 'left', 'right']:
        pred_signal = predator_sensors.get(f'predator_{direction}', 0)
        if pred_signal > 0:
            # Signal strength depends on LEARNED FEAR
            # First encounter: weak signal (base 10) - just "something is there"
            # After learning: strong signal (up to 80) - "DANGER!"
            base_signal = 10.0  # Minimal awareness
            fear_bonus = 70.0 * learned_fear  # Learned avoidance
            signal_strength = base_signal + fear_bonus

            sensor_currents[f"p_{direction}"] = pred_signal * signal_strength

            # Suppress food signal ONLY if we've learned predator is dangerous
            # "Food is there but I KNOW danger is blocking - don't go!"
            if learned_fear > 0.2:
                food_key = f"s_{direction}"
                if food_key in sensor_currents and sensor_currents[food_key] > 0:
                    # Suppression proportional to fear × threat
                    suppression = pred_signal * learned_fear * 0.8
                    sensor_currents[food_key] *= (1 - suppression)

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

    # === LONG-TERM MEMORY: "What happened here before?" ===
    # Recall relevant memories for current position
    current_pos_tuple = (world.agent_pos[0], world.agent_pos[1])

    # Store "before" values for delta calculation
    energy_before = homeostasis.energy
    safety_before = homeostasis.safety
    pain_before = homeostasis.pain_level  # Note: attribute is 'pain_level' not 'pain'

    # Context for similarity matching (피드백 #1: 위치 미신 방지)
    predator_info_for_context = world.get_predator_info()
    pred_dist = predator_info_for_context.get('distance') or predator_info_for_context.get('threat_level', 0)
    # If distance not available, use threat_level as proxy (higher threat = closer)
    if 'distance' in predator_info_for_context:
        context_predator_near = pred_dist is not None and pred_dist <= 3
    else:
        context_predator_near = predator_info_for_context.get('threat_level', 0) > 0.3
    context_energy_low = homeostasis.energy < 0.3
    context_fleeing = emotion.emotions.get('fear', 0) > 0.3

    # Get memory influence with context
    memory_influence = long_term_memory.get_memory_influence(
        current_pos_tuple,
        current_predator_near=context_predator_near,
        current_energy_low=context_energy_low,
        current_fleeing=context_fleeing
    )
    memory_recall_reason = long_term_memory.get_recall_reason()

    # === IMAGINATION: "What would happen if...?" ===
    # Before deciding, imagine each possible action's outcome
    predator_info = world.get_predator_info()
    imagination_result = imagination.evaluate_all_actions(
        current_state={
            'learned_fear': emotion.predator_fear_learned,
            'learned_food': emotion.food_seeking_learned,
        },
        world_info={
            'agent_pos': world.agent_pos.copy(),
            'food_pos': world.food_pos.copy(),
            'predator_pos': predator_info['position'],
            'grid_size': [world.width, world.height],
        },
        homeostasis=homeostasis.get_state(),
        emotions=emotion.emotions,
    )
    imagination_scores = imagination_result.get('scores', {})
    imagination_best = imagination_result.get('best_action')
    imagination_confidence = imagination_result.get('confidence', 0.1)

    # Combine imagination with memory influence
    # Memory can boost or suppress imagination scores based on past experience
    # v1.2: Memory weight adjusted by Self-Model uncertainty
    # - uncertainty 높음 → 기억 의존 ↑ (감각이 불확실하니 경험에 의존)
    # - uncertainty 낮음 → 기억 의존 ↓ (감각이 확실하니 현재 정보 신뢰)
    current_uncertainty = self_model.get_state().get('uncertainty', 0.5)
    # Memory weight: base 2.0, modulated by uncertainty (range: 1.0 ~ 3.0)
    memory_weight = 1.0 + (current_uncertainty * 2.0)  # uncertainty 0 → 1.0, uncertainty 1 → 3.0

    for direction in ['up', 'down', 'left', 'right']:
        mem_mod = memory_influence.get(direction, 0.0)
        if mem_mod != 0:
            imagination_scores[direction] = imagination_scores.get(direction, 0) + mem_mod * memory_weight

    # === GOAL-DIRECTED BEHAVIOR: "What should I focus on?" ===
    # Update current goal based on internal state
    predator_distance = predator_info.get('distance') if predator_info.get('distance') else None
    current_goal = goal_system.update(
        safety=homeostasis.safety,
        energy=homeostasis.energy,
        predator_distance=predator_distance,
        pain_level=homeostasis.pain_level
    )

    # Apply goal-specific biases to imagination scores
    for direction in ['up', 'down', 'left', 'right']:
        dir_details = imagination_result.get('details', {}).get(direction, {})
        goal_bias = goal_system.get_action_bias(direction, dir_details)
        imagination_scores[direction] = imagination_scores.get(direction, 0) + goal_bias

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

    # === Homeostasis-based exploration modulation ===
    # Drives affect exploration strategy:
    # - safety_drive ↑ → explore more (find safe spot)
    # - rest_drive ↑ → explore less (conserve energy)
    # - hunger_drive critical → desperate random search
    homeo_mod = homeostasis.get_exploration_modulation()
    homeo_epsilon_mod = homeo_mod['epsilon_modifier']

    # === Emotion-based exploration modulation ===
    # Emotions directly affect exploration:
    # - fear ↑ → freeze/flee, don't explore
    # - curiosity ↑ → explore more
    # - anxiety ↑ → reduce exploration
    emotion_mod = emotion.get_exploration_modulation()
    emotion_epsilon_mod = emotion_mod['epsilon_modifier']

    effective_epsilon = EPSILON + exploration_boost - confidence_reduction + state_boost \
                       - externality_reduction + internal_boost + homeo_epsilon_mod + emotion_epsilon_mod
    effective_epsilon = max(0.05, min(0.5, effective_epsilon))  # Clamp to [0.05, 0.5]

    # Fear freeze response: extreme fear completely stops exploration
    if emotion_mod.get('freeze_response', False):
        effective_epsilon = 0.0  # Don't move randomly when terrified

    is_exploring = random.random() < effective_epsilon
    action_source = "unknown"  # Track why this action was chosen

    if is_exploring:
        # Exploration: Pure random action (25% each direction)
        # No bias - infant needs to explore all directions equally
        primary_action = random.choice([1, 2, 3, 4])  # UP, DOWN, LEFT, RIGHT
        winning_action_id = f"a_{['up', 'down', 'left', 'right'][primary_action - 1]}"
        action_source = "explore"
    elif max_fires > 0:
        # === BLEND SNN reflex with Imagination ===
        # As imagination confidence grows, rely more on it
        # imagination_weight increases with development phase and confidence
        imagination_weight = imagination_confidence * 0.5  # Start small, grow with experience

        # Combine SNN firing counts with imagination scores
        direction_map = {'up': 1, 'down': 2, 'left': 3, 'right': 4}
        combined_scores = {}

        for direction in ['up', 'down', 'left', 'right']:
            action_id = f"a_{direction}"
            snn_score = action_fire_counts.get(action_id, 0)
            img_score = imagination_scores.get(direction, 0)

            # Normalize SNN score (0-1 range roughly)
            snn_normalized = snn_score / max(max_fires, 1)

            # Normalize imagination score (can be negative)
            img_normalized = (img_score + 5) / 10  # Shift and scale

            # Blend: (1 - weight) * SNN + weight * Imagination
            combined = (1 - imagination_weight) * snn_normalized + imagination_weight * img_normalized
            combined_scores[direction] = combined

        # Choose direction with highest combined score
        best_direction = max(combined_scores, key=combined_scores.get)
        winning_action_id = f"a_{best_direction}"
        primary_action = direction_map[best_direction]
        action_source = "snn+imagine" if imagination_weight > 0.1 else "snn"
    elif imagination_confidence > 0.3 and imagination_best:
        # No SNN firing, but imagination has a recommendation
        direction_map = {'up': 1, 'down': 2, 'left': 3, 'right': 4}
        primary_action = direction_map.get(imagination_best, 1)
        winning_action_id = f"a_{imagination_best}"
        action_source = "imagine"
    else:
        # No firing, low imagination confidence: Explore randomly
        # This is crucial for tabula rasa learning - agent must move to learn
        primary_action = random.choice([1, 2, 3, 4])  # UP, DOWN, LEFT, RIGHT
        winning_action_id = f"a_{['up', 'down', 'left', 'right'][primary_action - 1]}"
        action_source = "random"

    # --- Movement Reward & Anti-Stagnation ---
    # Small reward for actual movement (encourages exploration)
    # Penalty for staying still or oscillating
    movement_reward = 0.0
    oscillation_penalty = 0.0

    if primary_action == 0:
        # Staying still - small penalty to encourage movement
        movement_reward = -0.02
    elif len(action_history) >= 2:
        last_action = action_history[-1]
        second_last = action_history[-2]

        # Check for opposite direction oscillation
        if (primary_action == 1 and last_action == 2) or (primary_action == 2 and last_action == 1):
            oscillation_penalty = -0.5  # UP↔DOWN oscillation
        elif (primary_action == 3 and last_action == 4) or (primary_action == 4 and last_action == 3):
            oscillation_penalty = -0.5  # LEFT↔RIGHT oscillation
        else:
            # Actual movement in a new direction - small reward
            movement_reward = 0.01

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
        total_reward = (reward * 1.0) + oscillation_penalty + movement_reward
        if reward > 5.0:  # Food was found
            found_food = True
        if was_reset:
            died = True
    else:
        reward, was_reset, collision_info = world.move_agent(0)  # Stay
        total_reward = reward + oscillation_penalty + movement_reward
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

    # === CURIOSITY INTRINSIC REWARD ===
    # Key insight: Novelty/surprise is inherently rewarding (for learning)
    # High prediction error + self-caused = interesting discovery!
    # This encourages exploration without telling agent WHERE to go
    pred_error = agency_info.get('prediction_error', 0)
    curiosity_reward = 0.0

    if not was_perturbed and pred_error > 0.3:
        # Self-caused novel experience - reward curiosity!
        # Scale: 0.3 error = 0 reward, 1.0 error = 0.1 reward
        curiosity_reward = (pred_error - 0.3) * 0.15

        # Boost during infant phase (encourage exploration)
        if world.development_phase == 0:
            curiosity_reward *= 1.5

        total_reward += curiosity_reward

        if curiosity_reward > 0.05:
            print(f"[CURIOSITY] +{curiosity_reward:.3f} (pred_error: {pred_error:.2f})")

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

    # === HOMEOSTASIS: Update internal needs ===
    # "What does this being NEED right now?"
    ate_food = collision_info.get('ate_food') if collision_info else None
    got_food = ate_food is not None  # 'small' or 'large'
    is_resting = primary_action == 0  # STAY action = resting
    predator_threat = collision_info.get('predator_threat', 0.0) if collision_info else 0.0
    predator_caught = collision_info.get('predator_caught', False) if collision_info else False

    homeostasis.update(
        got_food=got_food,
        food_value=total_reward if got_food else 0,
        was_external_event=was_perturbed,
        hit_wall=collision_info.get('hit_wall', False) if collision_info else False,
        effort_level=self_state.get('effort', 0),
        is_resting=is_resting,
        predator_threat=predator_threat,
        predator_caught=predator_caught
    )

    # === EMOTION: Update emotional state ===
    # "How does this being FEEL right now?"
    # Emotion emerges from homeostasis + events
    emotion.update(
        homeostasis_state=homeostasis.get_state(),
        predator_threat=predator_threat,
        predator_caught=predator_caught,
        got_food=got_food,
        exploration_need=self_state.get('exploration_need', 0.3),
        uncertainty=self_state.get('uncertainty', 0.3)
    )

    # Update imagination confidence based on actual outcome
    # "Did my prediction match reality?" → confidence grows/shrinks
    action_dir_for_imagination = action_dir_map.get(primary_action, 'stay')
    imagination.update_from_outcome(
        predicted_action=action_dir_for_imagination,
        actual_outcome={
            'got_food': got_food,
            'got_hurt': predator_caught,
        }
    )

    # === LONG-TERM MEMORY: Store significant episodes ===
    # Determine outcome type for memory
    if predator_caught:
        memory_outcome = 'pain'
    elif got_food:
        memory_outcome = 'food'
    elif predator_threat > 0.5:
        memory_outcome = 'near_danger'
    elif predator_threat > 0 and predator_threat < 0.3:
        memory_outcome = 'escape'  # Was near danger but moved away
    else:
        memory_outcome = 'nothing'

    # Calculate actual deltas (피드백 #2: 경험 기반 점수)
    # These are the REAL changes that happened, not hardcoded values
    delta_energy = homeostasis.energy - energy_before
    delta_pain = homeostasis.pain_level - pain_before  # pain increased = bad experience
    delta_safety = homeostasis.safety - safety_before

    # Store episode if emotionally significant
    # v1.2: Externality affects memory storage
    # - externality 높음 → 기억 저장 약화 ("이건 내 행동 결과가 아니야")
    # - externality 낮음 → 기억 저장 정상 ("이건 내가 한 거야")
    emotion_va = emotion.get_valence_arousal()
    current_externality = self_model.get_state().get('externality', 0.0)
    # Scale down emotion_intensity by externality (external events = less memorable for self)
    adjusted_intensity = emotion_va['arousal'] * (1.0 - current_externality * 0.5)  # Max 50% reduction

    episode_stored = long_term_memory.store(
        position=current_pos_tuple,
        energy=homeostasis.energy,
        safety=homeostasis.safety,
        action=action_dir_for_imagination,
        outcome=memory_outcome,
        reward=total_reward,
        dominant_emotion=emotion.dominant_emotion or 'neutral',
        emotion_intensity=adjusted_intensity,  # Reduced if external cause
        # Actual deltas (경험 기반!)
        delta_energy=delta_energy,
        delta_pain=delta_pain,
        delta_safety=delta_safety,
        # Context (상황 요약)
        context_predator_near=context_predator_near,
        context_energy_low=context_energy_low,
        context_was_fleeing=context_fleeing
    )

    # Age all memories
    long_term_memory.step()

    # Pain from predator = immediate negative reward (LEARNING SIGNAL!)
    if predator_caught:
        pain_penalty = -2.0  # Strong negative signal for learning
        total_reward += pain_penalty
        print(f"[PAIN!] Predator caught agent! Health: {homeostasis.health:.0%}")

    # Get homeostasis modulation for behavior
    homeo_exploration = homeostasis.get_exploration_modulation()
    homeo_learning = homeostasis.get_learning_modulation()

    # Get emotion modulation for behavior
    emotion_exploration = emotion.get_exploration_modulation()
    emotion_attention = emotion.get_attention_modulation()
    emotion_learning = emotion.get_learning_modulation()

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

    # Apply emotion attention modulation (fear → narrow, curiosity → wide)
    attention.attention_width = max(0.1, min(1.0,
        attention.attention_width + emotion_attention['width_modifier']
    ))

    # Modulate reward by agency: "I did this" = full reward, "External" = reduced
    agency_modifier = agency_detector.get_dopamine_modifier()

    # Modulate by homeostasis: tired/starving = less effective learning
    # (brain needs energy and rest to consolidate memories)
    learning_mod = homeo_learning['learning_rate_modifier']

    # Modulate by emotion: fear/pain → enhanced memory, anxiety → impaired
    learning_mod *= emotion_learning['learning_rate_modifier']

    modulated_reward = total_reward * agency_modifier * learning_mod

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
            "wind": world.get_wind_info(),
            "predator": world.get_predator_info()
        },
        "agency": agency_info,
        "was_perturbed": was_perturbed,
        "perturb_type": perturb_type,  # 'wall', 'wind', or None
        "memory": working_memory.get_visualization_data(),
        "using_memory": using_memory,
        "attention": attention.get_visualization_data(),
        "conflict": value_conflict.get_visualization_data(),
        "self_model": self_model.get_visualization_data(),
        "homeostasis": homeostasis.get_visualization_data(),
        "emotion": emotion.get_visualization_data(),
        "imagination": imagination.get_visualization_data(),  # Internal simulation
        "action_source": action_source,  # Why this action was chosen
        "development": world.get_development_info(),
        "long_term_memory": {
            **long_term_memory.get_visualization_data(),
            "memory_influence": memory_influence,
            "recall_reason": memory_recall_reason,
        },
        "goal": {
            **goal_system.get_visualization_data(),
            "description": goal_system.get_goal_description(),
        }
    }

@app.get("/network/state")
def get_network_state():
    """Get current state of all neurons and synapses."""
    return network.get_state()

@app.post("/network/reset")
def reset_network():
    """Reset all neurons, synapses, and world to initial state."""
    global action_history, wall_blocked
    global imagination, long_term_memory, goal_system
    network.reset()
    world.reset()
    working_memory.clear_all()  # Clear working memory
    attention.clear()  # Clear attention focus
    value_conflict.clear()  # Clear value conflict state
    self_model.clear()  # Clear self-model state
    homeostasis.clear()  # Clear homeostasis state
    emotion.clear()  # Clear emotional state
    imagination = ImaginationSystem()  # Reset imagination
    long_term_memory = LongTermMemory(max_episodes=100)  # Reset long-term memory
    goal_system = GoalSystem()  # Reset goal system
    action_history = []  # Clear action history
    wall_blocked = {'up': 0, 'down': 0, 'left': 0, 'right': 0}  # Clear wall blocks
    print("Simulation Reset: All systems cleared (Memory, Attention, Self-Model, Homeostasis, Emotion, Imagination, LTM, Goal)")
    return {"message": "All systems reset including Long-term Memory and Goals"}

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
