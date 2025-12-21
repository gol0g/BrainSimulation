import random
from predator import Predator


class GridWorld:
    """
    Grid environment with developmental phases (Infant → Adult).

    Infant Phase: Protected environment for initial learning
    - Food spawns closer to agent
    - Predator is slower and starts farther away
    - Energy drains slower

    Adult Phase: Normal difficulty
    - Food spawns anywhere
    - Predator at full speed
    - Normal energy drain
    """

    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.agent_pos = [width // 2, height // 2]
        self.steps_since_fed = 0
        self.energy = 100.0

        # === DEVELOPMENTAL PHASE SYSTEM ===
        # Phase 0 = Infant (protected), Phase 1 = Adult (normal)
        self.development_phase = 0  # Start as infant
        self.phase_steps = 0  # Steps in current phase
        self.total_lifetime_steps = 0

        # Phase transition thresholds
        self.INFANT_DURATION = 2000  # Steps before transitioning to adult
        self.phase_transition_progress = 0.0  # 0.0 = full infant, 1.0 = full adult

        # === PHASE-DEPENDENT PARAMETERS ===
        # Infant values (protected) - 가중치가 균등(40)이라 랜덤 이동함
        # 생존을 보장하면서 경험 기회 제공
        self.INFANT_FOOD_MAX_DISTANCE = 1  # 음식이 1칸 거리에 스폰 (바로 옆!)
        self.INFANT_PREDATOR_MOVE_PROB = 0.05  # 5% 이동 (거의 안 움직임)
        self.INFANT_PREDATOR_CHASE_PROB = 0.02  # 2% 추적 (거의 안 쫓아옴)
        self.INFANT_PREDATOR_MIN_DISTANCE = 10  # 매우 멀리서 시작
        self.INFANT_ENERGY_DECAY = 0.003  # 매우 느린 에너지 감소 (300스텝에 1 감소)

        # Adult values (normal)
        self.ADULT_FOOD_MAX_DISTANCE = None  # Anywhere
        self.ADULT_PREDATOR_MOVE_PROB = 0.3  # 30% move chance
        self.ADULT_PREDATOR_CHASE_PROB = 0.2  # 20% chase chance
        self.ADULT_PREDATOR_MIN_DISTANCE = 5  # Normal distance
        self.ADULT_ENERGY_DECAY = 0.05  # Normal drain

        # Initialize with infant settings
        self.food_pos = self._place_food()

        # Large food system (Value Conflict) - DISABLED for now
        self.large_food_pos = None  # Position of large food (or None)
        self.large_food_reward = 25.0  # Much bigger reward
        self.small_food_reward = 10.0  # Regular food reward
        self.large_food_spawn_chance = 0.0  # DISABLED - was causing weight degradation
        self.large_food_min_distance = 3  # Must be at least this far from agent

        # Wind system - DISABLED during infant phase
        self.wind_direction = None  # None, 'up', 'down', 'left', 'right'
        self.wind_steps_remaining = 0
        self.steps_until_wind = random.randint(200, 400)  # Delayed first wind
        self.total_steps = 0

        # Predator system - starts with infant settings
        self.predator = Predator(
            world_width=width,
            world_height=height,
            chase_probability=self.INFANT_PREDATOR_CHASE_PROB,
            move_probability=self.INFANT_PREDATOR_MOVE_PROB,
            threat_radius=4
        )
        self.predator._place_random(avoid_pos=self.agent_pos, min_distance=self.INFANT_PREDATOR_MIN_DISTANCE)

    def _place_food(self):
        """Place food based on current development phase."""
        max_dist = self._get_food_max_distance()

        attempts = 0
        while attempts < 100:
            if max_dist is None:
                # Adult mode: anywhere on map
                pos = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
            else:
                # Infant mode: within max_dist MANHATTAN distance of agent
                # Generate valid Manhattan distance positions
                actual_dist = random.randint(1, max_dist)  # At least 1 tile away
                dx = random.randint(0, actual_dist)
                dy = actual_dist - dx  # Ensure |dx| + |dy| = actual_dist

                # Random signs
                if random.random() < 0.5:
                    dx = -dx
                if random.random() < 0.5:
                    dy = -dy

                pos = [
                    max(0, min(self.width - 1, self.agent_pos[0] + dx)),
                    max(0, min(self.height - 1, self.agent_pos[1] + dy))
                ]

            if pos != self.agent_pos:
                return pos
            attempts += 1

        # Fallback: just place somewhere
        return [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]

    def _get_food_max_distance(self):
        """Get maximum food distance based on development phase."""
        if self.development_phase == 0:
            # Infant: interpolate based on progress
            # Starts at INFANT distance, gradually increases
            infant_dist = self.INFANT_FOOD_MAX_DISTANCE
            adult_dist = max(self.width, self.height)  # Full map
            return int(infant_dist + (adult_dist - infant_dist) * self.phase_transition_progress)
        else:
            return None  # Adult: anywhere

    def _update_development_phase(self):
        """Update developmental phase based on lifetime experience."""
        self.total_lifetime_steps += 1
        self.phase_steps += 1

        if self.development_phase == 0:  # Infant phase
            # Gradually increase difficulty
            self.phase_transition_progress = min(1.0, self.phase_steps / self.INFANT_DURATION)

            # Update predator difficulty gradually
            move_prob = self.INFANT_PREDATOR_MOVE_PROB + \
                       (self.ADULT_PREDATOR_MOVE_PROB - self.INFANT_PREDATOR_MOVE_PROB) * self.phase_transition_progress
            chase_prob = self.INFANT_PREDATOR_CHASE_PROB + \
                        (self.ADULT_PREDATOR_CHASE_PROB - self.INFANT_PREDATOR_CHASE_PROB) * self.phase_transition_progress

            self.predator.move_probability = move_prob
            self.predator.chase_probability = chase_prob

            # Check for phase transition
            if self.phase_steps >= self.INFANT_DURATION:
                self.development_phase = 1
                self.phase_steps = 0
                self.phase_transition_progress = 1.0
                print(f"\n{'='*50}")
                print(f"[DEVELOPMENT] INFANT → ADULT phase transition!")
                print(f"Total lifetime: {self.total_lifetime_steps} steps")
                print(f"{'='*50}\n")

    def _get_energy_decay(self):
        """Get energy decay rate based on development phase."""
        if self.development_phase == 0:
            # Interpolate between infant and adult decay
            return self.INFANT_ENERGY_DECAY + \
                   (self.ADULT_ENERGY_DECAY - self.INFANT_ENERGY_DECAY) * self.phase_transition_progress
        else:
            return self.ADULT_ENERGY_DECAY

    def get_development_info(self):
        """Get development phase info for frontend display."""
        return {
            'phase': 'infant' if self.development_phase == 0 else 'adult',
            'progress': round(self.phase_transition_progress, 2),
            'lifetime_steps': self.total_lifetime_steps,
            'phase_steps': self.phase_steps
        }

    def _place_large_food(self):
        """Place large food at least min_distance away from agent."""
        attempts = 0
        while attempts < 50:
            pos = [random.randint(0, self.width - 1), random.randint(0, self.height - 1)]
            dist = abs(pos[0] - self.agent_pos[0]) + abs(pos[1] - self.agent_pos[1])
            if pos != self.agent_pos and pos != self.food_pos and dist >= self.large_food_min_distance:
                return pos
            attempts += 1
        return None  # Failed to place

    def _update_large_food(self):
        """Spawn large food occasionally."""
        if self.large_food_pos is None:
            if random.random() < self.large_food_spawn_chance:
                self.large_food_pos = self._place_large_food()
                if self.large_food_pos:
                    print(f"[LARGE FOOD] Spawned at {self.large_food_pos} (reward: {self.large_food_reward})")

    def _update_wind(self):
        """Update wind state each step."""
        self.total_steps += 1

        # Check if wind event should start
        if self.wind_direction is None:
            self.steps_until_wind -= 1
            if self.steps_until_wind <= 0:
                # Start wind event
                self.wind_direction = random.choice(['up', 'down', 'left', 'right'])
                self.wind_steps_remaining = random.randint(10, 25)  # Wind lasts 10-25 steps
                print(f"[WIND] Wind started: {self.wind_direction.upper()} for {self.wind_steps_remaining} steps")
        else:
            # Wind is active
            self.wind_steps_remaining -= 1
            if self.wind_steps_remaining <= 0:
                print(f"[WIND] Wind ended")
                self.wind_direction = None
                self.steps_until_wind = random.randint(80, 200)  # Next wind in 80-200 steps

    def move_agent(self, action):
        """
        Action: 0=Stay, 1=Up, 2=Down, 3=Left, 4=Right
        Returns: (reward, was_reset, collision_info)

        collision_info: {'hit_wall': bool, 'wall_direction': str or None, 'wind_push': str or None}
        """
        reward = -0.01  # Minor energy cost for existence
        was_reset = False
        collision_info = {
            'hit_wall': False,
            'wall_direction': None,
            'wind_push': None,
            'predator_threat': 0.0,
            'predator_caught': False
        }

        old_pos = self.agent_pos.copy()

        # Update developmental phase (infant → adult transition)
        self._update_development_phase()

        # Update wind state (disabled during early infant phase)
        if self.phase_transition_progress > 0.3:  # Wind starts at 30% progress
            self._update_wind()

        # Update large food spawning
        self._update_large_food()

        # Apply intended movement
        intended_pos = self.agent_pos.copy()
        if action == 1:  # Up
            intended_pos[1] = self.agent_pos[1] - 1
        elif action == 2:  # Down
            intended_pos[1] = self.agent_pos[1] + 1
        elif action == 3:  # Left
            intended_pos[0] = self.agent_pos[0] - 1
        elif action == 4:  # Right
            intended_pos[0] = self.agent_pos[0] + 1

        # Check wall collision
        if intended_pos[0] < 0:
            collision_info['hit_wall'] = True
            collision_info['wall_direction'] = 'left'
            intended_pos[0] = 0
        elif intended_pos[0] >= self.width:
            collision_info['hit_wall'] = True
            collision_info['wall_direction'] = 'right'
            intended_pos[0] = self.width - 1
        if intended_pos[1] < 0:
            collision_info['hit_wall'] = True
            collision_info['wall_direction'] = 'up'
            intended_pos[1] = 0
        elif intended_pos[1] >= self.height:
            collision_info['hit_wall'] = True
            collision_info['wall_direction'] = 'down'
            intended_pos[1] = self.height - 1

        self.agent_pos = intended_pos

        # Apply wind effect (30% chance to drift when wind is active)
        if self.wind_direction and random.random() < 0.3:
            wind_pos = self.agent_pos.copy()
            if self.wind_direction == 'up':
                wind_pos[1] = max(0, wind_pos[1] - 1)
            elif self.wind_direction == 'down':
                wind_pos[1] = min(self.height - 1, wind_pos[1] + 1)
            elif self.wind_direction == 'left':
                wind_pos[0] = max(0, wind_pos[0] - 1)
            elif self.wind_direction == 'right':
                wind_pos[0] = min(self.width - 1, wind_pos[0] + 1)

            if wind_pos != self.agent_pos:
                collision_info['wind_push'] = self.wind_direction
                self.agent_pos = wind_pos

        # Check for food (small)
        ate_food_type = None  # 'small', 'large', or None
        if self.agent_pos == self.food_pos:
            reward = self.small_food_reward
            self.food_pos = self._place_food()
            self.energy = min(100.0, self.energy + 15)
            self.steps_since_fed = 0
            ate_food_type = 'small'
            collision_info['ate_food'] = 'small'

        # Check for large food
        elif self.large_food_pos and self.agent_pos == self.large_food_pos:
            reward = self.large_food_reward
            self.large_food_pos = None  # Large food consumed
            self.energy = min(100.0, self.energy + 40)  # More energy
            self.steps_since_fed = 0
            ate_food_type = 'large'
            collision_info['ate_food'] = 'large'
            print(f"[LARGE FOOD] Consumed! Reward: {self.large_food_reward}")

        else:
            self.steps_since_fed += 1
            self.energy -= self._get_energy_decay()  # Phase-dependent decay
            collision_info['ate_food'] = None

        # Wall collision penalty
        if collision_info['hit_wall']:
            reward -= 0.1

        # --- Predator Step ---
        predator_info = self.predator.step(self.agent_pos)
        collision_info['predator_threat'] = self.predator.get_threat_level(self.agent_pos)
        collision_info['predator_caught'] = predator_info['caught_agent']

        # Caught by predator = big penalty
        if predator_info['caught_agent']:
            reward -= 3.0  # Significant penalty
            # Respawn predator away from agent
            self.predator.reset(self.agent_pos)

        # --- Auto-Reset (Death) Logic ---
        if self.energy <= 0:
            self.reset()
            reward = -5.0  # Penalty for dying
            was_reset = True

        return reward, was_reset, collision_info

    def get_sensory_input(self):
        """
        Simple sensory input: Direction to food
        Returns: [Up, Down, Left, Right] as float based on distance

        Both small and large food have distance-dependent signal strength.
        This allows fair comparison when deciding between options.
        """
        dx = self.food_pos[0] - self.agent_pos[0]
        dy = self.food_pos[1] - self.agent_pos[1]
        small_dist = abs(dx) + abs(dy)

        # Small food signal also distance-dependent (same formula as large)
        small_strength = max(0.3, 1.0 - small_dist * 0.1)

        sensors = {
            "food_up": small_strength if dy < 0 else 0.0,
            "food_down": small_strength if dy > 0 else 0.0,
            "food_left": small_strength if dx < 0 else 0.0,
            "food_right": small_strength if dx > 0 else 0.0,
        }

        # Add large food signals (same distance formula)
        if self.large_food_pos:
            ldx = self.large_food_pos[0] - self.agent_pos[0]
            ldy = self.large_food_pos[1] - self.agent_pos[1]
            large_dist = abs(ldx) + abs(ldy)

            # Large food signal strength decreases with distance
            large_strength = max(0.3, 1.0 - large_dist * 0.1)

            sensors["large_food_up"] = large_strength if ldy < 0 else 0.0
            sensors["large_food_down"] = large_strength if ldy > 0 else 0.0
            sensors["large_food_left"] = large_strength if ldx < 0 else 0.0
            sensors["large_food_right"] = large_strength if ldx > 0 else 0.0
        else:
            sensors["large_food_up"] = 0.0
            sensors["large_food_down"] = 0.0
            sensors["large_food_left"] = 0.0
            sensors["large_food_right"] = 0.0

        return sensors

    def get_food_info(self):
        """
        Get detailed food information for Value Conflict system.
        Returns info about both small and large food.
        """
        # Small food info
        dx = self.food_pos[0] - self.agent_pos[0]
        dy = self.food_pos[1] - self.agent_pos[1]
        small_dist = abs(dx) + abs(dy)

        # Determine primary direction to small food
        if abs(dx) > abs(dy):
            small_dir = 'right' if dx > 0 else 'left'
        elif abs(dy) > 0:
            small_dir = 'down' if dy > 0 else 'up'
        else:
            small_dir = None  # At food position

        small_food = {
            'direction': small_dir,
            'distance': small_dist,
            'reward': self.small_food_reward,
            'position': self.food_pos.copy()
        }

        # Large food info
        large_food = None
        if self.large_food_pos:
            ldx = self.large_food_pos[0] - self.agent_pos[0]
            ldy = self.large_food_pos[1] - self.agent_pos[1]
            large_dist = abs(ldx) + abs(ldy)

            if abs(ldx) > abs(ldy):
                large_dir = 'right' if ldx > 0 else 'left'
            elif abs(ldy) > 0:
                large_dir = 'down' if ldy > 0 else 'up'
            else:
                large_dir = None

            large_food = {
                'direction': large_dir,
                'distance': large_dist,
                'reward': self.large_food_reward,
                'position': self.large_food_pos.copy()
            }

        return {
            'small': small_food,
            'large': large_food
        }

    def get_wind_info(self):
        """Get current wind state for frontend display."""
        return {
            "active": self.wind_direction is not None,
            "direction": self.wind_direction,
            "steps_remaining": self.wind_steps_remaining
        }

    def get_predator_info(self):
        """Get predator state for frontend display."""
        return {
            "position": self.predator.pos.copy(),
            "threat_level": self.predator.get_threat_level(self.agent_pos),
            "threat_radius": self.predator.threat_radius,
            "last_action": self.predator.last_action
        }

    def get_predator_sensory_input(self):
        """
        Get sensory input about predator location.
        Similar to food sensors but for DANGER.
        Agent needs to SEE the predator to avoid it!

        Returns dict with predator_up, predator_down, etc.
        Signal strength increases with proximity (closer = stronger signal).
        """
        dx = self.predator.pos[0] - self.agent_pos[0]
        dy = self.predator.pos[1] - self.agent_pos[1]
        dist = abs(dx) + abs(dy)

        # Signal strength: stronger when closer (inverse of food logic)
        # At distance 0: strength = 1.0 (maximum danger)
        # At distance 5+: strength = 0.0 (can't see)
        max_perception = 5  # Can perceive predator up to 5 tiles away
        if dist >= max_perception:
            strength = 0.0
        else:
            strength = 1.0 - (dist / max_perception)

        return {
            "predator_up": strength if dy < 0 else 0.0,
            "predator_down": strength if dy > 0 else 0.0,
            "predator_left": strength if dx < 0 else 0.0,
            "predator_right": strength if dx > 0 else 0.0,
            "predator_distance": dist
        }

    def reset(self):
        self.agent_pos = [self.width // 2, self.height // 2]
        self.food_pos = self._place_food()
        self.large_food_pos = None  # Reset large food
        self.energy = 100.0
        self.steps_since_fed = 0
        # Don't reset wind - it's environmental
        # Reset predator to new position away from agent
        self.predator.reset(self.agent_pos)
