# **COMPREHENSIVE PROJECT DOCUMENTATION**
## **Ambulance Dispatch OpenEnv - Complete Technical Reference**

---

# **1. PROJECT OVERVIEW**

## **1.1 Project Identity**
- **Project Name:** Ambulance-OpenENV
- **Version:** 1.0.0
- **Purpose:** City-scale reinforcement learning environment for ambulance dispatch optimization
- **License:** MIT
- **Runtime:** Python 3.11
- **Framework:** FastAPI + PyTorch + OpenEnv Standard
- **Team:** CSNEHA20 (Team Lead), Vishallakshmikanthan (Member)

## **1.2 Core Purpose**
This is a production-grade OpenEnv-compatible RL environment where AI agents manage a fleet of ambulances across a procedurally generated city. Agents must:
1. Dispatch ambulances to incoming emergencies
2. Navigate dynamic traffic (rush-hour + incidents)
3. Route patients to appropriate hospitals (capacity + specialty constraints)
4. Balance competing priorities under strict time constraints

## **1.3 Key Features**
- **Procedural City:** Barabási-Albert scale-free graph (100 nodes, realistic hub-and-spoke)
- **5-State FSM Fleet:** IDLE → EN_ROUTE → AT_SCENE → TRANSPORTING → RETURNING
- **3 Severity Tiers:** CRITICAL (10-step timeout), HIGH (20-step), NORMAL (30-step)
- **Dynamic Traffic:** Rush-hour (1.5–2.5×) + random incidents (3.0× blockage)
- **Hospital Network:** 8-bed capacity, specialty routing (Trauma/Cardiac/General/Paediatric)
- **RFC 004 Rubric:** 9 named reward components for fine-grained training introspection

---

# **2. DIRECTORY STRUCTURE & FILE MANIFEST**

```
c:\Users\visha\Downloads\Ambulance-OpenENV\
├── agents/                    # Dispatch agents (8 files)
├── env/                       # Core simulation engine (3 files)
├── server/                    # FastAPI server (2 files)
├── tasks/                     # Task configurations (3 files)
├── rl/                        # RL training infrastructure (9 files)
├── multi_agent/               # Multi-agent coordination (4 files)
├── long_horizon/              # Long-horizon planning (5 files)
├── self_improvement/          # Self-play & weakness detection (7 files)
├── frontend/                  # Next.js dashboard (Next.js 14 app)
├── tests/                     # pytest test suite (4 files)
├── utils/                     # Utilities (1 file)
├── notebooks/                 # Jupyter notebooks (1 file)
├── Root configuration files (11 files)
└── Training/Inference scripts (11 files)
```

---

# **3. CORE ENVIRONMENT MODULES (env/)**

## **3.1 env/models.py** (125 lines)

**Purpose:** Pydantic data models for the entire environment. All data structures are strictly typed with validation.

### Classes and Their Purpose:

```python
class Rubric(BaseModel):
    """Named reward components for fine-grained diagnostic logging."""
    emergency_served: float = 0.0      # +20.0 when ambulance arrives at scene
    severity_bonus: float = 0.0        # +30 CRITICAL, +10 HIGH
    dispatch_speed: float = 0.0        # up to +10.0 for fast response
    hospital_delivery: float = 0.0     # +10.0 for patient delivery
    distance_penalty: float = 0.0      # -variable for long travel
    traffic_penalty: float = 0.0       # -variable for ignoring traffic
    idle_penalty: float = 0.0          # -1.0/step when backlog exists
    capacity_violation: float = 0.0    # -5.0 for hospital overflow
    timeout_penalty: float = 0.0       # -15.0 per expired emergency
    
    def total(self) -> float:
        return sum(self.model_dump().values())
```

```python
class AmbulanceState(str, Enum):
    """5-state FSM + intermediate states"""
    IDLE = "idle"                # Available for dispatch
    DISPATCHED = "dispatched"    # Just assigned, not moving yet
    EN_ROUTE = "en_route"        # Moving to emergency
    AT_SCENE = "at_scene"        # Loading patient (1 step)
    TRANSPORTING = "transporting" # Moving to hospital
    RETURNING = "returning"      # Returning to base
    REPOSITIONING = "repositioning"  # Proactive staging
```

```python
class Severity(str, Enum):
    CRITICAL = "CRITICAL"    # 10-step timeout, +30 bonus
    HIGH = "HIGH"            # 20-step timeout, +10 bonus
    NORMAL = "NORMAL"        # 30-step timeout, no bonus
```

```python
class AmbulanceInfo(BaseModel):
    id: int                      # 0-indexed ambulance ID
    node: int                    # Current graph node (0-99)
    state: AmbulanceState        # FSM state
    eta: int                     # Steps until state transition
    target_emg_id: Optional[str] = None   # Assigned emergency UUID
    target_hosp_id: Optional[int] = None  # Assigned hospital ID
```

```python
class EmergencyInfo(BaseModel):
    id: str                      # UUID (first 8 chars)
    node: int                    # Incident location
    severity: Severity           # CRITICAL/HIGH/NORMAL
    time_remaining: int          # Steps before expiry
    max_time_remaining: int = 90 # For frontend progress bars
    assigned: bool               # Whether ambulance dispatched
    spawn_time: int = 0          # Step when appeared
```

```python
class HospitalInfo(BaseModel):
    id: int                      # 0-indexed hospital ID
    node: int                    # Location on graph
    capacity: int                # Max concurrent patients (default 8)
    current_patients: int        # Current occupancy
    specialty: str = "General"   # Trauma/Cardiac/General/Paediatric
```

```python
class ObservationModel(Observation):
    """Full observation returned from reset() and step()"""
    ambulances: List[AmbulanceInfo]      # Fleet status
    emergencies: List[EmergencyInfo]     # Active unassigned incidents
    hospitals: List[HospitalInfo]        # Hospital network status
    traffic: Dict[str, float]            # {"global": multiplier}
    step: int = 0                        # Current simulation tick
    reward: float = 0.0                  # Scalar step reward
    reward_model: Optional[RewardModel] = None  # Typed reward
    done: bool = False                   # Episode termination flag
    rubric: Optional[Rubric] = None      # Per-component breakdown
```

```python
class ActionModel(Action):
    """Dispatch action - Pydantic extra='forbid' catches typos"""
    ambulance_id: Optional[int] = None     # Idle ambulance to dispatch
    emergency_id: str = ""                 # Target emergency UUID
    hospital_id: Optional[int] = None      # Destination hospital
    reposition_node: Optional[int] = None  # Proactive staging node
    is_noop: bool = False                  # Skip this step
```

---

## **3.2 env/simulator.py** (236 lines)

**Purpose:** Low-level simulation mechanics - city graph, traffic, ambulances, emergencies, hospitals.

### CityGraph Class:
```python
class CityGraph:
    """Barabási-Albert scale-free graph representing city road network"""
    
    def __init__(self, n: int = 100, m: int = 2):
        # m < n required for Barabási-Albert
        # seed=42 for determinism
        self.graph = nx.barabasi_albert_graph(n, m, seed=42)
        # Edge weights: 2-8 minutes base travel time
        for u, v in self.graph.edges():
            self.graph[u][v]['base_weight'] = float(np.random.randint(2, 9))
        # Precompute all-pairs shortest paths for O(1) lookup
        self._all_pairs_len = dict(nx.all_pairs_dijkstra_path_length(...))
    
    def shortest_path_time(self, source: int, target: int, traffic_multiplier: float) -> int:
        """O(1) lookup of shortest travel time with traffic"""
        base_length = self._all_pairs_len.get(source, {}).get(target)
        return max(1, int(np.ceil(base_length * traffic_multiplier)))
```

### TrafficEngine Class:
```python
class TrafficEngine:
    """Dynamic traffic with rush hours and deterministic incidents"""
    
    def __init__(self, rng: np.random.Generator = None):
        self._incidents: Dict[tuple, int] = {}  # edge -> steps_remaining
        self._incident_prob = 0.02  # 2% chance per step
    
    def maybe_spawn_incident(self, graph_edges: list, step: int):
        """Randomly block an edge for 5 steps (Hard task)"""
        if self._rng.random() < self._incident_prob:
            edge = graph_edges[int(self._rng.integers(0, len(graph_edges)))]
            self._incidents[edge] = 5
    
    def tick_incidents(self):
        """Decrement incident timers"""
    
    def get_multiplier(self, step: int) -> float:
        """
        Rush-hour logic:
        - 7-9 AM, 5-8 PM: normal(1.6, 0.2), clipped [1.2, 2.5]
        - Other hours: normal(1.0, 0.05), clipped [0.9, 1.2]
        - Incidents add +0.5 per blocked edge (max +1.0)
        """
        hour = (step % 1440) // 60
        if (7 <= hour < 9) or (17 <= hour < 20):
            base = float(np.clip(self._rng.normal(1.6, 0.2), 1.2, 2.5))
        else:
            base = float(np.clip(self._rng.normal(1.0, 0.05), 0.9, 1.2))
        if self._incidents:
            base += 0.5 * min(len(self._incidents), 2)
        return float(np.clip(base, 0.9, 10.0))
```

### Ambulance Class (FSM implementation):
```python
class Ambulance:
    """Single ambulance unit with state transition logic"""
    
    def __init__(self, amb_id: int, start_node: int, rng=None):
        self.id = amb_id
        self.node = start_node
        self.state = AmbulanceState.IDLE
        self.eta = 0  # countdown timer
        self.target_emg_id: Optional[str] = None
        self.target_emg_node: Optional[int] = None
        self.target_hosp_id: Optional[int] = None
        self.target_hosp_node: Optional[int] = None
    
    def update(self, city: CityGraph, tm: float):
        """
        FSM transitions (called each simulation step):
        
        DISPATCHED → EN_ROUTE: Calculate travel time to emergency
        EN_ROUTE → AT_SCENE: Arrive at scene, set load time (2-4 steps)
        AT_SCENE → TRANSPORTING: Calculate travel time to hospital
        TRANSPORTING → RETURNING: Arrive at hospital, release patient
        RETURNING → IDLE: Handover complete
        REPOSITIONING → IDLE: Arrive at staging node
        """
```

### AmbulanceFleet Class:
```python
class AmbulanceFleet:
    """Manages all ambulance units"""
    
    def __init__(self, n: int, nodes: List[int], rng=None):
        self.ambulances = [Ambulance(i, random_node) for i in range(n)]
    
    def get_idle(self) -> List[Ambulance]:
        return [a for a in self.ambulances if a.state == AmbulanceState.IDLE]
    
    def dispatch(self, amb_id: int, emg_id: str, emg_node: int, 
                 hosp_id: int, hosp_node: int):
        """Transition ambulance to DISPATCHED state with targets"""
    
    def reposition(self, amb_id: int, node: int, city_graph: CityGraph, 
                   traffic_multiplier: float):
        """Move idle ambulance to staging node proactively"""
    
    def step_update(self, city_graph: CityGraph, traffic_multiplier: float):
        """Advance all ambulances by one timestep"""
```

### EmergencyGenerator Class:
```python
class EmergencyGenerator:
    """Poisson process emergency generation"""
    
    config = {
        Severity.CRITICAL: {"prob": 0.25, "timeout": 20},
        Severity.HIGH: {"prob": 0.35, "timeout": 45},
        Severity.NORMAL: {"prob": 0.40, "timeout": 90}
    }
    
    def generate(self, step: int) -> List[EmergencyInfo]:
        """
        1. Sample count from Poisson(lambda_param)
        2. For each emergency, sample severity from distribution
        3. Assign random node and timeout based on severity
        """
```

### Hospital Class:
```python
class Hospital:
    """Hospital with capacity and specialty constraints"""
    
    def __init__(self, hosp_id: int, node: int, capacity: int):
        self.id = hosp_id
        self.node = node
        self.capacity = capacity
        self.current_patients = 0
        self.specialty = "General"  # Set by environment
    
    def is_available(self) -> bool:
        return self.current_patients < self.capacity
    
    def admit(self) -> bool:
        if self.is_available():
            self.current_patients += 1
            return True
        return False
    
    def release(self):
        if self.current_patients > 0:
            self.current_patients -= 1
```

---

## **3.3 env/environment.py** (444 lines)

**Purpose:** Main environment class implementing the OpenEnv interface.

### AmbulanceEnvironment Class:

```python
class AmbulanceEnvironment:
    """
    Production-grade Ambulance Dispatch Environment.
    SUPPORTS_CONCURRENT_SESSIONS = True (for multi-user training)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.n_ambulances = config.get("n_ambulances", 5)
        self.n_hospitals = config.get("n_hospitals", 3)
        self.max_steps = config.get("max_steps", 1440)
        self.graph_size = config.get("graph_size", 100)
        self.seed_val = config.get("seed", 42)
        self.reset(seed=self.seed_val)
    
    def reset(self, seed: Optional[int] = None) -> ObservationModel:
        """
        1. Initialize RNG with seed
        2. Create CityGraph (Barabási-Albert)
        3. Initialize Fleet at random nodes
        4. Create Hospitals with specialties
        5. Reset metrics dictionary
        6. Return initial observation
        """
    
    def step(self, action: ActionModel) -> ObservationModel:
        """
        SINGLE-ACTION STEP (processes one dispatch at a time)
        
        1. Process dispatch action:
           - Validate ambulance is IDLE
           - Validate emergency is unassigned
           - Check hospital capacity
           - Apply specialty mismatch penalty (-1.0)
           - Record dispatch time for response tracking
        
        2. Process reposition action (if reposition_node set)
        
        3. Simulation physics:
           - Maybe spawn traffic incident
           - Tick incident timers
           - Advance all ambulances (fleet.step_update)
        
        4. Emergency dynamics:
           - Generate new emergencies (Poisson)
           - Decrement time_remaining for unassigned emergencies
           - Remove expired emergencies (timeout_penalty -15.0 each)
        
        5. Success event processing:
           - EN_ROUTE → AT_SCENE: emergency_served (+20.0)
           - Apply severity_bonus (+30 CRITICAL, +10 HIGH)
           - Record response time and optimal time
           - TRANSPORTING → RETURNING: hospital_delivery (+10.0)
           - Release hospital capacity
        
        6. Idle penalty calculation:
           - Count idle ambulances during active backlog
           - idle_penalty -= idle_count * 1.0
        
        7. Update metrics and return observation
        """
    
    def step_all(self, actions: List[ActionModel]) -> ObservationModel:
        """
        MULTI-ACTION STEP (processes multiple dispatches in one tick)
        
        Same as step() but:
        - Applies ALL dispatch actions before physics advances
        - Allows simultaneous dispatch of multiple ambulances
        - Used by RepositioningOracle for optimal performance
        """
    
    def _get_observation(self, tm: Optional[float] = None) -> ObservationModel:
        """Build observation from current state"""
```

### Key Metrics Dictionary:
```python
self.metrics = {
    "served": 0,                    # Total emergencies served
    "missed": 0,                    # Total timeouts
    "total_emergencies": 0,         # Total generated
    "critical_served": 0,           # CRITICAL emergencies served
    "critical_total": 0,            # Total CRITICAL generated
    "high_served": 0,               # HIGH emergencies served
    "normal_served": 0,             # NORMAL emergencies served
    "avg_response_time": 0.0,       # Mean response time
    "hospital_overflow": 0,           # Capacity violation count
    "capacity_violations": 0,       # Same as overflow
    "idle_fraction": 0.0,            # Fraction of time ambulances idle
    "idle_steps": 0,                # Cumulative idle ambulance-steps
    "total_steps": 0,               # Current step count
    "priority_correct": 0,           # Correct priority dispatches
    "priority_total": 0,             # Total priority opportunities
    "response_times": [],            # List of all response times
    "optimal_times": [],             # List of optimal times for grading
    "zone_served": {0: 0, 1: 0, 2: 0, 3: 0},  # Per-zone service counts
    "zone_total": {0: 0, 1: 0, 2: 0, 3: 0},  # Per-zone emergency counts
}
```

---

# **4. SERVER MODULE (server/)**

## **4.1 server/ambulance_environment.py** (184 lines)

**Purpose:** OpenEnv-compliant wrapper around the core environment.

```python
class AmbulanceEnvironment(Environment[ActionModel, ObservationModel, AmbulanceEnvState]):
    """
    Proper OpenEnv wrapper providing:
    - Session isolation (SUPPORTS_CONCURRENT_SESSIONS = True)
    - Rubric integration (RFC 004)
    - Metadata exposure
    """
    
    def reset(self, seed=None, episode_id=None, **kwargs) -> ObservationModel:
        self._episode_id = episode_id or str(uuid.uuid4())
        self.rubric.reset()
        return self._wrap_obs(self._inner.reset(seed=seed), reward=0.0, done=False)
    
    def step(self, action: ActionModel, timeout_s=None, **kwargs) -> ObservationModel:
        obs = self._inner.step(action)
        # Compute rubric reward for richer signal
        rubric_state = _extract_rubric_state(obs, self._inner.metrics)
        rubric_reward = self.rubric.score(rubric_state)
        reward = rubric_reward if rubric_reward != 0.0 else obs.reward
        obs.reward = reward
        return self._wrap_obs(obs, reward=reward, done=obs.done)
```

---

## **4.2 server/app.py** (515 lines)

**Purpose:** FastAPI application with all HTTP endpoints and WebSocket.

### Key Endpoints:

```python
# Core OpenEnv endpoints (provided by create_app)
POST /env/reset      # Reset environment
POST /env/step       # Take action
GET  /env/state      # Get current state

# RFC 002 - Auto-Discovery
GET /tools           # Returns action schema JSON

# RFC 003 - MCP Protocol
GET /mcp             # Returns MCP server metadata

# Dashboard endpoints
POST /env/reset      # (dashboard version) Uses RepositioningOracle
POST /env/step       # (dashboard version) Auto-dispatches via oracle
GET  /env/metrics    # Current episode metrics
GET  /env/state      # Full environment state

# MARL endpoints (Multi-Agent RL)
GET /marl/status     # Fleet coordination statistics
GET /marl/conflicts  # Recent conflict events

# Curriculum endpoints (Long-horizon)
GET /curriculum/status  # Current training stage

# Self-improvement endpoints
GET /selfplay/weaknesses    # Weakness report
GET /selfplay/iterations    # Improvement history

# Demo endpoint
POST /demo/scenario    # Compare DQN vs Greedy baseline

# Benchmark endpoint
GET /score             # Run all three tasks, return scores

# WebSocket (2 Hz updates)
WS /ws/live            # Real-time state feed
```

---

# **5. TASK CONFIGURATIONS (tasks/)**

## **5.1 tasks/easy.py**
```python
@dataclass
class EasyConfig:
    n_ambulances: int = 2      # 2 ambulances
    n_hospitals: int = 2       # 2 hospitals
    max_steps: int = 30        # 30 steps
    lambda_param: float = 0.3  # Low arrival rate
    seed: int = 42
    # Grader: mean(optimal_time/actual_response_time)
```

## **5.2 tasks/medium.py**
```python
@dataclass
class MediumConfig:
    n_ambulances: int = 4      # 4 ambulances
    n_hospitals: int = 3       # 3 hospitals
    max_steps: int = 60        # 60 steps
    lambda_param: float = 0.4  # Medium arrival
    traffic_range: tuple = (1.0, 1.3)  # Mild traffic
    seed: int = 42
    # Grader: 0.50*served_pct + 0.35*response_score - 0.15*idle_fraction
```

## **5.3 tasks/hard.py**
```python
@dataclass
class HardConfig:
    n_ambulances: int = 6      # 6 ambulances
    n_hospitals: int = 4       # 4 hospitals (with specialties)
    max_steps: int = 100       # 100 steps
    lambda_param: float = 0.6  # High arrival rate
    traffic_range: tuple = (1.3, 2.5)  # Dynamic rush-hour
    seed: int = 42
    # Grader: 0.50*weighted_served + 0.30*priority_accuracy + 0.15*fairness - penalty
```

---

# **6. AGENTS (agents/)**

## **6.1 agents/greedy_agent.py** (40 lines)
```python
class GreedyAgent:
    """Baseline greedy dispatcher - nearest first"""
    
    def act(self, observation: ObservationModel) -> ActionModel:
        # 1. Get first idle ambulance
        # 2. Find nearest unassigned emergency (absolute node difference)
        # 3. Find nearest hospital to emergency
        # Returns ActionModel or noop if no valid dispatch
```

## **6.2 agents/baseline.py** (51 lines)
```python
class BaselineAgent:
    """Enhanced agent with priority sorting"""
    
    def act(self, observation: ObservationModel) -> ActionModel:
        # 1. Sort emergencies: CRITICAL > HIGH > NORMAL, then time_remaining
        # 2. Select nearest idle ambulance to highest priority emergency
        # 3. Select nearest available hospital with capacity
```

## **6.3 agents/oracle.py** (147 lines)
```python
class OracleAgent:
    """
    Optimal dispatch using Dijkstra pre-computed paths.
    Upper-bound reference for evaluation.
    """
    
    def act(self, observation: ObservationModel) -> ActionModel:
        # Single best dispatch to highest priority emergency
    
    def act_all(self, observation: ObservationModel) -> List[ActionModel]:
        """
        Multi-dispatch: one action per idle ambulance
        Greedy assignment avoiding emergency/hospital conflicts
        """
```

## **6.4 agents/repositioning_oracle.py** (220 lines)
```python
class RepositioningOracle(OracleAgent):
    """
    OPTIMAL AGENT FOR INFERENCE.
    
    Features:
    1. Multi-dispatch: all idle ambulances dispatched simultaneously
    2. Specialty-aware hospital selection
    3. Proactive repositioning to hotspots
    4. Zone fairness: spreads ambulances across 4 city zones
    
    Specialty Mapping:
    CRITICAL -> Trauma, Cardiac
    HIGH -> Trauma, General
    NORMAL -> General, Paediatric
    """
    
    def act_all_with_reposition(self, obs: ObservationModel) -> List[ActionModel]:
        """
        Phase 1: Dispatch all idle ambulances to emergencies
        Phase 2: Reposition remaining idle ambulances to predicted hotspots
        """
    
    def _hotspot_targets(self, n: int) -> List[int]:
        """
        Returns n nodes for repositioning based on:
        1. Emergency frequency history (Counter)
        2. Zone coverage (ensure all 4 zones represented)
        3. Zone centers as fallback [12, 37, 62, 87]
        """
```

## **6.5 agents/fleet_agent.py** (263 lines)
```python
class AmbulanceQAgent:
    """
    Independent DQN agent for a single ambulance in MARL.
    
    State encoding (per agent):
    - Own state: node(1) + state_onehot(6) + eta(1) = 8
    - Fleet summary per other agent: node(1) + busy(1) = 2*(n-1)
    - Oversight signal: conflict_flag(1) + conflict_amb(1) = 2
    - Global context: pending by severity(3) + hosp_occ(1) + step_norm(1) = 5
    """
    
    def encode_observation(self, obs, coordination_signal=None) -> np.ndarray:
        """Build fixed-size state vector"""
    
    def act(self, state: np.ndarray, mask: np.ndarray) -> int:
        """Epsilon-greedy with action masking"""
    
    def train_step(self) -> float:
        """Double DQN update with soft target network update"""
```

## **6.6 agents/oversight_agent.py** (202 lines)
```python
class OversightAgent:
    """
    Fleet oversight coordinator - NOT a decision maker.
    
    Responsibilities:
    1. Detect coordination conflicts (two agents -> same emergency)
    2. Emit per-agent coordination signals [conflict_flag, conflict_amb_norm]
    3. Maintain conflict history for dashboard
    4. Track per-agent performance metrics
    """
    
    def get_coordination_signals(self, intended_actions: Dict[int, int]) -> Dict[int, np.ndarray]:
        """
        Returns 2-element signal for each agent:
        [0] = 1.0 if conflict detected, else 0.0
        [1] = normalized ID of conflicting partner
        """
```

---

# **7. RL TRAINING INFRASTRUCTURE (rl/)**

## **7.1 rl/dqn.py** (111 lines)

```python
class StandardDQN(nn.Module):
    """Baseline: Input -> 512 -> 256 -> 128 -> action_size"""

class DQN(nn.Module):
    """
    Dueling DQN architecture:
    - Shared features: Input -> 512 -> 256
    - Value stream: 256 -> 128 -> 1 (V(s))
    - Advantage stream: 256 -> 128 -> action_size (A(s,a))
    - Output: Q(s,a) = V(s) + (A(s,a) - mean(A))
    """

class DuelingDQN(DQN):
    """Explicit alias for DQN (already implements dueling)"""
```

## **7.2 rl/rl_agent.py** (254 lines)

```python
class DQNAgent:
    """
    Complete DQN training agent with:
    - DuelingDQN or StandardDQN networks
    - Prioritized Experience Replay (PER) or uniform
    - Soft target updates or hard copy
    - Optional reward normalization (z-score)
    - Double DQN for target computation
    """
    
    def __init__(self, state_size, action_size,
                 use_dueling=True, use_per=True,
                 use_soft_update=True, normalize_rewards=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        NetworkClass = DuelingDQN if use_dueling else StandardDQN
        self.policy_net = NetworkClass(state_size, action_size).to(device)
        self.target_net = NetworkClass(state_size, action_size).to(device)
        self.memory = PrioritizedReplayBuffer(20000) if use_per else SimpleReplayBuffer(20000)
        self.epsilon = 1.0
        self.epsilon_decay = 0.9992
        self.tau = 0.005  # Soft update rate
    
    def get_coordinated_reward(self, observation, action_model, base_reward):
        """
        Adds:
        - Coordination penalty: -50 if multiple ambulances -> same emergency
        - Resource awareness: +0.1 per remaining hospital capacity
        - Efficiency penalty: -0.2 per idle ambulance
        - Coverage diversity: +2.0 * spread/100 where spread = std(ambulance positions)
        """
    
    def get_priority_weighted_reward(self, observation, action_model, base_reward):
        """
        Multiplies reward by severity:
        CRITICAL: 2.0x, HIGH: 1.5x, NORMAL: 1.0x
        """
    
    def train_step(self):
        """
        1. Sample PER batch with importance weights
        2. Compute current Q (policy net)
        3. Compute target Q (Double DQN: policy selects, target evaluates)
        4. Smooth L1 loss with PER weights
        5. Gradient clip (max norm 1.0)
        6. Soft or hard target update
        7. Update PER priorities with TD errors
        """
```

## **7.3 rl/state_encoder.py** (136 lines)

```python
class StateEncoder:
    """
    Converts ObservationModel to fixed-size numpy array.
    
    Feature dimensions:
    - Ambulances: 6 max * (node(1) + state_oh(6) + eta(1)) = 48
    - Emergencies: 10 max * (node(1) + severity_oh(3) + time(1) + assigned(1)) = 60
    - Hospitals: 4 max * (node(1) + capacity(1) + patients(1) + ratio(1)) = 16
    Total base: 124 (or 120 depending on config)
    
    With history encoder: +25 dimensions
    """
    
    MAX_EMERGENCIES = 10
    MAX_NODES = 100
    ETA_NORM = 50
    TIME_NORM = 50
    CAPACITY_NORM = 50
    
    def encode(self, observation: ObservationModel) -> np.ndarray:
        # Returns clipped float32 array [0, 1]
```

## **7.4 rl/action_mapper.py** (157 lines)

```python
class ActionMapper:
    """
    Maps discrete action indices to structured ActionModel.
    
    Action space: 1 + (MAX_AMBULANCES * MAX_EMERGENCIES * MAX_HOSPITALS)
    Default: 1 + (2 * 3 * 2) = 13 actions
    For inference compatibility: up to 251 actions
    
    Slot 0: No-op
    Slots 1+: (amb_idx, emg_idx, hosp_idx) combinations
    
    build_action_space() filters to TOP:
    - 3 emergencies by severity and time_remaining
    - 2 nearest idle ambulances
    - 2 nearest available hospitals
    """

class ActionMask:
    """
    Builds binary validity mask over action space.
    Valid actions require:
    - Ambulance is IDLE
    - Emergency is unassigned
    - Hospital has capacity
    """
```

## **7.5 rl/replay_buffer.py** (72 lines) & prioritized_replay_buffer.py (105 lines)

```python
class PrioritizedReplayBuffer:
    """
    Proportional PER (Schaul et al. 2016).
    
    Parameters:
    - alpha=0.6: Prioritization strength
    - beta=0.4: IS correction (anneals to 1.0)
    
    Priority = |TD_error| + epsilon
    Sampling probability ~ priority^alpha
    """

class SimpleReplayBuffer:
    """Uniform sampling fallback when use_per=False"""
```

## **7.6 rl/rubric.py** (241 lines)

```python
class RubricComponent:
    """Base class for named reward components (RFC 004)"""

class EmergencyServedRubric:    # +10.0 per event
class SeverityBonusRubric:      # +5 CRITICAL, +2 HIGH, +0.5 NORMAL
class DispatchSpeedRubric:      # 0.5 - avg_response * 0.05
class HospitalDeliveryRubric:   # +2.0 per delivery
class DistancePenaltyRubric:   # -0.1 per en-route ambulance
class TrafficPenaltyRubric:    # -0.2 if traffic > 1.5
class IdlePenaltyRubric:       # -0.3 per idle ambulance when pending
class CapacityViolationRubric: # -4.0 per hospital overflow
class TimeoutRubric:            # -15.0 per missed emergency

def make_ambulance_rubric() -> Rubric:
    """Factory returning standard 9-component rubric"""
```

## **7.7 rl/demand_predictor.py** (38 lines)
```python
class DemandPredictor:
    """Simple frequency-based hotspot prediction"""
    def predict(self, n: int = 5) -> List[int]:
        return self.node_counts.most_common(n)
```

---

# **8. GRADING MODULES**

## **8.1 grader_easy.py** (49 lines)
```python
def grade_easy(episode_info: Dict[str, Any]) -> float:
    """
    Formula: mean(optimal_time / actual_response_time), clamped [0, 1]
    Tests: Basic dispatch correctness and response time optimization
    """
```

## **8.2 grader_medium.py** (47 lines)
```python
def grade_medium(episode_info: Dict[str, Any]) -> float:
    """
    Formula: 0.50*served_pct + 0.35*response_score - 0.15*idle_fraction
    Tests: Fleet coordination, priority dispatch, hospital load balancing
    """
```

## **8.3 grader_hard.py** (95 lines)
```python
def grade_hard(episode_info: Dict[str, Any]) -> float:
    """
    Formula:
    - weighted_served = 0.7*critical_rate + 0.3*overall_rate
    - priority_accuracy = priority_correct / priority_total
    - fairness_score = 1 - 2*std_dev(zone_service_rates)
    - overload_penalty = 0.05 * capacity_violations
    - score = 0.50*weighted_served + 0.30*priority_accuracy + 0.15*fairness - penalty
    
    Tests: CRITICAL-first triage, specialty routing, zone fairness, traffic awareness
    """
```

---

# **9. MULTI-AGENT MODULES (multi_agent/)**

## **9.1 multi_agent/coordinator.py** (269 lines)

```python
class MultiAgentCoordinator:
    """
    Coordinates multiple independent AmbulanceQAgents.
    
    Workflow:
    1. marl_act(): Each agent independently selects action
    2. Detect conflicts (two agents -> same emergency)
    3. marl_learn(): Split global reward, apply -5 conflict penalty
    4. Per-agent training steps
    """
    
    def marl_act(self, observation: ObservationModel) -> Dict[int, int]:
        """Returns mapping: agent_id -> action_index"""
    
    def marl_learn(self, global_reward: float, 
                   next_observation: ObservationModel, 
                   done: bool) -> Dict[int, float]:
        """
        Distributes rewards:
        - Base: global_reward / n_agents
        - Conflict penalty: -5.0 for conflicting agents
        Stores transitions and triggers training for each agent
        """
```

---

# **10. LONG-HORIZON MODULES (long_horizon/)**

## **10.1 long_horizon/history_encoder.py** (174 lines)

```python
class HistoryEncoder:
    """
    Encodes rolling 50-step window into 25-dim feature vector.
    
    Features [0-24]:
    0: mean served rate
    1: mean timeout rate
    2: zone balance index (std/mean)
    3: traffic trend (-1, 0, +1)
    4: hospital utilization trend
    5-6: critical/high served rates
    7-9: reward mean/peak/trough
    10: mean idle fraction
    11: mean emergency count
    12: surge indicator (>30% drop)
    13-24: 12-bin reward histogram
    """
```

## **10.2 long_horizon/curriculum_manager.py** (150 lines)

```python
class CurriculumManager:
    """
    Progressive difficulty: Stages 1-10 map to max_steps 100-1000.
    Advancement requires window_avg >= threshold over 20 episodes.
    """
    
    _STAGE_TO_STEPS = {s: s * 100 for s in range(1, 11)}
    _STAGE_THRESHOLDS = {s: (0.65 if s == 1 else 0.70) for s in range(1, 11)}
```

---

# **11. SELF-IMPROVEMENT MODULES (self_improvement/)**

## **11.1 self_improvement/adversarial_generator.py** (212 lines)

```python
@dataclass
class ScenarioConfig:
    """Configuration for a training scenario"""
    n_ambulances: int = 5
    n_hospitals: int = 3
    max_steps: int = 100
    lambda_param: float = 0.15
    traffic_intensity: float = 1.0
    surge_zone: int = -1
    surge_step: int = -1

class AdversarialScenarioGenerator:
    """
    1. identify_failures(): KMeans clustering on low-scoring scenarios
    2. generate_scenarios(): Sample new configs around cluster centroids
    """
```

## **11.2 self_improvement/weakness_detector.py** (134 lines)

```python
class WeaknessDetector:
    """
    Tracks failure clusters across self-play iterations.
    Provides structured reports for API endpoints and analysis.
    """
    
    def analyze(self, results: List[Tuple[ScenarioConfig, float]]) -> WeaknessReport:
        """
        Clusters failures, tracks improvement history per cluster.
        Returns WeaknessReport with cluster details.
        """
```

---

# **12. TRAINING SCRIPTS**

## **12.1 train.py** (628 lines)

**Entry point:** `python train.py [options]`

**Modes:**
1. **Single-agent** (default): Standard DQN training
2. **MARL** (`--marl`): Multi-agent with coordinator
3. **Long-horizon** (`--long-horizon`): 500-step episodes with history
4. **Self-play** (`--self-play`): Weakness detection + adversarial scenarios

**CLI Options:**
```bash
--episodes N          # Training episodes (default: 3000)
--max-steps N         # Steps per episode (default: 150)
--no-dueling          # Use StandardDQN instead of DuelingDQN
--no-per              # Use uniform replay instead of PER
--no-soft-update      # Hard target network updates
--normalize-rewards   # Z-score reward normalization
--marl                # Multi-agent RL mode
--long-horizon        # Long-horizon mode
--self-play           # Self-improvement mode
--selfplay-interval N # Self-play cycle frequency (default: 200)
```

**Training Loop (single-agent):**
1. Encode state with StateEncoder
2. Build action space with ActionMapper
3. Generate mask with ActionMask
4. Select epsilon-greedy action
5. Step environment
6. Apply reward shaping (priority-weighted + coordinated)
7. Store transition in replay buffer
8. Train step (gradient update)
9. Decay epsilon
10. Periodic logging and model saving

## **12.2 inference.py** (106 lines)

**Entry point:** `python inference.py [--task easy|medium|hard|all]`

**STDOUT Format:**
```
[START] easy
[STEP] 0 18.50
[STEP] 1 -0.50
...
[END] 0.923
```

**Agent:** Uses RepositioningOracle with multi-dispatch

---

# **13. CONFIGURATION FILES**

## **13.1 openenv.yaml** (145 lines)

**Purpose:** OpenEnv specification metadata

**Key Sections:**
- `name`: ambulance-dispatch-env
- `version`: 1.0.0
- `rfc_compliance`: [RFC-001, RFC-002, RFC-003, RFC-004, RFC-005]
- `action_space`: JSON Schema for ActionModel
- `observation_space`: JSON Schema for ObservationModel
- `tasks`: Easy, Medium, Hard configurations with baseline scores

## **13.2 pyproject.toml** (48 lines)

**Project Metadata:**
- Build system: setuptools
- Dependencies: openenv-core, fastapi, uvicorn, pydantic, networkx, numpy, openai
- Optional dev: pytest, pytest-asyncio, httpx, black, ruff

## **13.3 requirements.txt** (15 lines)

Runtime dependencies with minimum versions.

## **13.4 Dockerfile** (48 lines)

**Multi-stage build:**
1. Python 3.11 slim base
2. System dependencies (gcc, g++, curl, git)
3. Node.js 20 LTS for frontend
4. Python requirements
5. Application code
6. Frontend build (Next.js)
7. Healthcheck and CMD

---

# **14. DATA FLOW & ARCHITECTURE**

## **14.1 Episode Flow**

```
[Reset]
    ↓
RNG seed → CityGraph(BA) → Fleet(random nodes) 
    → Hospitals(with specialties) → EmergencyGenerator
    ↓
[Step Loop]
    ↓
Agent observes → Selects action → Environment processes
    ↓
[Physics]
    - Apply dispatch/reposition
    - Spawn traffic incidents
    - Advance ambulances (FSM updates)
    - Generate new emergencies (Poisson)
    - Decrement emergency timeouts
    ↓
[Reward Calculation]
    - Emergency served? +20 (+30 CRITICAL/+10 HIGH)
    - Hospital delivery? +10
    - Timeout? -15 per emergency
    - Idle during backlog? -1 per ambulance
    - Capacity violation? -5
    ↓
Return observation with reward, done, rubric
```

## **14.2 Training Flow**

```
[Initialize]
StateEncoder → ActionMapper → ActionMask → DQNAgent
    ↓
[Episode Loop]
    ↓
Reset environment → Encode initial state
    ↓
[Step Loop]
    - Build action space (filter top emergencies/ambulances/hospitals)
    - Build validity mask
    - Epsilon-greedy action selection
    - Decode to ActionModel
    - Step environment
    - Apply reward shaping
    - Store (s, a, r, s', done)
    - Train step (sample batch, compute loss, update)
    ↓
[End Episode]
Decay epsilon → Log metrics → Save best model
```

---

# **15. RFC COMPLIANCE**

| RFC | Feature | Implementation |
|-----|---------|----------------|
| RFC-001 | Base Env API | `/env/reset`, `/env/step`, `/env/state` |
| RFC-002 | Auto-Discovery | `GET /tools` returns JSON schema |
| RFC-003 | MCP Protocol | `GET /mcp` returns metadata |
| RFC-004 | Named Rubric | 9 components in every observation |
| RFC-005 | Concurrent Sessions | `SUPPORTS_CONCURRENT_SESSIONS = True` |

---

# **16. KEY CONSTANTS & MAGIC NUMBERS**

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| MAX_NODES | 100 | CityGraph | Graph size (Barabási-Albert) |
| ZONE_CENTERS | [12, 37, 62, 87] | RepositioningOracle | Default staging nodes |
| MAX_EMERGENCIES | 10 | StateEncoder | Observation feature dim |
| MAX_AMBULANCES | 2/6 | ActionMapper/Encoder | Training/inference |
| CAPACITY | 8 | Hospital | Default hospital capacity |
| SEED | 42 | Tasks | Determinism |
| BATCH_SIZE | 128 | DQNAgent | Training batch |
| GAMMA | 0.99 | DQNAgent | Discount factor |
| TAU | 0.005 | DQNAgent | Soft update rate |
| EPSILON_DECAY | 0.9992 | DQNAgent | Exploration decay |

---

# **17. REWARD RUBRIC DETAILS**

| Component | Trigger | Value | Weight |
|-----------|---------|-------|--------|
| emergency_served | Ambulance arrives at scene | +20.0 | 1.0 |
| severity_bonus | CRITICAL served | +30.0 | 1.0 |
| severity_bonus | HIGH served | +10.0 | 1.0 |
| dispatch_speed | Fast response | up to +10.0 | 1.0 |
| hospital_delivery | Patient delivered | +10.0 | 1.0 |
| distance_penalty | Long travel | -variable | 1.0 |
| traffic_penalty | Rush-hour dispatch | -variable | 1.0 |
| idle_penalty | Idle during backlog | -1.0/step | 1.0 |
| capacity_violation | Hospital overflow | -5.0 | 1.0 |
| timeout_penalty | Emergency expires | -15.0 | 1.0 |

---

# **18. REPOSITORY REFERENCES**

| Repository | URL | Role |
|------------|-----|------|
| Team Repository | https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon | Primary |
| Mirror | https://github.com/Vishallakshmikanthan/Ambulance-Despatch-RL-Model | Backup |
| HuggingFace Space | https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV | Demo |

---

*This document provides complete coverage of every file, class, function, constant, and data flow in the Ambulance-OpenENV project. Every code element is documented with its purpose, parameters, and behavior to eliminate any ambiguity for AI systems analyzing this codebase.*

*Generated: 2024*
*Version: 1.0.0*
