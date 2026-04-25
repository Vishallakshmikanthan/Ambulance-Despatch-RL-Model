import uuid
import numpy as np
import random
import asyncio
from typing import Tuple, Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from env.models import (
    ObservationModel, ActionModel, RewardModel,
    AmbulanceState, Severity, AmbulanceInfo, EmergencyInfo, HospitalInfo,
    AmbulanceEnvState, Rubric, RewardValidator
)
from env.simulator import CityGraph, TrafficEngine, AmbulanceFleet, EmergencyGenerator, Hospital

# Thread pool for CPU-bound Dijkstra computations
_executor = ThreadPoolExecutor(max_workers=4)

class AmbulanceEnvironment:
    """
    Production-grade Ambulance Dispatch Environment following OpenEnv RFC standards.
    Supports concurrent sessions, async I/O, and named reward rubrics.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.n_ambulances = self.config.get("n_ambulances", 5)
        self.n_hospitals = self.config.get("n_hospitals", 3)
        self.max_steps = self.config.get("max_steps", 1440)
        self.graph_size = self.config.get("graph_size", 100)
        
        self.seed_val = self.config.get("seed", 42)
        self.reset(seed=self.seed_val)

    def reset(self, seed: Optional[int] = None) -> ObservationModel:
        if seed is not None:
            self.seed_val = seed

        self.rng = np.random.default_rng(self.seed_val)
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)

        self.episode_id = str(uuid.uuid4())
        self.city_graph = CityGraph(n=self.graph_size)
        self.nodes = list(self.city_graph.graph.nodes())
        self.fleet = AmbulanceFleet(n=self.n_ambulances, nodes=self.nodes, rng=self.rng)
        self.generator = EmergencyGenerator(
            nodes=self.nodes,
            lambda_param=self.config.get("lambda_param", 0.15),
            rng=self.rng
        )
        self.traffic = TrafficEngine(rng=self.rng)
        
        # Specialties for Rule #8
        specs = ["Trauma", "Cardiac", "General", "Paediatric"]
        self.hospitals = {}
        for i in range(self.n_hospitals):
            node = int(self.rng.choice(self.nodes))
            spec = specs[i % len(specs)]
            self.hospitals[i] = Hospital(hosp_id=i, node=node, capacity=8)
            self.hospitals[i].specialty = spec

        self.step_count = 0
        self.active_emergencies: List[EmergencyInfo] = []
        self.last_rubric = Rubric()
        self._noop_streak = 0  # escalating no-op penalty counter

        # Zone bounds split node space into 4 equal quartiles
        q = self.graph_size // 4
        self._zone_bounds = [(0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, self.graph_size)]

        self.metrics = {
            "served": 0, "missed": 0, "total_emergencies": 0,
            "critical_served": 0, "critical_total": 0,
            "high_served": 0, "normal_served": 0,
            "avg_response_time": 0.0, "hospital_overflow": 0,
            "capacity_violations": 0,
            "idle_fraction": 0.0, "idle_steps": 0, "total_steps": 0,
            "priority_correct": 0, "priority_total": 0,
            "response_times": [], "optimal_times": [],
            "zone_served": {0: 0, 1: 0, 2: 0, 3: 0},
            "zone_total":  {0: 0, 1: 0, 2: 0, 3: 0},
        }
        self._dispatch_times: Dict[str, int] = {}
        self._dispatch_nodes: Dict[str, int] = {}  # emg_id -> ambulance node at dispatch
        self._response_times: List[float] = []
        self._optimal_times: List[float] = []
        self._idle_steps = 0
        self.last_info: Dict[str, Any] = {}
        self.last_reward: float = 0.0
        self.last_done: bool = False
        self.last_action: Optional[ActionModel] = None
        self._reward_validator = RewardValidator()

        return self._get_observation()

    async def reset_async(self, seed: Optional[int] = None, **kwargs) -> ObservationModel:
        return self.reset(seed=seed)

    @property
    def state(self) -> AmbulanceEnvState:
        tm = self.traffic.get_multiplier(self.step_count)
        return AmbulanceEnvState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            metrics=self.metrics,
            ambulances=[a.to_info().model_dump() for a in self.fleet.ambulances],
            hospitals=[h.to_info().model_dump() for h in self.hospitals.values()],
            emergencies=[e.model_dump() for e in self.active_emergencies],
            traffic_multiplier=tm,
            rubric=self.last_rubric
        )

    async def step_async(self, action: ActionModel, **kwargs) -> ObservationModel:
        return self.step(action)

    def step(self, action: ActionModel) -> ObservationModel:
        self.step_count += 1
        rubric = Rubric()
        tm = self.traffic.get_multiplier(self.step_count)

        # --- Anti-reward hacking: action validation ---
        if action is None:
            obs_err = self._get_observation()
            obs_err.reward = -10.0
            obs_err.done = True
            self.last_info = {"error": "invalid_action", "reward_success": 0.0, "reward_efficiency": 0.0, "reward_penalty": -10.0}
            self.last_reward = -10.0
            self.last_done = True
            return obs_err

        # --- Independent Check 1: invalid reference penalty ---
        amb_ids = {a.id for a in self.fleet.ambulances}
        emg_ids = {e.id for e in self.active_emergencies}
        ref_penalty = self._reward_validator.check_references(
            action, amb_ids, emg_ids, set(self.hospitals.keys())
        )
        rubric.validation_penalty += ref_penalty

        # --- Independent Check 2: loop / exploit detection ---
        loop_penalty = self._reward_validator.check_loop(action)
        rubric.validation_penalty += loop_penalty

        if not action.is_noop and action.ambulance_id is not None:
            self.last_action = action

        # 1. Process Dispatch Action
        if action.ambulance_id is not None and not action.is_noop:
            amb = next((a for a in self.fleet.ambulances if a.id == action.ambulance_id), None)
            emg = next((e for e in self.active_emergencies if e.id == action.emergency_id and not e.assigned), None)
            hosp = self.hospitals.get(action.hospital_id)

            if amb and emg and hosp and amb.state == AmbulanceState.IDLE:
                if hosp.is_available():
                    # Specialty mismatch penalty (Hard task feature)
                    specialty_map = {
                        "CRITICAL": ["Trauma", "Cardiac"],
                        "HIGH": ["Trauma", "General"],
                        "NORMAL": ["General", "Paediatric"],
                    }
                    preferred = specialty_map.get(emg.severity.value, [])
                    if preferred and hosp.specialty not in preferred:
                        rubric.capacity_violation -= 1.0  # specialty mismatch penalty

                    self.fleet.dispatch(amb.id, emg.id, emg.node, hosp.id, hosp.node)
                    emg.assigned = True
                    self._dispatch_times[emg.id] = self.step_count
                    self._dispatch_nodes[emg.id] = amb.node  # for optimal time computation

                    # RFC Feature: Dispatch Speed Rubric
                    wait_time = self.step_count - emg.spawn_time
                    rubric.dispatch_speed = max(0, 10.0 - wait_time * 0.5)
                    self._noop_streak = 0
                else:
                    rubric.capacity_violation -= 5.0
                    self.metrics["hospital_overflow"] += 1
                    self.metrics["capacity_violations"] = self.metrics.get("capacity_violations", 0) + 1
                    self._noop_streak += 1
            else:
                # Invalid action / dispatching non-idle ambulance
                # Do NOT penalise pure-reposition actions that have no emergency_id
                if action.reposition_node is None:
                    self._noop_streak += 1
                    rubric.idle_penalty -= 2.0 + self._noop_streak * 0.5  # escalating
        else:
            # No-op: escalating penalty when emergencies are active
            if self.active_emergencies:
                self._noop_streak += 1
                rubric.idle_penalty -= 0.5 * self._noop_streak
            else:
                self._noop_streak = 0

        # --- REPOSITION LOGIC (processes reposition_node if set) ---
        if (
            action.reposition_node is not None
            and action.ambulance_id is not None
            and not action.is_noop
        ):
            _amb = next(
                (a for a in self.fleet.ambulances if a.id == action.ambulance_id),
                None,
            )
            if _amb and _amb.state == AmbulanceState.IDLE:
                tm_current = self.traffic.get_multiplier(self.step_count)
                self.fleet.reposition(
                    action.ambulance_id,
                    action.reposition_node,
                    self.city_graph,
                    tm_current,
                )

        # 2. Simulation Physics (with deterministic incident events for Hard task)
        pre_states = {a.id: a.state for a in self.fleet.ambulances}
        graph_edges = list(self.city_graph.graph.edges())
        self.traffic.maybe_spawn_incident(graph_edges, self.step_count)
        self.traffic.tick_incidents()
        self.fleet.step_update(self.city_graph, tm)
        post_states = {a.id: a.state for a in self.fleet.ambulances}

        # 3. Emergency Dynamics
        new_emgs = self.generator.generate(self.step_count)
        for ne in new_emgs:
            ne.spawn_time = self.step_count
            # Track critical_total for grader_hard
            if ne.severity == Severity.CRITICAL:
                self.metrics["critical_total"] = self.metrics.get("critical_total", 0) + 1
            # Track priority_total at spawn for CRITICAL and HIGH (grader accuracy denominator)
            if ne.severity in (Severity.CRITICAL, Severity.HIGH):
                self.metrics["priority_total"] = self.metrics.get("priority_total", 0) + 1
            # Assign zone for fairness tracking
            ne_zone = next(
                (z for z, (lo, hi) in enumerate(self._zone_bounds) if lo <= ne.node < hi),
                0
            )
            self.metrics["zone_total"][ne_zone] = self.metrics["zone_total"].get(ne_zone, 0) + 1
        self.active_emergencies.extend(new_emgs)
        self.metrics["total_emergencies"] += len(new_emgs)

        missed = 0
        rem = []
        for e in self.active_emergencies:
            if not e.assigned:
                e.time_remaining -= 1
                if e.time_remaining <= 0:
                    missed += 1
                    continue
            rem.append(e)
        self.active_emergencies = rem
        rubric.timeout_penalty -= missed * 15.0
        self.metrics["missed"] += missed

        # 4. Success Event Processing
        for amb in self.fleet.ambulances:
            if pre_states[amb.id] == AmbulanceState.EN_ROUTE and post_states[amb.id] == AmbulanceState.AT_SCENE:
                rubric.emergency_served += 20.0
                self.metrics["served"] += 1

                emg = next((e for e in self.active_emergencies if e.id == amb.target_emg_id), None)
                if emg:
                    if emg.severity == Severity.CRITICAL:
                        rubric.severity_bonus += 30.0
                        self.metrics["critical_served"] += 1
                        self.metrics["priority_correct"] = self.metrics.get("priority_correct", 0) + 1
                    elif emg.severity == Severity.HIGH:
                        rubric.severity_bonus += 10.0
                        self.metrics["high_served"] += 1
                        self.metrics["priority_correct"] = self.metrics.get("priority_correct", 0) + 1
                    else:
                        self.metrics["normal_served"] += 1
                        # NORMAL is not a priority dispatch — excluded from priority_correct
                    # Zone tracking for fairness metric
                    emg_zone = next(
                        (z for z, (lo, hi) in enumerate(self._zone_bounds) if lo <= emg.node < hi),
                        0
                    )
                    self.metrics["zone_served"][emg_zone] = self.metrics["zone_served"].get(emg_zone, 0) + 1
                    self.active_emergencies = [e for e in self.active_emergencies if e.id != amb.target_emg_id]

                # Record response time and optimal time for graders
                dispatch_t = self._dispatch_times.get(amb.target_emg_id or "")
                if dispatch_t is not None:
                    response_t = float(self.step_count - dispatch_t)
                    self._response_times.append(response_t)
                    self.metrics["response_times"] = self._response_times
                    self.metrics["avg_response_time"] = float(
                        sum(self._response_times) / len(self._response_times)
                    )
                    # Compute optimal (no-traffic) time for grade_easy
                    src_node = self._dispatch_nodes.get(amb.target_emg_id or "")
                    if src_node is not None and emg is not None:
                        try:
                            opt_t = float(self.city_graph.shortest_path_time(src_node, emg.node, 1.0))
                        except Exception:
                            opt_t = response_t
                        self._optimal_times.append(max(1.0, opt_t))
                        self.metrics["optimal_times"] = self._optimal_times

            if pre_states[amb.id] == AmbulanceState.TRANSPORTING and post_states[amb.id] == AmbulanceState.RETURNING:
                rubric.hospital_delivery += 10.0
                hosp = self.hospitals.get(amb.target_hosp_id)
                if hosp: hosp.release()

        # 5. Idle penalty and idle_fraction tracking
        idle_count = len([a for a in self.fleet.ambulances if a.state == AmbulanceState.IDLE])
        if self.active_emergencies and idle_count > 0:
            rubric.idle_penalty -= idle_count * 1.0
            self._idle_steps += idle_count

        # Update idle_fraction and idle_steps metrics
        self.metrics["total_steps"] = self.step_count
        self.metrics["idle_steps"] = self._idle_steps
        total_ambulance_steps = self.n_ambulances * self.step_count
        self.metrics["idle_fraction"] = (
            self._idle_steps / total_ambulance_steps if total_ambulance_steps > 0 else 0.0
        )

        self.last_rubric = rubric
        done = self.step_count >= self.max_steps
        obs = self._get_observation(tm)

        # --- Independent Check 3: reward inflation cap (applied after all bonuses computed) ---
        rubric.validation_penalty += RewardValidator.clip_positive(rubric)

        # --- Fairness score: reward zone-balanced coverage ---
        zone_served = self.metrics["zone_served"]
        zone_total = self.metrics["zone_total"]
        zone_rates = [
            zone_served.get(z, 0) / max(zone_total.get(z, 1), 1)
            for z in range(4)
        ]
        if max(zone_rates) > 0:
            rubric.fairness_score = min(zone_rates) / max(zone_rates) * 5.0

        obs.reward = rubric.total()
        obs.done = done
        obs.rubric = rubric
        obs.reward_model = RewardModel.from_rubric(
            rubric,
            served=self.metrics["served"],
            missed=self.metrics["missed"],
        )

        # Multi-component reward breakdown (judge visible)
        reward_success = rubric.emergency_served + rubric.severity_bonus + rubric.hospital_delivery
        reward_efficiency = rubric.dispatch_speed + rubric.fairness_score
        reward_penalty = (
            rubric.idle_penalty + rubric.capacity_violation
            + rubric.timeout_penalty + rubric.distance_penalty
            + rubric.traffic_penalty + rubric.validation_penalty
        )
        self.last_info = {
            "reward_success": reward_success,
            "reward_efficiency": reward_efficiency,
            "reward_penalty": reward_penalty,
            "ref_penalty": ref_penalty,
            "loop_penalty": loop_penalty,
            "rubric": rubric.model_dump(),
        }
        self.last_reward = obs.reward
        self.last_done = done
        return obs

    def step_all(self, actions: List[ActionModel]) -> ObservationModel:
        """
        Process multiple dispatch/reposition actions in a SINGLE simulation tick.
        Useful for multi-ambulance dispatch without wasting simulation steps.
        All actions are applied before physics advances.
        """
        self.step_count += 1
        rubric = Rubric()
        tm = self.traffic.get_multiplier(self.step_count)

        for action in actions:
            if action.is_noop or action.ambulance_id is None:
                continue
            # Handle reposition
            if action.reposition_node is not None and not action.emergency_id:
                _amb = next((a for a in self.fleet.ambulances if a.id == action.ambulance_id), None)
                if _amb and _amb.state == AmbulanceState.IDLE:
                    self.fleet.reposition(action.ambulance_id, action.reposition_node, self.city_graph, tm)
                continue
            # Handle dispatch
            amb = next((a for a in self.fleet.ambulances if a.id == action.ambulance_id), None)
            emg = next((e for e in self.active_emergencies if e.id == action.emergency_id and not e.assigned), None)
            hosp = self.hospitals.get(action.hospital_id)
            if amb and emg and hosp and amb.state == AmbulanceState.IDLE:
                if hosp.is_available():
                    specialty_map = {
                        "CRITICAL": ["Trauma", "Cardiac"],
                        "HIGH": ["Trauma", "General"],
                        "NORMAL": ["General", "Paediatric"],
                    }
                    preferred = specialty_map.get(emg.severity.value, [])
                    if preferred and hosp.specialty not in preferred:
                        rubric.capacity_violation -= 1.0
                    self.fleet.dispatch(amb.id, emg.id, emg.node, hosp.id, hosp.node)
                    emg.assigned = True
                    self._dispatch_times[emg.id] = self.step_count
                    self._dispatch_nodes[emg.id] = amb.node
                    wait_time = self.step_count - emg.spawn_time
                    rubric.dispatch_speed += max(0, 10.0 - wait_time * 0.5)
                    self._noop_streak = 0
                else:
                    rubric.capacity_violation -= 5.0
                    self.metrics["hospital_overflow"] += 1
                    self.metrics["capacity_violations"] = self.metrics.get("capacity_violations", 0) + 1

        # Physics tick (same as step())
        pre_states = {a.id: a.state for a in self.fleet.ambulances}
        graph_edges = list(self.city_graph.graph.edges())
        self.traffic.maybe_spawn_incident(graph_edges, self.step_count)
        self.traffic.tick_incidents()
        self.fleet.step_update(self.city_graph, tm)
        post_states = {a.id: a.state for a in self.fleet.ambulances}

        new_emgs = self.generator.generate(self.step_count)
        for ne in new_emgs:
            ne.spawn_time = self.step_count
            if ne.severity == Severity.CRITICAL:
                self.metrics["critical_total"] = self.metrics.get("critical_total", 0) + 1
            # Track priority_total at spawn for CRITICAL and HIGH (grader accuracy denominator)
            if ne.severity in (Severity.CRITICAL, Severity.HIGH):
                self.metrics["priority_total"] = self.metrics.get("priority_total", 0) + 1
            ne_zone = next((z for z, (lo, hi) in enumerate(self._zone_bounds) if lo <= ne.node < hi), 0)
            self.metrics["zone_total"][ne_zone] = self.metrics["zone_total"].get(ne_zone, 0) + 1
        self.active_emergencies.extend(new_emgs)
        self.metrics["total_emergencies"] += len(new_emgs)

        missed = 0
        rem = []
        for e in self.active_emergencies:
            if not e.assigned:
                e.time_remaining -= 1
                if e.time_remaining <= 0:
                    missed += 1
                    continue
            rem.append(e)
        self.active_emergencies = rem
        rubric.timeout_penalty -= missed * 15.0
        self.metrics["missed"] += missed

        for amb in self.fleet.ambulances:
            if pre_states[amb.id] == AmbulanceState.EN_ROUTE and post_states[amb.id] == AmbulanceState.AT_SCENE:
                rubric.emergency_served += 20.0
                self.metrics["served"] += 1
                emg = next((e for e in self.active_emergencies if e.id == amb.target_emg_id), None)
                if emg:
                    if emg.severity == Severity.CRITICAL:
                        rubric.severity_bonus += 30.0
                        self.metrics["critical_served"] += 1
                        self.metrics["priority_correct"] = self.metrics.get("priority_correct", 0) + 1
                    elif emg.severity == Severity.HIGH:
                        rubric.severity_bonus += 10.0
                        self.metrics["high_served"] += 1
                        self.metrics["priority_correct"] = self.metrics.get("priority_correct", 0) + 1
                    else:
                        self.metrics["normal_served"] += 1
                        # NORMAL is not a priority dispatch — excluded from priority_correct
                    emg_zone = next((z for z, (lo, hi) in enumerate(self._zone_bounds) if lo <= emg.node < hi), 0)
                    self.metrics["zone_served"][emg_zone] = self.metrics["zone_served"].get(emg_zone, 0) + 1
                    self.active_emergencies = [e for e in self.active_emergencies if e.id != amb.target_emg_id]
                dispatch_t = self._dispatch_times.get(amb.target_emg_id or "")
                if dispatch_t is not None:
                    response_t = float(self.step_count - dispatch_t)
                    self._response_times.append(response_t)
                    self.metrics["response_times"] = self._response_times
                    self.metrics["avg_response_time"] = float(sum(self._response_times) / len(self._response_times))
                    src_node = self._dispatch_nodes.get(amb.target_emg_id or "")
                    if src_node is not None and emg is not None:
                        try:
                            opt_t = float(self.city_graph.shortest_path_time(src_node, emg.node, 1.0))
                        except Exception:
                            opt_t = response_t
                        self._optimal_times.append(max(1.0, opt_t))
                        self.metrics["optimal_times"] = self._optimal_times

            if pre_states[amb.id] == AmbulanceState.TRANSPORTING and post_states[amb.id] == AmbulanceState.RETURNING:
                rubric.hospital_delivery += 10.0
                hosp = self.hospitals.get(amb.target_hosp_id)
                if hosp:
                    hosp.release()

        idle_count = len([a for a in self.fleet.ambulances if a.state == AmbulanceState.IDLE])
        if self.active_emergencies and idle_count > 0:
            rubric.idle_penalty -= idle_count * 1.0
            self._idle_steps += idle_count

        self.metrics["total_steps"] = self.step_count
        self.metrics["idle_steps"] = self._idle_steps
        total_ambulance_steps = self.n_ambulances * self.step_count
        self.metrics["idle_fraction"] = (self._idle_steps / total_ambulance_steps if total_ambulance_steps > 0 else 0.0)

        self.last_rubric = rubric
        done = self.step_count >= self.max_steps
        obs = self._get_observation(tm)

        # Inflation cap + fairness (mirrored from step())
        rubric.validation_penalty += RewardValidator.clip_positive(rubric)
        zone_served = self.metrics["zone_served"]
        zone_total = self.metrics["zone_total"]
        zone_rates = [
            zone_served.get(z, 0) / max(zone_total.get(z, 1), 1)
            for z in range(4)
        ]
        if max(zone_rates) > 0:
            rubric.fairness_score = min(zone_rates) / max(zone_rates) * 5.0

        obs.reward = rubric.total()
        obs.done = done
        obs.rubric = rubric

        reward_success = rubric.emergency_served + rubric.severity_bonus + rubric.hospital_delivery
        reward_efficiency = rubric.dispatch_speed + rubric.fairness_score
        reward_penalty = (
            rubric.idle_penalty + rubric.capacity_violation
            + rubric.timeout_penalty + rubric.distance_penalty
            + rubric.traffic_penalty + rubric.validation_penalty
        )
        self.last_info = {
            "reward_success": reward_success,
            "reward_efficiency": reward_efficiency,
            "reward_penalty": reward_penalty,
            "rubric": rubric.model_dump(),
        }
        self.last_reward = obs.reward
        self.last_done = done
        return obs

    def _get_observation(self, tm: Optional[float] = None) -> ObservationModel:
        if tm is None: tm = self.traffic.get_multiplier(self.step_count)
        return ObservationModel(
            ambulances=[amb.to_info() for amb in self.fleet.ambulances],
            emergencies=[e for e in self.active_emergencies if not e.assigned],
            hospitals=[h.to_info() for h in self.hospitals.values()],
            traffic={"global": tm},
            step=self.step_count
        )
