from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from openenv.core.env_server import Observation, Action, State


class Rubric(BaseModel):
    """Named reward components for fine-grained diagnostic logging."""
    emergency_served: float = 0.0
    severity_bonus: float = 0.0
    dispatch_speed: float = 0.0
    hospital_delivery: float = 0.0
    distance_penalty: float = 0.0
    traffic_penalty: float = 0.0
    idle_penalty: float = 0.0
    capacity_violation: float = 0.0
    timeout_penalty: float = 0.0
    fairness_score: float = 0.0
    validation_penalty: float = 0.0  # anti-reward-hacking deductions

    def total(self) -> float:
        return sum(self.model_dump().values())


class RewardValidator:
    """
    Independent reward-signal validator.

    Enforces three independent checks on every step so that reward signals
    remain honest and cannot be exploited:

    1. **Invalid-reference penalty** — ambulance_id / emergency_id / hospital_id
       that do not exist in the live environment incur a flat −10 each.
    2. **Loop-exploit detection** — tracks a rolling window of the last
       ``window`` action fingerprints.  Penalty escalates with repetition:
       ``−5 × repeat_count``, capped at ``−50``.
    3. **Reward-inflation cap** — a single step can award at most
       ``max_positive`` in positive reward components (emergency_served +
       severity_bonus + hospital_delivery + dispatch_speed).  Anything above
       the cap is clipped.  This prevents a single manufactured mega-step from
       inflating episode scores.
    """

    MAX_POSITIVE_PER_STEP: float = 80.0   # CRITICAL serve + delivery ceiling
    LOOP_WINDOW: int = 6                   # fingerprints remembered
    LOOP_BASE_PENALTY: float = -5.0
    LOOP_CAP: float = -50.0

    def __init__(self) -> None:
        self._action_history: list = []    # rolling fingerprint window

    def reset(self) -> None:
        self._action_history.clear()

    # ------------------------------------------------------------------
    # Check 1 — invalid references
    # ------------------------------------------------------------------
    def check_references(
        self,
        action: "ActionModel",
        ambulance_ids: set,
        emergency_ids: set,
        hospital_ids: set,
    ) -> float:
        """Return total penalty for any invalid ID references in the action."""
        penalty = 0.0
        if action.is_noop or action.ambulance_id is None:
            return penalty
        if action.ambulance_id not in ambulance_ids:
            penalty -= 10.0
        if action.emergency_id and action.emergency_id not in emergency_ids:
            penalty -= 10.0
        if action.hospital_id is not None and action.hospital_id not in hospital_ids:
            penalty -= 10.0
        return penalty

    # ------------------------------------------------------------------
    # Check 2 — loop / exploit detection
    # ------------------------------------------------------------------
    def check_loop(self, action: "ActionModel") -> float:
        """Penalise repeated identical dispatch actions (loop exploit)."""
        if action.is_noop or action.ambulance_id is None:
            return 0.0
        fingerprint = (action.ambulance_id, action.emergency_id, action.hospital_id)
        repeat_count = self._action_history.count(fingerprint)
        self._action_history.append(fingerprint)
        if len(self._action_history) > self.LOOP_WINDOW:
            self._action_history.pop(0)
        if repeat_count > 0:
            return max(self.LOOP_BASE_PENALTY * repeat_count, self.LOOP_CAP)
        return 0.0

    # ------------------------------------------------------------------
    # Check 3 — reward inflation cap
    # ------------------------------------------------------------------
    @staticmethod
    def clip_positive(rubric: "Rubric") -> float:
        """
        Return an additional penalty if the positive components of this step
        exceed the per-step ceiling.  Does NOT mutate the rubric.
        """
        positive = (
            rubric.emergency_served
            + rubric.severity_bonus
            + rubric.hospital_delivery
            + rubric.dispatch_speed
        )
        excess = positive - RewardValidator.MAX_POSITIVE_PER_STEP
        return -max(0.0, excess)


class AmbulanceState(str, Enum):
    IDLE = "idle"
    DISPATCHED = "dispatched"
    EN_ROUTE = "en_route"
    AT_SCENE = "at_scene"
    TRANSPORTING = "transporting"
    RETURNING = "returning"
    REPOSITIONING = "repositioning"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    NORMAL = "NORMAL"


class AmbulanceInfo(BaseModel):
    id: int
    node: int
    state: AmbulanceState
    eta: int
    target_emg_id: Optional[str] = None
    target_hosp_id: Optional[int] = None


class EmergencyInfo(BaseModel):
    id: str
    node: int
    severity: Severity
    time_remaining: int
    max_time_remaining: int = 90  # set at spawn time for frontend progress bars
    assigned: bool
    spawn_time: int = 0


class HospitalInfo(BaseModel):
    id: int
    node: int
    capacity: int
    current_patients: int
    specialty: str = "General"


# --- OpenEnv base-type subclasses ---

class ObservationModel(Observation):
    """Full environment observation returned from reset() and step()."""

    ambulances: List[AmbulanceInfo] = Field(default_factory=list)
    emergencies: List[EmergencyInfo] = Field(default_factory=list)
    hospitals: List[HospitalInfo] = Field(default_factory=list)
    traffic: Dict[str, float] = Field(default_factory=dict)
    step: int = 0
    reward: float = 0.0
    reward_model: Optional["RewardModel"] = None  # typed reward — populated each step()
    done: bool = False
    rubric: Optional[Rubric] = None


class ActionModel(Action):
    """Dispatch action sent by the agent each step.

    extra='forbid' is inherited from Action base — any unknown field raises
    a Pydantic validation error, catching typos in inference.py immediately.
    """

    ambulance_id: Optional[int] = None
    emergency_id: str = ""
    hospital_id: Optional[int] = None
    reposition_node: Optional[int] = None
    is_noop: bool = False


class AmbulanceEnvState(State):
    """Extended env state exposed via the state property."""

    episode_id: str
    step_count: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    ambulances: List[Dict[str, Any]] = Field(default_factory=list)
    hospitals: List[Dict[str, Any]] = Field(default_factory=list)
    emergencies: List[Dict[str, Any]] = Field(default_factory=list)
    traffic_multiplier: float = 1.0
    rubric: Optional[Rubric] = None


class RewardModel(BaseModel):
    """Typed reward returned per step — exposes scalar value and all named components."""
    value: float
    components: Dict[str, float]
    emergencies_served: int
    emergencies_missed: int

    @classmethod
    def from_rubric(cls, rubric: "Rubric", served: int, missed: int) -> "RewardModel":
        return cls(
            value=rubric.total(),
            components=rubric.model_dump(),
            emergencies_served=served,
            emergencies_missed=missed,
        )
