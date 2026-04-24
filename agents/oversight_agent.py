"""
oversight_agent.py — Fleet Oversight Agent.

The OversightAgent does NOT make dispatch decisions. Its role is to:
  1. Observe the full fleet state every step.
  2. Detect coordination conflicts (two agents intending the same target emergency).
  3. Emit per-agent coordination signals that modulate Q-value landscapes.
  4. Maintain a conflict history for the dashboard.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.models import ObservationModel, AmbulanceState


class ConflictEvent:
    """A detected conflict between two ambulance agents."""
    __slots__ = ("step", "agent_a", "agent_b", "emergency_id", "resolved")

    def __init__(self, step: int, agent_a: int, agent_b: int, emergency_id: str):
        self.step = step
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.emergency_id = emergency_id
        self.resolved = False

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "emergency_id": self.emergency_id,
            "resolved": self.resolved,
        }


class OversightAgent:
    """
    Fleet oversight coordinator.

    Workflow each step
    ------------------
    1. Call `observe(obs)` with the current environment observation.
    2. Call `get_coordination_signals(intended_actions)` where `intended_actions`
       is a dict mapping agent_id → action_index (emergency slot or noop).
    3. Pass the returned per-agent signals to each AmbulanceQAgent.encode_observation().
    4. After the environment steps, call `record_outcome(step, rewards)`.
    """

    _CONFLICT_HISTORY_SIZE = 200

    def __init__(self, n_agents: int, max_emergencies: int = 10):
        self.n_agents = n_agents
        self.max_emergencies = max_emergencies

        # Ring buffer of recent conflicts (for dashboard)
        self._conflict_history: deque[ConflictEvent] = deque(maxlen=self._CONFLICT_HISTORY_SIZE)

        # Running counters
        self.total_conflicts = 0
        self.total_resolutions = 0
        self.step_count = 0

        # Per-agent performance tracking
        self._agent_rewards: Dict[int, List[float]] = {i: [] for i in range(n_agents)}

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def observe(self, obs: ObservationModel) -> None:
        """Ingest the current observation (called once per env step)."""
        self.step_count += 1
        self._last_obs = obs

    def get_coordination_signals(
        self,
        intended_actions: Dict[int, int],
    ) -> Dict[int, np.ndarray]:
        """
        Compute a 2-element coordination signal for each agent.

        Signal layout
        -------------
        [0] conflict_flag     : 1.0 if this agent's intended action conflicts, else 0.0
        [1] conflict_amb_norm : normalized id of the conflicting ambulance (0.0 = none)

        Parameters
        ----------
        intended_actions : dict  agent_id → action_index
                          action_index 0..max_emergencies-1 = emergency slot index
                          action_index max_emergencies = noop
        """
        signals: Dict[int, np.ndarray] = {i: np.zeros(2, dtype=np.float32) for i in range(self.n_agents)}

        # Build mapping: action_index → list of agents that chose it
        action_to_agents: Dict[int, List[int]] = {}
        for agent_id, action in intended_actions.items():
            action_to_agents.setdefault(action, []).append(agent_id)

        # Detect conflicts (two+ agents targeting same non-noop emergency slot)
        for action, agents in action_to_agents.items():
            if action >= self.max_emergencies:
                continue  # noop — no conflict
            if len(agents) < 2:
                continue  # no conflict

            # Flag all involved agents
            for agent_id in agents:
                conflict_partners = [a for a in agents if a != agent_id]
                primary_partner = conflict_partners[0]  # flag first partner
                signals[agent_id][0] = 1.0
                signals[agent_id][1] = float(primary_partner) / max(self.n_agents - 1, 1)

            # Record conflict event
            emg_id = self._slot_to_emergency_id(action)
            event = ConflictEvent(
                step=self.step_count,
                agent_a=agents[0],
                agent_b=agents[1],
                emergency_id=emg_id,
            )
            self._conflict_history.append(event)
            self.total_conflicts += 1

        return signals

    def record_outcome(self, step: int, rewards: Dict[int, float]) -> None:
        """Record per-agent rewards after the step is resolved."""
        for agent_id, r in rewards.items():
            if agent_id in self._agent_rewards:
                self._agent_rewards[agent_id].append(r)

        # Mark latest conflict as resolved if rewards are positive overall
        if self._conflict_history:
            latest = self._conflict_history[-1]
            if not latest.resolved and latest.step == step:
                team_reward = sum(rewards.values())
                if team_reward > 0:
                    latest.resolved = True
                    self.total_resolutions += 1

    # ------------------------------------------------------------------
    # Dashboard / metrics
    # ------------------------------------------------------------------

    def get_conflict_history(self, last_n: int = 20) -> List[dict]:
        """Return the most recent `last_n` conflict events as dicts."""
        events = list(self._conflict_history)
        return [e.to_dict() for e in events[-last_n:]]

    def get_agent_metrics(self) -> Dict[int, dict]:
        """Per-agent summary statistics."""
        out = {}
        for agent_id, rewards in self._agent_rewards.items():
            if rewards:
                out[agent_id] = {
                    "total_reward": float(np.sum(rewards)),
                    "avg_reward": float(np.mean(rewards)),
                    "episodes": len(rewards),
                }
            else:
                out[agent_id] = {"total_reward": 0.0, "avg_reward": 0.0, "episodes": 0}
        return out

    def get_status(self) -> dict:
        """Summary dict for the /marl/status endpoint."""
        return {
            "step_count": self.step_count,
            "total_conflicts": self.total_conflicts,
            "total_resolutions": self.total_resolutions,
            "conflict_rate": (
                self.total_conflicts / max(self.step_count, 1)
            ),
            "agent_metrics": self.get_agent_metrics(),
        }

    def reset(self) -> None:
        """Reset between episodes (keeps history for dashboard continuity)."""
        self.step_count = 0
        for k in self._agent_rewards:
            self._agent_rewards[k] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _slot_to_emergency_id(self, slot: int) -> str:
        """Map emergency slot index back to emergency id if observation available."""
        try:
            emgs = self._last_obs.emergencies
            unassigned = [e for e in emgs if not e.assigned]
            if slot < len(unassigned):
                return unassigned[slot].id
        except Exception:
            pass
        return f"slot_{slot}"
