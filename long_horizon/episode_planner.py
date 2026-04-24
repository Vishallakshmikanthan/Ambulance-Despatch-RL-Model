"""
episode_planner.py — Provides an LLM/RL agent with an explicit planning interface.

At episode start the agent is asked to produce a high-level plan.
Every 100 steps the plan is fed back as a reminder in the observation context.

Usage
-----
planner = EpisodePlanner(n_windows=10, window_size=100)
plan = planner.request_plan(obs)          # at episode start (LLM mode)
context = planner.get_context(step)       # every 100 steps
planner.record_window_score(score)        # after each 100-step window
summary = planner.get_summary()          # end of episode
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from env.models import ObservationModel, Severity


@dataclass
class EpisodePlan:
    """High-level plan produced at the start of a long-horizon episode."""
    target_served_pct: float = 0.85       # overall goal
    reserve_ambulances: int = 1           # keep N ambulances idle at all times
    zone_priorities: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    hospital_strategy: str = "load_balance"  # or "nearest" or "specialty_first"
    raw_text: str = ""                    # LLM-generated text plan

    def to_dict(self) -> dict:
        return {
            "target_served_pct": self.target_served_pct,
            "reserve_ambulances": self.reserve_ambulances,
            "zone_priorities": self.zone_priorities,
            "hospital_strategy": self.hospital_strategy,
            "raw_text": self.raw_text,
        }


class EpisodePlanner:
    """
    Gives the agent an explicit planning context across a long episode.

    Parameters
    ----------
    n_windows  : int   Number of 100-step windows (e.g. 10 for 1000-step episode).
    window_size: int   Steps per window.
    """

    def __init__(self, n_windows: int = 10, window_size: int = 100):
        self.n_windows = n_windows
        self.window_size = window_size
        self._plan: Optional[EpisodePlan] = None
        self._window_scores: List[float] = []
        self._current_window = 0
        self._step = 0

    # ------------------------------------------------------------------
    # Planning interface
    # ------------------------------------------------------------------

    def request_plan(self, obs: ObservationModel) -> EpisodePlan:
        """
        Generate an initial plan from the observation.
        For LLM mode: returns a structured plan the LLM should fill.
        For RL mode: auto-generates a plan based on heuristics.
        """
        plan = self._heuristic_plan(obs)
        self._plan = plan
        self._window_scores = []
        self._current_window = 0
        self._step = 0
        return plan

    def get_context(self, step: int) -> dict:
        """Return the plan reminder context for the current step."""
        if self._plan is None:
            return {}

        window_idx = step // self.window_size
        steps_in_window = step % self.window_size
        remaining_windows = self.n_windows - window_idx

        return {
            "plan": self._plan.to_dict(),
            "current_window": window_idx,
            "steps_in_window": steps_in_window,
            "remaining_windows": remaining_windows,
            "window_scores_so_far": list(self._window_scores),
            "on_track": self._is_on_track(),
        }

    def record_step(self, step: int) -> None:
        """Track step progress; detect window transitions."""
        self._step = step
        new_window = step // self.window_size
        if new_window > self._current_window:
            self._current_window = new_window

    def record_window_score(self, score: float) -> None:
        """Record performance score for the completed window."""
        self._window_scores.append(score)

    def get_summary(self) -> dict:
        """End-of-episode summary."""
        return {
            "plan": self._plan.to_dict() if self._plan else {},
            "window_scores": list(self._window_scores),
            "avg_score": float(np.mean(self._window_scores)) if self._window_scores else 0.0,
            "target_met": (
                float(np.mean(self._window_scores)) >= (self._plan.target_served_pct if self._plan else 0.85)
                if self._window_scores else False
            ),
        }

    # ------------------------------------------------------------------
    # Heuristic planning (RL mode)
    # ------------------------------------------------------------------

    def _heuristic_plan(self, obs: ObservationModel) -> EpisodePlan:
        """Auto-generate a sensible plan based on current observation."""
        n_amb = len(obs.ambulances)
        reserve = max(1, n_amb // 4)

        # Prioritize zones with most emergencies
        from collections import Counter
        zone_counts = Counter()
        q = 25  # quarter of 100-node graph
        for e in obs.emergencies:
            zone = min(3, e.node // q)
            zone_counts[zone] += 1
        zone_priorities = [z for z, _ in zone_counts.most_common(4)]
        # Fill missing zones
        for z in range(4):
            if z not in zone_priorities:
                zone_priorities.append(z)

        return EpisodePlan(
            target_served_pct=0.85,
            reserve_ambulances=reserve,
            zone_priorities=zone_priorities,
            hospital_strategy="load_balance",
            raw_text=(
                f"Keep {reserve} ambulance(s) in reserve. "
                f"Prioritize zones {zone_priorities[:2]}. "
                f"Route to least-utilized hospital."
            ),
        )

    def _is_on_track(self) -> bool:
        if not self._window_scores or self._plan is None:
            return True  # optimistic default
        return float(np.mean(self._window_scores)) >= self._plan.target_served_pct * 0.9
