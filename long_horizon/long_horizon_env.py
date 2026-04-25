"""
long_horizon_env.py — Extends AmbulanceEnvironment to 1000-step episodes with:
  - Delayed consequence mechanics (scheduled surge events)
  - 50-step history encoding
  - Episode-level goal tracking
  - Recovery mode detection
  - EpisodePlanner integration
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.environment import AmbulanceEnvironment
from env.models import ActionModel, ObservationModel
from long_horizon.history_encoder import HistoryEncoder
from long_horizon.episode_planner import EpisodePlanner


# Scheduled surge events: (start_step, end_step, zone, lambda_multiplier)
# These are NOT directly observable — only inferable from time-of-day features.
_DEFAULT_SURGE_SCHEDULE = [
    (200, 280, 0, 2.5),   # Zone 0 surge at step 200
    (400, 480, 2, 2.0),   # Zone 2 surge at step 400
    (600, 680, 1, 2.8),   # Zone 1 surge at step 600
    (800, 880, 3, 2.2),   # Zone 3 surge at step 800
]


class LongHorizonAmbulanceEnvironment(AmbulanceEnvironment):
    """
    Long-horizon ambulance environment with curriculum-compatible max_steps.

    Extra observation fields added to the base ObservationModel:
      obs.reward_model.history_vec  — 25-dim HistoryEncoder output
      obs.step                       — already present

    New config keys
    ---------------
    max_steps          : int   Episode length (default 1000).
    history_window     : int   Steps in history buffer (default 50).
    window_size        : int   Steps per planning window (default 100).
    enable_surges      : bool  Whether to inject scheduled surge events.
    surge_schedule     : list  Override the default surge schedule.
    episode_goal_pct   : float Target served percentage for episode goal.
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        # Default to long-horizon config
        cfg.setdefault("max_steps", 1000)
        cfg.setdefault("n_ambulances", 5)
        cfg.setdefault("n_hospitals", 3)

        super().__init__(cfg)

        self._history_window = cfg.get("history_window", 50)
        self._window_size = cfg.get("window_size", 100)
        self._enable_surges = cfg.get("enable_surges", True)
        self._surge_schedule = cfg.get("surge_schedule", _DEFAULT_SURGE_SCHEDULE)
        self._episode_goal_pct = cfg.get("episode_goal_pct", 0.85)
        self._n_windows = self.max_steps // self._window_size

        self.history_encoder = HistoryEncoder(window=self._history_window)
        self.episode_planner = EpisodePlanner(
            n_windows=self._n_windows,
            window_size=self._window_size,
        )

        self._window_start_served = 0
        self._window_start_total = 0
        self._window_scores: List[float] = []
        self._recovery_mode = False
        self._recovery_threshold = 0.35  # if window score drops below this

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> ObservationModel:
        obs = super().reset(seed=seed)
        if not hasattr(self, 'history_encoder'):
            return obs
        self.history_encoder.reset()
        self._window_scores = []
        self._window_start_served = 0
        self._window_start_total = 0
        self._recovery_mode = False

        plan = self.episode_planner.request_plan(obs)
        obs.__dict__["_lh_plan"] = plan.to_dict()
        return obs

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: ActionModel) -> Tuple[ObservationModel, float, bool, dict]:
        # Inject surge effects if enabled
        if self._enable_surges:
            self._apply_surge()

        result = super().step(action)
        # Unpack: parent returns ObservationModel; some callers expect 4-tuple
        if isinstance(result, tuple):
            obs, reward, done, info = result
        else:
            obs = result
            reward = obs.reward
            done = obs.done
            info = {}

        # Update history encoder
        self.history_encoder.update(obs, reward, self.metrics)
        history_vec = self.history_encoder.encode()

        # Episode planner step tracking
        self.episode_planner.record_step(self.step_count)

        # Window boundary: compute window score
        if self.step_count > 0 and self.step_count % self._window_size == 0:
            score = self._compute_window_score()
            self._window_scores.append(score)
            self.episode_planner.record_window_score(score)
            self._window_start_served = self.metrics.get("served", 0)
            self._window_start_total = (
                self.metrics.get("served", 0) + self.metrics.get("missed", 0)
            )
            # Recovery mode detection
            if score < self._recovery_threshold:
                self._recovery_mode = True
                reward -= 5.0  # recovery mode penalty signal
            elif self._recovery_mode and score > self._recovery_threshold * 1.5:
                self._recovery_mode = False

        # Attach long-horizon extras to info dict
        info.update({
            "history_vec": history_vec.tolist(),
            "window_scores": list(self._window_scores),
            "recovery_mode": self._recovery_mode,
            "episode_goal_pct": self._episode_goal_pct,
            "planner_context": self.episode_planner.get_context(self.step_count),
        })

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Long-horizon metrics
    # ------------------------------------------------------------------

    def get_window_performance(self) -> Dict[str, Any]:
        """Return per-window performance for the frontend timeline chart."""
        return {
            "window_scores": list(self._window_scores),
            "n_windows": self._n_windows,
            "window_size": self._window_size,
            "recovery_mode": self._recovery_mode,
            "episode_goal_pct": self._episode_goal_pct,
            "on_track": (
                float(np.mean(self._window_scores)) >= self._episode_goal_pct * 0.85
                if self._window_scores else True
            ),
            "planner_summary": self.episode_planner.get_summary(),
        }

    def get_augmented_state_size(self, base_state_size: int) -> int:
        """Returns base_state_size + history feature dims."""
        return base_state_size + self.history_encoder.FEATURE_DIM

    def encode_augmented_state(self, base_state: np.ndarray) -> np.ndarray:
        """Concatenate base state with history encoding."""
        history_vec = self.history_encoder.encode()
        return np.concatenate([base_state, history_vec]).astype(np.float32)

    # ------------------------------------------------------------------
    # Surge mechanics
    # ------------------------------------------------------------------

    def _apply_surge(self) -> None:
        """Temporarily boost emergency arrival rate during scheduled surges."""
        for (start, end, zone, multiplier) in self._surge_schedule:
            if start <= self.step_count <= end:
                # Increase lambda for emergencies in the surge zone
                base_lambda = self.config.get("lambda_param", 0.15)
                # Apply temporarily by nudging the generator's lambda
                if hasattr(self.generator, "lambda_param"):
                    self.generator.lambda_param = base_lambda * multiplier
                return
        # Reset to base lambda outside surge windows
        base_lambda = self.config.get("lambda_param", 0.15)
        if hasattr(self.generator, "lambda_param"):
            self.generator.lambda_param = base_lambda

    # ------------------------------------------------------------------
    # Window score computation
    # ------------------------------------------------------------------

    def _compute_window_score(self) -> float:
        """Score for the just-completed window: served / (served + missed)."""
        served = self.metrics.get("served", 0)
        missed = self.metrics.get("missed", 0)
        window_served = served - self._window_start_served
        window_total = (served + missed) - self._window_start_total
        if window_total <= 0:
            return 0.0
        return float(window_served / window_total)
