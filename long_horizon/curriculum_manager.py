"""
curriculum_manager.py — Tracks training stage and controls episode length progression.

Stages 1-10 map to max_steps 100-1000 (increments of 100).
Advancement requires average score >= threshold over last 20 episodes.
"""
from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional


_STAGE_TO_STEPS = {s: s * 100 for s in range(1, 11)}
_STAGE_THRESHOLDS = {s: (0.65 if s == 1 else 0.70) for s in range(1, 11)}


class CurriculumManager:
    """
    Manages curriculum-based training stages.

    Parameters
    ----------
    initial_stage : int       Starting stage (1-10).
    window        : int       Episodes to average over before checking advancement.
    output_dir    : str       Where to save curriculum_progress.csv.
    """

    def __init__(
        self,
        initial_stage: int = 1,
        window: int = 20,
        output_dir: str = "outputs/curriculum",
    ):
        self.stage = initial_stage
        self.window = window
        self._score_buffer: Deque[float] = deque(maxlen=window)
        self._episode = 0
        self._transitions: List[dict] = []

        self._out_dir = Path(output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._out_dir / "curriculum_progress.csv"
        self._init_csv()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_steps(self) -> int:
        return _STAGE_TO_STEPS.get(self.stage, 1000)

    @property
    def threshold(self) -> float:
        return _STAGE_THRESHOLDS.get(self.stage, 0.70)

    @property
    def at_max_stage(self) -> bool:
        return self.stage >= 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_episode(self, score: float) -> bool:
        """
        Record episode score. Returns True if stage advanced this call.
        """
        self._episode += 1
        self._score_buffer.append(score)

        advanced = False
        if self.should_advance():
            self.advance()
            advanced = True

        self._append_csv({
            "episode": self._episode,
            "stage": self.stage,
            "max_steps": self.max_steps,
            "score": round(score, 4),
            "window_avg": round(self._window_avg(), 4),
            "threshold": self.threshold,
            "advanced": advanced,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        return advanced

    def should_advance(self) -> bool:
        """Return True when the window average meets the threshold."""
        if len(self._score_buffer) < self.window:
            return False
        if self.at_max_stage:
            return False
        return self._window_avg() >= self.threshold

    def advance(self) -> None:
        """Increment stage and clear score buffer."""
        old_stage = self.stage
        self.stage = min(self.stage + 1, 10)
        self._score_buffer.clear()
        transition = {
            "from_stage": old_stage,
            "to_stage": self.stage,
            "episode": self._episode,
            "new_max_steps": self.max_steps,
        }
        self._transitions.append(transition)
        print(
            f"[Curriculum] Stage {old_stage} → {self.stage} "
            f"(max_steps={self.max_steps}) at episode {self._episode}"
        )

    def get_progress(self) -> dict:
        """Return current progress summary."""
        return {
            "stage": self.stage,
            "max_steps": self.max_steps,
            "threshold": self.threshold,
            "window_avg": round(self._window_avg(), 4),
            "episode": self._episode,
            "transitions": self._transitions,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _window_avg(self) -> float:
        if not self._score_buffer:
            return 0.0
        return float(sum(self._score_buffer) / len(self._score_buffer))

    def _init_csv(self):
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "episode", "stage", "max_steps", "score",
                    "window_avg", "threshold", "advanced", "timestamp",
                ])
                writer.writeheader()

    def _append_csv(self, row: dict):
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
