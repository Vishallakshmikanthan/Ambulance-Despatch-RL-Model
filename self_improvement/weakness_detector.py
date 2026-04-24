"""
weakness_detector.py — Identifies and tracks agent failure patterns.

Uses AdversarialScenarioGenerator's clustering logic to produce a structured
weakness report consumed by:
  - SelfPlayTrainer  (for scenario targeting)
  - /selfplay/weaknesses  (API endpoint)
  - SelfImprovementView  (frontend component)
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from self_improvement.adversarial_generator import (
    AdversarialScenarioGenerator,
    ScenarioConfig,
)


class WeaknessReport:
    """Structured report of identified weaknesses."""

    def __init__(self, clusters: List[dict], iteration: int):
        self.clusters = clusters
        self.iteration = iteration

    def to_dict(self) -> dict:
        return {"iteration": self.iteration, "clusters": self.clusters}

    def top_n(self, n: int = 5) -> List[dict]:
        sorted_c = sorted(self.clusters, key=lambda c: c.get("avg_score", 1.0))
        return sorted_c[:n]


class WeaknessDetector:
    """
    Detects and tracks agent failure clusters across self-improvement iterations.

    Parameters
    ----------
    failure_threshold : float  Score below which an episode is a failure.
    history_size : int         How many iterations to keep in history.
    """

    def __init__(self, failure_threshold: float = 0.5, history_size: int = 50):
        self.failure_threshold = failure_threshold
        self._generator = AdversarialScenarioGenerator(
            failure_threshold=failure_threshold,
            n_clusters=5,
            noise_std=0.05,
        )
        self._iteration = 0
        self._reports: deque[WeaknessReport] = deque(maxlen=history_size)

        # Per-cluster improvement tracking: cluster_id → list of avg_scores over iterations
        self._cluster_history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        results: List[Tuple[ScenarioConfig, float]],
    ) -> WeaknessReport:
        """
        Analyze a batch of (config, score) pairs and produce a weakness report.

        Automatically increments the iteration counter.
        """
        self._iteration += 1
        clusters = self._generator.identify_failures(results)

        # Augment cluster dicts with improvement history
        for c in clusters:
            key = f"cluster_{c['cluster_id']}_lambda{c['centroid_lambda']:.2f}"
            if key not in self._cluster_history:
                self._cluster_history[key] = []
            self._cluster_history[key].append(c["avg_score"])
            c["improvement_history"] = list(self._cluster_history[key][-10:])
            c["cluster_key"] = key
            c["is_improving"] = self._is_improving(self._cluster_history[key])

        report = WeaknessReport(clusters=clusters, iteration=self._iteration)
        self._reports.append(report)
        return report

    def get_history(self) -> List[dict]:
        """Return all weakness reports as dicts."""
        return [r.to_dict() for r in self._reports]

    def get_latest(self) -> Optional[WeaknessReport]:
        return self._reports[-1] if self._reports else None

    def get_improvement_summary(self) -> dict:
        """
        For each tracked cluster, compute overall improvement.
        Used by the frontend expert-gap chart.
        """
        summary = {}
        for key, history in self._cluster_history.items():
            if len(history) < 2:
                summary[key] = {"trend": "insufficient_data", "delta": 0.0}
            else:
                delta = history[-1] - history[0]
                summary[key] = {
                    "trend": "improving" if delta > 0.05 else ("stable" if abs(delta) < 0.05 else "degrading"),
                    "delta": float(delta),
                    "history": list(history),
                }
        return summary

    def generate_targeted_scenarios(
        self,
        n: int = 10,
        base_config: Optional[ScenarioConfig] = None,
    ) -> List[ScenarioConfig]:
        """Delegate to AdversarialScenarioGenerator after identifying failures."""
        return self._generator.generate_scenarios(n=n, base_config=base_config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_improving(history: List[float]) -> bool:
        if len(history) < 3:
            return False
        recent = history[-3:]
        return recent[-1] > recent[0] + 0.03
