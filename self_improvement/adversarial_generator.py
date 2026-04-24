"""
adversarial_generator.py — Generates targeted training scenarios by identifying
failure clusters in the current agent's performance.

Workflow
--------
1. Run current agent on N random scenarios → collect (config, score) pairs.
2. identify_failures() clusters low-scoring configs using KMeans on feature vecs.
3. generate_scenarios() samples new configs centred on failure clusters.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ScenarioConfig:
    """Lightweight config for a single training scenario."""
    n_ambulances: int = 5
    n_hospitals: int = 3
    max_steps: int = 100
    seed: int = 42
    lambda_param: float = 0.15   # emergency arrival rate
    traffic_intensity: float = 1.0  # multiplier on traffic weights
    surge_zone: int = -1           # -1 = no surge, 0-3 = zone index
    surge_step: int = -1           # step at which surge starts
    graph_size: int = 100
    extra_tags: List[str] = field(default_factory=list)

    def to_env_dict(self) -> dict:
        return {
            "n_ambulances": self.n_ambulances,
            "n_hospitals": self.n_hospitals,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "lambda_param": self.lambda_param,
            "graph_size": self.graph_size,
        }

    def to_feature_vector(self) -> np.ndarray:
        """Convert to fixed-size float vector for clustering."""
        return np.array([
            self.n_ambulances / 10.0,
            self.n_hospitals / 5.0,
            self.max_steps / 200.0,
            self.lambda_param / 0.5,
            self.traffic_intensity / 3.0,
            (self.surge_zone + 1) / 5.0,
            (self.surge_step + 1) / 200.0,
        ], dtype=np.float32)


class AdversarialScenarioGenerator:
    """
    Identifies failure clusters and generates targeted training scenarios.

    Parameters
    ----------
    failure_threshold : float   Scores below this are considered failures.
    n_clusters : int            K for KMeans clustering.
    noise_std : float           Gaussian noise std when generating new scenarios.
    """

    def __init__(
        self,
        failure_threshold: float = 0.5,
        n_clusters: int = 3,
        noise_std: float = 0.05,
    ):
        self.failure_threshold = failure_threshold
        self.n_clusters = n_clusters
        self.noise_std = noise_std
        self._cluster_centres: Optional[np.ndarray] = None
        self._failure_configs: List[ScenarioConfig] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify_failures(
        self,
        results: List[Tuple[ScenarioConfig, float]],
    ) -> List[dict]:
        """
        Cluster failure scenarios to identify weakness patterns.

        Parameters
        ----------
        results : list of (ScenarioConfig, score) tuples.

        Returns
        -------
        List of cluster summary dicts with centroid info and average score.
        """
        failures = [(cfg, sc) for cfg, sc in results if sc < self.failure_threshold]
        if len(failures) < 2:
            self._cluster_centres = None
            self._failure_configs = [cfg for cfg, _ in failures]
            return []

        self._failure_configs = [cfg for cfg, _ in failures]
        vecs = np.array([cfg.to_feature_vector() for cfg in self._failure_configs])

        k = min(self.n_clusters, len(failures))
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(vecs)
            self._cluster_centres = km.cluster_centers_
        except ImportError:
            # Fallback: use failure configs directly as single cluster
            self._cluster_centres = vecs[:k]
            labels = np.zeros(len(vecs), dtype=int)

        clusters = []
        failure_scores = [sc for _, sc in failures]
        for c_idx in range(k):
            mask = labels == c_idx
            c_configs = [self._failure_configs[i] for i in range(len(self._failure_configs)) if mask[i]]
            c_scores  = [failure_scores[i]         for i in range(len(failure_scores))        if mask[i]]
            if not c_configs:
                continue
            # Describe the dominant pattern
            avg_lambda = np.mean([c.lambda_param for c in c_configs])
            avg_traffic = np.mean([c.traffic_intensity for c in c_configs])
            surge_zones = [c.surge_zone for c in c_configs if c.surge_zone >= 0]
            clusters.append({
                "cluster_id": c_idx,
                "size": len(c_configs),
                "avg_score": float(np.mean(c_scores)),
                "centroid_lambda": float(avg_lambda),
                "centroid_traffic": float(avg_traffic),
                "dominant_surge_zone": int(max(set(surge_zones), key=surge_zones.count)) if surge_zones else -1,
                "contributing_factors": _describe_factors(avg_lambda, avg_traffic, surge_zones),
            })

        return clusters

    def generate_scenarios(
        self,
        n: int = 10,
        base_config: Optional[ScenarioConfig] = None,
    ) -> List[ScenarioConfig]:
        """
        Generate new training scenarios targeting identified failure clusters.

        If no clusters have been identified yet, returns randomly varied configs.
        """
        if self._cluster_centres is None or len(self._cluster_centres) == 0:
            return [self._random_scenario(base_config) for _ in range(n)]

        generated = []
        for i in range(n):
            # Cycle through cluster centres
            centre = self._cluster_centres[i % len(self._cluster_centres)]
            noisy = centre + np.random.normal(0, self.noise_std, size=centre.shape)
            noisy = np.clip(noisy, 0.0, 1.0)
            cfg = self._vector_to_config(noisy, base_config)
            cfg.seed = random.randint(0, 10_000)
            generated.append(cfg)
        return generated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_scenario(self, base: Optional[ScenarioConfig] = None) -> ScenarioConfig:
        cfg = copy.deepcopy(base) if base else ScenarioConfig()
        cfg.seed = random.randint(0, 10_000)
        cfg.lambda_param = random.uniform(0.1, 0.35)
        cfg.traffic_intensity = random.uniform(0.8, 2.5)
        cfg.surge_zone = random.choice([-1, 0, 1, 2, 3])
        cfg.surge_step = random.randint(20, 80) if cfg.surge_zone >= 0 else -1
        return cfg

    def _vector_to_config(
        self, vec: np.ndarray, base: Optional[ScenarioConfig] = None
    ) -> ScenarioConfig:
        cfg = copy.deepcopy(base) if base else ScenarioConfig()
        cfg.n_ambulances = max(2, min(8, int(round(vec[0] * 10))))
        cfg.n_hospitals   = max(2, min(5, int(round(vec[1] * 5))))
        cfg.max_steps     = max(50, min(200, int(round(vec[2] * 200))))
        cfg.lambda_param  = float(np.clip(vec[3] * 0.5, 0.05, 0.5))
        cfg.traffic_intensity = float(np.clip(vec[4] * 3.0, 0.5, 3.0))
        surge = int(round(vec[5] * 5)) - 1
        cfg.surge_zone    = int(np.clip(surge, -1, 3))
        surge_step = int(round(vec[6] * 200)) - 1
        cfg.surge_step    = int(np.clip(surge_step, -1, 199))
        return cfg


# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------

def _describe_factors(avg_lambda: float, avg_traffic: float, surge_zones: list) -> List[str]:
    factors = []
    if avg_lambda > 0.25:
        factors.append("high_emergency_rate")
    if avg_traffic > 1.8:
        factors.append("heavy_traffic")
    if surge_zones:
        factors.append(f"zone_{max(set(surge_zones), key=surge_zones.count)}_surge")
    if not factors:
        factors.append("general_overload")
    return factors
