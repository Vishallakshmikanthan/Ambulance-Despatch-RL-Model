"""
history_encoder.py — Encodes a rolling 50-step observation history into
a compact 25-dimensional feature vector for long-horizon state representations.

Features
--------
0  : mean served rate over window
1  : mean timeout rate
2  : zone balance index (std across zone_served / mean)
3  : traffic trend  (-1 decreasing, 0 stable, +1 increasing)
4  : hospital utilization trend
5  : critical served rate
6  : high served rate
7  : mean reward over window (normalized)
8  : peak reward in window (normalized)
9  : trough reward in window (normalized)
10 : mean idle fraction
11 : mean emergency count per step
12 : surge indicator (served rate drop > 30% from peak)
13-24 : 12-bin histogram of per-step rewards (normalized)
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from env.models import ObservationModel


class _StepSummary:
    __slots__ = (
        "served", "missed", "critical_served", "high_served",
        "reward", "idle_fraction", "emg_count",
        "zone_served", "hosp_util", "traffic",
    )

    def __init__(self):
        self.served = 0
        self.missed = 0
        self.critical_served = 0
        self.high_served = 0
        self.reward = 0.0
        self.idle_fraction = 0.0
        self.emg_count = 0
        self.zone_served: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        self.hosp_util = 0.0
        self.traffic = 1.0


class HistoryEncoder:
    """
    Maintains a sliding window of step summaries and encodes them.

    Parameters
    ----------
    window : int   Number of steps to keep (default 50).
    """

    FEATURE_DIM = 25

    def __init__(self, window: int = 50):
        self.window = window
        self._buffer: Deque[_StepSummary] = deque(maxlen=window)
        self._prev_metrics: Optional[dict] = None

    def update(
        self,
        obs: ObservationModel,
        reward: float,
        env_metrics: Optional[dict] = None,
    ) -> None:
        """
        Add a new step to the history buffer.

        Parameters
        ----------
        obs         : Current observation.
        reward      : Scalar reward received this step.
        env_metrics : Optional env.metrics dict for richer features.
        """
        s = _StepSummary()
        s.reward = reward
        s.emg_count = len(obs.emergencies)
        s.traffic = obs.traffic.get("global", 1.0) if isinstance(obs.traffic, dict) else 1.0

        if obs.ambulances:
            idle = sum(1 for a in obs.ambulances if a.state.value == "idle")
            s.idle_fraction = idle / len(obs.ambulances)

        if obs.hospitals:
            s.hosp_util = float(np.mean([
                h.current_patients / max(h.capacity, 1) for h in obs.hospitals
            ]))

        if env_metrics:
            s.served = env_metrics.get("served", 0)
            s.missed = env_metrics.get("missed", 0)
            s.critical_served = env_metrics.get("critical_served", 0)
            s.high_served = env_metrics.get("high_served", 0)
            s.zone_served = env_metrics.get("zone_served", {0: 0, 1: 0, 2: 0, 3: 0})

        self._buffer.append(s)

    def encode(self) -> np.ndarray:
        """Return the 25-dim feature vector for the current window."""
        if not self._buffer:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        buf = list(self._buffer)
        rewards = np.array([s.reward for s in buf], dtype=np.float32)
        reward_norm = 100.0  # clip scale

        served   = np.array([s.served for s in buf], dtype=np.float32)
        missed   = np.array([s.missed for s in buf], dtype=np.float32)
        total    = served + missed + 1e-8

        features = np.zeros(self.FEATURE_DIM, dtype=np.float32)

        # 0: mean served rate
        features[0] = float(np.mean(served / total))
        # 1: mean timeout rate
        features[1] = float(np.mean(missed / total))
        # 2: zone balance index
        zone_totals = np.array([
            np.sum([s.zone_served.get(z, 0) for s in buf]) for z in range(4)
        ], dtype=np.float32)
        if zone_totals.mean() > 0:
            features[2] = float(np.std(zone_totals) / (np.mean(zone_totals) + 1e-8))
        # 3: traffic trend
        if len(buf) >= 10:
            early_t = np.mean([s.traffic for s in buf[:5]])
            late_t  = np.mean([s.traffic for s in buf[-5:]])
            features[3] = float(np.sign(late_t - early_t))
        # 4: hospital utilization trend
        hosp_utils = np.array([s.hosp_util for s in buf])
        if len(hosp_utils) >= 4:
            features[4] = float(np.sign(np.mean(hosp_utils[-4:]) - np.mean(hosp_utils[:4])))
        # 5: critical served rate
        crit_served = np.array([s.critical_served for s in buf])
        features[5] = float(np.mean(crit_served) / 10.0)
        # 6: high served rate
        high_served = np.array([s.high_served for s in buf])
        features[6] = float(np.mean(high_served) / 10.0)
        # 7: mean reward (normalized)
        features[7] = float(np.mean(rewards) / reward_norm)
        # 8: peak reward
        features[8] = float(np.max(rewards) / reward_norm)
        # 9: trough reward
        features[9] = float(np.min(rewards) / reward_norm)
        # 10: mean idle fraction
        features[10] = float(np.mean([s.idle_fraction for s in buf]))
        # 11: mean emergency count per step
        features[11] = float(np.mean([s.emg_count for s in buf]) / 10.0)
        # 12: surge indicator (served rate dropped >30% from peak in last 10 steps)
        if len(buf) >= 10:
            peak_rate = float(np.max(served[-10:] / (total[-10:])))
            curr_rate = float((served[-1]) / (total[-1]))
            features[12] = 1.0 if (peak_rate - curr_rate) > 0.30 else 0.0
        # 13-24: 12-bin histogram of rewards
        hist, _ = np.histogram(
            np.clip(rewards, -reward_norm, reward_norm),
            bins=12,
            range=(-reward_norm, reward_norm),
        )
        features[13:25] = hist.astype(np.float32) / max(len(buf), 1)

        return features

    def reset(self) -> None:
        self._buffer.clear()
        self._prev_metrics = None
