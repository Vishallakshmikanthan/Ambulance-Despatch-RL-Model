"""
expert_agent.py — Adaptive Expert Agent for imitation learning.

The expert's difficulty scales with the learning agent's training stage:
  Stage 0 (early)   : Greedy nearest-ambulance dispatch.
  Stage 1 (medium)  : Oracle dispatch (Dijkstra optimal, perfect info).
  Stage 2 (advanced): Adversarial oracle — deliberately avoids easy solutions
                       to push the learning agent further.

Usage
-----
expert = ExpertAgent(stage=0)
action = expert.act(obs, city_graph=env.city_graph.graph)
# If learning agent struggles, add expert trajectory to replay:
expert.get_imitation_trajectory(obs_list, env)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx

from agents.oracle import OracleAgent
from env.models import ActionModel, AmbulanceState, ObservationModel, Severity

_SEVERITY_RANK = {Severity.CRITICAL: 3, Severity.HIGH: 2, Severity.NORMAL: 1}


class ExpertAgent:
    """
    Adaptive expert agent that escalates difficulty as the learner improves.

    Parameters
    ----------
    stage : int     0 = greedy, 1 = oracle, 2 = adversarial oracle
    """

    def __init__(self, stage: int = 0):
        self.stage = stage
        self._oracle = OracleAgent()

    def set_stage(self, stage: int) -> None:
        self.stage = max(0, min(2, stage))
        if self.stage >= 1:
            pass  # oracle already initialised

    def act(
        self,
        obs: ObservationModel,
        city_graph: Optional[nx.Graph] = None,
    ) -> ActionModel:
        if self.stage == 0:
            return self._greedy_act(obs)
        elif self.stage == 1:
            self._oracle._graph = city_graph
            return self._oracle.act(obs)
        else:
            return self._adversarial_act(obs, city_graph)

    # ------------------------------------------------------------------
    # Per-stage strategies
    # ------------------------------------------------------------------

    def _greedy_act(self, obs: ObservationModel) -> ActionModel:
        """Greedy: nearest idle ambulance to highest-priority emergency."""
        idle = [a for a in obs.ambulances if a.state == AmbulanceState.IDLE]
        unassigned = [e for e in obs.emergencies if not e.assigned]
        if not idle or not unassigned:
            return ActionModel(ambulance_id=None, emergency_id="", is_noop=True)

        target = max(unassigned, key=lambda e: (_SEVERITY_RANK.get(e.severity, 0), -e.time_remaining))
        best_amb = min(idle, key=lambda a: abs(a.node - target.node))
        best_hosp = min(
            obs.hospitals,
            key=lambda h: (h.current_patients, abs(h.node - target.node)),
        )
        return ActionModel(
            ambulance_id=best_amb.id,
            emergency_id=target.id,
            hospital_id=best_hosp.id,
        )

    def _adversarial_act(
        self, obs: ObservationModel, city_graph: Optional[nx.Graph]
    ) -> ActionModel:
        """
        Adversarial oracle: uses oracle logic but routes to the SECOND best
        hospital (not nearest) to force the learning agent to handle suboptimal
        hand-offs and hospital load imbalances.
        """
        self._oracle._graph = city_graph
        base = self._oracle.act(obs)
        if base.is_noop or base.hospital_id is None:
            return base

        # Find second-best hospital
        available = sorted(
            obs.hospitals,
            key=lambda h: (h.current_patients, abs(h.node - (base.hospital_id or 0))),
        )
        if len(available) > 1:
            second_best = available[1]
            return ActionModel(
                ambulance_id=base.ambulance_id,
                emergency_id=base.emergency_id,
                hospital_id=second_best.id,
            )
        return base

    # ------------------------------------------------------------------
    # Imitation trajectory collection
    # ------------------------------------------------------------------

    def collect_trajectory(
        self,
        env,
        n_steps: int = 50,
        city_graph=None,
    ) -> List[Tuple[object, ActionModel, float, object, bool]]:
        """
        Run expert in the given env for `n_steps` and return SARS transitions.

        Returns list of (state_vec, action_model, reward, next_state_vec, done).
        Note: env must already be reset before calling this.
        """
        from rl.state_encoder import StateEncoder
        encoder = StateEncoder()
        transitions = []

        obs = env._get_observation()
        for _ in range(n_steps):
            state_vec = encoder.encode(obs)
            action = self.act(obs, city_graph=city_graph)
            next_obs, reward, done, _ = env.step(action)
            next_vec = encoder.encode(next_obs)
            transitions.append((state_vec, action, reward, next_vec, done))
            obs = next_obs
            if done:
                break
        return transitions
