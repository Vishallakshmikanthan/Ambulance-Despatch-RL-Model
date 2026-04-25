"""
Oracle agent — uses pre-computed all-pairs shortest paths for optimal dispatch.
This is the upper-bound agent used in inference.py for maximum scores.
"""
from __future__ import annotations
from typing import Optional, Dict, List
import networkx as nx
from env.models import ActionModel, AmbulanceState, ObservationModel, Severity

_SEVERITY_RANK = {Severity.CRITICAL: 3, Severity.HIGH: 2, Severity.NORMAL: 1}


class OracleAgent:
    """
    Optimal-dispatch oracle with multi-dispatch capability.
    Uses pre-computed all-pairs shortest paths for O(1) distance lookups.
    Falls back to |node_a - node_b| approximation if not provided.
    """

    def __init__(
        self,
        city_graph: Optional[nx.Graph] = None,
        all_pairs_len: Optional[Dict] = None,
    ):
        self._graph = city_graph
        self._all_pairs = all_pairs_len

    def bind_env(self, env) -> "OracleAgent":
        """Bind to a live environment to access its pre-computed path cache."""
        try:
            self._all_pairs = env.city_graph._all_pairs_len
        except Exception:
            pass
        return self

    def _dist(self, src: int, dst: int) -> float:
        if src == dst:
            return 0.0
        if self._all_pairs is not None:
            return float(self._all_pairs.get(src, {}).get(dst, abs(src - dst)))
        if self._graph is not None:
            try:
                return float(nx.shortest_path_length(self._graph, src, dst))
            except Exception:
                pass
        return float(abs(src - dst))

    def act(self, observation: ObservationModel) -> ActionModel:
        """Single best dispatch action for the highest-priority emergency."""
        idle_ambs = [a for a in observation.ambulances if a.state == AmbulanceState.IDLE]
        unassigned = [e for e in observation.emergencies if not e.assigned]

        if not idle_ambs or not unassigned:
            return ActionModel(ambulance_id=None, emergency_id="", hospital_id=None, is_noop=True)

        # Sort emergencies: CRITICAL first, then soonest expiry
        sorted_emgs = sorted(
            unassigned,
            key=lambda e: (_SEVERITY_RANK.get(e.severity, 0), -e.time_remaining),
            reverse=True,
        )
        target_emg = sorted_emgs[0]

        best_amb = min(idle_ambs, key=lambda a: self._dist(a.node, target_emg.node))

        available = [h for h in observation.hospitals if h.current_patients < h.capacity]
        if not available:
            available = list(observation.hospitals)
        best_hosp = min(
            available,
            key=lambda h: (h.current_patients, self._dist(target_emg.node, h.node)),
        )

        return ActionModel(
            ambulance_id=best_amb.id,
            emergency_id=target_emg.id,
            hospital_id=best_hosp.id,
        )

    def act_all(self, observation: ObservationModel) -> List[ActionModel]:
        """
        Return one action per idle ambulance, greedily pairing each to
        the best remaining unassigned emergency.
        """
        idle_ambs = sorted(
            [a for a in observation.ambulances if a.state == AmbulanceState.IDLE],
            key=lambda a: a.id,
        )
        unassigned = sorted(
            [e for e in observation.emergencies if not e.assigned],
            key=lambda e: (_SEVERITY_RANK.get(e.severity, 0), -e.time_remaining),
            reverse=True,
        )

        if not idle_ambs or not unassigned:
            return [ActionModel(is_noop=True)]

        available_hosps = [h for h in observation.hospitals if h.current_patients < h.capacity]
        if not available_hosps:
            available_hosps = list(observation.hospitals)

        actions = []
        used_emg_ids: set = set()
        used_hosp_ids_count: Dict[int, int] = {}

        for amb in idle_ambs:
            best_emg = None
            best_score = float("inf")
            for emg in unassigned:
                if emg.id in used_emg_ids:
                    continue
                sev_bonus = {3: -100, 2: -50, 1: 0}.get(_SEVERITY_RANK.get(emg.severity, 0), 0)
                score = self._dist(amb.node, emg.node) + sev_bonus
                if score < best_score:
                    best_score = score
                    best_emg = emg

            if best_emg is None:
                break

            best_hosp = None
            best_h_score = float("inf")
            for h in available_hosps:
                used = used_hosp_ids_count.get(h.id, 0)
                remaining = h.capacity - h.current_patients - used
                if remaining <= 0:
                    continue
                h_score = self._dist(best_emg.node, h.node) + h.current_patients * 0.1
                if h_score < best_h_score:
                    best_h_score = h_score
                    best_hosp = h

            if best_hosp is None:
                best_hosp = available_hosps[0] if available_hosps else None
            if best_hosp is None:
                break

            actions.append(ActionModel(
                ambulance_id=amb.id,
                emergency_id=best_emg.id,
                hospital_id=best_hosp.id,
            ))
            used_emg_ids.add(best_emg.id)
            used_hosp_ids_count[best_hosp.id] = used_hosp_ids_count.get(best_hosp.id, 0) + 1

        return actions if actions else [ActionModel(is_noop=True)]
