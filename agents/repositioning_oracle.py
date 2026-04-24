# agents/repositioning_oracle.py
"""
RepositioningOracle — extends OracleAgent with:
1. Multi-dispatch: serves all possible emergencies simultaneously.
2. Repositioning: keeps idle ambulances at predicted hotspots (zero idle_steps).
3. Specialty routing: matches hospital type to emergency severity.
4. Zone fairness: spreads repositioned ambulances across all city zones.
"""
from __future__ import annotations
from collections import Counter
from typing import Dict, List, Optional, Set

from env.models import ActionModel, AmbulanceState, ObservationModel, Severity
from agents.oracle import OracleAgent

_SEVERITY_RANK = {Severity.CRITICAL: 3, Severity.HIGH: 2, Severity.NORMAL: 1}

SPECIALTY_MAP = {
    "CRITICAL": ["Trauma", "Cardiac"],
    "HIGH":     ["Trauma", "General"],
    "NORMAL":   ["General", "Paediatric"],
}

ZONE_CENTERS = [12, 37, 62, 87]  # Representative nodes for each of the 4 city zones


class RepositioningOracle(OracleAgent):
    """
    Optimal dispatcher with repositioning and specialty awareness.
    Use this agent in inference.py for maximum scores on all three tasks.
    """

    def __init__(self, all_pairs_len: Optional[Dict] = None, n_nodes: int = 100,
                 enable_reposition: bool = True):
        super().__init__(all_pairs_len=all_pairs_len)
        self.n_nodes = n_nodes
        self.enable_reposition = enable_reposition
        self._emergency_node_history: List[int] = []

    def bind_env(self, env) -> "RepositioningOracle":
        """Bind to the environment's pre-computed path cache."""
        try:
            self._all_pairs = env.city_graph._all_pairs_len
            self.n_nodes = getattr(env, "graph_size", 100)
        except Exception:
            pass
        return self

    # ------------------------------------------------------------------
    # Hospital selection: specialty + capacity + distance
    # ------------------------------------------------------------------

    def _best_hospital(
        self,
        emg,
        observation: ObservationModel,
        used_capacity: Dict[int, int],
    ):
        """
        Score each hospital. Lower score = better choice.
        Factors: specialty match (big bonus), remaining capacity, distance.
        """
        sev_str = emg.severity.value if hasattr(emg.severity, "value") else str(emg.severity)
        preferred = SPECIALTY_MAP.get(sev_str, [])

        best_hosp = None
        best_score = float("inf")

        for h in observation.hospitals:
            used = used_capacity.get(h.id, 0)
            remaining = h.capacity - h.current_patients - used
            if remaining <= 0:
                continue
            specialty_penalty = 0.0 if h.specialty in preferred else 25.0
            dist = self._dist(emg.node, h.node)
            occupancy_penalty = h.current_patients * 0.5
            score = dist + specialty_penalty + occupancy_penalty
            if score < best_score:
                best_score = score
                best_hosp = h

        if best_hosp is None:
            # No available hospital — take least-full as emergency fallback
            hospitals = sorted(
                observation.hospitals,
                key=lambda h: (h.current_patients, self._dist(emg.node, h.node))
            )
            best_hosp = hospitals[0] if hospitals else None

        return best_hosp

    # ------------------------------------------------------------------
    # Demand tracking and hotspot prediction
    # ------------------------------------------------------------------

    def _record(self, obs: ObservationModel):
        """Record emergency locations for hotspot estimation."""
        for emg in obs.emergencies:
            self._emergency_node_history.append(emg.node)
        if len(self._emergency_node_history) > 500:
            self._emergency_node_history = self._emergency_node_history[-500:]

    def _hotspot_targets(self, n: int) -> List[int]:
        """Return n hotspot nodes. Falls back to zone-spread if insufficient history."""
        if len(self._emergency_node_history) < 10:
            return [ZONE_CENTERS[i % 4] for i in range(n)]
        counts = Counter(self._emergency_node_history)
        top = [node for node, _ in counts.most_common(n * 2)]
        # Ensure coverage of all 4 zones
        result = []
        covered_zones: Set[int] = set()
        for node in top:
            zone = min(3, node * 4 // self.n_nodes)
            if zone not in covered_zones or len(result) < n:
                result.append(node)
                covered_zones.add(zone)
            if len(result) >= n:
                break
        # Fill remaining slots with zone centers
        for i in range(n):
            if len(result) >= n:
                break
            result.append(ZONE_CENTERS[i % 4])
        return result[:n]

    # ------------------------------------------------------------------
    # Core multi-dispatch with reposition
    # ------------------------------------------------------------------

    def act_all_with_reposition(self, obs: ObservationModel) -> List[ActionModel]:
        """
        Returns one ActionModel per idle ambulance:
          - Dispatches to highest-priority unserved emergency.
          - Repositions remaining idle ambulances to hotspots.

        Call env.step(action) for each returned action.
        """
        self._record(obs)

        idle_ambs = sorted(
            [a for a in obs.ambulances if a.state == AmbulanceState.IDLE],
            key=lambda a: a.id,
        )
        unassigned = sorted(
            [e for e in obs.emergencies if not e.assigned],
            key=lambda e: (_SEVERITY_RANK.get(e.severity, 0), -e.time_remaining),
            reverse=True,
        )

        if not idle_ambs:
            return [ActionModel(is_noop=True)]

        actions: List[ActionModel] = []
        used_emg_ids: Set[str] = set()
        used_capacity: Dict[int, int] = {}

        # Phase 1: Dispatch to emergencies
        for amb in idle_ambs:
            best_emg = None
            best_score = float("inf")
            for emg in unassigned:
                if emg.id in used_emg_ids:
                    continue
                sev_bonus = {3: -1000, 2: -500, 1: 0}.get(
                    _SEVERITY_RANK.get(emg.severity, 0), 0
                )
                d = self._dist(amb.node, emg.node)
                urgency = -emg.time_remaining * 5
                score = d + sev_bonus + urgency
                if score < best_score:
                    best_score = score
                    best_emg = emg

            if best_emg is None:
                continue  # No emergency for this ambulance

            best_hosp = self._best_hospital(best_emg, obs, used_capacity)
            if best_hosp is None:
                continue

            actions.append(ActionModel(
                ambulance_id=amb.id,
                emergency_id=best_emg.id,
                hospital_id=best_hosp.id,
            ))
            used_emg_ids.add(best_emg.id)
            used_capacity[best_hosp.id] = used_capacity.get(best_hosp.id, 0) + 1

        # Phase 2: Repositioning (only when enable_reposition=True).
        # For easy/hard tasks: repos ambulances to hotspot targets to reduce response time.
        # Disabled for medium task where served_pct (0.50 weight) matters more than
        # idle_fraction (0.15 weight): long repos (9+ steps) blocks dispatch and hurts
        # served_pct far more than idle_fraction is improved.
        dispatched_ids = {a.ambulance_id for a in actions if a.ambulance_id is not None}
        remaining_idle = [a for a in idle_ambs if a.id not in dispatched_ids]
        # Emergencies still unserved after Phase 1 (Phase 1 claimed used_emg_ids)
        active_unassigned = [e for e in obs.emergencies
                             if not e.assigned and e.id not in used_emg_ids]

        if self.enable_reposition and remaining_idle and not active_unassigned:
            targets = self._hotspot_targets(max(len(remaining_idle) * 2, 8))
            used_targets: Set[int] = set()
            for amb in remaining_idle:
                target = next(
                    (t for t in targets if t != amb.node and t not in used_targets),
                    None,
                )
                if target is None:
                    continue
                used_targets.add(target)
                actions.append(ActionModel(
                    ambulance_id=amb.id,
                    emergency_id="",
                    hospital_id=None,
                    reposition_node=target,
                    is_noop=False,
                ))

        return actions if actions else [ActionModel(is_noop=True)]
