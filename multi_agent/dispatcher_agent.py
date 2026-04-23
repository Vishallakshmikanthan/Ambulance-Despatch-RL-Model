from env.models import ObservationModel, AmbulanceState, Severity

_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.NORMAL: 2,
}

_DEFAULT_WEIGHTS = {
    "CRITICAL": 1.0,
    "HIGH": 1.0,
    "NORMAL": 1.0,
}


class DispatcherAgent:
    """
    Global decision agent: selects the highest-priority emergency
    and the nearest idle ambulance.
    """

    def select_candidate(self, observation: ObservationModel, weights: dict = None):
        """
        Select top priority emergency and candidate ambulance.

        Logic:
        - prioritize CRITICAL > HIGH > NORMAL scaled by adaptive weights
        - then lowest time_remaining
        - choose nearest idle ambulance

        Returns dict {"ambulance_id": int, "emergency_id": str} or None.
        """
        if weights is None:
            weights = _DEFAULT_WEIGHTS

        idle_ambs = [a for a in observation.ambulances if a.state == AmbulanceState.IDLE]
        unassigned = [e for e in observation.emergencies if not e.assigned]

        if not idle_ambs or not unassigned:
            return None

        def priority_score(e):
            base_priority = _SEVERITY_RANK.get(e.severity, 3)
            weight = weights.get(e.severity.value if hasattr(e.severity, "value") else str(e.severity), 1.0)
            return base_priority * weight, e.time_remaining

        target_emg = min(unassigned, key=priority_score)

        best_amb = min(
            idle_ambs,
            key=lambda a: abs(a.node - target_emg.node),
        )

        return {
            "ambulance_id": best_amb.id,
            "emergency_id": target_emg.id,
        }
