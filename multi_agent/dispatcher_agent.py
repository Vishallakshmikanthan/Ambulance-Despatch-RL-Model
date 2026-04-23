from env.models import ObservationModel, AmbulanceState, Severity

_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.NORMAL: 2,
}


class DispatcherAgent:
    """
    Global decision agent: selects the highest-priority emergency
    and the nearest idle ambulance.
    """

    def select_candidate(self, observation: ObservationModel):
        """
        Select top priority emergency and candidate ambulance.

        Logic:
        - prioritize CRITICAL > HIGH > NORMAL
        - then lowest time_remaining
        - choose nearest idle ambulance

        Returns dict {"ambulance_id": int, "emergency_id": str} or None.
        """
        idle_ambs = [a for a in observation.ambulances if a.state == AmbulanceState.IDLE]
        unassigned = [e for e in observation.emergencies if not e.assigned]

        if not idle_ambs or not unassigned:
            return None

        target_emg = min(
            unassigned,
            key=lambda e: (_SEVERITY_RANK.get(e.severity, 3), e.time_remaining),
        )

        best_amb = min(
            idle_ambs,
            key=lambda a: abs(a.node - target_emg.node),
        )

        return {
            "ambulance_id": best_amb.id,
            "emergency_id": target_emg.id,
        }
