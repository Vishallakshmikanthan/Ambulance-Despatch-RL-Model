from env.models import ObservationModel


class AmbulanceAgent:
    """
    Local decision agent: refines a partial action by selecting the best hospital.
    """

    def refine_action(self, observation: ObservationModel, partial_action: dict):
        """
        Given ambulance_id and emergency_id:
        - choose best hospital
        - avoid full hospitals
        - prefer nearest valid hospital

        Returns dict with ambulance_id, emergency_id, hospital_id.
        """
        amb_id = partial_action.get("ambulance_id")
        emg_id = partial_action.get("emergency_id")

        if amb_id is None or not emg_id:
            return partial_action

        emergency = next(
            (e for e in observation.emergencies if e.id == emg_id), None
        )
        if emergency is None:
            return partial_action

        available_hosps = [
            h for h in observation.hospitals if h.current_patients < h.capacity
        ]
        if not available_hosps:
            available_hosps = list(observation.hospitals)

        best_hosp = min(
            available_hosps,
            key=lambda h: abs(h.node - emergency.node),
        )

        return {
            "ambulance_id": amb_id,
            "emergency_id": emg_id,
            "hospital_id": best_hosp.id,
        }
