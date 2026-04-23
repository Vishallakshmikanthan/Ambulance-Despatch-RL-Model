from multi_agent.dispatcher_agent import DispatcherAgent
from multi_agent.ambulance_agent import AmbulanceAgent
from multi_agent.planner import LookaheadPlanner
from env.models import ActionModel, ObservationModel, AmbulanceState, Severity

_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.NORMAL: 2,
}


class MultiAgentCoordinator:
    """
    Master controller that orchestrates the multi-agent pipeline:
    1. Dispatcher selects ambulance + emergency
    2. Ambulance agent selects hospital
    3. Returns final ActionModel
    """

    def __init__(self):
        self.dispatcher = DispatcherAgent()
        self.ambulance_agent = AmbulanceAgent()
        self.planner = LookaheadPlanner(horizon=3)

    def _build_candidates(self, observation: ObservationModel) -> list:
        """Generate 2–3 candidate ActionModels for lookahead evaluation."""
        candidates = []

        idle_ambs = [a for a in observation.ambulances if a.state == AmbulanceState.IDLE]
        unassigned = sorted(
            [e for e in observation.emergencies if not e.assigned],
            key=lambda e: (_SEVERITY_RANK.get(e.severity, 3), e.time_remaining),
        )

        if idle_ambs and unassigned:
            # primary: best emergency
            primary_emg = unassigned[0]
            best_amb = min(idle_ambs, key=lambda a: abs(a.node - primary_emg.node))
            primary_partial = {"ambulance_id": best_amb.id, "emergency_id": primary_emg.id}
            primary_full = self.ambulance_agent.refine_action(observation, primary_partial)
            candidates.append(ActionModel(
                ambulance_id=primary_full.get("ambulance_id"),
                emergency_id=primary_full.get("emergency_id"),
                hospital_id=primary_full.get("hospital_id"),
            ))

            # alternative: next best emergency (if available)
            if len(unassigned) > 1:
                alt_emg = unassigned[1]
                alt_amb = min(idle_ambs, key=lambda a: abs(a.node - alt_emg.node))
                alt_partial = {"ambulance_id": alt_amb.id, "emergency_id": alt_emg.id}
                alt_full = self.ambulance_agent.refine_action(observation, alt_partial)
                candidates.append(ActionModel(
                    ambulance_id=alt_full.get("ambulance_id"),
                    emergency_id=alt_full.get("emergency_id"),
                    hospital_id=alt_full.get("hospital_id"),
                ))

        # noop as fallback candidate
        candidates.append(ActionModel(is_noop=True))

        return candidates

    def act(self, observation: ObservationModel, env=None) -> ActionModel:
        """
        Full pipeline:
        1. Dispatcher selects ambulance + emergency
        2. Ambulance agent selects hospital
        3. Evaluate candidates via lookahead planner (if env provided)
        4. Return best ActionModel
        """
        partial = self.dispatcher.select_candidate(observation)

        if partial is None:
            return ActionModel(is_noop=True)

        full = self.ambulance_agent.refine_action(observation, partial)
        original_action = ActionModel(
            ambulance_id=full.get("ambulance_id"),
            emergency_id=full.get("emergency_id"),
            hospital_id=full.get("hospital_id"),
        )

        if env is None:
            return original_action

        try:
            candidates = self._build_candidates(observation)

            best_action = None
            best_score = float("-inf")

            for action in candidates:
                score = self.planner.simulate(env, action)
                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action if best_action is not None else original_action

        except Exception:
            return original_action
