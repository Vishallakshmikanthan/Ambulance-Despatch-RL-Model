from multi_agent.dispatcher_agent import DispatcherAgent
from multi_agent.ambulance_agent import AmbulanceAgent
from env.models import ActionModel, ObservationModel


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

    def act(self, observation: ObservationModel) -> ActionModel:
        """
        Full pipeline:
        1. Dispatcher selects ambulance + emergency
        2. Ambulance agent selects hospital
        3. Return final ActionModel
        """
        partial = self.dispatcher.select_candidate(observation)

        if partial is None:
            return ActionModel(is_noop=True)

        full = self.ambulance_agent.refine_action(observation, partial)

        return ActionModel(
            ambulance_id=full.get("ambulance_id"),
            emergency_id=full.get("emergency_id"),
            hospital_id=full.get("hospital_id"),
        )
