from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from multi_agent.dispatcher_agent import DispatcherAgent
from multi_agent.ambulance_agent import AmbulanceAgent
from multi_agent.planner import LookaheadPlanner
from self_improvement.performance_analyzer import PerformanceAnalyzer
from self_improvement.strategy_adapter import StrategyAdapter
from agents.fleet_agent import AmbulanceQAgent
from rl.action_mapper import ActionMapper
from rl.action_mask import ActionMask
from env.models import ActionModel, ObservationModel, AmbulanceState, Severity

_SEVERITY_RANK = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.NORMAL: 2,
}

_ADAPTATION_INTERVAL = 10

# Penalty applied to each conflicting agent when two agents target the same emergency
_CONFLICT_PENALTY = 5.0


class MultiAgentCoordinator:
    """
    Multi-agent coordinator.

    Each ambulance is controlled by an independent AmbulanceQAgent that selects
    actions from a shared discrete action space (emergency slots + noop).
    The coordinator:
      1. Builds a per-step action space from the current observation.
      2. Lets each agent independently select an action.
      3. Detects conflicts (two agents dispatching to the same emergency) and
         applies a conflict penalty (Step 7.4).
      4. Splits the global environment reward equally among all agents (Step 7.3).
      5. Stores transitions and triggers individual learning steps.

    The original heuristic pipeline (DispatcherAgent / LookaheadPlanner) is
    preserved for the legacy ``act()`` single-action interface used by the
    existing server and evaluation scripts.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, n_ambulances: int = 2, action_size: int = 11):
        self.n_ambulances = n_ambulances
        self.action_size = action_size

        # Independent Q-agents — one per ambulance (Step 7.2)
        self.fleet_agents: Dict[int, AmbulanceQAgent] = {
            i: AmbulanceQAgent(agent_id=i, n_agents=n_ambulances, action_size=action_size)
            for i in range(n_ambulances)
        }

        # Shared action mapper / mask builder
        self.mapper = ActionMapper()
        self.mask_builder = ActionMask()

        # Legacy heuristic pipeline (kept for backward-compat)
        self.dispatcher = DispatcherAgent()
        self.ambulance_agent = AmbulanceAgent()
        self.planner = LookaheadPlanner(horizon=3)
        self.analyzer = PerformanceAnalyzer()
        self.adapter = StrategyAdapter()
        self._step_count = 0

        # Per-agent transition buffers (populated by marl_act, consumed by marl_learn)
        self._last_states: Dict[int, np.ndarray] = {}
        self._last_actions: Dict[int, int] = {}
        self.conflicts = 0

    # ------------------------------------------------------------------ #
    #  Episode management                                                  #
    # ------------------------------------------------------------------ #

    def reset(self):
        """Clear per-episode state."""
        self.analyzer.reset()
        self._step_count = 0
        self._last_states.clear()
        self._last_actions.clear()

    def record_step(self, reward, info):
        """Record a completed env step and periodically adapt strategy."""
        self.analyzer.record(reward, info)
        self._step_count += 1
        if self._step_count % _ADAPTATION_INTERVAL == 0:
            metrics = self.analyzer.get_metrics()
            self.adapter.update(metrics)

    # ------------------------------------------------------------------ #
    #  Multi-agent RL interface (Steps 7.2 – 7.4)                        #
    # ------------------------------------------------------------------ #

    def marl_act(self, observation: ObservationModel) -> Dict[int, int]:
        """
        Each fleet agent independently selects an action index.

        Returns
        -------
        actions : dict mapping agent_id -> action_index
        """
        self.mapper.build_action_space(observation)
        mask = self.mask_builder.build_mask(self.mapper)

        actions: Dict[int, int] = {}
        for agent_id, agent in self.fleet_agents.items():
            state = agent.encode_observation(observation)
            action_idx = agent.act(state, mask)
            actions[agent_id] = action_idx
            self._last_states[agent_id] = state
            self._last_actions[agent_id] = action_idx

        return actions

    def marl_learn(
        self,
        global_reward: float,
        next_observation: ObservationModel,
        done: bool,
    ) -> Dict[int, float]:
        """
        Distribute rewards and trigger per-agent learning.

        Step 7.3 — team reward split: reward_i = global_reward / n_agents
        Step 7.4 — conflict penalty: if agents chose the same emergency, each
                   conflicting agent receives an additional –5 penalty.

        Returns
        -------
        agent_rewards : dict mapping agent_id -> reward used for learning
        """
        n = max(len(self.fleet_agents), 1)
        base_reward = global_reward / n

        # Detect conflicts (multiple agents targeting the same non-noop emergency)
        action_to_agents: Dict[int, List[int]] = {}
        for agent_id, action_idx in self._last_actions.items():
            action_to_agents.setdefault(action_idx, []).append(agent_id)

        conflicting_agents = set()
        # action_idx == 0 is typically noop — exclude it from conflict detection
        for action_idx, agents_list in action_to_agents.items():
            if action_idx != 0 and len(agents_list) > 1:
                conflicting_agents.update(agents_list)

        if conflicting_agents:
            self.conflicts += 1
            print(f"[CONFLICT DETECTED] Total: {self.conflicts}")

        self.mapper.build_action_space(next_observation)
        mask = self.mask_builder.build_mask(self.mapper)
        agent_rewards: Dict[int, float] = {}

        for agent_id, agent in self.fleet_agents.items():
            reward_i = base_reward
            if agent_id in conflicting_agents:
                reward_i -= _CONFLICT_PENALTY

            next_state = agent.encode_observation(next_observation)
            agent.remember(
                self._last_states[agent_id],
                self._last_actions[agent_id],
                reward_i,
                next_state,
                done,
            )
            agent.train_step()
            agent_rewards[agent_id] = reward_i

        return agent_rewards

    def decode_actions(self, actions: Dict[int, int]) -> Dict[int, ActionModel]:
        """Convert agent action indices to ActionModel objects."""
        decoded: Dict[int, ActionModel] = {}
        for agent_id, idx in actions.items():
            try:
                decoded[agent_id] = self.mapper.decode(idx)
            except Exception:
                decoded[agent_id] = ActionModel(is_noop=True)
        return decoded

    # ------------------------------------------------------------------ #
    #  Legacy single-action interface (unchanged)                         #
    # ------------------------------------------------------------------ #

    def _build_candidates(self, observation: ObservationModel) -> list:
        """Generate 2–3 candidate ActionModels for lookahead evaluation."""
        candidates = []
        weights = self.adapter.get_weights()

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
        1. Dispatcher selects ambulance + emergency (with adaptive weights)
        2. Ambulance agent selects hospital
        3. Evaluate candidates via lookahead planner (if env provided)
        4. Return best ActionModel
        """
        weights = self.adapter.get_weights()
        partial = self.dispatcher.select_candidate(observation, weights=weights)

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
