"""
fleet_agent.py — Independent Q-Learning agent for a single ambulance in the fleet.

Each ambulance in the fleet gets its own AmbulanceQAgent with:
  - DuelingDQN policy/target networks
  - PrioritizedReplayBuffer
  - Local state encoder (own obs + compressed fleet summary + oversight signal)
  - Independent Adam optimizer
"""
from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.dqn import DuelingDQN
from rl.prioritized_replay_buffer import PrioritizedReplayBuffer
from env.models import ObservationModel, AmbulanceState, Severity

# Local obs dims: node(1) + state_onehot(6) + eta(1) = 8
# Fleet summary per other agent: node(1) + busy(1) = 2  →  (n-1)*2
# Oversight coordination signal: conflict flag(1) + conflict_amb_id(1) = 2
# Global context: pending_critical(1) + pending_high(1) + pending_normal(1) +
#                 avg_hosp_occ(1) + step_norm(1) = 5
_BASE_LOCAL = 8
_GLOBAL_CTX = 5
_OVERSIGHT_SIG = 2


def _build_obs_size(n_agents: int) -> int:
    fleet_summary = (n_agents - 1) * 2
    return _BASE_LOCAL + fleet_summary + _OVERSIGHT_SIG + _GLOBAL_CTX


class AmbulanceQAgent:
    """
    Independent DQN agent representing a single ambulance unit.

    Parameters
    ----------
    agent_id : int          Index of this ambulance within the fleet (0-based).
    n_agents : int          Total fleet size.
    action_size : int       Number of discrete actions (emergency slots + noop).
    lr : float              Learning-rate for Adam.
    gamma : float           Discount factor.
    capacity : int          Replay buffer capacity.
    """

    def __init__(
        self,
        agent_id: int,
        n_agents: int,
        action_size: int = 11,   # 10 emergency slots + 1 noop
        lr: float = 5e-4,
        gamma: float = 0.99,
        capacity: int = 20_000,
    ):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.action_size = action_size
        self.gamma = gamma

        self.obs_size = _build_obs_size(n_agents)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingDQN(self.obs_size, action_size).to(self.device)
        self.target_net = DuelingDQN(self.obs_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(capacity=capacity)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9992
        self.tau = 0.005
        self.batch_size = 64
        self.step_count = 0

        # Track the agent's last intended action for conflict detection
        self.intended_action: int | None = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def encode_observation(
        self,
        obs: ObservationModel,
        coordination_signal: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Encode the full observation into a fixed-size vector for this agent.

        coordination_signal : 2-element array [conflict_flag, conflict_amb_id_norm]
                              provided by the OversightAgent each step.
        """
        features: list[float] = []

        # --- Own ambulance state ---
        my_amb = next((a for a in obs.ambulances if a.id == self.agent_id), None)
        if my_amb is not None:
            features.append(float(my_amb.node) / 100.0)
            state_oh = [0.0] * 6
            state_map = {
                AmbulanceState.IDLE: 0, AmbulanceState.DISPATCHED: 1,
                AmbulanceState.EN_ROUTE: 2, AmbulanceState.AT_SCENE: 3,
                AmbulanceState.TRANSPORTING: 4, AmbulanceState.RETURNING: 5,
            }
            state_oh[state_map.get(my_amb.state, 0)] = 1.0
            features.extend(state_oh)
            features.append(float(my_amb.eta) / 50.0)
        else:
            features.extend([0.0] * _BASE_LOCAL)

        # --- Fleet summary (other agents) ---
        for a in obs.ambulances:
            if a.id == self.agent_id:
                continue
            features.append(float(a.node) / 100.0)
            features.append(0.0 if a.state == AmbulanceState.IDLE else 1.0)

        # Pad fleet summary if fewer agents present than expected
        present_others = sum(1 for a in obs.ambulances if a.id != self.agent_id)
        missing = (self.n_agents - 1) - present_others
        features.extend([0.0] * (missing * 2))

        # --- Oversight coordination signal ---
        if coordination_signal is not None and len(coordination_signal) >= 2:
            features.append(float(coordination_signal[0]))
            features.append(float(coordination_signal[1]))
        else:
            features.extend([0.0, 0.0])

        # --- Global context ---
        emgs = obs.emergencies
        features.append(sum(1 for e in emgs if e.severity == Severity.CRITICAL and not e.assigned) / 10.0)
        features.append(sum(1 for e in emgs if e.severity == Severity.HIGH and not e.assigned) / 10.0)
        features.append(sum(1 for e in emgs if e.severity == Severity.NORMAL and not e.assigned) / 10.0)
        if obs.hospitals:
            avg_occ = np.mean([h.current_patients / max(h.capacity, 1) for h in obs.hospitals])
        else:
            avg_occ = 0.0
        features.append(float(avg_occ))
        features.append(float(obs.step) / 1000.0)

        vec = np.array(features, dtype=np.float32)
        # Safety: clip to expected size
        expected = self.obs_size
        if len(vec) < expected:
            vec = np.pad(vec, (0, expected - len(vec)))
        elif len(vec) > expected:
            vec = vec[:expected]
        return vec

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, mask: np.ndarray | None = None) -> int:
        """
        Epsilon-greedy action selection.

        mask : binary array of length action_size; 0 means action is invalid.
        """
        if mask is None:
            mask = np.ones(self.action_size, dtype=np.float32)

        if random.random() < self.epsilon:
            valid = np.where(mask == 1)[0]
            action = int(np.random.choice(valid)) if len(valid) > 0 else 0
        else:
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.policy_net.eval()
            with torch.no_grad():
                q = self.policy_net(t).cpu().numpy()[0]
            self.policy_net.train()
            q[mask == 0] = -1e9
            action = int(np.argmax(q))

        self.intended_action = action
        return action

    # ------------------------------------------------------------------
    # Memory & training
    # ------------------------------------------------------------------

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """Run one gradient-descent step. Returns loss or None if not ready."""
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        # Handle both PER (tuple with weights/indices) and simple buffers
        if len(batch) == 3:
            (states, actions, rewards, next_states, dones), weights, indices = batch
        else:
            states, actions, rewards, next_states, dones = batch
            weights = np.ones(self.batch_size, dtype=np.float32)
            indices = None

        states_t      = torch.tensor(np.array(states),      dtype=torch.float32).to(self.device)
        actions_t     = torch.tensor(actions,               dtype=torch.long).to(self.device)
        rewards_t     = torch.tensor(rewards,               dtype=torch.float32).to(self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_t       = torch.tensor(dones,                 dtype=torch.float32).to(self.device)
        weights_t     = torch.tensor(weights,               dtype=torch.float32).to(self.device)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q       = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q     = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        td_errors = (current_q - target_q).detach().abs().cpu().numpy()
        loss = (weights_t * nn.functional.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target network
        for tp, op in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

        # Update PER priorities if applicable
        if indices is not None and hasattr(self.memory, "update_priorities"):
            self.memory.update_priorities(indices, td_errors + 1e-6)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.step_count += 1
        return float(loss.item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)
        self.step_count = ckpt.get("step_count", 0)
