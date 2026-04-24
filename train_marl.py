"""
train_marl.py — Multi-Agent Reinforcement Learning training script.

Trains one AmbulanceQAgent per ambulance in the fleet, coordinated by an
OversightAgent that detects conflicts and emits coordination signals.

Usage
-----
    python train_marl.py --episodes 500 --n-ambulances 5 --output-dir outputs/marl

Outputs
-------
  outputs/marl/agent_{i}.pt          — saved model for each agent
  outputs/marl/coordination_metrics.csv
  outputs/marl/marl_reward_curve.png
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.environment import AmbulanceEnvironment
from env.models import AmbulanceState, Severity
from agents.fleet_agent import AmbulanceQAgent
from agents.oversight_agent import OversightAgent
from rl.action_mapper import ActionMapper
from rl.action_mask import ActionMask


MAX_EMERGENCIES = 10  # action slots (10 emergency slots + 1 noop = 11)
ACTION_SIZE = MAX_EMERGENCIES + 1


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def build_action_mask(obs, mapper: ActionMapper) -> np.ndarray:
    mapper.build_action_space(obs)
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    mask[MAX_EMERGENCIES] = 1.0  # noop always valid
    unassigned = [e for e in obs.emergencies if not e.assigned]
    for i in range(min(len(unassigned), MAX_EMERGENCIES)):
        mask[i] = 1.0
    return mask


def action_idx_to_model(idx: int, obs, mapper: ActionMapper):
    """Map an action index to an ActionModel."""
    from env.models import ActionModel, AmbulanceState
    if idx >= MAX_EMERGENCIES:
        return ActionModel(ambulance_id=None, emergency_id="", is_noop=True)

    unassigned = [e for e in obs.emergencies if not e.assigned]
    if idx >= len(unassigned):
        return ActionModel(ambulance_id=None, emergency_id="", is_noop=True)

    target_emg = unassigned[idx]
    idle_ambs = [a for a in obs.ambulances if a.state == AmbulanceState.IDLE]
    if not idle_ambs:
        return ActionModel(ambulance_id=None, emergency_id="", is_noop=True)

    best_amb = min(idle_ambs, key=lambda a: abs(a.node - target_emg.node))
    best_hosp = min(
        obs.hospitals,
        key=lambda h: (h.current_patients, abs(h.node - target_emg.node)),
    )
    return ActionModel(
        ambulance_id=best_amb.id,
        emergency_id=target_emg.id,
        hospital_id=best_hosp.id,
    )


def compute_score(metrics: dict) -> float:
    served = metrics.get("served", 0)
    missed = metrics.get("missed", 0)
    total = served + missed
    return float(served / total) if total > 0 else 0.0


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = {
        "n_ambulances": args.n_ambulances,
        "n_hospitals": args.n_hospitals,
        "max_steps": args.max_steps,
        "seed": 42,
    }
    env = AmbulanceEnvironment(env_cfg)

    # Create one agent per ambulance
    agents = [
        AmbulanceQAgent(
            agent_id=i,
            n_agents=args.n_ambulances,
            action_size=ACTION_SIZE,
        )
        for i in range(args.n_ambulances)
    ]
    oversight = OversightAgent(n_agents=args.n_ambulances)
    mapper = ActionMapper()

    # CSV logging
    csv_path = out_dir / "coordination_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "team_reward", "conflict_rate",
            *[f"agent_{i}_reward" for i in range(args.n_ambulances)],
            "score", "epsilon_mean",
        ])

    episode_team_rewards = []
    episode_scores = []

    print(f"MARL Training: {args.n_ambulances} agents, {args.episodes} episodes")

    for episode in range(args.episodes):
        obs = env.reset()
        oversight.reset()

        team_reward = 0.0
        agent_rewards = {i: 0.0 for i in range(args.n_ambulances)}
        step_conflicts = 0

        for step in range(args.max_steps):
            # Oversight observes
            oversight.observe(obs)

            # Each agent selects intended action
            intended: dict = {}
            mask = build_action_mask(obs, mapper)
            for agent in agents:
                coord_sig = np.zeros(2, dtype=np.float32)  # pre-step, no signal yet
                state_vec = agent.encode_observation(obs, coord_sig)
                intended[agent.agent_id] = agent.act(state_vec, mask)

            # Oversight computes coordination signals
            coord_signals = oversight.get_coordination_signals(intended)
            step_conflicts += sum(1 for s in coord_signals.values() if s[0] > 0)

            # Re-encode with coordination signals and pick final actions
            final_states = {}
            final_actions = {}
            for agent in agents:
                sig = coord_signals[agent.agent_id]
                state_vec = agent.encode_observation(obs, sig)
                final_states[agent.agent_id] = state_vec
                final_actions[agent.agent_id] = agent.act(state_vec, mask)

            # Team selects consensus action: use agent 0's choice (leader)
            # In full MARL, each agent acts independently on its assigned emergency
            # Here we pick the highest-priority non-conflicting action
            leader_action_idx = _resolve_team_action(final_actions, coord_signals)
            action_model = action_idx_to_model(leader_action_idx, obs, mapper)

            next_obs, reward, done, info = env.step(action_model)
            if not isinstance(info, dict):
                info = {}

            # Distribute reward to agents
            team_r = reward
            team_reward += team_r
            per_agent_r = {
                i: (team_r / args.n_ambulances) + (
                    -1.0 if coord_signals[i][0] > 0 else 0.5
                )
                for i in range(args.n_ambulances)
            }

            # Store transitions and train each agent
            next_mask = build_action_mask(next_obs, mapper)
            for agent in agents:
                sig = coord_signals[agent.agent_id]
                next_state_vec = agent.encode_observation(next_obs, sig)
                agent.remember(
                    final_states[agent.agent_id],
                    final_actions[agent.agent_id],
                    per_agent_r[agent.agent_id],
                    next_state_vec,
                    done,
                )
                agent.train_step()
                agent_rewards[agent.agent_id] += per_agent_r[agent.agent_id]

            oversight.record_outcome(step, per_agent_r)
            obs = next_obs
            if done:
                break

        score = compute_score(env.metrics)
        episode_team_rewards.append(team_reward)
        episode_scores.append(score)

        conf_rate = step_conflicts / max(env.step_count, 1)
        eps_mean = float(np.mean([a.epsilon for a in agents]))

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, round(team_reward, 2), round(conf_rate, 4),
                *[round(agent_rewards[i], 2) for i in range(args.n_ambulances)],
                round(score, 4), round(eps_mean, 4),
            ])

        if episode % 50 == 0:
            print(
                f"Episode {episode:4d} | TeamReward={team_reward:7.1f} "
                f"Score={score:.3f} Conflicts={step_conflicts} Eps={eps_mean:.3f}"
            )

    # Save models
    for agent in agents:
        agent.save(str(out_dir / f"agent_{agent.agent_id}.pt"))
    print(f"Models saved to {out_dir}/")

    # Plot reward curve
    _plot_rewards(episode_team_rewards, episode_scores, out_dir)

    print(f"MARL training complete. Metrics: {csv_path}")


def _resolve_team_action(
    final_actions: dict,
    coord_signals: dict,
) -> int:
    """Pick the non-conflicting action from agent 0; fall back to noop."""
    a0 = final_actions.get(0, MAX_EMERGENCIES)
    if coord_signals.get(0, np.zeros(2))[0] > 0:
        return MAX_EMERGENCIES  # conflict detected — noop
    return a0


def _plot_rewards(team_rewards, scores, out_dir: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    window = 20
    smoothed = np.convolve(team_rewards, np.ones(window) / window, mode="valid")
    ax1.plot(team_rewards, alpha=0.3, color="#3b82f6", label="Raw")
    ax1.plot(range(window - 1, len(team_rewards)), smoothed, color="#3b82f6", label=f"{window}-ep MA")
    ax1.set_title("MARL Team Reward over Training")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Team Reward")
    ax1.legend()
    ax1.grid(alpha=0.3)

    smoothed_s = np.convolve(scores, np.ones(window) / window, mode="valid")
    ax2.plot(scores, alpha=0.3, color="#10b981", label="Raw Score")
    ax2.plot(range(window - 1, len(scores)), smoothed_s, color="#10b981", label=f"{window}-ep MA")
    ax2.axhline(0.7, color="#ef4444", linestyle="--", label="Target 0.70")
    ax2.set_title("Episode Score (served/total) over Training")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Score [0, 1]")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "marl_reward_curve.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"Reward curve saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--n-ambulances", type=int, default=5)
    parser.add_argument("--n-hospitals", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="outputs/marl")
    args = parser.parse_args()
    train(args)
