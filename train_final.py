"""
train_final.py — Hackathon final training script.
Adds moving-average improvement curve, anti-reward-hacking validation,
multi-component reward breakdown, and per-step reward logging.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.environment import AmbulanceEnvironment as AmbulanceEnv
from rl.state_encoder import StateEncoder
from rl.action_mapper import ActionMapper
from rl.action_mask import ActionMask
from rl.rl_agent import DQNAgent


# ---------------------------------------------------------------------------
# FIX 1 — Moving average helper
# ---------------------------------------------------------------------------

def compute_moving_avg(rewards, window=20):
    avg = []
    for i in range(len(rewards)):
        if i < window:
            avg.append(sum(rewards[:i+1]) / (i+1))
        else:
            avg.append(sum(rewards[i-window:i]) / window)
    return avg


def plot_results(rewards, output_path="reward_curve.png"):
    moving_avg = compute_moving_avg(rewards)

    plt.figure()
    plt.plot(rewards, alpha=0.3, label="Raw Reward")
    plt.plot(moving_avg, linewidth=2, label="Moving Avg (20)")
    plt.title("Training Improvement Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Reward curve saved to {output_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    episodes = 500
    max_steps = 150

    env = AmbulanceEnv()
    encoder = StateEncoder()
    mapper = ActionMapper()
    mask_builder = ActionMask()

    obs = env.reset()
    mapper.build_action_space(obs)
    state_size = len(encoder.encode(obs))
    action_size = mapper.size()

    agent = DQNAgent(state_size, action_size)

    rewards: list[float] = []
    best_reward = -float("inf")

    os.makedirs("outputs/final", exist_ok=True)

    print("=== Final Training ===")

    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            mapper.build_action_space(obs)
            mask = mask_builder.build_mask(mapper)
            state = encoder.encode(obs)
            action_index = agent.act(state, mask)
            action = mapper.decode(action_index)

            # FIX 2 — env validates the action internally; access info via env.last_info
            next_obs = env.step(action)
            reward = env.last_reward
            done = env.last_done
            info = env.last_info

            # FIX 4 — log reward components each step
            print(
                f"[STEP] ep={episode+1} step={step+1} "
                f"reward={reward:.2f} "
                f"success={info.get('reward_success', 0):.1f} "
                f"eff={info.get('reward_efficiency', 0):.1f} "
                f"penalty={info.get('reward_penalty', 0):.1f}"
            )

            next_state = encoder.encode(next_obs)
            agent.store(state, action_index, reward, next_state, done)
            agent.train_step()

            obs = next_obs
            total_reward += reward
            if done:
                break

        agent.decay_epsilon(episode)
        rewards.append(total_reward)
        moving_avg = float(np.mean(rewards[-20:]))

        if (episode + 1) % 10 == 0:
            print(f"[Episode {episode+1}] Total={total_reward:.1f} Avg(20)={moving_avg:.1f} ε={agent.epsilon:.3f}")

        if moving_avg > best_reward:
            best_reward = moving_avg
            torch.save(agent.policy_net.state_dict(), "outputs/final/best_model.pt")

    # FIX 1 — print summary and plot
    print(f"[TRAINING COMPLETE] Initial Reward: {rewards[0]:.2f}")
    print(f"[TRAINING COMPLETE] Final Reward: {rewards[-1]:.2f}")
    plot_results(rewards, output_path="outputs/final/reward_curve.png")
    # Also save to repo root so README can reference it directly
    plot_results(rewards, output_path="reward_curve.png")
    print(f"Best Avg(20) Reward: {best_reward:.2f}")


if __name__ == "__main__":
    main()
