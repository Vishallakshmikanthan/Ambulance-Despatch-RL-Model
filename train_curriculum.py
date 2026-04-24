"""
train_curriculum.py — Curriculum learning training script.

Progresses through 10 stages (max_steps 100 → 1000) using CurriculumManager.
Saves curriculum_progress.csv and a stage-progression chart.

Usage
-----
    python train_curriculum.py --episodes 2000 --output-dir outputs/curriculum
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env.environment import AmbulanceEnvironment
from rl.state_encoder import StateEncoder
from rl.action_mapper import ActionMapper
from rl.action_mask import ActionMask
from rl.rl_agent import DQNAgent
from long_horizon.curriculum_manager import CurriculumManager
from long_horizon.long_horizon_env import LongHorizonAmbulanceEnvironment


def compute_score(metrics: dict) -> float:
    served = metrics.get("served", 0)
    missed = metrics.get("missed", 0)
    total = served + missed
    return float(served / total) if total > 0 else 0.0


def make_env(max_steps: int, n_ambulances: int = 5, n_hospitals: int = 3) -> LongHorizonAmbulanceEnvironment:
    return LongHorizonAmbulanceEnvironment({
        "n_ambulances": n_ambulances,
        "n_hospitals": n_hospitals,
        "max_steps": max_steps,
        "seed": np.random.randint(0, 10_000),
        "enable_surges": True,
    })


def train(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    curriculum = CurriculumManager(initial_stage=1, output_dir=str(out_dir))

    encoder = StateEncoder()
    mapper = ActionMapper()
    mask_builder = ActionMask()

    # Build initial env to get state/action sizes
    env = make_env(curriculum.max_steps, args.n_ambulances, args.n_hospitals)
    obs = env.reset()
    mapper.build_action_space(obs)
    state = encoder.encode(obs)

    # Augmented state includes history encoding
    base_size = len(state)
    aug_size = env.get_augmented_state_size(base_size)
    action_size = mapper.size()

    agent = DQNAgent(aug_size, action_size, use_dueling=True, use_per=True)

    episode_rewards = []
    episode_scores = []
    episode_stages = []
    best_score = 0.0

    print(f"Curriculum Training: {args.episodes} episodes, initial stage={curriculum.stage}")

    for episode in range(args.episodes):
        # Recreate env with current curriculum max_steps
        env = make_env(curriculum.max_steps, args.n_ambulances, args.n_hospitals)
        obs = env.reset()

        total_reward = 0.0

        for step in range(curriculum.max_steps):
            base_state = encoder.encode(obs)
            state = env.encode_augmented_state(base_state)

            mapper.build_action_space(obs)
            mask = mask_builder.build_mask(mapper)

            # Pad/truncate mask to match action_size
            if len(mask) < action_size:
                mask = np.pad(mask, (0, action_size - len(mask)))
            elif len(mask) > action_size:
                mask = mask[:action_size]

            action_idx = agent.act(state, mask)
            action_model = mapper.decode(action_idx)

            result = env.step(action_model)
            if isinstance(result, tuple):
                next_obs, reward, done, info = result
            else:
                next_obs = result
                reward = next_obs.reward
                done = next_obs.done
                info = {}

            next_base = encoder.encode(next_obs)
            next_state = env.encode_augmented_state(next_base)

            agent.remember(state, action_idx, reward, next_state, done)
            agent.replay()

            total_reward += reward
            obs = next_obs

            if done:
                break

        score = compute_score(env.metrics)
        advanced = curriculum.record_episode(score)
        episode_rewards.append(total_reward)
        episode_scores.append(score)
        episode_stages.append(curriculum.stage)

        if score > best_score:
            best_score = score
            agent.policy_net.eval()
            import torch
            torch.save(agent.policy_net.state_dict(), str(out_dir / "best_model.pt"))
            agent.policy_net.train()

        if episode % 50 == 0 or advanced:
            tag = " [ADVANCED]" if advanced else ""
            print(
                f"Episode {episode:4d} | Stage={curriculum.stage} "
                f"MaxSteps={curriculum.max_steps} | "
                f"Reward={total_reward:7.1f} Score={score:.3f}{tag}"
            )

    _plot_curriculum(episode_rewards, episode_scores, episode_stages, out_dir)
    print(f"Curriculum training complete. Final stage: {curriculum.stage}")
    print(f"Best score: {best_score:.4f}")


def _plot_curriculum(rewards, scores, stages, out_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    window = 20

    # Reward
    sm_r = np.convolve(rewards, np.ones(window) / window, mode="valid")
    axes[0].plot(rewards, alpha=0.3, color="#3b82f6")
    axes[0].plot(range(window - 1, len(rewards)), sm_r, color="#3b82f6")
    axes[0].set_title("Episode Reward (Curriculum)")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.3)

    # Score
    sm_s = np.convolve(scores, np.ones(window) / window, mode="valid")
    axes[1].plot(scores, alpha=0.3, color="#10b981")
    axes[1].plot(range(window - 1, len(scores)), sm_s, color="#10b981")
    axes[1].axhline(0.7, color="#ef4444", linestyle="--", label="Target 0.70")
    axes[1].set_title("Episode Score")
    axes[1].set_ylabel("Score [0, 1]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Stage progression
    axes[2].step(range(len(stages)), stages, color="#f59e0b", where="post")
    axes[2].set_title("Curriculum Stage Progression")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Stage (1-10)")
    axes[2].set_yticks(range(1, 11))
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "curriculum_progress_chart.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"Curriculum chart saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--n-ambulances", type=int, default=5)
    parser.add_argument("--n-hospitals", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="outputs/curriculum")
    args = parser.parse_args()
    train(args)
