"""
train_selfplay.py — Self-improvement training loop.

Runs iterative self-play: evaluate → detect weaknesses → generate targeted
scenarios → train → repeat.

Usage
-----
    python train_selfplay.py --iterations 20 --output-dir outputs/selfplay
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
from self_improvement.self_play_trainer import SelfPlayTrainer
from self_improvement.adversarial_generator import ScenarioConfig


def compute_score(metrics: dict) -> float:
    served = metrics.get("served", 0)
    missed = metrics.get("missed", 0)
    total = served + missed
    return float(served / total) if total > 0 else 0.0


def train(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = StateEncoder()
    mapper = ActionMapper()
    mask_builder = ActionMask()

    # Build agent
    env0 = AmbulanceEnvironment({"n_ambulances": 5, "n_hospitals": 3, "max_steps": 100})
    obs0 = env0.reset()
    mapper.build_action_space(obs0)
    state0 = encoder.encode(obs0)
    state_size = len(state0)
    action_size = mapper.size()

    agent = DQNAgent(state_size, action_size, use_dueling=True, use_per=True)

    def env_factory(cfg: dict) -> AmbulanceEnvironment:
        return AmbulanceEnvironment(cfg)

    def action_mapper_fn(obs):
        mapper.build_action_space(obs)
        mask = mask_builder.build_mask(mapper)
        state = encoder.encode(obs)
        action_idx = agent.act(state, mask)
        action_model = mapper.decode(action_idx)
        return action_idx, action_model, mask

    base_cfg = ScenarioConfig(
        n_ambulances=5, n_hospitals=3, max_steps=100
    )

    trainer = SelfPlayTrainer(
        env_factory=env_factory,
        agent=agent,
        action_mapper=action_mapper_fn,
        score_fn=compute_score,
        n_eval=args.n_eval,
        targeted_episodes=args.targeted_episodes,
        expert_gap_threshold=0.30,
        output_dir=str(out_dir),
        base_config=base_cfg,
    )

    metrics = trainer.run(n_iterations=args.iterations, verbose=True)

    # Plot improvement curve
    _plot_selfplay(metrics, out_dir)

    # Save agent
    import torch
    torch.save(agent.policy_net.state_dict(), str(out_dir / "selfplay_agent.pt"))
    print(f"Self-play training complete. Agent saved to {out_dir}/selfplay_agent.pt")


def _plot_selfplay(metrics: list, out_dir: Path):
    if not metrics:
        return

    iterations = [m["iteration"] for m in metrics]
    eval_scores = [m["avg_eval_score"] for m in metrics]
    expert_scores = [m["expert_score"] for m in metrics]
    gaps = [m["expert_gap"] for m in metrics]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(iterations, eval_scores, "o-", color="#10b981", label="Agent Score")
    axes[0].plot(iterations, expert_scores, "s--", color="#f59e0b", label="Expert Score")
    axes[0].fill_between(iterations, eval_scores, expert_scores, alpha=0.15, color="#6366f1")
    axes[0].set_title("Self-Improvement: Agent vs Expert Score")
    axes[0].set_ylabel("Score [0, 1]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(iterations, gaps, "o-", color="#ef4444", label="Expert Gap")
    axes[1].axhline(0.0, color="#6b7280", linestyle="--")
    axes[1].set_title("Expert Gap (shrinking = improvement)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Gap")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "selfplay_improvement.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"Self-play chart saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--targeted-episodes", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="outputs/selfplay")
    args = parser.parse_args()
    train(args)
