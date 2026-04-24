from __future__ import annotations

from typing import Any, Dict


class AutoEvaluator:
    """Runs baseline and advanced agents on the same config and compares results."""

    def __init__(self, env_class, baseline_agent, advanced_agent):
        self.env_class = env_class
        self.baseline = baseline_agent
        self.advanced = advanced_agent

    def run_episode(self, agent, config) -> Dict[str, Any]:
        """Run one full episode and return reward/served totals."""
        cfg_dict = config.to_dict() if hasattr(config, "to_dict") else config
        env = self.env_class(cfg_dict)

        obs = env.reset(seed=cfg_dict.get("seed", 42))
        total_reward = 0.0
        served = 0
        done = False

        while not done:
            # Support both act() and act_all_with_reposition() interfaces
            if hasattr(agent, "act_all_with_reposition"):
                actions = agent.act_all_with_reposition(obs)
                obs = env.step_all(actions)
                reward = float(obs.reward or 0.0)
                done = bool(obs.done)
            else:
                action = agent.act(obs)
                obs = env.step_all([action])
                reward = float(obs.reward or 0.0)
                done = bool(obs.done)

            total_reward += reward
            if reward > 0:
                served += 1

        return {
            "reward": total_reward,
            "served": served,
            "metrics": dict(env.metrics),
        }

    def evaluate(self, config) -> Dict[str, Any]:
        """Evaluate both agents and return comparison dict."""
        base = self.run_episode(self.baseline, config)
        adv = self.run_episode(self.advanced, config)
        improvement = adv["reward"] - base["reward"]

        return {
            "baseline": base,
            "advanced": adv,
            "improvement": improvement,
        }
