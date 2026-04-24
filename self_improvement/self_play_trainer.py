"""
self_play_trainer.py — Outer self-improvement loop.

Algorithm
---------
For each iteration:
  1. Evaluate current agent on N_EVAL random scenarios.
  2. WeaknessDetector identifies failure clusters.
  3. AdversarialScenarioGenerator creates targeted scenarios.
  4. ExpertAgent provides imitation data when learner underperforms.
  5. Train on targeted scenarios for TARGETED_EPISODES episodes.
  6. Log metrics to CSV.

Usage
-----
    trainer = SelfPlayTrainer(env_factory=..., agent=...)
    trainer.run(n_iterations=20)
"""
from __future__ import annotations

import csv
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from self_improvement.adversarial_generator import AdversarialScenarioGenerator, ScenarioConfig
from self_improvement.weakness_detector import WeaknessDetector
from self_improvement.expert_agent import ExpertAgent
from rl.state_encoder import StateEncoder


class SelfPlayTrainer:
    """
    Self-improvement training loop combining:
      - Failure detection and targeted scenario generation
      - Expert-in-the-loop imitation learning
      - Adaptive curriculum

    Parameters
    ----------
    env_factory : callable  () → AmbulanceEnvironment instance
    agent       : DQNAgent  The learning RL agent.
    action_mapper : callable  (ObservationModel) → (action_index, ActionModel, mask)
    score_fn    : callable  (env_metrics) → float  Maps env metrics to [0, 1] score.
    n_eval      : int  Episodes per evaluation batch.
    targeted_episodes : int  Training episodes on targeted scenarios per iteration.
    expert_gap_threshold : float  If learner/expert gap > this, add imitation data.
    output_dir  : str  Where to save CSVs and checkpoints.
    """

    def __init__(
        self,
        env_factory: Callable,
        agent,
        action_mapper: Callable,
        score_fn: Callable,
        n_eval: int = 20,
        targeted_episodes: int = 30,
        expert_gap_threshold: float = 0.30,
        output_dir: str = "outputs/selfplay",
        base_config: Optional[ScenarioConfig] = None,
    ):
        self.env_factory = env_factory
        self.agent = agent
        self.action_mapper = action_mapper
        self.score_fn = score_fn
        self.n_eval = n_eval
        self.targeted_episodes = targeted_episodes
        self.expert_gap_threshold = expert_gap_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_config = base_config or ScenarioConfig()

        self.detector = WeaknessDetector(failure_threshold=0.5)
        self.generator = AdversarialScenarioGenerator(failure_threshold=0.5)
        self.expert = ExpertAgent(stage=0)

        self.iteration_scores: List[float] = []
        self.encoder = StateEncoder()

        self._csv_path = self.output_dir / "selfplay_iterations.csv"
        self._init_csv()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, n_iterations: int = 20, verbose: bool = True) -> List[dict]:
        """Run the full self-improvement loop for n_iterations."""
        all_metrics = []

        for iteration in range(1, n_iterations + 1):
            t0 = time.time()

            # --- Phase 1: Evaluation ---
            eval_results = self._evaluate(self.n_eval)
            avg_score = float(np.mean([sc for _, sc in eval_results]))

            # --- Phase 2: Expert evaluation ---
            expert_score = self._evaluate_expert(n=5)

            # --- Phase 3: Weakness detection ---
            report = self.detector.analyze(eval_results)
            n_weaknesses = len(report.clusters)

            # --- Phase 4: Escalate expert if learner catching up ---
            if avg_score > 0.65 and self.expert.stage == 0:
                self.expert.set_stage(1)
            elif avg_score > 0.80 and self.expert.stage == 1:
                self.expert.set_stage(2)

            # --- Phase 5: Generate targeted scenarios ---
            targeted = self.generator.generate_scenarios(
                n=self.targeted_episodes, base_config=self.base_config
            )

            # --- Phase 6: Train on targeted scenarios ---
            imitation_used = 0
            train_rewards = []
            for scenario_cfg in targeted:
                env = self.env_factory(scenario_cfg.to_env_dict())
                episode_reward = 0.0

                # Optionally add expert imitation data
                gap = expert_score - avg_score
                if gap > self.expert_gap_threshold:
                    expert_transitions = self.expert.collect_trajectory(env, n_steps=20)
                    for s, am, r, ns, done in expert_transitions:
                        # Map ActionModel to action index using action_mapper logic
                        action_idx = self._actionmodel_to_idx(am, env._get_observation())
                        self.agent.memory.push(s, action_idx, r * 2.0, ns, done)
                    imitation_used += len(expert_transitions)
                    env.reset(seed=scenario_cfg.seed)

                obs = env._get_observation()
                for _ in range(scenario_cfg.max_steps):
                    state = self.encoder.encode(obs)
                    action_idx, action_model, mask = self.action_mapper(obs)
                    next_obs, reward, done, _ = env.step(action_model)
                    next_state = self.encoder.encode(next_obs)
                    self.agent.remember(state, action_idx, reward, next_state, done)
                    self.agent.replay()  # type: ignore
                    episode_reward += reward
                    obs = next_obs
                    if done:
                        break
                train_rewards.append(episode_reward)

            elapsed = time.time() - t0
            metrics = {
                "iteration": iteration,
                "avg_eval_score": round(avg_score, 4),
                "expert_score": round(expert_score, 4),
                "expert_gap": round(expert_score - avg_score, 4),
                "n_weaknesses": n_weaknesses,
                "imitation_transitions": imitation_used,
                "avg_train_reward": round(float(np.mean(train_rewards)) if train_rewards else 0.0, 4),
                "expert_stage": self.expert.stage,
                "elapsed_s": round(elapsed, 2),
            }
            all_metrics.append(metrics)
            self._append_csv(metrics)

            total_reward = metrics["avg_train_reward"]
            self.iteration_scores.append(total_reward)
            print(f"[IMPROVEMENT] Iteration {iteration} Score: {total_reward}")

            if verbose:
                print(
                    f"[SelfPlay] Iter {iteration:3d} | "
                    f"EvalScore={avg_score:.3f} ExpertScore={expert_score:.3f} "
                    f"Gap={expert_score-avg_score:.3f} | "
                    f"Weaknesses={n_weaknesses} ImitationData={imitation_used} | "
                    f"{elapsed:.1f}s"
                )

        return all_metrics

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate(self, n: int) -> List[tuple]:
        """Evaluate current agent on n random scenarios."""
        results = []
        for _ in range(n):
            seed = random.randint(0, 100_000)
            cfg = ScenarioConfig(seed=seed, max_steps=self.base_config.max_steps)
            env = self.env_factory(cfg.to_env_dict())
            obs = env.reset(seed=seed)
            for _ in range(cfg.max_steps):
                state = self.encoder.encode(obs)
                _, action_model, mask = self.action_mapper(obs)
                obs, reward, done, _ = env.step(action_model)
                if done:
                    break
            score = self.score_fn(env.metrics)
            results.append((cfg, score))
        return results

    def _evaluate_expert(self, n: int = 5) -> float:
        """Evaluate expert agent on n scenarios."""
        scores = []
        for _ in range(n):
            seed = random.randint(0, 100_000)
            cfg = ScenarioConfig(seed=seed, max_steps=self.base_config.max_steps)
            env = self.env_factory(cfg.to_env_dict())
            env.reset(seed=seed)
            obs = env._get_observation()
            for _ in range(cfg.max_steps):
                action = self.expert.act(obs)
                obs, _, done, _ = env.step(action)
                if done:
                    break
            scores.append(self.score_fn(env.metrics))
        return float(np.mean(scores)) if scores else 0.0

    def _actionmodel_to_idx(self, action_model, obs) -> int:
        """Convert an ActionModel to an integer action index (best effort)."""
        if action_model.is_noop or not action_model.emergency_id:
            return 10  # noop slot
        unassigned = [e for e in obs.emergencies if not e.assigned]
        for i, emg in enumerate(unassigned[:10]):
            if emg.id == action_model.emergency_id:
                return i
        return 10

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

    def _init_csv(self):
        if not self._csv_path.exists():
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "iteration", "avg_eval_score", "expert_score", "expert_gap",
                    "n_weaknesses", "imitation_transitions", "avg_train_reward",
                    "expert_stage", "elapsed_s",
                ])
                writer.writeheader()

    def _append_csv(self, row: dict):
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)
