"""
inference.py — Ambulance Dispatch multi-agent inference script.

STDOUT FORMAT (strict):
    [START] task_name
    [STEP]  step_number reward
    [END]   score
"""

import random
import numpy as np

from env.environment import AmbulanceEnvironment
from multi_agent.coordinator import MultiAgentCoordinator
from tasks.configs import EasyConfig, MediumConfig, HardConfig
from tasks.graders import grade_easy, grade_medium, grade_hard


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def run_task(task_name, config_class, grader):
    print(f"[START] {task_name}", flush=True)

    set_seed(42)

    cfg = config_class()
    env = AmbulanceEnvironment(cfg.to_dict())
    agent = MultiAgentCoordinator()

    obs = env.reset(seed=cfg.seed)

    history = []
    step = 0
    done = False

    while not done and step < cfg.max_steps:
        action = agent.act(obs)
        obs = env.step(action)

        reward = float(obs.reward or 0.0)
        done = bool(obs.done)

        history.append({"reward": reward, "info": {}})
        agent.record_step(reward, {})

        print(f"[STEP] {step} {reward}", flush=True)
        step += 1

    score = grader(history)
    print(f"[END] {score}", flush=True)

    agent.reset()


if __name__ == "__main__":
    run_task("easy", EasyConfig, grade_easy)
    run_task("medium", MediumConfig, grade_medium)
    run_task("hard", HardConfig, grade_hard)
