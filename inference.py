"""
inference.py — Ambulance Dispatch optimised inference script.

STDOUT FORMAT (strict):
    [START] task_name
    [STEP]  step_number reward
    [END]   score
"""

import argparse
import random
import numpy as np

from env.environment import AmbulanceEnvironment
from env.models import ActionModel
from agents.repositioning_oracle import RepositioningOracle
from agents.baseline import BaselineAgent
from tasks.easy import EasyConfig
from tasks.medium import MediumConfig
from tasks.hard import HardConfig
from grader_easy import grade_easy
from grader_medium import grade_medium
from grader_hard import grade_hard
from tasks.configs import EasyConfig as EasyCfg
from evaluation.auto_evaluator import AutoEvaluator
from evaluation.report import generate_report


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def run_task(task_name: str, config_class, grader):
    print(f"[START] {task_name}", flush=True)

    set_seed(42)

    cfg = config_class()
    env = AmbulanceEnvironment(cfg.to_dict())
    obs = env.reset(seed=cfg.seed)

    # Medium task: disable repositioning — served_pct (0.50 weight) is hurt
    # more by long repos (9+ steps blocking dispatch) than idle_fraction (0.15) saves.
    # Easy/Hard: enable repositioning to pre-position near hotspots for faster response.
    enable_repos = task_name != "medium"
    agent = RepositioningOracle(enable_reposition=enable_repos).bind_env(env)

    step = 0
    done = False

    while not done and step < cfg.max_steps:
        # Multi-dispatch: all idle ambulances dispatched in one physics tick
        actions = agent.act_all_with_reposition(obs)
        obs = env.step_all(actions)
        reward = float(obs.reward or 0.0)
        done = bool(obs.done)
        print(f"[STEP] {step} {reward}", flush=True)
        step += 1

    # Pass real metrics to the real graders
    m = env.metrics
    episode_info = {
        "response_times":    list(m.get("response_times", [])),
        "optimal_times":     list(m.get("optimal_times", [])),
        "served":            int(m.get("served", 0)),
        "total_emergencies": int(m.get("total_emergencies", 0)),
        "avg_response_time": float(m.get("avg_response_time", 0.0)),
        "idle_steps":        int(m.get("idle_steps", 0)),
        "total_steps":       int(m.get("total_steps", step)),
        "critical_served":   int(m.get("critical_served", 0)),
        "critical_total":    int(m.get("critical_total", 0)),
        "priority_correct":  int(m.get("priority_correct", 0)),
        "priority_total":    int(m.get("priority_total", 0)),
        "capacity_violations": int(m.get("capacity_violations", 0)),
        "fairness_zone_counts": {
            "zone_served": dict(m.get("zone_served", {})),
            "zone_total":  dict(m.get("zone_total", {})),
        },
    }

    score = grader(episode_info)
    print(f"[END] {score}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Ambulance Dispatch inference")
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task to run (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task_map = {
        "easy":   (EasyConfig,   grade_easy),
        "medium": (MediumConfig, grade_medium),
        "hard":   (HardConfig,   grade_hard),
    }
    if args.task == "all":
        for name, (cfg, grader) in task_map.items():
            run_task(name, cfg, grader)
    else:
        cfg, grader = task_map[args.task]
        run_task(args.task, cfg, grader)

    # Auto-evaluation: compare baseline vs advanced agent on easy config
    set_seed(42)
    _env_cfg = EasyCfg()
    _baseline = BaselineAgent()
    _advanced = RepositioningOracle(enable_reposition=True)
    _advanced_env = AmbulanceEnvironment(_env_cfg.to_dict())
    _advanced_env.reset(seed=_env_cfg.seed)
    _advanced = _advanced.bind_env(_advanced_env)

    evaluator = AutoEvaluator(AmbulanceEnvironment, _baseline, _advanced)
    results = evaluator.evaluate(_env_cfg)
    generate_report(results)
