"""
Quick analysis script — runs all tasks with all agents and prints a comparison table.
Run: python notebooks/run_analysis.py
"""
import random
import numpy as np
import sys
sys.path.insert(0, '.')

from env.environment import AmbulanceEnvironment
from agents.greedy_agent import GreedyAgent
from agents.baseline import BaselineAgent
from agents.oracle import OracleAgent
from agents.repositioning_oracle import RepositioningOracle
from tasks.easy import EasyConfig
from tasks.medium import MediumConfig
from tasks.hard import HardConfig
from grader_easy import grade_easy
from grader_medium import grade_medium
from grader_hard import grade_hard


def run_agent(agent_class, config_class, grader, seed=42, use_step_all=False,
              agent_kwargs=None):
    random.seed(seed)
    np.random.seed(seed)
    cfg = config_class()
    env = AmbulanceEnvironment(cfg.to_dict())
    obs = env.reset(seed=cfg.seed)

    kwargs = agent_kwargs or {}
    if hasattr(agent_class, 'bind_env'):
        agent = agent_class(**kwargs).bind_env(env)
    else:
        agent = agent_class(**kwargs)

    done = False
    step = 0
    while not done and step < cfg.max_steps:
        if use_step_all and hasattr(agent, 'act_all_with_reposition'):
            actions = agent.act_all_with_reposition(obs)
            obs = env.step_all(actions)
        else:
            action = agent.act(obs)
            obs = env.step(action)
        done = bool(obs.done)
        step += 1

    m = env.metrics
    episode_info = {
        "response_times": list(m.get("response_times", [])),
        "optimal_times":  list(m.get("optimal_times", [])),
        "served": int(m.get("served", 0)),
        "total_emergencies": int(m.get("total_emergencies", 0)),
        "avg_response_time": float(m.get("avg_response_time", 0.0)),
        "idle_steps": int(m.get("idle_steps", 0)),
        "total_steps": int(m.get("total_steps", step)),
        "critical_served": int(m.get("critical_served", 0)),
        "critical_total": int(m.get("critical_total", 0)),
        "priority_correct": int(m.get("priority_correct", 0)),
        "priority_total": int(m.get("priority_total", 0)),
        "capacity_violations": int(m.get("capacity_violations", 0)),
        "fairness_zone_counts": {
            "zone_served": dict(m.get("zone_served", {})),
            "zone_total":  dict(m.get("zone_total", {})),
        },
    }
    return grader(episode_info), m


def main():
    agents = [
        ("GreedyAgent",         GreedyAgent,         False),
        ("BaselineAgent",       BaselineAgent,        False),
        ("OracleAgent",         OracleAgent,          False),
        ("RepositioningOracle", RepositioningOracle,  True),
    ]
    tasks = [
        ("Easy",   EasyConfig,   grade_easy),
        ("Medium", MediumConfig, grade_medium),
        ("Hard",   HardConfig,   grade_hard),
    ]

    print("\n" + "=" * 70)
    print(" AMBULANCE DISPATCH — AGENT COMPARISON (seed=42)")
    print("=" * 70)
    print(f"{'Agent':<25} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Avg':>8}")
    print("-" * 70)

    for agent_name, agent_cls, use_all in agents:
        scores = []
        for task_name, task_cfg, grader in tasks:
            try:
                # Match inference.py: RepositioningOracle disables repos for medium
                agent_kwargs = {}
                if agent_cls is RepositioningOracle and task_name == "Medium":
                    agent_kwargs = {"enable_reposition": False}
                score, _ = run_agent(agent_cls, task_cfg, grader,
                                     use_step_all=use_all, agent_kwargs=agent_kwargs)
                scores.append(score)
            except Exception as e:
                print(f"  [{agent_name}/{task_name}] ERROR: {e}")
                scores.append(0.0)
        avg = sum(scores) / len(scores)
        print(f"{agent_name:<25} {scores[0]:>8.3f} {scores[1]:>8.3f} {scores[2]:>8.3f} {avg:>8.3f}")

    print("=" * 70)
    print("\nTo reproduce: python inference.py")
    print("Live scores:  GET http://localhost:7860/score\n")


if __name__ == "__main__":
    main()
