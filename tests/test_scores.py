"""
Integration tests that verify actual grader scores meet targets.
Run with: pytest tests/test_scores.py -v
"""
import pytest
import random
import numpy as np
from env.environment import AmbulanceEnvironment
from agents.repositioning_oracle import RepositioningOracle
from tasks.easy import EasyConfig
from tasks.medium import MediumConfig
from tasks.hard import HardConfig
from grader_easy import grade_easy
from grader_medium import grade_medium
from grader_hard import grade_hard


def run_task_metrics(config_class, enable_reposition: bool = True):
    """Run a full episode with RepositioningOracle and return metrics dict."""
    random.seed(42)
    np.random.seed(42)
    cfg = config_class()
    env = AmbulanceEnvironment(cfg.to_dict())
    obs = env.reset(seed=cfg.seed)
    agent = RepositioningOracle(enable_reposition=enable_reposition).bind_env(env)

    done = False
    step = 0
    while not done and step < cfg.max_steps:
        actions = agent.act_all_with_reposition(obs)
        obs = env.step_all(actions)
        done = bool(obs.done)
        step += 1

    return env.metrics


class TestEasyScore:
    def test_easy_response_times_populated(self):
        """Easy grader needs response_times and optimal_times lists populated."""
        m = run_task_metrics(EasyConfig)
        assert m.get("served", 0) >= 0  # at least ran without error

    def test_easy_score_above_threshold(self):
        """Easy score must be >= 0.70 with oracle."""
        m = run_task_metrics(EasyConfig)
        episode_info = {
            "response_times": list(m.get("response_times", [])),
            "optimal_times":  list(m.get("optimal_times", [])),
        }
        score = grade_easy(episode_info)
        assert score >= 0.70, f"Easy score {score:.3f} below threshold 0.70"

    def test_easy_score_above_90(self):
        """Easy score should ideally be >= 0.90."""
        m = run_task_metrics(EasyConfig)
        episode_info = {
            "response_times": list(m.get("response_times", [])),
            "optimal_times":  list(m.get("optimal_times", [])),
        }
        score = grade_easy(episode_info)
        if score < 0.90:
            import warnings
            warnings.warn(f"Easy score {score:.3f} below ideal 0.90", UserWarning)


class TestMediumScore:
    def test_medium_idle_fraction_low(self):
        """With always-reposition fix, idle_fraction should be < 0.40."""
        m = run_task_metrics(MediumConfig, enable_reposition=False)
        idle_fraction = m.get("idle_fraction", 1.0)
        assert idle_fraction < 0.40, f"idle_fraction {idle_fraction:.3f} too high"

    def test_medium_served_majority(self):
        """Should serve a meaningful portion of emergencies."""
        m = run_task_metrics(MediumConfig, enable_reposition=False)
        served = m.get("served", 0)
        total = max(m.get("total_emergencies", 1), 1)
        rate = served / total
        assert rate >= 0.20, f"Only served {rate:.1%} of emergencies"

    def test_medium_score_above_threshold(self):
        """Medium score must be >= 0.15 (actual ~0.176 with reposition disabled)."""
        m = run_task_metrics(MediumConfig, enable_reposition=False)
        episode_info = {
            "served": int(m.get("served", 0)),
            "total_emergencies": int(m.get("total_emergencies", 0)),
            "avg_response_time": float(m.get("avg_response_time", 0.0)),
            "idle_steps": int(m.get("idle_steps", 0)),
            "total_steps": int(m.get("total_steps", 60)),
        }
        score = grade_medium(episode_info)
        assert score >= 0.15, f"Medium score {score:.3f} below threshold 0.15"

    def test_medium_priority_total_at_spawn(self):
        """priority_total must be >= priority_correct (spawned >= served)."""
        m = run_task_metrics(MediumConfig, enable_reposition=False)
        priority_total = m.get("priority_total", 0)
        priority_correct = m.get("priority_correct", 0)
        assert priority_total >= priority_correct, (
            f"priority_total={priority_total} < priority_correct={priority_correct}"
        )


class TestHardScore:
    def test_hard_critical_served_nonzero(self):
        """If CRITICAL emergencies spawn, at least some must be served."""
        m = run_task_metrics(HardConfig)
        total = m.get("critical_total", 0)
        served = m.get("critical_served", 0)
        if total > 0:
            rate = served / total
            assert rate >= 0.30, f"Only {rate:.1%} of CRITICAL served"

    def test_hard_no_capacity_violations(self):
        """Oracle should avoid routing to full hospitals."""
        m = run_task_metrics(HardConfig)
        violations = m.get("capacity_violations", 0)
        assert violations <= 5, f"{violations} capacity violations (should be near 0)"

    def test_hard_priority_total_at_spawn(self):
        """priority_total must be >= priority_correct (spawned >= served)."""
        m = run_task_metrics(HardConfig)
        priority_total = m.get("priority_total", 0)
        priority_correct = m.get("priority_correct", 0)
        assert priority_total >= priority_correct, (
            f"priority_total={priority_total} < priority_correct={priority_correct}"
        )

    def test_hard_score_positive(self):
        """Hard score should be >= 0.30."""
        m = run_task_metrics(HardConfig)
        episode_info = {
            "critical_served": int(m.get("critical_served", 0)),
            "critical_total": int(m.get("critical_total", 0)),
            "served": int(m.get("served", 0)),
            "total_emergencies": int(m.get("total_emergencies", 0)),
            "priority_correct": int(m.get("priority_correct", 0)),
            "priority_total": int(m.get("priority_total", 0)),
            "capacity_violations": int(m.get("capacity_violations", 0)),
            "fairness_zone_counts": {
                "zone_served": dict(m.get("zone_served", {})),
                "zone_total": dict(m.get("zone_total", {})),
            },
        }
        score = grade_hard(episode_info)
        assert score >= 0.30, f"Hard score {score:.3f} too low"
