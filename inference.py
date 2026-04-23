"""
inference.py — Ambulance Dispatch RL Environment baseline inference script.

MANDATORY environment variables:
    API_BASE_URL    The API endpoint for the LLM (default provided).
    MODEL_NAME      The model identifier to use for inference (default provided).
    HF_TOKEN        Your Hugging Face / API key (NO default — must be set).

STDOUT FORMAT (strict — any deviation causes scoring failure):
    [START] task=<task_name> env=ambulance-dispatch model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import argparse
import json
import os
import sys
import logging
import re
import time
from typing import List, Optional

from openai import OpenAI

from env.environment import AmbulanceEnvironment
from env.models import ActionModel, ObservationModel
from tasks.easy import EasyConfig
from tasks.medium import MediumConfig
from tasks.hard import HardConfig
from grader_easy import grade_easy
from grader_medium import grade_medium
from grader_hard import grade_hard
from tasks.graders import grade_easy as history_grade_easy
from tasks.graders import grade_medium as history_grade_medium
from tasks.graders import grade_hard as history_grade_hard

_HISTORY_GRADERS = {
    "easy": history_grade_easy,
    "medium": history_grade_medium,
    "hard": history_grade_hard,
}

# ---------------------------------------------------------------------------
# Logging — all debug/info to stderr; stdout is ONLY [START]/[STEP]/[END]
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Environment variables — per spec checklist
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if using from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# Also accept OPENAI_API_KEY as a fallback alias for the key
_API_KEY = HF_TOKEN or os.getenv("OPENAI_API_KEY")

BENCHMARK = "ambulance-dispatch"
MAX_STEPS_TIMEOUT_S: float = 360.0  # per-task wall-clock timeout (3 × 360 = 18 min < 20 min)

# ---------------------------------------------------------------------------
# OpenAI client — must use OpenAI client for all LLM calls
# ---------------------------------------------------------------------------
_OPENAI_CLIENT: Optional[OpenAI] = None
if _API_KEY:
    try:
        _OPENAI_CLIENT = OpenAI(api_key=_API_KEY, base_url=API_BASE_URL)
        logging.info("OpenAI client initialised: base_url=%s model=%s", API_BASE_URL, MODEL_NAME)
    except Exception as exc:
        logging.warning("OpenAI client init failed: %s — using greedy fallback.", exc)
else:
    logging.warning("HF_TOKEN not set — using greedy agent fallback.")


# ---------------------------------------------------------------------------
# Structured stdout logging — EXACT format from sample inference script
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Greedy fallback agent (used when LLM is unavailable)
# ---------------------------------------------------------------------------

def _greedy_action(obs: ObservationModel) -> ActionModel:
    """Pick nearest idle ambulance → highest-priority unassigned emergency → available hospital."""
    idle = [a for a in obs.ambulances if a.state == "idle"]
    unassigned = [e for e in obs.emergencies if not e.assigned]
    if not idle or not unassigned:
        return ActionModel(is_noop=True)
    _priority = {"CRITICAL": 0, "HIGH": 1, "NORMAL": 2}
    emg = sorted(unassigned, key=lambda e: (_priority.get(e.severity, 9), e.time_remaining))[0]
    amb = min(idle, key=lambda a: abs(a.node - emg.node))
    available_hosps = [h for h in obs.hospitals if h.current_patients < h.capacity]
    if not available_hosps:
        available_hosps = obs.hospitals
    hosp = min(available_hosps, key=lambda h: abs(h.node - emg.node))
    return ActionModel(ambulance_id=amb.id, emergency_id=emg.id, hospital_id=hosp.id)


def _llm_action(obs: ObservationModel) -> ActionModel:
    """Ask the LLM to pick a dispatch action. Falls back to greedy on failure."""
    if _OPENAI_CLIENT is None:
        return _greedy_action(obs)

    obs_json = json.dumps(obs.model_dump(), default=str)[:8000]
    system_content = (
        "You are an expert ambulance dispatch system. "
        "Given the current environment observation as JSON, respond with ONLY a JSON object "
        "containing the dispatch action. Fields: ambulance_id (int or null), "
        "emergency_id (string), hospital_id (int or null), is_noop (bool). "
        "Pick an idle ambulance (state='idle') and send it to the most critical unassigned "
        "emergency (assigned=false), choosing the nearest available hospital. "
        "If no idle ambulances or no unassigned emergencies, set is_noop=true."
    )
    try:
        resp = _OPENAI_CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": obs_json},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        parsed = json.loads(raw)
        action = ActionModel(
            ambulance_id=parsed.get("ambulance_id"),
            emergency_id=str(parsed.get("emergency_id", "")),
            hospital_id=parsed.get("hospital_id"),
            is_noop=bool(parsed.get("is_noop", False)),
        )
        if action.ambulance_id is not None:
            if not any(a.id == action.ambulance_id for a in obs.ambulances):
                return _greedy_action(obs)
        return action
    except Exception as exc:
        logging.warning("LLM action failed: %s — using greedy.", exc)
        return _greedy_action(obs)


# ---------------------------------------------------------------------------
# Action → compact string for [STEP] log line
# ---------------------------------------------------------------------------

def _action_str(action: ActionModel) -> str:
    """Format action as compact string for log line."""
    if action.is_noop:
        return "noop()"
    return f"dispatch(amb={action.ambulance_id},emg='{action.emergency_id}',hosp={action.hospital_id})"


# ---------------------------------------------------------------------------
# Main task runner
# ---------------------------------------------------------------------------

def run_task(task_name: str, config_class, grader_func) -> None:
    config = config_class()
    env = AmbulanceEnvironment(config.to_dict())

    rewards: List[float] = []
    history: List[dict] = []
    steps_taken = 0
    score = 0.0
    success = False

    logging.info("[START] %s", task_name)
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(seed=config.seed)
        task_start = time.monotonic()

        for step in range(1, config.max_steps + 1):
            if obs.done:
                break

            # Wall-clock timeout
            if time.monotonic() - task_start > MAX_STEPS_TIMEOUT_S:
                logging.warning("Task %s timed out after %.1fs", task_name, MAX_STEPS_TIMEOUT_S)
                break

            action = _llm_action(obs) if _OPENAI_CLIENT else _greedy_action(obs)
            obs = env.step(action)

            reward = float(obs.reward or 0.0)
            done = bool(obs.done)
            error = None

            rewards.append(reward)
            history.append({"reward": reward, "info": {"step": step, "done": done}})
            steps_taken = step

            logging.info("[STEP] %d %s", step, f"{reward:.2f}")
            log_step(step=step, action=_action_str(action), reward=reward, done=done, error=error)

            if done:
                break

        # Build episode_info for grader
        metrics = env.metrics
        episode_info = {
            "response_times": metrics.get("response_times", []),
            "optimal_times": metrics.get("optimal_times", []),
            "total_emergencies": metrics.get("total_emergencies", 0),
            "served": metrics.get("served", 0),
            "avg_response_time": metrics.get("avg_response_time", 0.0),
            "idle_steps": metrics.get("idle_steps", 0),
            "total_steps": steps_taken,
            "critical_total": metrics.get("critical_total", 0),
            "critical_served": metrics.get("critical_served", 0),
            "priority_correct": metrics.get("priority_correct", 0),
            "priority_total": max(metrics.get("priority_total", 0), 1),
            "capacity_violations": metrics.get("capacity_violations", 0),
            "fairness_zone_counts": {
                "zone_served": metrics.get("zone_served", {}),
                "zone_total": metrics.get("zone_total", {}),
            },
            "metrics": metrics,
        }

        score = grader_func(episode_info)
        score = float(max(0.0, min(1.0, score)))
        success = score > 0.0

        # History-based score using tasks/graders.py
        history_grader = _HISTORY_GRADERS.get(task_name)
        if history_grader is not None and history:
            history_score = history_grader(history)
            logging.info("[END] history_score=%.4f", history_score)
            print(f"[END] SCORE: {history_score:.4f}", flush=True)

    except Exception as exc:
        logging.critical("Task %s crashed: %s", task_name, exc, exc_info=True)
        # Must always emit [END] even on exception
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


_TASK_REGISTRY = {
    "easy":   (EasyConfig,   grade_easy),
    "medium": (MediumConfig, grade_medium),
    "hard":   (HardConfig,   grade_hard),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ambulance-dispatch inference.")
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task(s) to evaluate (default: all)",
    )
    args = parser.parse_args()

    tasks_to_run = (
        list(_TASK_REGISTRY.items())
        if args.task == "all"
        else [(args.task, _TASK_REGISTRY[args.task])]
    )
    for name, (config_cls, grader) in tasks_to_run:
        run_task(name, config_cls, grader)
