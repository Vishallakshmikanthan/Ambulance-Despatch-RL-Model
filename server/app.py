"""
FastAPI application entry point — served by uvicorn on port 7860.

Start locally:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

Environment variables:
    ENABLE_WEB_INTERFACE=true   Activate the built-in OpenEnv Gradio UI at /web
    HF_TOKEN                    HuggingFace API token
    API_BASE_URL                LLM API base URL
    MODEL_NAME                  LLM model identifier
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from openenv.core.env_server import create_app

from env.models import ActionModel, ObservationModel
from env.environment import AmbulanceEnvironment as _RawEnv
from server.ambulance_environment import AmbulanceEnvironment
from agents.greedy_agent import GreedyAgent
from tasks.easy import EasyConfig
from tasks.medium import MediumConfig
from tasks.hard import HardConfig

# ---------------------------------------------------------------------------
# Core OpenEnv application
# ---------------------------------------------------------------------------

# create_app accepts a *factory* — each WebSocket session gets a new instance,
# providing the session isolation required by SUPPORTS_CONCURRENT_SESSIONS=True.
app: FastAPI = create_app(
    env=AmbulanceEnvironment,
    action_cls=ActionModel,
    observation_cls=ObservationModel,
    env_name="ambulance-dispatch",
    max_concurrent_envs=10,
)

# ---------------------------------------------------------------------------
# CORS — allow the React dashboard (or any origin) to reach the API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Serve built React dashboard at /dashboard (built to frontend/dist/)
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if _STATIC_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_STATIC_DIR), html=True), name="dashboard")

# ---------------------------------------------------------------------------
# RFC 002 — Auto-discovery endpoint
# GET /tools  →  JSON schema of the action space
# ---------------------------------------------------------------------------
from fastapi.responses import JSONResponse

@app.get("/tools", tags=["RFC-002 Auto-Discovery"])
async def list_tools() -> JSONResponse:
    """Return the action-space JSON schema so generic agents can discover
    available actions without prior environment knowledge (RFC 002)."""
    schema = ActionModel.model_json_schema()
    return JSONResponse(
        content={
            "tools": [
                {
                    "name": "dispatch",
                    "description": (
                        "Dispatch an idle ambulance to an emergency and specify the "
                        "destination hospital.  Set is_noop=true to skip this step."
                    ),
                    "input_schema": schema,
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# RFC 003 — MCP Server exposure
# ---------------------------------------------------------------------------

@app.get("/mcp", tags=["RFC-003 MCP"])
async def mcp_server_info():
    """Returns MCP (Model Context Protocol) server metadata (RFC 003)."""
    return {
        "mcp_version": "0.1.0",
        "description": "Ambulance Dispatch RL Environment MCP Server",
        "capabilities": ["tools", "resources"],
        "endpoints": {
            "dispatch": "/env/step",
            "reset": "/env/reset",
            "state": "/env/state"
        }
    }

# ---------------------------------------------------------------------------
# Episode Replay & Trajectory Storage (Feature 12)
# Capped at 50 episodes to prevent unbounded memory growth.
# ---------------------------------------------------------------------------
_MAX_TRAJECTORIES = 50
trajectories = {}

@app.get("/episodes", tags=["Replay"])
async def list_episodes():
    return [{"id": k, "steps": len(v)} for k, v in trajectories.items()]

@app.get("/episodes/{episode_id}", tags=["Replay"])
async def get_episode(episode_id: str):
    return trajectories.get(episode_id, [])

@app.get("/", tags=["Health"])
async def root() -> dict:
    return {"status": "ok", "environment": "ambulance-dispatch", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "healthy", "environment": "ambulance-dispatch", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Dashboard: Stateful /env/* endpoints
# The frontend calls /env/reset and /env/step (proxied here by Next.js).
# These maintain a single persistent environment + greedy agent so the
# dashboard shows live dispatching behaviour across steps.
# ---------------------------------------------------------------------------

_TASK_CONFIGS = {
    "easy":   EasyConfig,
    "medium": MediumConfig,
    "hard":   HardConfig,
}

_dash_env: Optional[_RawEnv] = None
_dash_agent = GreedyAgent()        # fallback for manual actions
_dash_repo_oracle = None           # lazy-bound after reset
_dash_last_obs: Optional[ObservationModel] = None
_current_trajectory: list = []
_current_episode_id: str = ""


@app.post("/env/reset", tags=["Dashboard"])
async def dashboard_reset(body: dict = Body(default={})):
    global _dash_env, _dash_last_obs, _current_trajectory, _current_episode_id, _dash_repo_oracle
    task_name = (body.get("task_name") or "easy").lower()
    cfg = _TASK_CONFIGS.get(task_name, EasyConfig)().to_dict()
    _dash_env = _RawEnv(cfg)
    obs = _dash_env.reset()
    _dash_last_obs = obs
    _current_episode_id = _dash_env.episode_id
    _current_trajectory = []
    from agents.repositioning_oracle import RepositioningOracle
    enable_repos = task_name != "medium"
    _dash_repo_oracle = RepositioningOracle(enable_reposition=enable_repos).bind_env(_dash_env)
    return obs.model_dump()


@app.post("/env/step", tags=["Dashboard"])
async def dashboard_step(body: dict = Body(default={})):
    global _dash_env, _dash_last_obs, _current_trajectory
    if _dash_env is None:
        raise HTTPException(status_code=400, detail="Call /env/reset first")
    # Guard: do not advance a finished episode
    if _dash_last_obs is not None and getattr(_dash_last_obs, 'done', False):
        return _dash_last_obs.model_dump() if hasattr(_dash_last_obs, 'model_dump') else _dash_last_obs

    # Use request body action if provided, otherwise use RepositioningOracle
    action_body = body.get("action") if body else None
    if action_body and isinstance(action_body, dict) and not action_body.get("is_noop", True):
        try:
            action = ActionModel(**action_body)
            obs = _dash_env.step(action)
        except Exception:
            actions = _dash_repo_oracle.act_all_with_reposition(_dash_last_obs) if _dash_repo_oracle else [_dash_agent.act(_dash_last_obs or _dash_env._get_observation())]
            obs = _dash_env.step_all(actions) if _dash_repo_oracle else _dash_env.step(actions[0])
    elif _dash_repo_oracle and _dash_last_obs:
        actions = _dash_repo_oracle.act_all_with_reposition(_dash_last_obs)
        obs = _dash_env.step_all(actions)
    else:
        obs = _dash_env.step(_dash_agent.act(_dash_last_obs or _dash_env._get_observation()))

    _dash_last_obs = obs

    # Store step in trajectory
    _current_trajectory.append({
        "step": _dash_env.step_count,
        "obs": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    })

    # On episode end, persist trajectory (cap at _MAX_TRAJECTORIES)
    if obs.done and _current_episode_id:
        if len(trajectories) >= _MAX_TRAJECTORIES:
            oldest = next(iter(trajectories))
            del trajectories[oldest]
        trajectories[_current_episode_id] = list(_current_trajectory)

    return obs.model_dump()


@app.get("/env/metrics", tags=["Dashboard"])
async def dashboard_metrics():
    """Return current episode metrics for richer dashboard KPIs."""
    global _dash_env
    if _dash_env is None:
        return {"metrics": {}, "episode_id": None}
    return {"metrics": _dash_env.metrics, "episode_id": _dash_env.episode_id}


@app.get("/env/state", tags=["Dashboard"])
async def dashboard_state():
    global _dash_env
    if _dash_env is None:
        raise HTTPException(status_code=400, detail="Call /env/reset first")
    return _dash_env.state.model_dump()


# ---------------------------------------------------------------------------
# MARL endpoints — Multi-Agent coordination status
# ---------------------------------------------------------------------------

# Lazily initialised oversight agent (singleton for the server process)
_oversight_agent = None

def _get_oversight():
    global _oversight_agent
    if _oversight_agent is None:
        from agents.oversight_agent import OversightAgent
        _oversight_agent = OversightAgent(n_agents=5)
    return _oversight_agent


@app.get("/marl/status", tags=["MARL"])
async def marl_status():
    """Return current fleet coordination statistics from the OversightAgent."""
    return _get_oversight().get_status()


@app.get("/marl/conflicts", tags=["MARL"])
async def marl_conflicts(last_n: int = 20):
    """Return the last N conflict events detected by the OversightAgent."""
    return _get_oversight().get_conflict_history(last_n=last_n)


# ---------------------------------------------------------------------------
# Curriculum endpoints — Long-Horizon planning progress
# ---------------------------------------------------------------------------

_curriculum_manager = None

def _get_curriculum():
    global _curriculum_manager
    if _curriculum_manager is None:
        from long_horizon.curriculum_manager import CurriculumManager
        _curriculum_manager = CurriculumManager(initial_stage=1)
    return _curriculum_manager


@app.get("/curriculum/status", tags=["Curriculum"])
async def curriculum_status():
    """Return the current curriculum stage and episode progress."""
    return _get_curriculum().get_progress()


# ---------------------------------------------------------------------------
# Self-Improvement endpoints — weakness detection
# ---------------------------------------------------------------------------

_weakness_detector = None

def _get_weakness_detector():
    global _weakness_detector
    if _weakness_detector is None:
        from self_improvement.weakness_detector import WeaknessDetector
        _weakness_detector = WeaknessDetector()
    return _weakness_detector


@app.get("/selfplay/weaknesses", tags=["Self-Improvement"])
async def selfplay_weaknesses():
    """Return the latest weakness report (clusters of failing scenarios)."""
    wd = _get_weakness_detector()
    report = wd.get_latest()
    if report is None:
        return {"clusters": [], "iteration": 0, "message": "No weaknesses analyzed yet"}
    return report.to_dict()


@app.get("/selfplay/iterations", tags=["Self-Improvement"])
async def selfplay_iterations():
    """Return the improvement history across all self-play iterations."""
    wd = _get_weakness_detector()
    return wd.get_improvement_summary()


# ---------------------------------------------------------------------------
# Demo endpoint — compare trained vs baseline on a given scenario
# ---------------------------------------------------------------------------

@app.post("/demo/scenario", tags=["Demo"])
async def demo_scenario(body: dict = Body(default={})):
    """
    Run a scenario against the trained DQN agent and a greedy baseline.
    Returns both scores for comparison.

    Request body (all optional):
        n_ambulances (int), n_hospitals (int), max_steps (int), seed (int)
    """
    from agents.greedy_agent import GreedyAgent
    from rl.state_encoder import StateEncoder
    from rl.action_mapper import ActionMapper
    from rl.action_mask import ActionMask

    n_amb   = int(body.get("n_ambulances", 4))
    n_hosp  = int(body.get("n_hospitals", 3))
    steps   = int(body.get("max_steps", 80))
    seed    = int(body.get("seed", 0))

    cfg = {"n_ambulances": n_amb, "n_hospitals": n_hosp, "max_steps": steps, "seed": seed}

    def run_greedy(cfg: dict) -> float:
        env = _RawEnv(cfg)
        obs = env.reset(seed=seed)
        agent = GreedyAgent()
        for _ in range(steps):
            action = agent.act(obs)
            obs = env.step(action)
            if obs.done:
                break
        served = env.metrics.get("served", 0)
        missed = env.metrics.get("missed", 0)
        total = served + missed
        return round(served / total, 4) if total else 0.0

    def run_dqn(cfg: dict) -> float:
        import torch, pathlib
        from rl.rl_agent import DQNAgent
        model_path = pathlib.Path("outputs/marl/agent_0.pt")
        env = _RawEnv(cfg)
        obs = env.reset(seed=seed)
        encoder = StateEncoder()
        mapper = ActionMapper()
        mask_builder = ActionMask()
        state0 = encoder.encode(obs)
        mapper.build_action_space(obs)
        action_size = mapper.size()
        agent = DQNAgent(len(state0), action_size, use_dueling=True, use_per=False)
        if model_path.exists():
            agent.policy_net.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        agent.epsilon = 0.0  # evaluation mode
        for _ in range(steps):
            state = encoder.encode(obs)
            mapper.build_action_space(obs)
            mask = mask_builder.build_mask(mapper)
            idx = agent.act(state, mask)
            action = mapper.decode(idx)
            obs = env.step(action)
            if obs.done:
                break
        served = env.metrics.get("served", 0)
        missed = env.metrics.get("missed", 0)
        total = served + missed
        return round(served / total, 4) if total else 0.0

    greedy_score = run_greedy(cfg)
    dqn_score = run_dqn(cfg)

    return {
        "config": cfg,
        "greedy_score": greedy_score,
        "dqn_score": dqn_score,
        "improvement": round(dqn_score - greedy_score, 4),
    }


# ---------------------------------------------------------------------------
# RFC WebSocket live feed — pushes state at 2 Hz
# ---------------------------------------------------------------------------

import asyncio
from fastapi import WebSocket, WebSocketDisconnect


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket live feed — pushes environment state at 2 Hz (every 500ms).
    Clients receive JSON with: step, ambulances, emergencies, hospitals,
    traffic, reward, done, metrics.
    """
    await websocket.accept()
    try:
        while True:
            global _dash_env, _dash_last_obs
            if _dash_env is not None and _dash_last_obs is not None:
                payload = {
                    "step":        _dash_last_obs.step,
                    "reward":      _dash_last_obs.reward,
                    "done":        _dash_last_obs.done,
                    "ambulances":  [a.model_dump() for a in _dash_last_obs.ambulances],
                    "emergencies": [e.model_dump() for e in _dash_last_obs.emergencies],
                    "hospitals":   [h.model_dump() for h in _dash_last_obs.hospitals],
                    "traffic":     _dash_last_obs.traffic,
                    "metrics":     _dash_env.metrics if _dash_env else {},
                }
                await websocket.send_json(payload)
            await asyncio.sleep(0.5)  # 2 Hz
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# /score benchmark endpoint — runs all three tasks and returns live scores
# ---------------------------------------------------------------------------

@app.get("/score", tags=["Benchmark"])
async def benchmark_score():
    """
    Run a quick benchmark with RepositioningOracle agent and return scores.
    Warning: This is CPU-intensive and takes ~2-5 seconds.
    """
    import random as _random
    import numpy as _np
    from agents.repositioning_oracle import RepositioningOracle
    from tasks.easy import EasyConfig
    from tasks.medium import MediumConfig
    from tasks.hard import HardConfig
    from grader_easy import grade_easy
    from grader_medium import grade_medium
    from grader_hard import grade_hard

    results = {}
    tasks = [
        ("easy",   EasyConfig(),   grade_easy),
        ("medium", MediumConfig(), grade_medium),
        ("hard",   HardConfig(),   grade_hard),
    ]

    for task_name, cfg, grader in tasks:
        _random.seed(42)
        _np.random.seed(42)
        env = _RawEnv(cfg.to_dict())
        obs = env.reset(seed=cfg.seed)
        enable_repos = task_name != "medium"
        agent = RepositioningOracle(enable_reposition=enable_repos).bind_env(env)
        done = False
        step = 0
        while not done and step < cfg.max_steps:
            actions = agent.act_all_with_reposition(obs)
            obs = env.step_all(actions)   # ONE physics tick — correct multi-dispatch
            done = bool(obs.done)
            step += 1
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
        results[task_name] = {"score": score, "metrics": m}

    return {
        "scores": {k: v["score"] for k, v in results.items()},
        "agent": "RepositioningOracle",
        "seed": 42,
        "details": results,
    }


# ---------------------------------------------------------------------------
# Entry point — required by openenv validate & [project.scripts]
# ---------------------------------------------------------------------------

def main():
    """Start the uvicorn server — used by `ambulance-server` console script."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        workers=int(os.getenv("WORKERS", "4")),
    )


if __name__ == "__main__":
    main()

