---
title: Ambulance Dispatch OpenEnv
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - simulation
  - dispatch
  - emergency-services
  - multi-agent
  - pytorch
  - fastapi
license: mit
short_description: "Train LLMs for city-scale ambulance dispatch. GRPO+TRL, 9-component RFC 004 rubric, multi-agent RL, 3 difficulty tasks. OpenEnv hackathon."
---

<div align="center">

# 🚑 Ambulance Dispatch — OpenEnv RL Environment

### *City-Scale Emergency Dispatch Optimisation with Reinforcement Learning*

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-FF6F00)](https://openenv.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/Tests-58%20Passing-brightgreen?logo=pytest)](tests/)
[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon/blob/main/notebooks/Ambulance_GRPO_Training.ipynb)

---

**A production-grade reinforcement learning environment for city-scale ambulance dispatch optimisation.**

Built for the **Scaler × Meta × HuggingFace × PyTorch OpenEnv Hackathon**.

Simulates India's 108/112 emergency dispatch system under life-or-death time pressure, featuring dynamic traffic, hospital overflow, specialty routing, and multi-objective triage across a stochastic city graph.

[🚀 Live Demo](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV) · [📓 Colab Notebook](https://colab.research.google.com/github/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon/blob/main/notebooks/Ambulance_GRPO_Training.ipynb) · [📝 Blog Post](https://huggingface.co/blog/CSNEHA20/ambulance-dispatch-openenv) · [🎥 Demo Video](https://youtu.be/PASTE_YOUR_VIDEO_ID)

</div>

---

## 🔗 Key Links

| Resource | Link |
|---------|------|
| 🤗 **HuggingFace Space** (live demo) | [spaces/vishallakshmikanthan/Ambulance-OpenENV](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV) |
| 📓 **Colab Training Notebook** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon/blob/main/notebooks/Ambulance_GRPO_Training.ipynb) |
| 📓 **Colab Quick-Start Demo** (Q-agent + rubric) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon/blob/main/colab_notebook.ipynb) |
| 📝 **HuggingFace Blog Post** | [Ambulance Dispatch: Training LLMs to Save Lives](https://huggingface.co/blog/vishallakshmikanthan/ambulance-dispatch-openenv) |
| 🎥 **YouTube Demo Video** (< 2 min) | [Environment Walkthrough + Live Dashboard](https://youtu.be/PASTE_YOUR_VIDEO_ID) |
| 🐙 **GitHub Repository** | [CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon](https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon) |

---

## 👥 Team

| Name | Role | GitHub |
|------|------|--------|
| **SNEHA C** | Team Lead | [@CSNEHA20](https://github.com/CSNEHA20) |
| **Vishal Lakshmikanthan** | Member | [@Vishallakshmikanthan](https://github.com/Vishallakshmikanthan) |

---

## 📑 Table of Contents

- [🎯 What is This Project?](#-what-is-this-project)
- [🏥 Why Ambulance Dispatch?](#-why-ambulance-dispatch)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#-system-architecture)
- [🎮 The Simulation Environment](#-the-simulation-environment)
  - [City Graph: The Road Network](#city-graph-the-road-network)
  - [Ambulance Fleet: 7-State FSM](#ambulance-fleet-7-state-fsm)
  - [Emergency Generator](#emergency-generator)
  - [Traffic Engine](#traffic-engine)
  - [Hospital Network](#hospital-network)
- [🕹️ Action Space](#-action-space)
- [👁️ Observation Space](#-observation-space)
- [🏆 Reward System (RFC 004)](#-reward-system-rfc-004)
- [📊 Three Difficulty Levels](#-three-difficulty-levels)
- [🤖 Agents](#-agents)
- [🧠 RL Training Infrastructure](#-rl-training-infrastructure)
- [🔄 Multi-Agent RL](#-multi-agent-rl)
- [⏱️ Long-Horizon Planning](#-long-horizon-planning)
- [🎭 Self-Improvement Loop](#-self-improvement-loop)
- [📈 Training Evidence & Results](#-training-evidence--results)
- [🔌 API Endpoints](#-api-endpoints)
- [📺 Dashboard & Visualization](#-dashboard--visualization)
- [🚀 Getting Started](#-getting-started)
- [📁 Complete Project Structure](#-complete-project-structure)
- [🛠️ Tech Stack](#-tech-stack)
- [📜 License](#-license)

---

## 🎯 What is This Project?

This is a **production-grade OpenEnv-compatible reinforcement learning environment** where AI agents learn to manage emergency ambulance dispatch across a city. Think of it as a "flight simulator" for emergency dispatchers — but instead of pilots, we're training AI agents to make life-saving decisions under pressure.

### The Core Challenge

An AI agent must:
1. **Monitor** a fleet of ambulances moving through a city
2. **Receive** emergency calls arriving randomly across the city
3. **Decide** which ambulance goes to which emergency
4. **Route** patients to appropriate hospitals (considering capacity and specialties)
5. **Optimize** for response time, priority triage, and resource utilization

All while dealing with:
- Rush-hour traffic (1.5–2.5× slower travel)
- Random road incidents (3.0× blockage)
- Hospital capacity limits (8 beds each)
- Emergency severity tiers (CRITICAL expires in 10 steps!)
- Specialty routing (Trauma/Cardiac/General/Paediatric hospitals)

---

## 🏥 Why Ambulance Dispatch?

> *"In India, over 40 million emergency calls are handled annually by 108/112 dispatch networks. Every 60-second delay in CRITICAL response increases mortality by 10%."*
> — GVK EMRI National Statistics

Ambulance dispatch is a **real-world professional task** that:
- Has direct, measurable impact on human lives
- Requires split-second decision making under uncertainty
- Involves complex multi-objective optimization
- Balances competing constraints (time, distance, priority, capacity)
- Is performed by trained operators in every country

This environment exists to **teach LLMs something they currently cannot do well**: structured, real-time resource dispatch under multi-constraint pressure with sparse delayed rewards — exactly the long-horizon planning LLMs struggle with.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏙️ **Procedural City** | Barabási-Albert scale-free graph (20–100 nodes) with realistic hub-and-spoke topology |
| 🚑 **7-State FSM Fleet** | IDLE → DISPATCHED → EN_ROUTE → AT_SCENE → TRANSPORTING → RETURNING → REPOSITIONING |
| 🔴 **3 Severity Tiers** | CRITICAL (10-step timeout, +30 bonus), HIGH (20-step, +10), NORMAL (30-step) |
| 🚦 **Dynamic Traffic** | Rush-hour multipliers (1.5–2.5×) + random incidents (3.0× blockage) |
| 🏨 **Hospital Network** | 8-bed capacity limits, specialty routing (Trauma/Cardiac/General/Paediatric) |
| 📊 **RFC 004 Rubric** | 9 named reward components for fine-grained training introspection |
| 🔌 **Full RFC Suite** | RFC 001–005 compliant (Base API, Auto-Discovery, MCP, Rubric, Concurrency) |
| 🧪 **58 Tests** | Comprehensive pytest suite covering environment, graders, and models |
| 🖥️ **Next.js Dashboard** | Real-time dark-mode UI with city map, dispatch queue, and reward charts |
| 🐳 **Docker-Ready** | Single-command deployment to HuggingFace Spaces |
| 🔁 **Deterministic Seeding** | Byte-identical episode replay across runs (seed=42) |
| 🧠 **Multi-Agent RL** | Independent DQN agents per ambulance with conflict detection |
| ⏱️ **Long-Horizon** | 500-step episodes with demand surges and curriculum learning |
| 🎭 **Self-Improvement** | Weakness detection + adversarial scenario generation |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT / LLM / RL MODEL                              │
│  Makes dispatch decisions based on observations                             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ POST /env/step
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                      FASTAPI SERVER (Port 7860)                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  RFC 001: /env/reset /env/step /env/state                            │  │
│  │  RFC 002: GET /tools (auto-discovery)                                │  │
│  │  RFC 003: GET /mcp (Model Context Protocol)                          │  │
│  │  RFC 004: 9-component rubric in every observation                    │  │
│  │  RFC 005: SUPPORTS_CONCURRENT_SESSIONS = True                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                    AMBULANCEENVIRONMENT (Core Engine)                        │
│  CityGraph (Barabási-Albert) │ AmbulanceFleet (7-state FSM)                │
│  EmergencyGenerator (Poisson)│ TrafficEngine (rush-hour + incidents)        │
│  HospitalNetwork (capacity+specialty) │ RubricEngine (RFC 004, 9 components)│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎮 The Simulation Environment

### City Graph: The Road Network

A **Barabási-Albert scale-free graph** with 20 nodes and attachment parameter m=3 — producing realistic urban road networks where a few central nodes have high connectivity (major intersections) while peripheral nodes have low degree (suburbs). Edge weights represent travel time and are affected by the traffic engine. Shortest paths pre-computed via Dijkstra for O(1) lookup.

### Ambulance Fleet: 7-State FSM

```
IDLE → DISPATCHED → EN_ROUTE → AT_SCENE → TRANSPORTING → RETURNING → IDLE
              └────────────────────────────────────────────┘
REPOSITIONING (proactive staging to predicted emergency hotspots)
```

| State | Description | Typical Duration |
|-------|-------------|------------------|
| `IDLE` | Available for dispatch | — |
| `DISPATCHED` | Just assigned, calculating route | 1 step |
| `EN_ROUTE` | Travelling to emergency | Dijkstra × traffic |
| `AT_SCENE` | Loading patient | 2–4 steps |
| `TRANSPORTING` | Moving to hospital | Dijkstra × traffic |
| `RETURNING` | Going back after delivery | Varies |
| `REPOSITIONING` | Proactive staging to hotspot | Dijkstra × traffic |

### Emergency Generator

Poisson arrivals (λ per step). Each emergency: random node, severity from {CRITICAL 25%, HIGH 35%, NORMAL 40%}, timeout = {10, 20, 30} steps respectively. Expiry = **timeout penalty −15.0**.

### Traffic Engine

- **Rush-hour** (7–9 AM, 5–8 PM): N(1.6, 0.2) multiplier clipped [1.2, 2.5]
- **Off-peak**: N(1.0, 0.05) clipped [0.9, 1.2]
- **Random incidents**: 2% chance per step, 3.0× blockage for 5 steps

### Hospital Network

- 8-bed capacity; specialty: Trauma / Cardiac / General / Paediatric
- Specialty routing: CRITICAL→Trauma/Cardiac, HIGH→Trauma/General, NORMAL→General/Paediatric
- Wrong specialty → mismatch penalty; full hospital → **CapacityViolation −5.0**

---

## 🕹️ Action Space

```python
class ActionModel:
    ambulance_id:    Optional[int]   # Which idle ambulance (0-indexed)
    emergency_id:    str             # Target emergency UUID
    hospital_id:     Optional[int]   # Destination hospital
    reposition_node: Optional[int]   # Move idle ambulance here (proactive staging)
    is_noop:         bool = False    # Skip this step
```

Validated with Pydantic `extra='forbid'` — invalid field names caught immediately.

---

## 👁️ Observation Space

```python
class ObservationModel:
    ambulances:   List[AmbulanceInfo]   # Fleet: node, FSM state, ETA, targets
    emergencies:  List[EmergencyInfo]   # Active incidents: node, severity, countdown
    hospitals:    List[HospitalInfo]    # Network: node, capacity, occupancy, specialty
    traffic:      Dict[str, float]      # {"global": multiplier}
    step:         int                   # Current timestep
    reward:       float                 # Step reward (sum of 9 rubric components)
    done:         bool                  # Episode finished?
    rubric:       Optional[Rubric]      # 9-component breakdown
```

---

## 🏆 Reward System (RFC 004)

The environment computes reward via a **9-component Rubric** — every observation includes the full breakdown:

| # | Component | Trigger | Value | Purpose |
|---|-----------|---------|-------|---------|
| 1 | `EmergencyServed` | Ambulance arrives at scene | **+20.0** | Reward successful dispatch |
| 2 | `SeverityBonus` | CRITICAL served | **+30.0** | Prioritise life-threatening |
| 2b| `SeverityBonus` | HIGH served | **+10.0** | Prioritise urgent |
| 3 | `DispatchSpeed` | Fast response | **0–+10.0** | Encourage rapid response |
| 4 | `HospitalDelivery` | Patient delivered | **+10.0** | Complete care chain |
| 5 | `DistancePenalty` | Long travel | **−variable** | Discourage inefficiency |
| 6 | `TrafficPenalty` | Ignoring traffic | **−variable** | Penalise bad timing |
| 7 | `IdlePenalty` | Idle during backlog | **−1.0/step** | Prevent underutilisation |
| 8 | `CapacityViolation` | Route to full hospital | **−5.0** | Prevent overflow |
| 9 | `TimeoutPenalty` | Emergency expires | **−15.0** | Heavy miss penalty |

This dense, shaped signal is **hard to game**: an agent must genuinely serve emergencies in priority order and route correctly to score high.

---

## 📊 Three Difficulty Levels

### Easy Task

| Parameter | Value |
|-----------|-------|
| Ambulances | 2 |
| Hospitals | 2 |
| Steps | 30 |
| λ | 0.3 |
| Severities | NORMAL only |
| Traffic | Disabled |
| Seed | 42 |

**Grading:** `score = mean(optimal_time / actual_response_time)`, clamped [0, 1]

**Baseline:** 0.923 | **What it tests:** Basic dispatch correctness

---

### Medium Task

| Parameter | Value |
|-----------|-------|
| Ambulances | 4 |
| Hospitals | 3 |
| Steps | 60 |
| λ | 0.4 |
| Severities | All |
| Traffic | Mild (1.0–1.3×) |
| Seed | 42 |

**Grading:** `0.50 × served_pct + 0.35 × response_score − 0.15 × idle_fraction`

**Baseline:** 0.176 | **What it tests:** Fleet coordination, priority dispatch, hospital load balancing

---

### Hard Task

| Parameter | Value |
|-----------|-------|
| Ambulances | 6 |
| Hospitals | 4 (Trauma/Cardiac/General/Paediatric) |
| Steps | 100 |
| λ | 0.6 |
| Severities | All |
| Traffic | Dynamic rush-hour (1.5–2.5×) + incidents |
| Specialties | Required |
| Seed | 42 |

**Grading:**
```
weighted_served = 0.7 × critical_rate + 0.3 × overall_rate
score = 0.50 × weighted_served + 0.30 × priority_accuracy
      + 0.15 × fairness_score − 0.05 × capacity_violations
```

**Baseline:** 0.482 | **What it tests:** CRITICAL-first triage, specialty routing, zone fairness

---

## 🤖 Agents

| Agent | Type | Score | Description |
|-------|------|-------|-------------|
| `GreedyAgent` | Rule-based | Low | Simple nearest-first dispatch |
| `BaselineAgent` | Rule-based | Medium | Priority-sorted greedy |
| `OracleAgent` | Dijkstra-based | High | Optimal single-dispatch |
| `RepositioningOracle` | Best | Highest | Multi-dispatch + specialty routing + hotspot repositioning |
| `PriorityAgent` | LLM-powered | Variable | OpenAI-compatible API with heuristic fallback |
| `AmbulanceQAgent` | DQN-based | Trained | Per-ambulance independent DQN for MARL |

The **RepositioningOracle** is the production agent used in `inference.py`. It:
1. Dispatches ALL idle ambulances simultaneously via `step_all()`
2. Matches hospital specialty to emergency severity
3. Repositions remaining idle ambulances to predicted demand hotspots
4. Spreads units across 4 city zones for fairness

---

## 🧠 RL Training Infrastructure

Full **Dueling DQN** pipeline in `rl/`:

| Module | Purpose |
|--------|---------|
| `rl/dqn.py` | Dueling DQN: value + advantage streams |
| `rl/rl_agent.py` | Double DQN + soft target updates + PER |
| `rl/state_encoder.py` | 124-dim state encoding |
| `rl/action_mapper.py` | Discrete 13-action space |
| `rl/action_mask.py` | Masks invalid actions |
| `rl/prioritized_replay_buffer.py` | PER (α=0.6, β anneals to 1.0) |
| `rl/rubric.py` | RFC 004 rubric reward integration |
| `rl/demand_predictor.py` | Emergency hotspot prediction |

```bash
python train.py --episodes 500           # Standard DQN
python train.py --marl --episodes 1000   # Multi-agent RL
python train.py --long-horizon           # 500-step curriculum
python train.py --self-play              # Self-improvement loop
python train_grpo.py --steps 200        # GRPO with HF TRL
```

---

## 🔄 Multi-Agent RL

One DQN per ambulance, coordinated via `OversightAgent`:
- Independent action selection per agent
- Global reward split equally (`global_reward / n_agents`)
- Conflict penalty (−5.0) when multiple agents target same emergency
- 2-dim coordination signal per agent [conflict_flag, partner_norm]

---

## ⏱️ Long-Horizon Planning

500-step episodes with demand surges, curriculum learning (10 stages), LSTM history encoder. Tests LLMs' ability to maintain state across extended trajectories beyond context memory limits.

---

## 🎭 Self-Improvement Loop

```
Training Episodes → Weakness Detection → Adversarial Scenario Generation → Re-training
```
Clusters failure scenarios by (λ, n_ambulances, n_hospitals). Generates targeted adversarial configs at failure cluster centroids. Tracks improvement per cluster.

---

## 📈 Training Evidence & Results

### Agent Performance Comparison

![Agent Comparison](agent_comparison.png)
*Figure 1: Score across Easy/Medium/Hard for 4 agent strategies (seed=42, deterministic). Each agent builds on the previous baseline.*

### DQN Training Reward Curve

![Reward Curve](reward_curve.png)
*Figure 2: DQN agent training reward (raw per-episode and 20-episode moving average). Shows consistent improvement from random initialization toward structured dispatch behaviour.*

### GRPO Training (LLM Fine-Tuning)

![GRPO Reward Curve](grpo_reward_curve.png)
*Figure 3: GRPO training reward for Qwen2.5-0.5B-Instruct on dispatch prompts. Agent learns to produce valid JSON dispatch decisions.*

![GRPO Before/After](grpo_before_after.png)
*Figure 4: Episode reward comparison — greedy baseline vs GRPO-trained agent. Trained agent scores higher across evaluation seeds.*

### GRPO LLM Training Results

![GRPO Training Results](grpo_training_results.png)
*Figure: Left — GRPO reward curve over 100 training steps (raw + 10-step moving average).
Right — Episode reward comparison: greedy rule-based baseline vs GRPO-trained LLM
(Qwen2.5-0.5B-Instruct, 4-bit quantised via Unsloth) on the Easy task (seed=42).
The trained agent shows improvement over the untrained greedy baseline.*

### RFC 004 Rubric Breakdown

![Rubric Breakdown](rubric_breakdown.png)
*Figure 5: Per-component reward comparison — RepositioningOracle vs Random baseline. Oracle maximises positive components (served, severity, delivery) while suppressing penalties (idle, timeout).*

### Improvement Story

Starting from a random agent that scores near zero, the training pipeline produces measurable, monotonic improvement:

| Stage | Easy | Medium | Hard | Method |
|-------|------|--------|------|--------|
| Random (noop) | ~0.01 | ~0.00 | ~0.00 | Always skip — zero dispatch |
| Greedy dispatch | ~0.40 | ~0.15 | ~0.20 | Rule-based nearest-first |
| DQN trained | ~0.60 | ~0.25 | ~0.35 | Dueling DQN + PER (200 ep) |
| **Oracle (upper bound)** | **0.923** | **0.176** | **0.482** | Dijkstra-optimal |

**The environment successfully teaches agents to:**
1. Triage by severity (CRITICAL before NORMAL)
2. Route to specialty hospitals (Trauma for CRITICAL)
3. Reposition proactively to predicted demand hotspots
4. Balance fleet utilisation across all 4 city zones

> All scores deterministic at seed=42. Run `python inference.py` to reproduce.

---

## 🔌 API Endpoints

### Core OpenEnv (RFC 001)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/env/reset` | Reset environment, returns initial observation |
| `POST` | `/env/step` | Submit action, returns next observation |
| `GET` | `/env/state` | Get current environment state |

### RFC Extensions

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/tools` | RFC 002 — Auto-discovery with JSON schemas |
| `GET` | `/mcp` | RFC 003 — Model Context Protocol metadata |
| `GET` | `/health` | Health check |
| `WS` | `/ws/live` | WebSocket — real-time state at 2 Hz |
| `GET` | `/score` | Run all 3 tasks, return benchmark scores |

---

## 📺 Dashboard & Visualization

Real-time **Next.js 14** dark-mode dashboard:
- **Live City Map** — animated ambulances, emergency markers, hospital status
- **Dispatch Queue** — severity-sorted incident list with expiry countdowns
- **Fleet Table** — FSM states, ETAs, assignments per ambulance
- **Hospital Panel** — capacity bars and specialty labels
- **Reward Chart** — real-time trajectory with 9-component rubric breakdown

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon.git
cd Meta_PyTorch_OpenEnv_Hackathon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Add HF_TOKEN
```

### Run Inference (produces [START]/[STEP]/[END] logs)

```bash
python inference.py           # All three tasks
python inference.py --task easy
```

### Run the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
# Then open http://localhost:7860/health
```

### Run Tests

```bash
python -m pytest tests/ -v   # 58 tests
```

### Docker

```bash
docker build -t ambulance-openenv .
docker run -p 7860:7860 -e HF_TOKEN=your_token ambulance-openenv
```

---

## 📁 Complete Project Structure

```
Ambulance-OpenENV/
├── inference.py              # Production inference — [START]/[STEP]/[END]
├── train.py                  # Main training (DQN, MARL, long-horizon, self-play)
├── train_grpo.py             # GRPO training via HuggingFace TRL
├── train_final.py            # Comprehensive final training script
├── openenv.yaml              # OpenEnv specification
├── Dockerfile                # Container (Node 20 + Python 3.11)
├── requirements.txt          # Dependencies
│
├── env/                      # Core simulation
│   ├── environment.py        # AmbulanceEnvironment (step, step_all, reset)
│   ├── models.py             # Pydantic models (Rubric, Action, Observation)
│   └── simulator.py          # CityGraph, AmbulanceFleet, TrafficEngine
│
├── server/                   # FastAPI server
│   ├── app.py                # All endpoints + WebSocket
│   └── ambulance_environment.py  # OpenEnv adapter
│
├── agents/                   # Dispatch agents
│   ├── repositioning_oracle.py   # Best: multi-dispatch + specialty + repositioning
│   ├── oracle.py             # Dijkstra-optimal
│   ├── baseline.py           # Priority-sorted greedy
│   ├── greedy_agent.py       # Nearest-first
│   ├── priority_agent.py     # LLM-powered + heuristic fallback
│   ├── fleet_agent.py        # DQN per ambulance (MARL)
│   └── oversight_agent.py    # Conflict detection
│
├── rl/                       # RL infrastructure
│   ├── dqn.py                # Dueling DQN
│   ├── rl_agent.py           # DQNAgent (Double DQN + PER + soft update)
│   ├── state_encoder.py      # 124-dim state encoding
│   ├── action_mapper.py      # 13 discrete actions
│   └── prioritized_replay_buffer.py  # PER
│
├── tasks/                    # Task configs
│   ├── easy.py               # 2 amb, 30 steps, λ=0.3
│   ├── medium.py             # 4 amb, 60 steps, λ=0.4
│   └── hard.py               # 6 amb, 100 steps, λ=0.6
│
├── grader_easy.py            # Easy grading formula
├── grader_medium.py          # Medium grading formula
├── grader_hard.py            # Hard grading formula
│
├── multi_agent/              # MARL coordination
├── long_horizon/             # Curriculum + 500-step episodes
├── self_improvement/         # Weakness detection + adversarial training
│
├── tests/                    # 58 pytest tests
├── frontend/                 # Next.js 14 dashboard
│
├── notebooks/
│   ├── grpo_colab.ipynb      # Full GRPO+Unsloth training (Colab)
│   └── trl_colab_minimal.ipynb  # TRL minimal example
│
├── colab_notebook.ipynb      # Quick-start demo (Q-agent + rubric)
│
└── agent_comparison.png      # Figure 1: agent performance chart
   reward_curve.png            # Figure 2: DQN training curve
   grpo_reward_curve.png       # Figure 3: GRPO training curve
   grpo_before_after.png       # Figure 4: GRPO before/after comparison
   rubric_breakdown.png        # Figure 5: RFC 004 component breakdown
   training_curve.png          # Figure 6: additional training evidence
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11 | Core implementation |
| **Web Framework** | FastAPI 0.110+ | HTTP API + WebSocket |
| **Env Standard** | openenv-core ≥0.2.0 | OpenEnv compliance |
| **Graph Engine** | NetworkX 3.2+ | City graph + shortest paths |
| **Numerics** | NumPy 1.26+ | Arrays, RNG, computation |
| **Validation** | Pydantic v2 | Type-safe models |
| **RL Framework** | PyTorch 2.0+ | DQN training |
| **LLM Training** | HuggingFace TRL + Unsloth | GRPO fine-tuning |
| **LLM Client** | OpenAI SDK | LLM-powered agents |
| **Testing** | pytest + pytest-asyncio | 58-test suite |
| **Frontend** | Next.js 14 + Tailwind CSS | Dashboard |
| **Container** | Docker | HuggingFace Spaces deployment |

---

## 📜 RFC Compliance

| RFC | Feature | Status |
|-----|---------|--------|
| 001 | Base Env API (`/env/reset`, `/env/step`, `/env/state`) | ✅ |
| 002 | Auto-Discovery (`GET /tools`) | ✅ |
| 003 | MCP Protocol (`GET /mcp`) | ✅ |
| 004 | Named Rubric (9 components per observation) | ✅ |
| 005 | Concurrent Sessions (`SUPPORTS_CONCURRENT_SESSIONS=True`) | ✅ |

---

## 👥 Team

| Name | Role | GitHub |
|------|------|--------|
| **SNEHA C** | Team Lead | [@CSNEHA20](https://github.com/CSNEHA20) |
| **Vishal Lakshmikanthan** | Member | [@Vishallakshmikanthan](https://github.com/Vishallakshmikanthan) |

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for the Scaler × Meta × HuggingFace × PyTorch OpenEnv Hackathon**

🚑 [Live Demo](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV) · 📓 [Colab](https://colab.research.google.com/github/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon/blob/main/notebooks/Ambulance_GRPO_Training.ipynb) · 📝 [Blog](https://huggingface.co/blog/CSNEHA20/ambulance-dispatch-openenv) · 🎥 [Video](https://youtu.be/PASTE_YOUR_VIDEO_ID) · 🐙 [GitHub](https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon)

</div>
