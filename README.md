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
short_description: "City-scale RL ambulance dispatch — OpenEnv. Easy=0.923 | Medium=0.176 | Hard=0.482. 9-component RFC 004 rubric, multi-agent, dynamic traffic."
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

---

**A production-grade reinforcement learning environment for city-scale ambulance dispatch optimisation.**

Built for the **Scaler × Meta × HuggingFace × PyTorch OpenEnv Hackathon**.

Simulates India's 108/112 emergency dispatch system under life-or-death time pressure, featuring dynamic traffic, hospital overflow, specialty routing, and multi-objective triage across a stochastic city graph.

[Live Demo](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV) · [Report Bug](https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon/issues) · [Documentation](#table-of-contents)

</div>

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
  - [Easy Task](#easy-task)
  - [Medium Task](#medium-task)
  - [Hard Task](#hard-task)
- [🤖 Agents](#-agents)
- [🧠 RL Training Infrastructure](#-rl-training-infrastructure)
- [🔄 Multi-Agent RL](#-multi-agent-rl)
- [⏱️ Long-Horizon Planning](#-long-horizon-planning)
- [🎭 Self-Improvement Loop](#-self-improvement-loop)
- [🔌 API Endpoints](#-api-endpoints)
- [📺 Dashboard & Visualization](#-dashboard--visualization)
- [🚀 Getting Started](#-getting-started)
- [📁 Complete Project Structure](#-complete-project-structure)
- [🛠️ Tech Stack](#-tech-stack)
- [👥 Team](#-team)
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
- Rush-hour traffic (1.5-2.5x slower travel)
- Random road incidents (3.0x blockage)
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

By creating an RL environment for this task, we can:
- Train AI agents to assist or augment human dispatchers
- Research optimal dispatch policies
- Simulate "what-if" scenarios (disasters, pandemics, infrastructure changes)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🏙️ **Procedural City** | Barabási-Albert scale-free graph (20-100 nodes) with realistic hub-and-spoke topology |
| 🚑 **7-State FSM Fleet** | IDLE → DISPATCHED → EN_ROUTE → AT_SCENE → TRANSPORTING → RETURNING → REPOSITIONING |
| 🔴 **3 Severity Tiers** | CRITICAL (10-step timeout, +30 bonus), HIGH (20-step, +10), NORMAL (30-step) |
| 🚦 **Dynamic Traffic** | Rush-hour multipliers (1.5–2.5×) + random incidents (3.0× blockage) |
| 🏨 **Hospital Network** | 8-bed capacity limits, specialty routing (Trauma/Cardiac/General/Paediatric) |
| 📊 **RFC 004 Rubric** | 9 named reward components for fine-grained training introspection |
| 🔌 **Full RFC Suite** | RFC 001–005 compliant (Base API, Auto-Discovery, MCP, Rubric, Concurrency) |
| 🧪 **58 Tests** | Comprehensive pytest suite covering environment, graders, and models |
| 🖥️ **Next.js Dashboard** | Real-time dark-mode UI with city map, dispatch queue, and reward charts |
| 🐳 **Docker-Ready** | Single-command deployment to HuggingFace Spaces |
| 🔁 **Deterministic Seeding** | Byte-identical episode replay across runs |
| 🧠 **Multi-Agent RL** | Independent DQN agents per ambulance with conflict detection |
| ⏱️ **Long-Horizon** | 500-step episodes with demand surges and curriculum learning |
| 🎭 **Self-Improvement** | Weakness detection + adversarial scenario generation |

---

## 🏗️ System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT / LLM / RL MODEL                              │
│  Makes dispatch decisions based on observations                            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ POST /env/step
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                      FASTAPI SERVER (Port 7860)                            │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  Core OpenEnv Endpoints    │  RFC Extensions  │  Dashboard Endpoints   │  │
│  │  • POST /env/reset         │  • GET /tools    │  • GET /marl/status    │  │
│  │  • POST /env/step          │  • GET /mcp      │  • GET /curriculum/... │  │
│  │  • GET /env/state          │  • WS /ws/live   │  • GET /selfplay/...  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────────┐
│                    AMBULANCEENVIRONMENT (Core Engine)                       │
│                                                                              │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────────────┐    │
│  │  CityGraph   │  │  AmbulanceFleet │  │   EmergencyGenerator       │    │
│  │  (NetworkX)  │  │  (7-state FSM)  │  │   (Poisson λ arrivals)     │    │
│  │  • 20 nodes  │  │  • n ambulances │  │   • CRITICAL/HIGH/NORMAL   │    │
│  │  • BA graph  │  │  • state machine│  │   • Random node placement  │    │
│  │  • Dijkstra  │  │  • ETA tracking │  │   • Time-limited           │    │
│  └──────────────┘  └─────────────────┘  └──────────────────────────┘    │
│                                                                              │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────────────┐    │
│  │ TrafficEngine│  │ HospitalNetwork │  │      RubricEngine         │    │
│  │ • Rush-hour  │  │ • 8-bed cap     │  │  9 reward components       │    │
│  │ • Incidents  │  │ • Specialty     │  │  per-step calculation      │    │
│  │ • Multipliers│  │ • Overflow      │  │  (RFC 004 compliant)       │    │
│  └──────────────┘  └─────────────────┘  └──────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Dependencies

```
env/models.py ───────────────────────────────────────┐
     │                                               │
     ▼                                               ▼
env/simulator.py (CityGraph, Ambulance, etc.)  server/models (Action/Observation)
     │                                               │
     └──────────────┬────────────────────────────────┘
                    │
                    ▼
          env/environment.py (AmbulanceEnvironment)
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   agents/*.py   rl/*.py    server/app.py
   (decision    (training  (HTTP API +
    makers)      agents)    WebSocket)
```

---

## 🎮 The Simulation Environment

### City Graph: The Road Network

The city is modeled as a **Barabási-Albert scale-free graph** — a mathematical model that produces realistic urban road networks where:
- A few central nodes have high connectivity (major intersections, downtown)
- Most nodes have low connectivity (suburbs, residential areas)
- Creates natural "hubs" that match real city topology

**Technical Details:**
- Default: 20 nodes, attachment parameter m=3
- Each edge has a base travel time (2-8 minutes)
- Shortest paths pre-computed via Dijkstra for O(1) lookup
- Traffic multipliers dynamically adjust travel times

### Ambulance Fleet: 7-State FSM

Each ambulance follows a strict **finite state machine** with 7 states:

```
┌──────┐ dispatch  ┌───────────┐  travel   ┌──────────┐
│ IDLE │──────────▶│ DISPATCHED│──────────▶│ EN_ROUTE │
└──────┘           └───────────┘           └────┬─────┘
   ▲                                            │
   │ return                                     │ arrive
   │                                            ▼
┌──────┐           ┌───────────┐           ┌──────────┐
│RETURN│◀──────────│TRANSPORTING│◀─────────│ AT_SCENE │
│_ING  │   deliver  └───────────┘   load    └──────────┘
└──────┘

Special state:
┌─────────────┐
│REPOSITIONING│  ← Proactive staging to hotspots
└─────────────┘
```

**State Descriptions:**
| State | Description | Typical Duration |
|-------|-------------|------------------|
| `IDLE` | Available for dispatch at current location | — |
| `DISPATCHED` | Just assigned, calculating route | 1 step |
| `EN_ROUTE` | Travelling to emergency scene | Dijkstra time × traffic |
| `AT_SCENE` | Loading patient | 2-4 steps |
| `TRANSPORTING` | Moving patient to hospital | Dijkstra time × traffic |
| `RETURNING` | Going back to base after delivery | Varies |
| `REPOSITIONING` | Proactive staging to predicted hotspot | Dijkstra time |

### Emergency Generator

Emergencies arrive via a **Poisson process** (like real phone calls):
- λ (lambda) controls arrival rate (e.g., 0.3 = ~0.3 emergencies per step)
- Each emergency has:
  - **Location**: Random node on the graph
  - **Severity**: CRITICAL (25%), HIGH (35%), NORMAL (40%)
  - **Time Limit**: CRITICAL=10 steps, HIGH=20, NORMAL=30
  - **Specialty Need**: Implicitly determined by severity

If an emergency times out unserved → **timeout penalty (-15.0)**

### Traffic Engine

Two traffic effects make the environment dynamic:

**1. Rush-Hour Multipliers:**
- 7-9 AM and 5-8 PM: Normal distribution μ=1.6, σ=0.2, clipped [1.2, 2.5]
- Other hours: Normal μ=1.0, σ=0.05, clipped [0.9, 1.2]

**2. Random Incidents:**
- 2% chance per step to spawn a road incident
- Random edge blocked for 5 steps
- Adds +0.5 to traffic multiplier (max +1.0 from incidents)

### Hospital Network

**Hospital Properties:**
- **Capacity**: 8 concurrent patients (default)
- **Current Patients**: Real-time occupancy
- **Specialty**: Trauma, Cardiac, General, or Paediatric

**Specialty Routing Logic:**
```python
CRITICAL → Trauma or Cardiac (life-threatening)
HIGH     → Trauma or General (serious but stable)
NORMAL   → General or Paediatric (routine)
```

Dispatching to the wrong specialty incurs a penalty. Dispatching to a full hospital triggers **CapacityViolation (-5.0)** and requires re-routing.

---

## 🕹️ Action Space

The agent sends one action per step (or batch with `step_all()`):

```python
class ActionModel:
    ambulance_id: Optional[int]     # Which idle ambulance (0-indexed)
    emergency_id: str                 # Target emergency UUID
    hospital_id: Optional[int]        # Destination hospital
    reposition_node: Optional[int]    # Move idle ambulance here (optional)
    is_noop: bool = False             # Skip this step
```

**Action Validation (Pydantic extra='forbid'):**
- Invalid ambulance_id → -10 penalty
- Invalid emergency_id → -10 penalty
- Invalid hospital_id → -10 penalty
- Dispatching busy ambulance → Action rejected
- Dispatching to full hospital → CapacityViolation penalty

**Example Actions:**
```python
# Dispatch ambulance 0 to emergency "abc-123", send to hospital 1
action = ActionModel(
    ambulance_id=0,
    emergency_id="abc-123",
    hospital_id=1
)

# Reposition ambulance 2 to node 15 (proactive staging)
action = ActionModel(
    ambulance_id=2,
    emergency_id="",
    hospital_id=None,
    reposition_node=15,
    is_noop=False
)

# Skip this step (let simulation advance)
action = ActionModel(is_noop=True)
```

---

## 👁️ Observation Space

```python
class ObservationModel:
    ambulances: List[AmbulanceInfo]      # Fleet status
    emergencies: List[EmergencyInfo]     # Active incidents
    hospitals: List[HospitalInfo]        # Hospital network
    traffic: Dict[str, float]            # {"global": multiplier}
    step: int                            # Current timestep
    reward: float                        # Step reward
    done: bool                           # Episode finished?
    rubric: Optional[Rubric]             # 9-component breakdown
```

**Nested Types:**

```python
class AmbulanceInfo:
    id: int                      # 0, 1, 2, ...
    node: int                    # Current graph position
    state: AmbulanceState        # idle/en_route/at_scene/...
    eta: int                     # Steps until next state
    target_emg_id: Optional[str] # Assigned emergency
    target_hosp_id: Optional[int] # Assigned hospital

class EmergencyInfo:
    id: str                      # UUID (first 8 chars shown)
    node: int                    # Incident location
    severity: Severity           # CRITICAL/HIGH/NORMAL
    time_remaining: int          # Steps before timeout
    max_time_remaining: int      # For progress bars
    assigned: bool               # Ambulance dispatched?
    spawn_time: int              # When it appeared

class HospitalInfo:
    id: int                      # 0, 1, 2, ...
    node: int                    # Location
    capacity: int                # Max beds (8)
    current_patients: int        # Occupied beds
    specialty: str               # Trauma/Cardiac/General/Paediatric
```

---

## 🏆 Reward System (RFC 004)

The environment computes reward via a **9-component Rubric**:

| # | Component | Trigger | Value | Purpose |
|---|-----------|---------|-------|---------|
| 1 | `EmergencyServed` | Ambulance arrives at scene | **+20.0** | Reward successful dispatch |
| 2 | `SeverityBonus` | CRITICAL served | **+30.0** | Prioritize life-threatening |
| 2b| `SeverityBonus` | HIGH served | **+10.0** | Prioritize urgent cases |
| 3 | `DispatchSpeed` | Fast response (low wait) | **0–+10.0** | Encourage rapid response |
| 4 | `HospitalDelivery` | Patient delivered | **+10.0** | Complete care chain |
| 5 | `DistancePenalty` | Long travel distance | **−variable** | Discourage inefficient routes |
| 6 | `TrafficPenalty` | Ignoring traffic | **−variable** | Penalize bad timing |
| 7 | `IdlePenalty` | Ambulance idle during backlog | **−1.0/step** | Prevent underutilization |
| 8 | `CapacityViolation` | Route to full hospital | **−5.0** | Prevent overflow |
| 9 | `TimeoutPenalty` | Emergency expires | **−15.0** | Heavy penalty for misses |

**Additional Components:**
- `fairness_score`: Zone coverage equity (hard task)
- `validation_penalty`: Anti-exploit deductions

**Reward Capping:**
- Maximum positive per step: 80.0 (prevents reward hacking)
- Loop detection: −5 × repeat count (capped at −50)

---

## 📊 Three Difficulty Levels

### Easy Task

| Parameter | Value |
|-----------|-------|
| Ambulances | 2 |
| Hospitals | 2 |
| Steps | 30 |
| Arrival Rate (λ) | 0.3 |
| Severities | NORMAL only |
| Traffic | Disabled |
| Seed | 42 |

**Grading:** Mean of (optimal_time / actual_response_time), clamped [0, 1]

**Baseline Score:** 0.923

**What it Tests:** Basic dispatch correctness. Just pick the nearest idle ambulance and nearest hospital.

---

### Medium Task

| Parameter | Value |
|-----------|-------|
| Ambulances | 4 |
| Hospitals | 3 |
| Steps | 60 |
| Arrival Rate (λ) | 0.4 |
| Severities | CRITICAL, HIGH, NORMAL |
| Traffic | Mild (1.0–1.3×) |
| Seed | 42 |

**Grading:**
```
score = 0.50 × served_percentage
      + 0.35 × response_score
      - 0.15 × idle_fraction
```

**Baseline Score:** 0.176

**What it Tests:** Fleet coordination, priority dispatch (CRITICAL first), hospital load balancing.

**Strategy Note:** Repositioning is DISABLED for medium task because the time cost (9+ steps blocking dispatch) outweighs the idle_fraction benefit (0.15 weight).

---

### Hard Task

| Parameter | Value |
|-----------|-------|
| Ambulances | 6 |
| Hospitals | 4 (Trauma, Cardiac, General, Paediatric) |
| Steps | 100 |
| Arrival Rate (λ) | 0.6 |
| Severities | All three tiers |
| Traffic | Dynamic rush-hour (1.5–2.5×) + incidents |
| Specialty Routing | Required |
| Seed | 42 |

**Grading:**
```
critical_rate = critical_served / critical_total
overall_rate = served / total_emergencies
weighted_served = 0.7 × critical_rate + 0.3 × overall_rate

score = 0.50 × weighted_served
      + 0.30 × priority_accuracy
      + 0.15 × fairness_score
      - 0.05 × capacity_violations
```

**Baseline Score:** 0.482

**What it Tests:** CRITICAL-first triage, specialty routing, zone fairness across 4 city zones, traffic-aware planning.

---

## 🤖 Agents

### GreedyAgent (`agents/greedy_agent.py`)
**Type:** Rule-based baseline

Simple nearest-first dispatch:
1. Find first idle ambulance
2. Find nearest unassigned emergency (by node difference)
3. Find nearest hospital to that emergency

**Use case:** Minimal baseline, testing environment logic

---

### BaselineAgent (`agents/baseline.py`)
**Type:** Enhanced rule-based

Priority-sorted greedy:
1. Sort emergencies: CRITICAL > HIGH > NORMAL, then by time_remaining
2. Assign nearest idle ambulance to highest priority
3. Assign nearest available hospital

**Use case:** Better baseline that respects severity

---

### OracleAgent (`agents/oracle.py`)
**Type:** Optimal reference (Dijkstra-based)

Uses pre-computed shortest paths to make globally optimal assignments:
- Computes actual travel times (not just node differences)
- Optimal ambulance-to-emergency matching
- Upper-bound reference for evaluating other agents

**Use case:** Score ceiling estimation, validation

---

### RepositioningOracle (`agents/repositioning_oracle.py`)
**Type:** Best-performing agent (used in inference)

**Features:**
1. **Multi-dispatch**: All idle ambulances dispatched simultaneously
2. **Specialty-aware routing**: Matches hospital specialty to emergency severity
3. **Proactive repositioning**: Moves idle ambulances to predicted hotspots
4. **Zone fairness**: Spreads repositioned ambulances across all 4 city zones

**Hotspot Prediction:**
```python
ZONE_CENTERS = [12, 37, 62, 87]  # 4 city zones

# Predict hotspots based on:
1. Emergency frequency history (Counter)
2. Zone coverage (ensure all zones represented)
3. Zone centers as fallback
```

**Specialty Mapping:**
```python
CRITICAL → Trauma, Cardiac
HIGH     → Trauma, General
NORMAL   → General, Paediatric
```

**Use case:** Production inference, score maximization

---

### PriorityAgent (`agents/priority_agent.py`)
**Type:** LLM-powered with heuristic fallback

Uses OpenAI-compatible API for structured decision-making:
- Sends full observation to LLM
- LLM returns dispatch decisions in structured format
- Heuristic fallback if API fails or times out

**Use case:** LLM-based baselines, exploring LLM-as-agent approaches

---

### AmbulanceQAgent (`agents/fleet_agent.py`)
**Type:** DQN-based RL agent for MARL

Per-ambulance independent DQN:
- State encoding: 8 dims (own state) + 2×(n-1) (fleet summary) + 2 (oversight) + 5 (global)
- Epsilon-greedy with action masking
- Double DQN updates with soft target network

**Use case:** Multi-agent RL training

---

### OversightAgent (`agents/oversight_agent.py`)
**Type:** Fleet coordinator (non-decision maker)

Detects and signals coordination conflicts:
- Identifies when multiple agents target same emergency
- Emits per-agent coordination signals [conflict_flag, conflict_amb_norm]
- Maintains conflict history for dashboard

**Use case:** MARL coordination, conflict detection

---

## 🧠 RL Training Infrastructure

Located in `rl/` directory — complete Dueling DQN implementation:

### Core Modules

| File | Purpose |
|------|---------|
| `rl/dqn.py` | Dueling DQN network architecture (value + advantage streams) |
| `rl/rl_agent.py` | Training loop with soft target updates, Double DQN |
| `rl/state_encoder.py` | 120-dimensional state encoding from observations |
| `rl/action_mapper.py` | Maps discrete DQN outputs to structured actions |
| `rl/action_mask.py` | Masks invalid actions (busy ambulances, etc.) |
| `rl/replay_buffer.py` | Standard uniform replay buffer |
| `rl/prioritized_replay_buffer.py` | PER (Prioritized Experience Replay) |
| `rl/rubric.py` | RFC 004 Rubric integration for reward shaping |
| `rl/demand_predictor.py` | Hotspot prediction for proactive staging |

### State Encoding (120 dimensions)

```python
# Ambulances: 6 max × (node + state_onehot(6) + eta) = 48 dims
# Emergencies: 10 max × (node + severity_oh(3) + time + assigned) = 60 dims
# Hospitals: 4 max × (node + capacity + patients + ratio) = 16 dims
# Total: 124 → clipped to 120
```

### Training Features

- **Dueling DQN**: Separate value and advantage streams
- **Double DQN**: Reduce overestimation bias
- **Soft Target Update**: τ=0.005 for smooth target network updates
- **Prioritized Replay**: α=0.6, β anneals to 1.0
- **Reward Normalization**: Optional z-score normalization
- **Action Masking**: Only valid actions considered

### Running Training

```bash
# Standard single-agent training
python train.py --episodes 500

# Multi-agent RL (one DQN per ambulance)
python train.py --marl --episodes 1000

# Long-horizon (500-step episodes with demand surges)
python train.py --long-horizon --episodes 500

# With self-improvement loop
python train.py --self-play --selfplay-interval 200

# Disable enhancements
python train.py --no-dueling --no-per --no-soft-update
```

---

## 🔄 Multi-Agent RL

Located in `multi_agent/` — fleet coordination system:

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   MultiAgentCoordinator                      │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ AmbulanceQ  │  │ AmbulanceQ  │  │ AmbulanceQ  │  ...    │
│  │ Agent #0    │  │ Agent #1    │  │ Agent #2    │         │
│  │ (independent│  │ (independent│  │ (independent│         │
│  │  DQN)       │  │  DQN)       │  │  DQN)       │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          │                                 │
│         ┌────────────────▼────────────────┐                │
│         │       OversightAgent            │                │
│         │  (conflict detection & signals) │                │
│         └─────────────────────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

1. **Independent Agents**: Each ambulance has its own DQN
2. **Global Reward Splitting**: `global_reward / n_agents` per step
3. **Conflict Detection**: −5 penalty if multiple agents target same emergency
4. **Coordination Signals**: 2-dim vector per agent [conflict_flag, partner_id]

### Running MARL

```bash
python train_marl.py
# or
python train.py --marl
```

---

## ⏱️ Long-Horizon Planning

Located in `long_horizon/` — extended episodes with curriculum:

### Features

- **500-step episodes** (vs. 30/60/100 in standard tasks)
- **Demand surges**: Periodic spikes in emergency arrival rate
- **Curriculum learning**: Gradually increase difficulty
- **History encoding**: LSTM-based encoder for temporal patterns

### Curriculum Stages

1. **Stage 1**: Low arrival rate, no traffic
2. **Stage 2**: Medium arrival rate, mild traffic
3. **Stage 3**: High arrival rate, full traffic + incidents
4. **Stage 4**: Demand surges during rush hours

### Modules

| File | Purpose |
|------|---------|
| `long_horizon_env.py` | Extended environment with surge logic |
| `curriculum_manager.py` | Stage progression based on performance |
| `episode_planner.py` | Macro-level planning across episode |
| `history_encoder.py` | LSTM encoding of observation history |

### Running Long-Horizon Training

```bash
python train_curriculum.py
# or
python train.py --long-horizon
```

---

## 🎭 Self-Improvement Loop

Located in `self_improvement/` — autonomous agent improvement:

### The Loop

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Training   │───▶│  Weakness    │───▶│  Adversarial │
│   Episodes   │    │  Detection   │    │  Scenario    │
└──────────────┘    └──────────────┘    │  Generation  │
     ▲                                  └──────┬───────┘
     │                                         │
     └─────────────────────────────────────────┘
              (train on hard scenarios)
```

### Components

| File | Purpose |
|------|---------|
| `weakness_detector.py` | Identifies underperforming scenarios |
| `adversarial_generator.py` | Creates challenging scenarios |
| `self_play_trainer.py` | Training loop with self-play |
| `expert_agent.py` | Reference oracle for comparison |
| `performance_analyzer.py` | Metrics tracking and analysis |

### Adversarial Scenarios

The generator creates challenging conditions:
- **Clustered emergencies**: Multiple emergencies at nearby nodes
- **Low resource scenarios**: Few ambulances, many emergencies
- **Traffic nightmares**: High multipliers + incidents on critical edges
- **Hospital overload**: Simultaneous arrivals exceeding capacity

### Running Self-Play

```bash
python train_selfplay.py
# or
python train.py --self-play --selfplay-interval 200
```

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
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `WS` | `/ws/live` | WebSocket — real-time state at 2 Hz |

### Multi-Agent Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/marl/status` | Fleet coordination statistics |
| `GET` | `/marl/conflicts` | Recent conflict events |

### Long-Horizon Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/curriculum/status` | Current training stage |
| `GET` | `/curriculum/progress` | Stage completion metrics |

### Self-Improvement Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/selfplay/weaknesses` | Detected weakness report |
| `GET` | `/selfplay/iterations` | Improvement history |
| `POST` | `/selfplay/adversarial` | Generate custom adversarial scenario |

### Dashboard Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/dashboard/reset` | Reset with oracle auto-play |
| `POST` | `/dashboard/step` | Step with oracle dispatch |
| `GET` | `/dashboard/metrics` | Current episode metrics |
| `GET` | `/score` | Run all tasks, return scores |

---

## 📺 Dashboard & Visualization

### Next.js 14 Dashboard (`frontend/`)

Real-time dark-mode UI with multiple views:

**Views:**
1. **LIVE** — Real-time city map, dispatch queue, ambulance status
2. **MULTI-AGENT** — Per-agent DQN states, coordination signals
3. **LONG-HORIZON** — Curriculum stage, episode progress
4. **SELF-IMPROVE** — Weakness detection, adversarial scenarios

**Components:**
- `CityMap.jsx` — Hexagonal hub-and-spoke layout with animated ambulances
- `AmbulanceTable.jsx` — Fleet status with FSM states and ETAs
- `HospitalPanel.jsx` — Capacity bars and specialty labels
- `RewardChart.jsx` — Real-time reward trajectory with rubric breakdown
- `MultiAgentView.jsx` — MARL coordination visualization
- `LongHorizonView.jsx` — Curriculum progress
- `SelfImprovementView.jsx` — Self-play metrics

**Running the Dashboard:**

```bash
cd frontend
npm install
npm run dev      # Development (port 3000)
npm run build    # Production build (to dist/)
```

The FastAPI server automatically serves the built dashboard at `/dashboard`.

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for dashboard, optional)
- **Docker** (optional, for deployment)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon.git
cd Meta_PyTorch_OpenEnv_Hackathon

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template
cp .env.example .env
# Edit .env and add: HF_TOKEN, API_BASE_URL, MODEL_NAME
```

### Running Inference (Production Agent)

```bash
# Run all three tasks (outputs [START]/[STEP]/[END] logs)
python inference.py

# Run specific task
python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

### Running the Server

```bash
# Development (hot-reload)
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Production (4 workers)
uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4
```

Test: `curl http://localhost:7860/health`

### Running Training

```bash
# Basic DQN training (500 episodes)
python train.py --episodes 500

# Multi-agent RL
python train.py --marl --episodes 1000

# Long-horizon with curriculum
python train.py --long-horizon

# With self-improvement
python train.py --self-play --selfplay-interval 200

# Final training (comprehensive)
python train_final.py
```

### Running Tests

```bash
python -m pytest tests/ -v
```

All **58 tests** cover environment logic, graders, and model validation.

### Docker Deployment

```bash
# Build
docker build -t ambulance-openenv .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  ambulance-openenv
```

---

## 📁 Complete Project Structure

```
Ambulance-OpenENV/
│
├── 📄 Configuration Files
│   ├── openenv.yaml              # OpenEnv specification with tasks
│   ├── pyproject.toml            # Python project config, dependencies
│   ├── requirements.txt            # Pip dependencies
│   ├── Dockerfile                # Container build instructions
│   ├── .env.example              # Environment variable template
│   └── .gitignore                # Git ignore patterns
│
├── 📄 Main Scripts
│   ├── inference.py              # Production inference (uses RepositioningOracle)
│   ├── train.py                  # Main training script (DQN, MARL, self-play)
│   ├── train_final.py            # Comprehensive final training
│   ├── train_curriculum.py       # Long-horizon curriculum training
│   ├── train_marl.py             # Multi-agent RL training
│   ├── train_selfplay.py         # Self-improvement training
│   ├── train_grpo.py             # GRPO policy gradient training
│   ├── evaluate.py               # Checkpoint evaluation
│   ├── demo.py                   # Interactive demo
│   ├── compare.py                # Agent comparison benchmarks
│   └── app.py                    # Simple Gradio app entry
│
├── 📄 Grading Scripts
│   ├── grader_easy.py            # Easy task grading (optimal/actual ratios)
│   ├── grader_medium.py          # Medium task grading (served+response-idle)
│   └── grader_hard.py            # Hard task grading (critical+fairness+penalty)
│
├── 📂 env/                       # Core Simulation Engine
│   ├── __init__.py
│   ├── models.py                 # Pydantic models (Rubric, Action, Observation)
│   ├── simulator.py              # CityGraph, AmbulanceFleet, TrafficEngine, etc.
│   └── environment.py            # AmbulanceEnvironment main class
│
├── 📂 server/                    # FastAPI HTTP Server
│   ├── __init__.py
│   ├── ambulance_environment.py  # OpenEnv wrapper for core environment
│   └── app.py                    # FastAPI application with all endpoints
│
├── 📂 agents/                    # Dispatch Agents
│   ├── __init__.py
│   ├── baseline.py               # Priority-sorted baseline
│   ├── greedy_agent.py           # Simple nearest-first
│   ├── oracle.py                 # Dijkstra-based optimal
│   ├── repositioning_oracle.py   # Best agent: multi-dispatch + repositioning
│   ├── priority_agent.py         # LLM-powered with fallback
│   ├── fleet_agent.py            # DQN agent for MARL
│   └── oversight_agent.py        # Conflict detection coordinator
│
├── 📂 rl/                        # RL Training Infrastructure
│   ├── dqn.py                    # Dueling DQN network
│   ├── rl_agent.py               # DQNAgent training loop
│   ├── state_encoder.py          # 120-dim state encoding
│   ├── action_mapper.py          # Discrete → ActionModel mapping
│   ├── action_mask.py            # Invalid action masking
│   ├── replay_buffer.py          # Uniform replay buffer
│   ├── prioritized_replay_buffer.py  # PER implementation
│   ├── rubric.py                 # RFC 004 Rubric integration
│   └── demand_predictor.py       # Hotspot prediction
│
├── 📂 multi_agent/               # Multi-Agent Coordination
│   ├── ambulance_agent.py        # Individual ambulance agent wrapper
│   ├── coordinator.py            # MultiAgentCoordinator (central controller)
│   ├── dispatcher_agent.py       # Dispatch decision logic
│   └── planner.py                # Route planning utilities
│
├── 📂 long_horizon/              # Long-Horizon & Curriculum
│   ├── __init__.py
│   ├── curriculum_manager.py     # Difficulty stage progression
│   ├── episode_planner.py        # Macro-level planning
│   ├── history_encoder.py        # LSTM temporal encoding
│   └── long_horizon_env.py       # Extended 500-step environment
│
├── 📂 self_improvement/          # Self-Play & Adversarial Training
│   ├── __init__.py
│   ├── adversarial_generator.py  # Creates challenging scenarios
│   ├── expert_agent.py           # Reference oracle agent
│   ├── performance_analyzer.py   # Metrics tracking
│   ├── self_play_trainer.py      # Self-play training loop
│   ├── strategy_adapter.py       # Strategy adjustment
│   └── weakness_detector.py      # Identifies failure patterns
│
├── 📂 tasks/                     # Task Configurations
│   ├── __init__.py
│   ├── configs.py                # Unified config classes
│   ├── easy.py                   # Easy task config (2 amb, 30 steps)
│   ├── medium.py                 # Medium task config (4 amb, 60 steps)
│   ├── hard.py                   # Hard task config (6 amb, 100 steps)
│   └── graders.py                # Task graders interface
│
├── 📂 evaluation/                # Evaluation Framework
│   ├── __init__.py
│   ├── auto_evaluator.py         # Automated baseline vs advanced comparison
│   └── report.py                 # Report generation utilities
│
├── 📂 tests/                     # Test Suite (58 tests)
│   ├── __init__.py
│   ├── test_environment.py       # Environment logic tests
│   ├── test_graders.py           # Grader correctness tests
│   ├── test_models.py            # Pydantic model tests
│   └── test_action_reduction.py  # Action space pruning tests
│
├── 📂 utils/                     # Utilities
│   └── logger.py                 # Structured logging
│
├── 📂 frontend/                  # Next.js 14 Dashboard
│   ├── app/
│   │   ├── components/
│   │   │   ├── AmbulanceTable.jsx
│   │   │   ├── CityMap.jsx
│   │   │   ├── HospitalPanel.jsx
│   │   │   ├── LongHorizonView.jsx
│   │   │   ├── MultiAgentView.jsx
│   │   │   ├── RewardChart.jsx
│   │   │   └── SelfImprovementView.jsx
│   │   ├── globals.css
│   │   ├── layout.js
│   │   └── page.js               # Main dashboard
│   ├── public/
│   ├── next.config.js
│   ├── package.json
│   ├── postcss.config.js
│   └── tailwind.config.js
│
├── 📂 notebooks/                 # Jupyter Notebooks
│   ├── grpo_colab.ipynb          # GRPO training (Google Colab)
│   ├── trl_colab_minimal.ipynb   # TRL minimal example
│   └── run_analysis.py           # Notebook utilities
│
├── 📂 outputs/                   # Training Outputs
│   └── final/
│       └── reward_curve.png
│
└── 📄 Documentation
    ├── README.md                 # This file
    ├── PROJECT_DOCUMENTATION.md  # Detailed technical reference
    ├── colab_notebook.ipynb      # Colab training notebook
    └── colab_train.py            # Colab training script
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11 | Core implementation |
| **Web Framework** | FastAPI 0.110+ | HTTP API + WebSocket |
| **Server** | Uvicorn | ASGI server |
| **Env Standard** | openenv-core ≥0.2.0 | OpenEnv compliance |
| **Graph Engine** | NetworkX 3.2+ | City graph + shortest paths |
| **Numerics** | NumPy 1.26+ | Arrays, RNG, computation |
| **Validation** | Pydantic v2 | Type-safe models |
| **RL Framework** | PyTorch 2.0+ | DQN training |
| **LLM Client** | OpenAI SDK | LLM-powered agents |
| **Testing** | pytest + pytest-asyncio | Test suite |
| **Frontend** | Next.js 14 + Tailwind CSS | Dashboard |
| **Charts** | Chart.js | Reward visualization |
| **HTTP Client** | httpx | Async API calls |
| **Deployment** | Docker | Containerization |

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

🚑 [Live Demo](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV) · 📊 [HuggingFace Space](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV) · 🐙 [GitHub](https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon)

</div>
