---
title: "Ambulance Dispatch: Training LLMs to Save Lives with OpenEnv"
thumbnail: /blog/assets/ambulance-dispatch/thumbnail.png
authors:
  - user: CSNEHA20
  - user: Vishallakshmikanthan
---

# 🚑 Ambulance Dispatch: Training LLMs to Save Lives with OpenEnv

> *Every 60-second delay in CRITICAL ambulance response increases mortality by 10%.
> We built an RL environment to fix that.*

## The Problem

India handles over 40 million emergency calls annually through its 108/112 dispatch networks.
Human dispatchers must simultaneously decide:
- **Which ambulance** to send? (nearest vs. fastest vs. least busy)
- **Which hospital** is appropriate? (nearest vs. specialty match vs. capacity)
- **How to cover** the city proactively? (repositioning idle units)

These are exactly the kind of multi-step, resource-constrained decisions that LLMs struggle with —
but could master with the right training environment.

## The Environment

We built **Ambulance-OpenENV**, a fully OpenEnv-compliant RL environment featuring:

- 🏙️ **Procedural city** on a Barabási-Albert scale-free graph (realistic hub-and-spoke topology)
- 🚑 **5-state FSM** per ambulance: IDLE → EN_ROUTE → AT_SCENE → TRANSPORTING → RETURNING
- 🔴 **3 severity tiers**: CRITICAL (20-step timeout), HIGH (45 steps), NORMAL (90 steps)
- 🚦 **Dynamic traffic**: rush-hour multipliers (1.5–2.5×) + random road incidents
- 🏥 **Hospital specialties**: Trauma, Cardiac, General, Paediatric — wrong match = penalty
- 📊 **RFC 004 Rubric**: 9 named reward components for fine-grained training signal

## The Reward Signal

Instead of a single scalar, our RFC 004 Rubric provides **9 named components** every step:

```
+20.0  per emergency reached (ambulance arrives at scene)
+30.0  CRITICAL severity bonus | +10.0 HIGH severity bonus
+10.0  per successful hospital delivery
+ 7.5  fast dispatch speed bonus (inverse of wait time)
- 1.0  per idle ambulance-step when emergencies are active
-15.0  per emergency that expires unserved (timeout)
- 5.0  per routing to a full hospital (capacity violation)
```

This rich signal allows GRPO to learn the *why* behind each penalty, not just the total.

## Training

We train using two approaches:

### 1. DQN (Dueling DQN + Prioritized Experience Replay)

```bash
python train.py --episodes 500 --marl
```

The DQN agent learns a 124-dimensional state representation covering ambulance positions,
emergency urgency, hospital occupancy, and traffic conditions.

### 2. GRPO via HuggingFace TRL (future work)

```bash
python train_grpo.py --steps 100 --output-dir outputs/grpo
```

The GRPO training pipeline is implemented and ready. An LLM receives the observation as
structured JSON and must output a valid dispatch action. The 9-component rubric provides
separate reward signals, preventing reward hacking. Full LLM fine-tuning runs are planned
as the next step beyond this hackathon submission.

## Results

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| Random (noop) | ~0.01 | ~0.00 | ~0.00 |
| GreedyAgent | ~0.89 | ~0.18 | ~0.30 |
| RepositioningOracle | **0.923** | **0.176** | **0.482** |

The RepositioningOracle establishes the current upper bound. Full GRPO/LLM fine-tuning on this environment is the next step — the reward signal and training pipeline are ready.

## Try It

```bash
# Run the environment locally
git clone https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon.git Ambulance-OpenENV
cd Ambulance-OpenENV
pip install -r requirements.txt
python inference.py  # See [START]/[STEP]/[END] scores
```

Or open the [live HuggingFace Space](https://huggingface.co/spaces/vishallakshmikanthan/Ambulance-OpenENV)
to watch the real-time dispatch dashboard.

## Colab Notebook

Run training directly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon/blob/main/notebooks/Ambulance_GRPO_Training.ipynb)

## Code

[GitHub Repository](https://github.com/CSNEHA20/Meta_PyTorch_OpenEnv_Hackathon)
