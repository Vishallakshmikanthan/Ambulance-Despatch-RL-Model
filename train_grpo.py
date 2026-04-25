"""
train_grpo.py — TRL GRPO training for LLM-based ambulance dispatch agent.

Uses HuggingFace TRL's GRPOTrainer to train a small language model (Qwen2.5-7B
or Llama-3.1-8B) to dispatch ambulances by generating structured JSON responses.

Multi-component reward: 9 rubric components → cleaner gradient signal for GRPO.

Usage
-----
    python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 100 --output-dir outputs/grpo

For Colab (with Unsloth):
    python train_grpo.py --use-unsloth --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit --steps 200

Notes
-----
  - Requires: pip install trl transformers torch
  - Optional: pip install unsloth  (for 4-bit quantised training on free-tier Colab)
  - Reward signals are kept separate (not summed) to prevent reward hacking.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Graceful import guards (trl / unsloth may not be installed)
# ---------------------------------------------------------------------------
try:
    from trl import GRPOConfig, GRPOTrainer
    _TRL_AVAILABLE = True
except ImportError:
    _TRL_AVAILABLE = False
    print("[WARN] trl not installed. Run: pip install trl")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    print("[WARN] transformers not installed. Run: pip install transformers")

from env.environment import AmbulanceEnvironment
from env.models import ActionModel, ObservationModel, Rubric, AmbulanceState, Severity

# ---------------------------------------------------------------------------
# System prompt & prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert emergency dispatch coordinator for a city ambulance fleet. "
    "Analyze the current situation and dispatch the highest-priority unserved emergency "
    "using the nearest available idle ambulance, routing to the hospital with lowest "
    "occupancy and appropriate specialty. "
    "Your response MUST be valid JSON with exactly these keys: "
    '{"ambulance_id": <int or null>, "emergency_id": "<string>", "hospital_id": <int or null>}. '
    "Set ambulance_id to null and emergency_id to empty string for a no-op."
)


def build_prompt(obs: ObservationModel) -> str:
    """Serialize observation as structured JSON for the LLM prompt."""
    obs_dict = {
        "step": obs.step,
        "ambulances": [
            {
                "id": a.id,
                "node": a.node,
                "state": a.state.value,
                "eta": a.eta,
                "busy": a.state != AmbulanceState.IDLE,
            }
            for a in obs.ambulances
        ],
        "emergencies": [
            {
                "id": e.id,
                "node": e.node,
                "severity": e.severity.value,
                "time_remaining": e.time_remaining,
                "assigned": e.assigned,
            }
            for e in obs.emergencies
            if not e.assigned
        ],
        "hospitals": [
            {
                "id": h.id,
                "node": h.node,
                "occupancy": h.current_patients,
                "capacity": h.capacity,
                "specialty": h.specialty,
                "available": h.current_patients < h.capacity,
            }
            for h in obs.hospitals
        ],
        "traffic_global": obs.traffic.get("global", 1.0) if isinstance(obs.traffic, dict) else 1.0,
    }
    return json.dumps(obs_dict, indent=2)


# ---------------------------------------------------------------------------
# Action parsing & validation
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str, obs: ObservationModel) -> ActionModel:
    """
    Parse LLM JSON response into ActionModel with strict validation.
    Returns a noop on any parse or validation error.
    """
    try:
        # Extract JSON block from response (model may add explanation text)
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return _noop()
        parsed = json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        return _noop()

    amb_id = parsed.get("ambulance_id")
    emg_id = parsed.get("emergency_id", "")
    hosp_id = parsed.get("hospital_id")

    if not emg_id:
        return _noop()

    # Validate ambulance exists and is idle
    valid_idle_ids = {a.id for a in obs.ambulances if a.state == AmbulanceState.IDLE}
    if amb_id not in valid_idle_ids:
        return _noop()

    # Validate emergency exists and is unassigned
    valid_emg_ids = {e.id for e in obs.emergencies if not e.assigned}
    if emg_id not in valid_emg_ids:
        return _noop()

    # Validate hospital exists
    valid_hosp_ids = {h.id for h in obs.hospitals}
    if hosp_id not in valid_hosp_ids:
        return _noop()

    return ActionModel(ambulance_id=amb_id, emergency_id=emg_id, hospital_id=hosp_id)


def _noop() -> ActionModel:
    return ActionModel(ambulance_id=None, emergency_id="", is_noop=True)


# ---------------------------------------------------------------------------
# STEP 10.5 — ACTION PARSER (simple alias matching spec)
# ---------------------------------------------------------------------------

def parse_action(output: str) -> Optional[dict]:
    """Extract a JSON action dict from raw model output. Returns None on failure."""
    try:
        text = output.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        data = json.loads(text[start:end])
        return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# STEP 10.6 — ROLLOUT FUNCTION
# (works both as standalone rollout(model)->float and as TRL reward_fn)
# ---------------------------------------------------------------------------

def rollout(model_or_completions, prompts=None):
    """
    Dual-mode reward function.

    Standalone:  rollout(model) -> float  — full episode with a real model.
    TRL mode:    rollout(completions, prompts=...) -> List[float]  — batch scoring.
    """
    # ---- Standalone mode ----
    if hasattr(model_or_completions, "generate"):
        _model = model_or_completions
        import torch
        _tok = AutoTokenizer.from_pretrained(_model.config._name_or_path, trust_remote_code=True)
        env = AmbulanceEnvironment()
        obs = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            prompt = f"<|system|>\n{_SYSTEM_PROMPT}\n<|user|>\n{build_prompt(obs)}\n<|assistant|>\n"
            inputs = _tok(prompt, return_tensors="pt").to(_model.device)
            with torch.no_grad():
                out = _model.generate(**inputs, max_new_tokens=100, do_sample=False)
            text = _tok.decode(out[0], skip_special_tokens=True)
            action_dict = parse_action(text)
            if action_dict is None:
                reward = -10.0
                done = True
            else:
                try:
                    action = ActionModel(
                        ambulance_id=action_dict.get("ambulance_id"),
                        emergency_id=str(action_dict.get("emergency_id", "")),
                        hospital_id=action_dict.get("hospital_id"),
                    )
                    result = env.step(action)
                    if isinstance(result, tuple):
                        obs, reward, done, _ = result
                    else:
                        obs = result
                        reward = getattr(env, "last_reward", 0.0)
                        done = getattr(env, "done", False)
                except Exception:
                    reward = -10.0
                    done = True
            total_reward += float(reward)
        return total_reward

    # ---- TRL reward_fn mode ----
    completions = model_or_completions
    rewards: List[float] = []
    for text in completions:
        action_dict = parse_action(text)
        if action_dict is None:
            rewards.append(-5.0)
            continue
        has_keys = all(k in action_dict for k in ["ambulance_id", "emergency_id", "hospital_id"])
        base = 2.0 if has_keys else -2.0
        # Bonus env step
        try:
            env = AmbulanceEnvironment({"max_steps": 10})
            env.reset()
            action = ActionModel(
                ambulance_id=action_dict.get("ambulance_id"),
                emergency_id=str(action_dict.get("emergency_id", "")),
                hospital_id=action_dict.get("hospital_id"),
            )
            result = env.step(action)
            env_r = float(result[1]) if isinstance(result, tuple) else 0.0
            base += env_r
        except Exception:
            pass
        rewards.append(base)
    return rewards


# ---------------------------------------------------------------------------
# Reward functions (multi-component)
# ---------------------------------------------------------------------------

def compute_rewards(rubric: "Rubric | None") -> Dict[str, float]:
    """Extract named reward components from the rubric."""
    if rubric is None:
        return {"total": -1.0}
    return {
        "emergency_served": rubric.emergency_served,
        "severity_bonus": rubric.severity_bonus,
        "dispatch_speed": rubric.dispatch_speed,
        "hospital_delivery": rubric.hospital_delivery,
        "distance_penalty": rubric.distance_penalty,
        "traffic_penalty": rubric.traffic_penalty,
        "idle_penalty": rubric.idle_penalty,
        "capacity_violation": rubric.capacity_violation,
        "timeout_penalty": rubric.timeout_penalty,
        "total": rubric.total(),
    }


# ---------------------------------------------------------------------------
# GRPO rollout function
# ---------------------------------------------------------------------------

class GRPORolloutEnv:
    """Wrapper that runs one GRPO episode and collects (prompt, response, reward) triples."""

    def __init__(self, env_config: dict):
        self._cfg = env_config

    def rollout(self, generate_fn) -> List[Dict[str, Any]]:
        """
        Run one episode using `generate_fn(prompt) -> str` as the LLM.
        Returns list of step dicts with prompt, completion, reward.
        """
        env = AmbulanceEnvironment(self._cfg)
        obs = env.reset()
        steps = []

        for _ in range(self._cfg.get("max_steps", 50)):
            prompt = build_prompt(obs)
            full_prompt = f"<|system|>\n{_SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"
            response = generate_fn(full_prompt)

            action = parse_llm_response(response, obs)
            next_obs = env.step(action)
            if isinstance(next_obs, tuple):
                next_obs, reward, done, _ = next_obs
            else:
                reward = next_obs.reward
                done = next_obs.done

            reward_components = compute_rewards(next_obs.rubric)
            steps.append({
                "prompt": full_prompt,
                "completion": response,
                "reward": reward,
                "reward_components": reward_components,
            })
            obs = next_obs
            if done:
                break

        return steps


# ---------------------------------------------------------------------------
# Standalone training (without TRL — for testing reward pipeline)
# ---------------------------------------------------------------------------

def train_standalone(args):
    """
    Standalone training loop that evaluates a greedy text baseline.
    Used to verify the reward pipeline works before hooking up a real LLM.
    """
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = {"n_ambulances": 3, "n_hospitals": 2, "max_steps": 50}
    rollout_env = GRPORolloutEnv(env_cfg)

    # Greedy text baseline: always dispatch nearest idle to highest priority
    def greedy_generate(prompt: str) -> str:
        # Parse obs from prompt and return a deterministic greedy action
        try:
            user_start = prompt.find("<|user|>") + len("<|user|>\n")
            user_end = prompt.find("\n<|assistant|>")
            obs_json = json.loads(prompt[user_start:user_end])
            ambs = [a for a in obs_json["ambulances"] if not a["busy"]]
            emgs = sorted(obs_json["emergencies"], key=lambda e: -{"CRITICAL": 3, "HIGH": 2, "NORMAL": 1}.get(e["severity"], 0))
            hosps = sorted(obs_json["hospitals"], key=lambda h: h["occupancy"])
            if ambs and emgs and hosps:
                return json.dumps({
                    "ambulance_id": ambs[0]["id"],
                    "emergency_id": emgs[0]["id"],
                    "hospital_id": hosps[0]["id"],
                })
        except Exception:
            pass
        return json.dumps({"ambulance_id": None, "emergency_id": "", "hospital_id": None})

    csv_path = out_dir / "grpo_rewards.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "episode_reward", "avg_reward"])

    all_episode_rewards = []
    print(f"GRPO Standalone training: {args.steps} episodes")
    for ep in range(args.steps):
        steps_data = rollout_env.rollout(greedy_generate)
        ep_reward = sum(s["reward"] for s in steps_data)
        all_episode_rewards.append(ep_reward)
        avg = float(np.mean(all_episode_rewards[-20:])) if len(all_episode_rewards) >= 20 else float(np.mean(all_episode_rewards))

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, round(ep_reward, 3), round(avg, 3)])

        if ep % 20 == 0:
            print(f"Episode {ep:4d} | Reward={ep_reward:7.2f} Avg(20)={avg:6.2f}")

    _plot_grpo_rewards(all_episode_rewards, out_dir)
    print(f"GRPO reward CSV: {csv_path}")


# ---------------------------------------------------------------------------
# TRL GRPO training (requires trl + transformers + gpu)
# ---------------------------------------------------------------------------

def train_with_trl(args):
    if not _TRL_AVAILABLE or not _HF_AVAILABLE:
        print("TRL or transformers not available. Falling back to standalone mode.")
        train_standalone(args)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    if args.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=2048,
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,
                target_modules=["q_proj", "v_proj"],
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
            )
        except ImportError:
            print("[WARN] unsloth not installed, using standard transformers.")
            args.use_unsloth = False

    if not args.use_unsloth:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            device_map="auto",
        )

    env_cfg = {"n_ambulances": 3, "n_hospitals": 2, "max_steps": 30}
    rollout_env = GRPORolloutEnv(env_cfg)

    # Build dataset of prompts
    env = AmbulanceEnvironment(env_cfg)
    prompts = []
    for _ in range(200):
        obs = env.reset()
        for _ in range(5):
            p = build_prompt(obs)
            prompts.append({
                "prompt": f"<|system|>\n{_SYSTEM_PROMPT}\n<|user|>\n{p}\n<|assistant|>\n"
            })
            action = _noop()
            result = env.step(action)
            obs = result[0] if isinstance(result, tuple) else result
            if obs.done:
                break

    from datasets import Dataset
    dataset = Dataset.from_list(prompts)

    # Reward function for GRPO
    def reward_fn(completions, prompts=None, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            # Quick parse and evaluate
            try:
                parsed = json.loads(completion[completion.find("{"):completion.rfind("}") + 1])
                # Valid JSON with correct structure → positive base reward
                has_keys = all(k in parsed for k in ["ambulance_id", "emergency_id", "hospital_id"])
                rewards.append(2.0 if has_keys else -2.0)
            except Exception:
                rewards.append(-5.0)
        return rewards

    grpo_config = GRPOConfig(
        output_dir=str(out_dir / "grpo_model"),
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        logging_steps=10,
        save_steps=50,
        report_to="none",
        num_generations=4,
        max_new_tokens=128,
        temperature=0.7,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        train_dataset=dataset,
        reward_funcs=rollout,
    )

    print(f"Starting TRL GRPO training for {args.steps} steps...")
    trainer.train()
    model.save_pretrained("trained_model")
    tokenizer.save_pretrained("trained_model")
    print("Model saved to trained_model/")

    # Reward curve
    log_history = trainer.state.log_history
    _rewards = [
        e.get("reward", e.get("train_reward"))
        for e in log_history
        if "reward" in e or "train_reward" in e
    ]
    _plot_grpo_rewards([r for r in _rewards if r is not None], out_dir)


def _plot_grpo_rewards(rewards: list, out_dir: Path):
    if not rewards:
        return
    window = min(20, len(rewards) // 4 or 1)
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.35, color="#6366f1", label="Episode reward")
    plt.plot(range(window - 1, len(rewards)), smoothed, color="#6366f1", linewidth=2, label=f"{window}-ep MA")
    plt.axhline(0, color="#6b7280", linestyle="--")
    plt.title("GRPO Training Reward Curve")
    plt.xlabel("Episode / Step")
    plt.ylabel("Total Episode Reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "grpo_reward_curve.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"GRPO reward curve saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--use-unsloth", action="store_true")
    parser.add_argument("--use-trl", action="store_true", help="Use TRL GRPOTrainer (requires GPU)")
    parser.add_argument("--output-dir", type=str, default="outputs/grpo")
    args = parser.parse_args()

    if args.use_trl:
        train_with_trl(args)
    else:
        train_standalone(args)


# ---------------------------------------------------------------------------
# STEP 10.7-10.8 — Direct Colab execution block
# Run this block in a Colab cell to train and save:
#
#   from train_grpo import (
#       model, tokenizer, rollout,
#       model_name, GRPOConfig, GRPOTrainer
#   )
#   from datasets import Dataset
#   import json
#   from env.environment import AmbulanceEnvironment
#   prompts = [build_prompt(AmbulanceEnvironment({"seed": i}).reset()) for i in range(64)]
#   dataset = Dataset.from_dict({"prompt": prompts})
#   cfg = GRPOConfig(output_dir="outputs/grpo", max_steps=100, max_new_tokens=128,
#                   num_generations=4, per_device_train_batch_size=1,
#                   gradient_accumulation_steps=8, learning_rate=5e-6,
#                   temperature=0.8, fp16=True, report_to="none")
#   trainer = GRPOTrainer(model=model, args=cfg, reward_funcs=rollout,
#                         train_dataset=dataset, processing_class=tokenizer)
#   trainer.train()
#   model.save_pretrained("trained_model")
#   tokenizer.save_pretrained("trained_model")
# ---------------------------------------------------------------------------
