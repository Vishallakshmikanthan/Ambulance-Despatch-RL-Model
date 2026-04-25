"""
colab_train.py — TRL-compatible Colab training demo using GPT-2 + AmbulanceEnvironment.
Run in Google Colab or locally after: pip install transformers torch
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from env.environment import AmbulanceEnvironment

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.eval()

env = AmbulanceEnvironment()


def run_episode():
    obs = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        prompt = f"Observation: {obs}\nAction:"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Structured action — GPT-2 output is used as context only;
        # a real TRL loop would parse the generated text into an action.
        action = {"ambulance_id": 0, "emergency_id": "E1", "hospital_id": 0}

        obs = env.step(action)
        reward = env.last_reward
        done = env.last_done
        total_reward += reward

    return total_reward


if __name__ == "__main__":
    for i in range(5):
        reward = run_episode()
        print(f"Episode {i}, Reward: {reward:.2f}")
