import numpy as np
import torch

from env.environment import AmbulanceEnvironment as AmbulanceEnv
from rl.state_encoder import StateEncoder
from rl.action_mapper import ActionMapper
from rl.action_mask import ActionMask
from rl.dqn import DQN

def main():
    # 2. INITIALIZE COMPONENTS
    env = AmbulanceEnv()

    encoder = StateEncoder()
    mapper = ActionMapper()
    mask_builder = ActionMask()

    # Get dimensions from initial reset
    obs = env.reset()
    mapper.build_action_space(obs)
    state = encoder.encode(obs)

    state_size = len(state)
    action_size = mapper.size()

    # 3. LOAD MODEL
    print(f"Loading model with state_size={state_size}, action_size={action_size}...")
    model = DQN(state_size, action_size)
    try:
        model.load_state_dict(torch.load("dqn_model.pth", map_location=torch.device('cpu')))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: dqn_model.pth not found. Please run train.py first.")
        return

    model.eval()

    # 4. EVALUATION SETTINGS
    episodes = 20
    max_steps = 100

    total_rewards = []
    success_counts = []

    print(f"Starting evaluation for {episodes} episodes...")

    # 5. RUN EVALUATION
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        success = 0

        for step in range(max_steps):
            # Encode state
            state = encoder.encode(obs)

            # Build fixed-size mapping and mask
            mapper.build_action_space(obs)
            mask = mask_builder.build_mask(mapper)

            # Convert to tensor and get Q-values
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                q_values = model(state_tensor).detach().cpu().numpy()[0]

            # Apply action masking
            q_values[mask == 0] = -1e9

            # Select best valid action (greedy)
            action_index = int(np.argmax(q_values))

            # Decode to ActionModel
            action = mapper.decode(action_index)

            # Step environment
            next_obs, reward, done, info = env.step(action)

            total_reward += reward

            # Performance metric: Success if dispatch reward is high
            if reward > 5:
                success += 1

            obs = next_obs

            if done:
                break

        total_rewards.append(total_reward)
        success_counts.append(success)

        print(f"Episode {episode+1} Reward: {total_reward:.2f}, Successes: {success}")

    # 6. FINAL METRICS
    print("\n" + "="*25)
    print("===== FINAL RESULTS =====")
    print("="*25)
    print(f"Average Reward:    {np.mean(total_rewards):.2f}")
    print(f"Max Reward:        {np.max(total_rewards):.2f}")
    print(f"Min Reward:        {np.min(total_rewards):.2f}")
    print(f"Average Success:   {np.mean(success_counts):.2f}")
    print("="*25)

if __name__ == "__main__":
    print("Starting evaluation...")
    main()
