import numpy as np
import torch
import os
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe in headless / CI environments
import matplotlib.pyplot as plt

from env.environment import AmbulanceEnvironment as AmbulanceEnv
from rl.state_encoder import StateEncoder
from rl.action_mapper import ActionMapper
from rl.action_mask import ActionMask
from rl.rl_agent import DQNAgent
from rl.demand_predictor import DemandPredictor

def main():
    # 1. PARSE CLI ARGUMENTS
    parser = argparse.ArgumentParser(description="Train the Ambulance Dispatch DQN agent")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=150, help="Max environment steps per episode")
    parser.add_argument("--no-dueling", action="store_true", help="Disable Dueling DQN (use StandardDQN)")
    parser.add_argument("--no-per", action="store_true", help="Disable Prioritized Replay (use uniform)")
    parser.add_argument("--no-soft-update", action="store_true", help="Use hard target update instead of soft")
    parser.add_argument("--normalize-rewards", action="store_true", help="Enable z-score reward normalization")
    args = parser.parse_args()

    # 2. INITIALIZE COMPONENTS
    env = AmbulanceEnv()

    encoder = StateEncoder()
    mapper = ActionMapper()
    mask_builder = ActionMask()
    predictor = DemandPredictor()

    # Dummy initial observation to get dimensions
    obs = env.reset()
    mapper.build_action_space(obs)
    state = encoder.encode(obs)

    state_size = len(state)
    action_size = mapper.size()

    # Initialize DQN Agent with feature flags from CLI
    agent = DQNAgent(
        state_size,
        action_size,
        use_dueling=not args.no_dueling,
        use_per=not args.no_per,
        use_soft_update=not args.no_soft_update,
        normalize_rewards=args.normalize_rewards
    )

    # 3. TRAINING PARAMETERS
    episodes = args.episodes
    batch_size = 128
    max_steps = args.max_steps

    agent.batch_size = batch_size
    rewards_history = []
    episode_rewards = []
    best_reward = -np.inf

    print("Starting training loop...")

    # 4. TRAINING LOOP
    for episode in range(episodes):

        obs = env.reset()
        total_reward = 0

        for step in range(max_steps):
            # Encode current state
            state = encoder.encode(obs)

            # Build fixed-size action space mapping
            mapper.build_action_space(obs)

            # Generate validity mask
            mask = mask_builder.build_mask(mapper)

            # Select epsilon-greedy action
            action_index = agent.act(state, mask)

            # Decode to ActionModel
            action = mapper.decode(action_index)

            # Step environment
            next_obs, reward, done, info = env.step(action)

            # Apply Reward Shaping
            # Note: env.step() already includes coordination penalty and future-aware bonus.
            # get_priority_weighted_reward scales by severity for additional signal strength.
            # get_coordinated_reward adds coverage-diversity spread bonus (not in env.step).
            reward = agent.get_priority_weighted_reward(obs, action, reward)
            reward = agent.get_coordinated_reward(obs, action, reward)

            # Update demand predictor and hotspots
            predictor.update(next_obs)

            # Encode next state
            next_state = encoder.encode(next_obs)

            # Store transition in replay buffer (normalises reward if flag set)
            agent.store(state, action_index, reward, next_state, done)

            # Perform weight update session
            agent.train_step()

            # Advance state
            obs = next_obs
            total_reward += reward

            if done:
                break

        # Decay epsilon once per episode to match user milestones
        agent.decay_epsilon(episode)
        
        # Track history
        rewards_history.append(total_reward)
        episode_rewards.append(total_reward)
        moving_avg = np.mean(rewards_history[-50:])
        
        # Periodic logging
        if (episode + 1) % 1 == 0:
            success = info.get("metrics", {}).get("successful_dispatches", 0)
            print(f"Episode {episode+1}:")
            print(f"Reward: {total_reward:.1f}")
            print(f"Success: {success}")
            print(f"Avg Reward: {moving_avg:.1f}, Epsilon: {agent.epsilon:.3f}")
            print("-" * 20)
        
        # Save best model
        if moving_avg > best_reward:
            best_reward = moving_avg
            torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
            print(f"--- New Best Model Saved (Avg Reward: {best_reward:.1f}) ---")

    print(f"Training complete. Best Avg Reward: {best_reward:.1f}")

    # 5. PLOT RESULTS
    window = 50
    moving_avg_list = []

    for i in range(len(episode_rewards)):
        if i < window:
            moving_avg_list.append(np.mean(episode_rewards[:i+1]))
        else:
            # User specified range [i-window:i]
            moving_avg_list.append(np.mean(episode_rewards[i-window:i]))

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, color='skyblue', alpha=0.4, label='Raw Reward')
    plt.plot(moving_avg_list, color='royalblue', linewidth=2, label=f'Moving Avg ({window})')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Ambulance Dispatch Training: Learning Trend")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.close()
    print("Training curve saved to training_curve.png")

if __name__ == "__main__":
    main()
