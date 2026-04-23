import numpy as np
import torch
from env.environment import AmbulanceEnvironment as AmbulanceEnv
from rl.state_encoder import StateEncoder
from rl.action_mapper import ActionMapper
from rl.action_mask import ActionMask
from rl.dqn import DQN
from agents.greedy_agent import GreedyAgent

def run_rl_episodes(env, model, encoder, mapper, mask_builder, episodes=20, max_steps=150):
    total_rewards = []
    total_served = 0
    
    for _ in range(episodes):
        obs = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            state = encoder.encode(obs)
            mapper.build_action_space(obs)
            mask = mask_builder.build_mask(mapper)
            
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor).numpy()[0]
            
            q_values[mask == 0] = -1e9
            action_index = int(np.argmax(q_values))
            action = mapper.decode(action_index)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
        total_served += info['metrics']['served']
        
    return np.mean(total_rewards), total_served

def run_greedy_episodes(env, agent, episodes=20, max_steps=150):
    total_rewards = []
    total_served = 0
    
    for _ in range(episodes):
        obs = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                break
        total_rewards.append(episode_reward)
        total_served += info['metrics']['served']
        
    return np.mean(total_rewards), total_served

def main():
    env = AmbulanceEnv()
    encoder = StateEncoder()
    mapper = ActionMapper()
    mask_builder = ActionMask()
    
    # RL Agent Setup
    obs = env.reset()
    mapper.build_action_space(obs)
    state_size = len(encoder.encode(obs))
    action_size = mapper.size()
    
    model = DQN(state_size, action_size)
    try:
        model.load_state_dict(torch.load("dqn_model.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("Please train the model first.")
        return
    model.eval()
    
    # Greedy Agent Setup
    greedy_agent = GreedyAgent()
    
    # Run Comparisons
    print("Evaluating RL Agent...")
    rl_avg_reward, rl_total_served = run_rl_episodes(env, model, encoder, mapper, mask_builder)
    
    print("Evaluating Greedy Agent...")
    greedy_avg_reward, greedy_total_served = run_greedy_episodes(env, greedy_agent)
    
    # Final Output
    print(f"RL Agent Avg Reward: {rl_avg_reward:.2f}")
    print(f"RL Agent Total Served: {rl_total_served}")
    print(f"Greedy Agent Avg Reward: {greedy_avg_reward:.2f}")
    print(f"Greedy Agent Total Served: {greedy_total_served}")

if __name__ == "__main__":
    main()
