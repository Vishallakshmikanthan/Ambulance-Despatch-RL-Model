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
from agents.fleet_agent import AmbulanceQAgent
from multi_agent.coordinator import MultiAgentCoordinator
from long_horizon.long_horizon_env import LongHorizonAmbulanceEnvironment
from long_horizon.history_encoder import HistoryEncoder
from self_improvement.weakness_detector import WeaknessDetector
from self_improvement.adversarial_generator import AdversarialScenarioGenerator, ScenarioConfig
from self_improvement.self_play_trainer import SelfPlayTrainer

def main():
    # 1. PARSE CLI ARGUMENTS
    parser = argparse.ArgumentParser(description="Train the Ambulance Dispatch DQN agent")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=150, help="Max environment steps per episode")
    parser.add_argument("--no-dueling", action="store_true", help="Disable Dueling DQN (use StandardDQN)")
    parser.add_argument("--no-per", action="store_true", help="Disable Prioritized Replay (use uniform)")
    parser.add_argument("--no-soft-update", action="store_true", help="Use hard target update instead of soft")
    parser.add_argument("--normalize-rewards", action="store_true", help="Enable z-score reward normalization")
    parser.add_argument("--marl", action="store_true", help="Use multi-agent RL (one DQN per ambulance)")
    parser.add_argument("--long-horizon", action="store_true", help="Use LongHorizonAmbulanceEnvironment (500-step episodes with surges)")
    parser.add_argument("--self-play", action="store_true", help="Enable self-improvement loop (weakness detection + adversarial scenarios)")
    parser.add_argument("--selfplay-interval", type=int, default=200, help="Run self-play loop every N episodes")
    args = parser.parse_args()

    if args.marl:
        _train_marl(args)
    elif getattr(args, "long_horizon", False):
        _train_long_horizon(args)
    elif getattr(args, "self_play", False):
        _train_selfplay(args)
    else:
        _train_single(args)


# ---------------------------------------------------------------------------
# Multi-agent training loop (Steps 7.2 – 7.5)
# ---------------------------------------------------------------------------

def _train_marl(args):
    """
    MARL training loop.

    - One AmbulanceQAgent is created per ambulance in the fleet (Step 7.2).
    - Each agent independently selects an action each step.
    - Global env reward is split equally across agents (Step 7.3).
    - Agents targeting the same emergency receive an extra –5 penalty (Step 7.4).
    - Per-agent transitions are stored and a learning step is triggered each
      environment step (Step 7.5).
    """
    print("=== MARL Training Mode ===")

    env = AmbulanceEnv()
    obs = env.reset()

    n_ambulances = len(obs.ambulances)
    coordinator = MultiAgentCoordinator(n_ambulances=n_ambulances)

    episodes = args.episodes
    max_steps = args.max_steps

    team_rewards_history: list[float] = []
    best_avg = -float("inf")

    os.makedirs("outputs/marl", exist_ok=True)

    for episode in range(episodes):
        obs = env.reset()
        coordinator.reset()
        team_reward = 0.0

        for _ in range(max_steps):
            # Each agent picks an action independently
            actions = coordinator.marl_act(obs)

            # Decode each agent's action to an ActionModel and step the env
            # with the first valid non-noop action (the env accepts a single action)
            decoded = coordinator.decode_actions(actions)
            chosen_action = next(
                (a for a in decoded.values() if not getattr(a, "is_noop", False)),
                list(decoded.values())[0],
            )

            result = env.step(chosen_action)
            if isinstance(result, tuple):
                next_obs, reward, done, info = result
            else:
                next_obs = result
                reward = getattr(env, "last_reward", 0.0)
                done = getattr(env, "done", False)
                info = {}

            # Distribute reward and trigger per-agent learning
            coordinator.marl_learn(float(reward), next_obs, done)

            team_reward += float(reward)
            obs = next_obs
            if done:
                break

        team_rewards_history.append(team_reward)
        moving_avg = float(np.mean(team_rewards_history[-50:]))

        if (episode + 1) % 50 == 0:
            epsilons = [a.epsilon for a in coordinator.fleet_agents.values()]
            print(
                f"[MARL] Episode {episode+1} | Team Reward: {team_reward:.1f} "
                f"| Avg(50): {moving_avg:.1f} | ε: {min(epsilons):.3f}–{max(epsilons):.3f}"
            )

        if moving_avg > best_avg:
            best_avg = moving_avg
            for agent_id, agent in coordinator.fleet_agents.items():
                agent.save(f"outputs/marl/agent_{agent_id}.pt")
            print(f"--- New best saved (Avg: {best_avg:.1f}) ---")

    print(f"MARL training complete. Best Avg Reward: {best_avg:.1f}")

    # Plot team reward curve
    window = 50
    avgs = [
        float(np.mean(team_rewards_history[max(0, i - window):i + 1]))
        for i in range(len(team_rewards_history))
    ]
    plt.figure(figsize=(10, 5))
    plt.plot(team_rewards_history, color="skyblue", alpha=0.4, label="Team Reward")
    plt.plot(avgs, color="royalblue", linewidth=2, label=f"Moving Avg ({window})")
    plt.xlabel("Episodes")
    plt.ylabel("Team Reward")
    plt.title("MARL Ambulance Dispatch — Team Learning Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/marl/marl_training_curve.png", dpi=150)
    plt.close()
    print("Training curve saved to outputs/marl/marl_training_curve.png")


# ---------------------------------------------------------------------------
# Self-play training loop (Step 9.5)
# ---------------------------------------------------------------------------

def _train_selfplay(args):
    """
    Self-improvement training loop.

    Every --selfplay-interval episodes the SelfPlayTrainer:
      1. Evaluates the agent on N random scenarios.
      2. Identifies low-scoring failure clusters (WeaknessDetector).
      3. Generates adversarial configs targeting those clusters.
      4. Trains the agent on the new configs.
      5. Adds the surviving configs back into the training pool.
    """
    print("=== Self-Play Training Mode ===")

    base_config = ScenarioConfig(
        n_ambulances=5,
        n_hospitals=3,
        max_steps=args.max_steps,
    )

    encoder = StateEncoder()
    mapper = ActionMapper()
    mask_builder = ActionMask()

    def env_factory(cfg: ScenarioConfig = None):
        c = cfg.to_env_dict() if cfg is not None else {}
        return AmbulanceEnv(c)

    # Bootstrap dimensions from a fresh env
    _boot_env = env_factory()
    _boot_obs = _boot_env.reset()
    mapper.build_action_space(_boot_obs)
    state_size = len(encoder.encode(_boot_obs))
    action_size = mapper.size()

    agent = DQNAgent(
        state_size,
        action_size,
        use_dueling=not args.no_dueling,
        use_per=not args.no_per,
        use_soft_update=not args.no_soft_update,
        normalize_rewards=args.normalize_rewards,
    )

    def action_mapper_fn(obs):
        mapper.build_action_space(obs)
        mask = mask_builder.build_mask(mapper)
        state = encoder.encode(obs)
        idx = agent.act(state, mask)
        return idx, mapper.decode(idx), mask

    def score_fn(metrics: dict) -> float:
        served = metrics.get("served", 0)
        missed = metrics.get("missed", 0)
        total = served + missed
        return served / total if total > 0 else 0.0

    trainer = SelfPlayTrainer(
        env_factory=env_factory,
        agent=agent,
        action_mapper=action_mapper_fn,
        score_fn=score_fn,
        n_eval=20,
        targeted_episodes=30,
        output_dir="outputs/selfplay",
        base_config=base_config,
    )

    selfplay_interval = getattr(args, "selfplay_interval", 200)
    episodes = args.episodes
    max_steps = args.max_steps
    rewards_history: list[float] = []
    best_reward = -float("inf")
    # Seed training config pool
    config_pool: list[ScenarioConfig] = [base_config]

    os.makedirs("outputs/selfplay", exist_ok=True)

    for episode in range(episodes):
        # Pick a config from the pool (round-robin)
        cfg = config_pool[episode % len(config_pool)]
        env = env_factory(cfg)
        obs = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            mapper.build_action_space(obs)
            mask = mask_builder.build_mask(mapper)
            state = encoder.encode(obs)
            action_index = agent.act(state, mask)
            action = mapper.decode(action_index)

            result = env.step(action)
            if isinstance(result, tuple):
                next_obs, reward, done, info = result
            else:
                next_obs = result
                reward = getattr(env, "last_reward", 0.0)
                done = getattr(env, "done", False)
                info = {}

            next_state = encoder.encode(next_obs)
            agent.store(state, action_index, float(reward), next_state, done)
            agent.train_step()

            obs = next_obs
            total_reward += float(reward)
            if done:
                break

        agent.decay_epsilon(episode)
        rewards_history.append(total_reward)
        moving_avg = float(np.mean(rewards_history[-50:]))

        # --- Step 9.5: Self-play loop every N episodes ---
        if (episode + 1) % selfplay_interval == 0:
            print(f"[SelfPlay] Running improvement cycle at episode {episode+1}...")
            iteration_metrics = trainer.run(n_iterations=1, verbose=False)
            # trainer.run returns list of dicts with targeted ScenarioConfigs
            # Pull newly generated configs from the generator directly
            new_configs = trainer.generator.generate_scenarios(
                n=10, base_config=base_config
            )
            config_pool.extend(new_configs)
            print(f"[SelfPlay] Pool size: {len(config_pool)} configs")

        if (episode + 1) % 50 == 0:
            print(
                f"[SP] Episode {episode+1} | Reward: {total_reward:.1f} "
                f"| Avg(50): {moving_avg:.1f} | Pool: {len(config_pool)} | ε: {agent.epsilon:.3f}"
            )

        if moving_avg > best_reward:
            best_reward = moving_avg
            torch.save(agent.policy_net.state_dict(), "outputs/selfplay/best_model.pt")

    print(f"Self-play training complete. Best Avg: {best_reward:.1f}")

    window = 50
    avgs = [
        float(np.mean(rewards_history[max(0, i - window):i + 1]))
        for i in range(len(rewards_history))
    ]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, color="skyblue", alpha=0.4, label="Reward")
    plt.plot(avgs, color="royalblue", linewidth=2, label=f"Moving Avg ({window})")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Self-Play Ambulance Dispatch — Learning Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/selfplay/selfplay_training_curve.png", dpi=150)
    plt.close()
    print("Training curve saved to outputs/selfplay/selfplay_training_curve.png")


# ---------------------------------------------------------------------------
# Long-horizon training loop (Step 8.6)
# ---------------------------------------------------------------------------

def _train_long_horizon(args):
    """
    Training loop using LongHorizonAmbulanceEnvironment.
    History encoding is appended to the base state vector each step.
    """
    print("=== Long-Horizon Training Mode ===")

    lh_config = {
        "max_steps": 500,
        "n_ambulances": 5,
        "n_hospitals": 3,
        "enable_surges": True,
        "surge_schedule": [
            (100, 140, 0, 2.5),
            (250, 290, 2, 2.0),
            (400, 440, 1, 2.8),
        ],
    }

    env = LongHorizonAmbulanceEnvironment(lh_config)
    history_encoder = HistoryEncoder(window=50)

    encoder = StateEncoder()
    mapper = ActionMapper()
    mask_builder = ActionMask()

    obs = env.reset()
    history_encoder.reset() if hasattr(history_encoder, "reset") else None
    mapper.build_action_space(obs)
    base_state = encoder.encode(obs)
    history_features = history_encoder.encode()
    state_size = len(base_state) + len(history_features)
    action_size = mapper.size()

    agent = DQNAgent(
        state_size,
        action_size,
        use_dueling=not args.no_dueling,
        use_per=not args.no_per,
        use_soft_update=not args.no_soft_update,
        normalize_rewards=args.normalize_rewards,
    )

    episodes = args.episodes
    max_steps = lh_config["max_steps"]
    rewards_history: list[float] = []
    best_reward = -float("inf")

    os.makedirs("outputs/long_horizon", exist_ok=True)

    for episode in range(episodes):
        obs = env.reset()
        if hasattr(history_encoder, "reset"):
            history_encoder.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            base_state = encoder.encode(obs)
            history_features = history_encoder.encode()
            state = np.concatenate([base_state, history_features]).astype(np.float32)

            mapper.build_action_space(obs)
            mask = mask_builder.build_mask(mapper)
            action_index = agent.act(state, mask)
            action = mapper.decode(action_index)

            result = env.step(action)
            if isinstance(result, tuple):
                next_obs, reward, done, info = result
            else:
                next_obs = result
                reward = getattr(env, "last_reward", 0.0)
                done = getattr(env, "done", False)
                info = {}

            # Update history encoder with raw observation data
            history_encoder.update(next_obs, float(reward), getattr(env, "metrics", {}))

            next_base = encoder.encode(next_obs)
            next_history = history_encoder.encode()
            next_state = np.concatenate([next_base, next_history]).astype(np.float32)

            agent.store(state, action_index, float(reward), next_state, done)
            agent.train_step()

            obs = next_obs
            total_reward += float(reward)
            if done:
                break

        agent.decay_epsilon(episode)
        rewards_history.append(total_reward)
        moving_avg = float(np.mean(rewards_history[-50:]))

        if (episode + 1) % 50 == 0:
            print(
                f"[LH] Episode {episode+1} | Reward: {total_reward:.1f} "
                f"| Avg(50): {moving_avg:.1f} | ε: {agent.epsilon:.3f}"
            )

        if moving_avg > best_reward:
            best_reward = moving_avg
            torch.save(agent.policy_net.state_dict(), "outputs/long_horizon/best_model.pt")
            print(f"--- New best saved (Avg: {best_reward:.1f}) ---")

    print(f"Long-horizon training complete. Best Avg: {best_reward:.1f}")

    window = 50
    avgs = [
        float(np.mean(rewards_history[max(0, i - window):i + 1]))
        for i in range(len(rewards_history))
    ]
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, color="skyblue", alpha=0.4, label="Reward")
    plt.plot(avgs, color="royalblue", linewidth=2, label=f"Moving Avg ({window})")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Long-Horizon Ambulance Dispatch — Learning Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("outputs/long_horizon/lh_training_curve.png", dpi=150)
    plt.close()
    print("Training curve saved to outputs/long_horizon/lh_training_curve.png")


# ---------------------------------------------------------------------------
# Original single-agent training loop (unchanged)
# ---------------------------------------------------------------------------

def _train_single(args):
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

    # Self-play integration (Step 9.5) — active when --self-play flag set
    selfplay_enabled = getattr(args, "self_play", False)
    selfplay_interval = getattr(args, "selfplay_interval", 200)
    _sp_generator = AdversarialScenarioGenerator(failure_threshold=0.5) if selfplay_enabled else None
    _sp_detector = WeaknessDetector(failure_threshold=0.5) if selfplay_enabled else None
    _sp_results: list = []   # accumulates (ScenarioConfig, score) pairs between cycles
    _sp_base_config = ScenarioConfig(n_ambulances=env.n_ambulances,
                                     n_hospitals=env.n_hospitals,
                                     max_steps=max_steps) if selfplay_enabled else None

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

        # Step 9.5 — Self-play cycle every N episodes
        if selfplay_enabled:
            _sp_results.append((_sp_base_config, float(total_reward) / max(max_steps, 1)))
            if (episode + 1) % selfplay_interval == 0 and _sp_results:
                report = _sp_detector.analyze(_sp_results)
                new_configs = _sp_generator.generate_scenarios(n=10, base_config=_sp_base_config)
                print(f"[SelfPlay] Ep {episode+1}: {len(report.clusters)} weakness clusters, "
                      f"{len(new_configs)} new scenarios generated")
                # Run agent on a subset of new configs for targeted experience
                for sc in new_configs[:5]:
                    _sp_env = AmbulanceEnv(sc.to_env_dict())
                    _sp_obs = _sp_env.reset()
                    for _ in range(sc.max_steps):
                        mapper.build_action_space(_sp_obs)
                        _sp_mask = mask_builder.build_mask(mapper)
                        _sp_state = encoder.encode(_sp_obs)
                        _sp_idx = agent.act(_sp_state, _sp_mask)
                        _sp_result = _sp_env.step(mapper.decode(_sp_idx))
                        if isinstance(_sp_result, tuple):
                            _sp_next_obs, _sp_r, _sp_done, _ = _sp_result
                        else:
                            _sp_next_obs = _sp_result
                            _sp_r = getattr(_sp_env, "last_reward", 0.0)
                            _sp_done = getattr(_sp_env, "done", False)
                        _sp_next_state = encoder.encode(_sp_next_obs)
                        agent.store(_sp_state, _sp_idx, float(_sp_r), _sp_next_state, _sp_done)
                        agent.train_step()
                        _sp_obs = _sp_next_obs
                        if _sp_done:
                            break
                _sp_results.clear()
        
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
