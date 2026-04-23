import copy


class LookaheadPlanner:

    def __init__(self, horizon=3):
        self.horizon = horizon

    def simulate(self, env, action):
        """
        Simulate future rewards for a given action
        """
        sim_env = copy.deepcopy(env)

        total_reward = 0

        obs, reward, done, info = sim_env.step(action)
        total_reward += reward

        steps = 1

        while not done and steps < self.horizon:
            # simple greedy fallback during simulation
            from env.models import ActionModel
            noop_action = ActionModel(is_noop=True)
            obs, reward, done, info = sim_env.step(noop_action)
            total_reward += reward
            steps += 1

        return total_reward
