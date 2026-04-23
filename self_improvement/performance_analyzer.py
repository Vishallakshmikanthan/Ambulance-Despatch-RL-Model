class PerformanceAnalyzer:

    def __init__(self):
        self.history = []

    def record(self, reward, info):
        """
        Track performance metrics per step
        """
        self.history.append({
            "reward": reward,
            "info": info
        })

    def get_metrics(self):
        """
        Compute summary metrics
        """
        if not self.history:
            return {}

        total_reward = sum(h["reward"] for h in self.history)
        steps = len(self.history)

        return {
            "avg_reward": total_reward / steps,
            "steps": steps
        }

    def reset(self):
        self.history = []
