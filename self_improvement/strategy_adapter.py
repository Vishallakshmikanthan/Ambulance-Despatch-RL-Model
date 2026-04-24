class StrategyAdapter:

    def __init__(self):
        self.priority_weight = {
            "CRITICAL": 1.0,
            "HIGH": 1.0,
            "NORMAL": 1.0
        }

    def update(self, metrics):
        """
        Adjust strategy based on performance
        """
        if metrics.get("avg_reward", 0) < 0:
            self.priority_weight["CRITICAL"] *= 1.2
            self.priority_weight["HIGH"] *= 1.1

    def adjust_strategy(self, metrics):
        """Fine-grained weight adjustment based on avg_reward signal."""
        if metrics.get("avg_reward", 0) < 0:
            self.priority_weight["CRITICAL"] *= 1.5
        if metrics.get("avg_reward", 0) > 10:
            self.priority_weight["NORMAL"] *= 1.1
        print("[ADAPT] Updated weights:", self.priority_weight)

    def get_weights(self):
        return self.priority_weight
