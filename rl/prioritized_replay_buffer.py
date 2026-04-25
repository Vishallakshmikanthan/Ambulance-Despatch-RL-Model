import numpy as np
import random
from collections import deque


class SimpleReplayBuffer:
    """
    Uniform (non-prioritized) replay buffer.
    Used as fallback when use_per=False.
    """
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Returns: (samples, indices=None, weights=ones)
        Signature matches PrioritizedReplayBuffer for drop-in compatibility.
        """
        samples = random.sample(self.buffer, batch_size)
        indices = None
        weights = np.ones(batch_size, dtype=np.float32)
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """No-op: uniform buffer does not use priorities."""
        pass

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Proportional Prioritized Experience Replay (PER).
    Stores transitions with priorities based on absolute TD-error.

    Parameters:
        capacity  : max transitions to store
        alpha     : prioritisation strength (0 = uniform, 1 = full priority)
        beta      : importance-sampling correction factor (anneals toward 1.0)
    """
    def __init__(self, capacity: int = 20000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001

        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        """Store a transition. New transitions get max existing priority."""
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        """
        Sample a batch proportional to priority^alpha.

        Returns:
            samples : list of (s, a, r, s', done) tuples
            indices : array of sampled buffer positions (for priority update)
            weights : importance-sampling weights (float32 array)
        """
        if len(self.buffer) == 0:
            return [], [], []

        current_priorities = self.priorities[:len(self.buffer)]
        probs = current_priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # IS weights: (1/N * 1/P(i))^beta, normalised by max weight
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # Anneal beta toward 1.0 for full IS correction over time
        self.beta = min(1.0, self.beta + self.beta_increment)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """Update priorities after computing TD-errors."""
        for idx, prio in zip(batch_indices, batch_priorities):
            scalar = prio.item() if hasattr(prio, "item") else float(np.asarray(prio).flat[0])
            self.priorities[idx] = float(abs(scalar)) + 1e-6

    def __len__(self):
        return len(self.buffer)
