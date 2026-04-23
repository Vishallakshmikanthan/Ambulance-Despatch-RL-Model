def grade_easy(history):
    served = sum(1 for h in history if h["reward"] > 0)
    total = len(history)

    if total == 0:
        return 0.0

    return min(served / total, 1.0)


def grade_medium(history):
    total_reward = sum(h["reward"] for h in history)

    score = total_reward / (len(history) * 10)

    return max(0.0, min(score, 1.0))


def grade_hard(history):
    positive = sum(1 for h in history if h["reward"] > 5)
    negative = sum(1 for h in history if h["reward"] < 0)

    total = positive + negative

    if total == 0:
        return 0.0

    score = positive / total

    return max(0.0, min(score, 1.0))
