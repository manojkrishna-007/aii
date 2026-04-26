def compute_reward(state, consensus, step):

    base = 1.0 if consensus else 0.3

    memory_penalty = len(state["memory"]["failed"]) * 0.05

    coalition_score = (
        len(state["coalitions"]["growth"]) +
        len(state["coalitions"]["tech"]) -
        len(state["coalitions"]["risk"])
    ) * 0.05

    power_score = sum(state["power"].values()) / 10

    time_penalty = step * 0.02

    reward = base + coalition_score + power_score - memory_penalty - time_penalty

    return max(0.0, min(1.0, reward))


def compute_consensus(history):
    last = [h["message"] for h in history[-4:]]
    return len(set(last)) <= 2