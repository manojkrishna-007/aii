import json

class MetricsLogger:
    def __init__(self):
        self.rewards = []
        self.consensus_steps = []

        # behavior tracking (BEFORE / AFTER RL)
        self.behavior_before = {
            "aggressive": 0,
            "risk_averse": 0,
            "technical": 0,
            "analytical": 0,
            "neutral": 0
        }

        self.behavior_after = {
            "aggressive": 0,
            "risk_averse": 0,
            "technical": 0,
            "analytical": 0,
            "neutral": 0
        }

        self.phase = "before"

    def log_reward(self, reward):
        self.rewards.append(reward)

    def log_consensus(self, steps):
        self.consensus_steps.append(steps)

    def log_behavior(self, category):
        if self.phase == "before":
            self.behavior_before[category] += 1
        else:
            self.behavior_after[category] += 1

    def save(self):
        with open("metrics.json", "w") as f:
            json.dump({
                "rewards": self.rewards,
                "consensus_steps": self.consensus_steps,
                "behavior_before": self.behavior_before,
                "behavior_after": self.behavior_after
            }, f, indent=2)