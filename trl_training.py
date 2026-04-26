from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from env import BoardroomEnv
from metrics_logger import MetricsLogger
from behavior_tracker import classify_behavior
import torch
import matplotlib.pyplot as plt

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=PPOConfig(batch_size=4, learning_rate=1e-5)
)

env = BoardroomEnv()
logger = MetricsLogger()

before_times = []
after_times = []

def run_episode(phase):

    obs = env.reset("hard")
    done = False
    reward_last = 0

    while not done:

        act = tokenizer(str(obs), return_tensors="pt")
        out = model.generate(**act, max_length=60)
        action = tokenizer.decode(out[0])

        obs, reward, done, _ = env.step({
            "agent": "CEO",
            "message": action
        })

        trainer.step([action], [torch.tensor(reward.value)])

        reward_last = reward.value

        # 🧠 behavior logging
        category = classify_behavior(action)
        logger.log_behavior(category)

        # 🧠 time-to-consensus logging
        if reward.info and reward.info.get("time_to_consensus") is not None:
            ttc = reward.info["time_to_consensus"]

            if logger.phase == "before":
                before_times.append(ttc)
            else:
                after_times.append(ttc)

    logger.log_reward(reward_last)

# --------------------
# BEFORE TRAINING
# --------------------
logger.phase = "before"
for ep in range(8):
    run_episode("before")

# --------------------
# AFTER TRAINING
# --------------------
logger.phase = "after"
for ep in range(4):
    run_episode("after")

# --------------------
# SAVE METRICS
# --------------------
logger.save()

# --------------------
# REWARD CURVE
# --------------------
plt.plot(logger.rewards)
plt.title("RL Reward Curve")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.savefig("reward_curve.png")

# --------------------
# CONSENSUS IMPROVEMENT
# --------------------
plt.clf()
plt.plot(before_times, label="Before Training")
plt.plot(after_times, label="After Training")
plt.title("Time-to-Consensus Improvement")
plt.xlabel("Episodes")
plt.ylabel("Steps")
plt.legend()
plt.savefig("consensus_improvement.png")

# --------------------
# BEHAVIOR SHIFT
# --------------------
import numpy as np

labels = list(logger.behavior_before.keys())

before_vals = [logger.behavior_before[k] for k in labels]
after_vals = [logger.behavior_after[k] for k in labels]

x = np.arange(len(labels))

plt.clf()
plt.bar(x - 0.2, before_vals, width=0.4, label="Before")
plt.bar(x + 0.2, after_vals, width=0.4, label="After")

plt.xticks(x, labels)
plt.title("Behavior Shift After RL Training")
plt.legend()
plt.savefig("behavior_shift.png")