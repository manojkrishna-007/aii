from trl import PPOTrainer, PPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from env import BoardroomEnv
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=PPOConfig(batch_size=4, learning_rate=1e-5)
)

env = BoardroomEnv()

# PHASE 1: exploration (8 episodes)
for ep in range(8):

    obs = env.reset("hard")
    done = False

    while not done:

        act = tokenizer(str(obs), return_tensors="pt")
        out = model.generate(**act, max_length=60)
        action = tokenizer.decode(out[0])

        obs, reward, done, _ = env.step({
            "agent": "CEO",
            "message": action
        })

        trainer.step([action], [torch.tensor(reward.value)])

# PHASE 2: evaluation (4 episodes)
for ep in range(4):

    obs = env.reset("hard")
    done = False

    while not done:

        act = tokenizer(str(obs), return_tensors="pt")
        out = model.generate(**act, max_length=60)
        action = tokenizer.decode(out[0])

        obs, reward, done, _ = env.step({
            "agent": "CEO",
            "message": action
        })

        trainer.step([action], [torch.tensor(reward.value)])

    print("Reward:", reward.value)
    
plt.plot(rewards)
plt.title("RL Learning Curve")
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.savefig("reward_curve.png")