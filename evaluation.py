from env import BoardroomEnv

env = BoardroomEnv()

def run(mode):

    total_reward = 0

    for i in range(5):

        obs = env.reset("hard")
        env.set_task("Global expansion under risk")

        done = False

        while not done:

            obs, reward, done, _ = env.step({
                "agent": "CEO",
                "message": "Expand cautiously"
            })

            total_reward += reward.value

    return total_reward / 5


before = run("before")
after = run("after")

print("\nBEFORE TRAINING SCORE:", before)
print("AFTER TRAINING SCORE:", after)
print("IMPROVEMENT:", after - before)