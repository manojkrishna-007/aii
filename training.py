import requests

BASE = "http://localhost:7860"

for ep in range(50):
    state = requests.post(BASE + "/reset", json={"level":"hard"}).json()

    done = False

    while not done:
        action = {
            "agent": "CEO",
            "message": "We should expand carefully"
        }

        res = requests.post(BASE + "/step", json=action).json()

        reward = res["reward"]["value"]
        done = res["done"]

    print("Episode:", ep, "Reward:", reward)