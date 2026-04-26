from fastapi import FastAPI
from env import BoardroomEnv
from models import Action
from agents import call_llm

app = FastAPI()
env = BoardroomEnv()

@app.post("/reset")
def reset(level: str = "easy"):
    obs = env.reset(level)

    # 5th LLM generates task
    task = call_llm("Director", "Create boardroom problem in 10 words")
    env.set_task(task)

    return obs.dict()

@app.post("/step")
def step(action: dict):
    obs, reward, done, info = env.step(Action(**action))
    return {"observation": obs.dict(), "reward": reward.dict(), "done": done}

@app.get("/state")
def state():
    return env.state