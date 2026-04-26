from models import Observation, Action, Reward
from agents import call_llm
from tasks import TASKS
from reward import compute_reward, compute_consensus

class BoardroomEnv:

    def reset(self, level="easy"):

        cfg = TASKS[level]

        self.state = {
            "task": "",
            "history": [],
            "step_count": 0,
            "budget": cfg["budget"],
            "risk": cfg["risk"],
            "memory": {"successful": [], "failed": []},
            "coalitions": {"growth": [], "risk": [], "tech": []},
            "power": {"CEO": 1.5, "CFO": 1.2, "CTO": 1.0, "Investor": 1.3}
        }

        return Observation(**self.state)

    def set_task(self, task):
        self.state["task"] = task

    def step(self, action: Action):

        self.state["history"].append(action.dict())

        agent = ["CEO","CFO","CTO","Investor"][self.state["step_count"] % 4]

        msg = call_llm(agent, self.state["task"])

        self.state["history"].append({"agent": agent, "message": msg})

        if "risk" in msg:
            self.state["coalitions"]["risk"].append(agent)
        if "growth" in msg:
            self.state["coalitions"]["growth"].append(agent)
        if "tech" in msg:
            self.state["coalitions"]["tech"].append(agent)

        consensus = compute_consensus(self.state["history"])

        reward_val = compute_reward(self.state, consensus, self.state["step_count"])

        if reward_val < 0.5:
            self.state["memory"]["failed"].append(action.message)
        else:
            self.state["memory"]["successful"].append(action.message)

        self.state["step_count"] += 1

        done = consensus or self.state["step_count"] >= 12

        return Observation(**self.state), Reward(
            value=reward_val,
            done=done,
            info={"consensus": consensus}
        ), done, {}