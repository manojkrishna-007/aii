from pydantic import BaseModel
from typing import List, Dict, Any

class Observation(BaseModel):
    task: str
    history: List[Dict[str, str]]
    step_count: int
    budget: float
    risk: float
    memory: Dict[str, Any]
    coalitions: Dict[str, List[str]]
    power: Dict[str, float]

class Action(BaseModel):
    agent: str
    message: str

class Reward(BaseModel):
    value: float
    done: bool
    info: Dict[str, Any]