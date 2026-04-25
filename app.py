"""
app.py — HuggingFace Spaces entry point (also usable standalone).
For the full-featured server see server/app.py.
"""
from fastapi import FastAPI
from env.environment import AmbulanceEnvironment

app = FastAPI(title="Ambulance OpenEnv")

env = AmbulanceEnvironment()


@app.get("/")
def home():
    return {"status": "Ambulance OpenEnv Running"}


@app.get("/reset")
def reset():
    obs = env.reset()
    return {"state": str(obs)}


@app.post("/step")
def step():
    action = {
        "ambulance_id": 0,
        "emergency_id": "E1",
        "hospital_id": 0,
    }
    env.step(action)
    return {"state": str(env.last_info), "reward": env.last_reward}
