from fastapi import FastAPI
from .model import *
from .env_wrapper import server
import os

app = FastAPI()

VISUAL = os.environ.get("VISUAL", "false").lower() == "true"
if VISUAL:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/")
def hello():
    return "This is environment TextWorld."


@app.post("/create")
def create(body: CreateRequestBody):
    return server.create(games_dir=body.games_dir, max_steps=body.max_steps or 50)


@app.post("/step")
def step(body: StepRequestBody):
    return server.step(body.id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.id, data_idx=body.data_idx or 0, game_path=body.game_path)


@app.get("/observation")
def get_observation(id: int):
    return server.get_observation(id)


@app.get("/available_actions")
def get_available_actions(id: int):
    return server.get_available_actions(id)


@app.get("/detail")
def get_detailed_info(id: int):
    return server.get_detailed_info(id)


@app.post("/close")
def close(body: CloseRequestBody):
    return server.close(body.id)

