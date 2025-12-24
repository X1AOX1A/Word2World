from pydantic import BaseModel
from typing import Optional


class CreateRequestBody(BaseModel):
    games_dir: Optional[str] = None
    max_steps: Optional[int] = 50


class StepRequestBody(BaseModel):
    id: int
    action: str


class ResetRequestBody(BaseModel):
    id: int
    data_idx: Optional[int] = 0
    game_path: Optional[str] = None


class CloseRequestBody(BaseModel):
    id: int

