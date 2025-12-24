from typing import Any, Mapping
import os

import re
import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class TextworldEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": 'You are playing a text-based interactive fiction game (TextWorld).\nYou will receive observations describing the current state. When available, a list of admissible actions may be provided.\nAlways output strictly in the following format:\n"Thought:\n<your reasoning>\n\nAction:\n<the single action to take>"\nGuidelines:\n- Prefer actions from admissible commands when provided.\n- If no list is provided, issue a valid single command (e.g., "look", "inventory", "open door", "go north", "take key").\n- Avoid invalid or multiple actions in one step.\n',
            }
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "Understood. I will respond with one valid action per turn.",
            }
        ),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        games_dir: str = "data/textworld/games",
        max_steps: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        payload = {"games_dir": games_dir, "max_steps": max_steps}
        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout, json=payload)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        if "error" in ok:
            raise RequestException(f"Failed to create environment: {ok['error']}")
        self.env_id = ok["id"]
        self.info = {
            "observation": ok.get("observation", ""),
            "reward": 0,
            "done": False,
            "available_actions": [],
        }

    def __len__(self):
        return self.data_len

    # ------------------ Internal HTTP Wrappers ------------------ #
    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    # ------------------ Environment Interaction Methods ------------------ #
    def observe(self) -> str:
        # Prefer cached observation and include admissible actions if available
        obs = self.info.get("observation", "")
        actions = self.info.get("available_actions", []) or []
        if actions:
            return f"{obs}\nAVAILABLE ACTIONS: {', '.join(actions)}"
        return obs

    def step(self, action: str) -> StepOutput:
        # Extract the single 'Action:' line from the model's response
        action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", action, re.DOTALL)
        if len(action_matches) > 1:
            return StepOutput(
                state="Error: Only one 'Action' is allowed per response. Please adjust your response.",
                reward=0,
                done=False,
            )
        action = action_matches[-1] if action_matches else ""
        action = action.strip()
        if not action:
            return StepOutput(
                state="Error: Missing action. Please provide a single action.",
                reward=0,
                done=False,
            )

        response = self._post("step", {"action": action})
        if "error" in response:
            return StepOutput(
                state=f"Error from server: {response['error']}",
                reward=0,
                done=False,
            )
        # Keep full server info including TextWorld-specific score fields.
        self.info = {
            "observation": response.get("observation", ""),
            "reward": response.get("reward", 0),
            "done": response.get("done", False),
            "available_actions": response.get("available_actions", []),
            "score": response.get("score", 0),
            "max_score": response.get("max_score", 0),
            "won": response.get("won", False),
            "lost": response.get("lost", False),
        }

        # For success computation without touching common evaluator code:
        # On the final step, convert reward to 100 if final score == max_score, else 0.
        # This matches the default success rule (reward == 1 or reward == 100).
        # Intermediate steps report 0, only the final step contributes.
        score_val = float(response.get("score", 0) or 0)
        max_score_val = float(response.get("max_score", 0) or 0)
        done_flag = bool(self.info["done"])  # final step indicator
        if done_flag and max_score_val > 0 and score_val >= max_score_val:
            reported_reward = 100.0
        else:
            reported_reward = 0.0
        return StepOutput(
            state=self.info["observation"],
            reward=reported_reward,
            done=bool(done_flag),
        )

    def reset(self, data_idx: int = 0, game_path: Any = None) -> dict[str, Any]:
        # Accept SciWorld-style argument name for consistency
        payload: dict[str, Any] = {"data_idx": data_idx}
        if game_path is not None:
            payload["game_path"] = game_path
        response = self._post("reset", payload)
        # If reset failed on the server, fail fast to avoid stepping an uninitialized env.
        if isinstance(response, dict) and "error" in response:
            raise RequestException(f"Reset failed: {response['error']}")
        self.info.update(
            {
                "observation": response.get("observation", ""),
                "reward": 0,
                "done": False,
                "available_actions": response.get("available_actions", []),
            }
        )
        return response

    def close(self):
        try:
            return self._post("close", {})
        except Exception:
            return {"closed": False}


class TextworldTask(BaseTask):
    env_client_cls = TextworldEnvClient
    env_name = "TextWorld"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
