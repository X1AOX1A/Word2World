from typing import Any, Mapping
import requests
import re
from requests.exceptions import RequestException
from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class BabyAIEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": 'You are an exploration master that wants to finish every goal you are given. Every round I will give you an observation, and you have to respond an action and your thought based on the observation to finish the given task. You are placed in a room and you need to accomplish the given goal with actions.\n\nYou can use the following actions: \n\n- turn right \n\n- turn left \n\n- move forward \n\n- go to <obj> <id> \n\n- pick up <obj> <id> \n\n- go through <door> <id>: <door> must be an open door. \n\n- toggle and go through <door> <id>: <door> can be a closed door or a locked door. If you want to open a locked door, you need to carry a key that is of the same color as the locked door. \n\n- toggle: there is a closed or locked door right in front of you and you can toggle it.\nYour response should use the following format:\nThought:\n<Your Thought>\n\nAction:\n<Your Action>',
            }
        ),
        ConversationMessage(
            {
                "from": "gpt",
                "loss": False,
                "value": "OK. I'll follow your instructions and try my best to solve the task.",
            }
        ),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        self.env_id = ok["id"]

    def __len__(self):
        return self.data_len

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

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", action, re.DOTALL)
        if len(action_matches) > 1:
            return StepOutput(
                state="Error: Only one 'Action' is allowed per response. Please adjust your response.",
                reward=0,
                done=False,
            )
        action = action_matches[-1] if action_matches else ""
        action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
        action = " ".join(action.split()).strip()
        response = self._post("step", {"action": action})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "score": response["score"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["score"],
            done=response["done"],
        )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": data_idx})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "score": response["score"],
            "done": response["done"],
        }
        return response

    def close(self):
        response = self._post("close",{})
        return response

class BabyAITask(BaseTask):
    env_client_cls = BabyAIEnvClient
    env_name = "BabyAI"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
