from typing import Any, Mapping, Dict, List, Optional

import requests
from requests.exceptions import RequestException
from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput

class SearchQAEnvClient(BaseEnvClient):
    conversation_start = (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value":"""You must always reason inside <think>...</think> first; if you lack knowledge, issue a <search>...</search> and then stop; do not generate <information> or <answer> yet; wait for external input between <information>...</information> before continuing; resume only when new <information> is given; do not skip steps or anticipate answers early.""",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.id = 0
        data = dict()
        data['id'] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> Dict[str, Any]:
        question = self._get("observation")
        return question

    def step(self, action: str) -> StepOutput:
        # action is the original output of llm
        # print(f"Action: {action}")
        response = self._post("step", {"action": action})
        # print(response)
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, id: int) -> Dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response
    
    def close(self):
        response = self._post("close", {})
        return response

class SearchQATask(BaseTask):
    env_client_cls = SearchQAEnvClient
    env_name = "SearchQA"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
