"""
WebarenaEnvServer
"""

import json
import pickle
import re
from pathlib import Path
import time
from typing import Any, Optional

from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_id_based_action,
    create_stop_action,
)
from browser_env.actions import ActionParsingError
from browser_env.env_config import URL_MAPPINGS
from browser_env.helper_functions import RenderHelper, get_action_description
from browser_env.utils import Observation
from evaluation_harness import evaluator_router


from multiprocessing.connection import Connection
from typing import Any, TypedDict

from browser_env import (
    Action,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from evaluation_harness import evaluator_router
import asyncio

# ===== commands =====
class WASubprocessCommand(TypedDict, total=False):
    cmd: str
    data: Any


# ===== state =====


def wa_entrypoint(pipe: Connection):
    """Main entrypoint for the subprocess.

    Creates the environment and listens for commands from the pipe until a STOP command is received.

    POSTCONDITION: Every received command will send back exactly one response.
    RESTRICTION: No commands will be sent in parallel.
    """
    env_config_file = None
    wa_env = ScriptBrowserEnv(
        headless=True,
        slow_mo=0,
        observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": 1280, "height": 720},
        sleep_after_execution=1.5,
    )
    running = True

    while running:
        # recv
        retval = None
        data: WASubprocessCommand = pipe.recv()

        # process
        try:
            match data:
                case {"cmd": "close"}:
                    running = False
                    # retval = wa_env.close()
                # case {"cmd": "reset"}:
                #     retval = wa_env.reset()
                case {"cmd": "reset", "data": {"seed": seed, "options": options, "config_file": config_file}}:
                    env_config_file = config_file
                    retval = wa_env.reset(seed=seed, options=options)
                case {"cmd": "step", "data": {"action": action}}:
                    retval = wa_env.step(action)
                case {"cmd": "page"}:
                    retval = wa_env.page
                case {"cmd": "get_page_client", "data": {"page": page}}:
                    retval = wa_env.get_page_client(page)
                case {"cmd": "_get_obs_metadata"}:
                    retval = wa_env._get_obs_metadata()
                case {"cmd": "eval", "data": {"trajectory": trajectory}}:
                    if env_config_file:
                        evaluator = evaluator_router(env_config_file)
                        print(env_config_file)
                        retval = evaluator(
                        trajectory=trajectory,
                        config_file=env_config_file,
                        page=wa_env.page,
                        client=wa_env.get_page_client(wa_env.page),
                        )
                    else:
                        retval = 0.0
                case other:
                    print(f"!!! UNKNOWN COMMAND IN WA IPC !!!\n{other}")
                    raise ValueError("Unknown command")
        except Exception as e:
            retval = e

        # return
        pipe.send(retval)

    # clean up
    wa_env.close()


import multiprocessing


class PromptConstructor:
    """
    Construct prompt
    """

    def __init__(
        self,
        instruction_path: str | Path,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction = instruction

    def construct(
        self,
        trajectory: list,
        intent: str,
        meta_data: dict[str, Any] = {},
    ):
        """Construct prompt given the trajectory"""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]

        # input x
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        # make sure all keywords are replaced
        assert all([f"{{k}}" not in current for k in keywords])

        return current

    def _extract_action(self, response: str) -> str:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(f"Cannot parse action from response {response}")

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response


class WebarenaEnvServer:
    """
    WebarenaEnvServer
    """

    def __init__(self) -> None:
        self._max_id = 0
        self.envs = {}
        self.state_info = {}  # env_idx -> {"observation": obs, "info": info}
        self.trajectory = {}
        self.meta_data = {}
        self.intent = {}  # question in config_file
        self.prompt_constructor = PromptConstructor(
            instruction_path="./agent/prompts/jsons/p_cot_id_actree_2s.json"
        )
        self.wa_send = {}
        self.wa_recv = {}

    def send_command(self, idx: int, cmd: str, **data):
        """Send a command and retrieve its response."""
        if not self.envs[idx].is_alive():
            print(f"Error: Env {idx} is dead")
            raise RuntimeError(f"Error: Env {idx} is dead")
        msg = {"cmd": cmd, "data": data}
        try:
            self.wa_send[idx].send(msg)
            retval = self.wa_send[idx].recv()
            if isinstance(retval, Exception):
                raise retval
        except Exception as e:
            print(f"Error: failed in send command {cmd}: {e}")
            raise e
        return retval

    def create(self,env_idx) -> int:
        """
        Only call this create function once.
        """
        print(f"Creating env {env_idx}")
        (self.wa_send[env_idx], self.wa_recv[env_idx]) = (
            multiprocessing.Pipe()
        )
        self.envs[env_idx] = multiprocessing.Process(
            target=wa_entrypoint, args=(self.wa_recv[env_idx],)
        )
        self.envs[env_idx].start()

        # self.envs[self._max_id] = ScriptBrowserEnv(
        #     headless=True,
        #     slow_mo=100,
        #     observation_type="accessibility_tree",
        #     current_viewport_only=True,
        #     viewport_size={"width": 1280, "height": 720},
        # )
        self.trajectory[env_idx] = []
        self.meta_data[env_idx] = {}
        self.intent[env_idx] = ""
        # self.envs[env_idx].reset()
        try:
            self.send_command(env_idx, "reset" ,seed=None, options=None, config_file=None)
        except Exception as e:
            print(f"Env {env_idx} Error in reset: {e}")
            self.close(env_idx)
            raise e
        
        return env_idx

    def step(
        self,
        env_idx: int,
        response: str,
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any]]:
        """
        Return:
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )
        """
        # print(self.wa_send.keys())
        try:
            force_prefix = self.prompt_constructor.instruction["meta_data"].get(
                "force_prefix", ""
            )
            response = f"{force_prefix}{response}"
            parsed_response = self.prompt_constructor.extract_action(response)
            action = create_id_based_action(parsed_response)
            action["raw_prediction"] = response

            self.trajectory[env_idx].append(action)

            action_str = get_action_description(
                action,
                self.state_info[env_idx]["info"]["observation_metadata"],
                action_set_tag="id_accessibility_tree",
                prompt_constructor=self.prompt_constructor,
            )
            self.meta_data[env_idx]["action_history"].append(action_str)

            terminated = False
            # obs, reward, terminated, truncated, info = self.envs[env_idx].step(action)
            # print(f"idx{env_idx} action{action}\n")
            obs, reward, terminated, truncated, info = self.send_command(
                env_idx, "step", action=action
            )
            

            self.state_info[env_idx] = {"observation": obs, "info": info}
            self.trajectory[env_idx].append(self.state_info[env_idx])

            prompt = self.prompt_constructor.construct(
                self.trajectory[env_idx], self.intent[env_idx], self.meta_data[env_idx]
            )

            if terminated:
                # add a action place holder
                self.trajectory[env_idx].append(create_stop_action(""))

            if terminated or action["action_type"] == ActionTypes.STOP:
                del self.trajectory[env_idx][-1]
                terminated = True
                # evaluator = evaluator_router(self.config_file)
                # reward = evaluator(
                #     trajectory=self.trajectory[env_idx],
                #     config_file=self.config_file,
                #     page=self.envs[env_idx].page,
                #     client=self.envs[env_idx].get_page_client(self.envs[env_idx].page),
                # )
                # page = self.send_command(env_idx, "page")
                # reward = evaluator(
                #     trajectory=self.trajectory[env_idx],
                #     config_file=self.config_file,
                #     page=page,
                #     client=self.send_command(env_idx, "get_page_client", page=page),
                # )
                reward = self.send_command(
                    env_idx,
                    "eval",
                    trajectory=self.trajectory[env_idx],
                )
                

            return (prompt, reward, terminated, truncated, info)
        except Exception as e:
            print(f"Env {env_idx} Error in step: {e}")
            return (str(e), 0, False, False, None)

    def observation(self, env_idx) -> dict[str, Observation]:
        """
        Return
            {"text": text_obs, "image": image_obs}

        Example text:
        [4] RootWebArea 'Projects · Dashboard · GitLab' focused: True
        [12] link 'Skip to content'
        [28] link 'Dashboard'
        [2266] button '' hasPopup: menu expanded: False
        [63] textbox 'Search GitLab' required: False
        [61] generic 'Use the shortcut key <kbd>/</kbd> to start a search'
        [79] link 'Create new...'
        [95] link 'Issues'
                [97] generic '13 assigned issues'
        [101] link 'Merge requests'
                [104] generic '8 merge requests
        """
        return self.prompt_constructor.construct(
            self.trajectory[env_idx], self.intent[env_idx], self.meta_data[env_idx]
        )

    def observation_metadata(self, env_idx):
        """
        Return
        {
            "text": self.text_processor.meta_data,
            "image": self.image_processor.meta_data,
        }
        """
        # return self.envs[env_idx]._get_obs_metadata()
        try:
            ret = self.send_command(env_idx, "_get_obs_metadata")
        except Exception as e:
            print(f"Env {env_idx} Error in _get_obs_metadata: {e}")
            self.close(env_idx)
            raise e
        
        return ret

    def reset(
        self, env_idx, seed: int | None = None, options: dict[str, str] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        options={"config_file": config_file}
        Return:
            (observation, info)
        """
        print(f"Resetting env {env_idx}")
        self.config_file = Path(options["config_file"])
        with open(self.config_file) as f:
            _c = json.load(f)
            self.intent[env_idx] = _c["intent"]
        # obs, info = self.envs[env_idx].reset(seed=seed, options=options)
        try:
            obs, info = self.send_command(env_idx, "reset", seed=seed, options=options, config_file=self.config_file)
        except Exception as e:
            raise e

        self.trajectory[env_idx] = []
        self.state_info[env_idx] = {"observation": obs, "info": info}
        self.trajectory[env_idx].append(self.state_info[env_idx])

        self.meta_data[env_idx] = {"action_history": ["None"]}

        return (obs, info, _c["sites"], _c["intent"])

    def close(self, env_idx) -> None:
        if env_idx in self.envs:
            try:
                # Send close command to subprocess\
                self.send_command(env_idx, "close")
                
                # clean up pipes
                while self.wa_recv[env_idx].poll(timeout=5):
                    try:
                        _ = self.wa_recv[env_idx].recv()
                    except (EOFError, pickle.UnpicklingError):
                        print(f"env {env_idx} recv pipe clean up error: EOFError/UnpicklingError")
                        break
                    except Exception as e:
                        print(f"env {env_idx} recv pipe clean up error: {e}")
                        break
                
                # Close the communication pipes
                self.wa_send[env_idx].close()
                self.wa_recv[env_idx].close()
                
                # Terminate the process
                if self.envs[env_idx].is_alive():
                    self.envs[env_idx].terminate()
                    self.envs[env_idx].join(timeout=10)
                    if self.envs[env_idx].is_alive():
                        self.envs[env_idx].kill()

                # Remove from tracking dictionaries
                del self.envs[env_idx]
                del self.wa_send[env_idx]
                del self.wa_recv[env_idx]
                del self.trajectory[env_idx]
                del self.meta_data[env_idx]
                del self.intent[env_idx]
                del self.state_info[env_idx]

                print(f"Environment {env_idx} closed successfully.")
            except Exception as e:
                print(f"Error while closing environment {env_idx}: {e}")
        return 0


webarena_env_server = WebarenaEnvServer()