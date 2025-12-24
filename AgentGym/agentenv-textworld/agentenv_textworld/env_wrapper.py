import glob
import os
from pathlib import Path
import threading
from typing import Dict, List, Optional

from .environment import TextWorldEnv


class TextWorld_Wrapper:
    """Environment server wrapper for TextWorld.

    Manages multiple TextWorldEnv instances and exposes thread-safe methods for
    create, reset, step, observation, and close.
    """

    def __init__(self):
        self._max_id: int = 0
        self._lock = threading.Lock()
        self.env: Dict[int, TextWorldEnv] = {}
        self.info: Dict[int, dict] = {}
        self.ls: List[int] = []
        # Per-env: ordered list of game files and the base dir used for selection.
        self.games_map: Dict[int, List[str]] = {}
        self.games_base_dir: Dict[int, str] = {}

    def _scan_games(self, games_dir: Optional[str]) -> List[str]:
        """Find compiled TextWorld games under `games_dir` using the canonical naming.

        Only matches files named `textworld_*.z8`, sorted lexicographically for stability.
        Used for reporting and basic listing; selection is done via `path + id` directly.
        """
        if not games_dir:
            return []
        root = Path(games_dir)
        if not root.exists():
            return []
        files = [str(p.resolve()) for p in root.rglob("textworld_*.z8") if p.is_file()]
        files.sort()
        return files

    def create(self, games_dir: str = "data/textworld/games", max_steps: int = 50):
        try:
            # Fallbacks if client doesn't provide a games_dir
            with self._lock:
                env_id = self._max_id
                self._max_id += 1
            env = TextWorldEnv(max_steps=max_steps)
            self.env[env_id] = env
            self.info[env_id] = {
                "observation": "",
                "reward": 0.0,
                "done": False,
                "deleted": False,
                "available_actions": [],
            }
            games = self._scan_games(games_dir)
            self.games_map[env_id] = games
            self.games_base_dir[env_id] = games_dir
            print(f"[TextWorld] games_dir resolved to: {games_dir}")
            print(f"[TextWorld] found {len(self.games_map[env_id])} game files")
            self.ls.append(env_id)
            print(f"-------Env {env_id} created--------")
            return {"id": env_id, "games": len(self.games_map[env_id])}
        except Exception as e:
            return {"error": f"{e}"}

    def step(self, env_id: int, action: str):
        try:
            self._check_id(env_id)
            ob, reward, terminated, truncated, info = self.env[env_id].step(action)
            done = bool(terminated or truncated)
            payload = {
                "observation": ob,
                "reward": float(reward),
                "done": done,
                "available_actions": info.get("available_actions", []),
                "score": info.get("score", 0),
                "max_score": info.get("max_score", 0),
                "won": info.get("won", False),
                "lost": info.get("lost", False),
            }
            self.info[env_id].update(payload)
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload

    def reset(self, env_id: int, data_idx: int = 0, game_path: Optional[str] = None):
        try:
            self._check_id(env_id, allow_done=True)
            # Determine the game file to load using simple `path + id` logic.
            file_path: Optional[str] = None
            if game_path:
                if os.path.isdir(game_path):
                    base_dir = game_path
                    file_path = os.path.join(base_dir, f"textworld_{data_idx}.z8")
                else:
                    # If a file path is provided, use it directly
                    file_path = game_path
            else:
                # Use base_dir given at create()
                base_dir = self.games_base_dir.get(env_id)
                if not base_dir:
                    raise FileNotFoundError(
                        "No games directory known. Pass 'game_path' as a directory or set 'games_dir' in create()."
                    )
                file_path = os.path.join(base_dir, f"textworld_{data_idx}.z8")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Game file not found: {file_path}")

            ob, info = self.env[env_id].reset(file_path)
            payload = {
                "id": env_id,
                "observation": ob,
                "available_actions": info.get("available_actions", []),
                "done": False,
                "reward": 0.0,
            }
            self.info[env_id] = {
                **payload,
                "deleted": False,
            }
        except Exception as e:
            payload = {"error": str(e)}
        return payload

    def get_observation(self, env_id: int):
        try:
            self._check_id(env_id)
            return self.info[env_id]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_available_actions(self, env_id: int):
        try:
            self._check_id(env_id)
            return self.info[env_id].get("available_actions", [])
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_info(self, env_id: int):
        try:
            self._check_id(env_id)
            return self.info[env_id]
        except Exception as e:
            return {"error": str(e)}

    def close(self, env_id: int) -> dict:
        try:
            if env_id in self.ls:
                self.ls.remove(env_id)
            env = self.env.pop(env_id)
            env.close()
            self.info.pop(env_id, None)
            self.games_map.pop(env_id, None)
            self.games_base_dir.pop(env_id, None)
            print(f"-------Env {env_id} closed--------")
            return {"closed": True}
        except KeyError:
            return {"closed": False, "error": "Env not exist"}
        except Exception as e:
            return {"closed": False, "error": str(e)}

    def _check_id(self, env_id: int, allow_done: bool = False):
        if env_id not in self.info:
            raise NameError(f"The id {env_id} is not valid.")
        if self.info[env_id]["deleted"]:
            raise NameError(f"The task with environment {env_id} has been deleted.")
        if (not allow_done) and self.info[env_id].get("done"):
            raise NameError(f"The task with environment {env_id} has finished.")

    def __del__(self):
        for idx in list(self.ls):
            try:
                self.close(idx)
            except Exception:
                pass


server = TextWorld_Wrapper()
