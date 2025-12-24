import os
from typing import Any, Dict, List, Optional, Tuple


class TextWorldEnv:
    """A light wrapper around TextWorld Gym environments.

    This class defers actual TextWorld imports until first reset to avoid import
    errors when the library is not present in the current interpreter. It
    expects TextWorld to be installed in the Python env used to run this server.
    """

    def __init__(self, max_steps: int = 50) -> None:
        self._max_steps = max_steps
        self._env = None
        self._env_id = None
        self._request_infos = None
        self._steps = 0
        self._last_infos: Dict[str, Any] = {}
        # Track whether a successful reset has occurred.
        self._ready: bool = False

    def _ensure_imports(self):
        # Lazy import to avoid hard dependency at module import time
        global textworld
        import textworld  # type: ignore
        import textworld.gym  # type: ignore

        return textworld

    def _register_game(self, game_file: str):
        textworld = self._ensure_imports()

        if self._request_infos is None:
            self._request_infos = textworld.EnvInfos(
                admissible_commands=True,
                description=True,
                inventory=True,
                won=True,
                lost=True,
                score=True,
                max_score=True,
            )

        # Register and create a Gym env for the given game
        self._env_id = textworld.gym.register_game(game_file, self._request_infos)
        # Newer TextWorld exposes its own registry and make()
        self._env = textworld.gym.make(self._env_id)

    def reset(self, game_file: str) -> Tuple[str, Dict[str, Any]]:
        if not os.path.exists(game_file):
            raise FileNotFoundError(f"Game file not found: {game_file}")

        self.close()
        self._register_game(game_file)

        obs, infos = self._env.reset()
        # TextWorld returns (obs, infos). Normalize to str + dict
        self._steps = 0
        self._last_infos = dict(infos or {})
        self._ready = True
        return str(obs), self._format_info(obs, infos, reward=0.0, done=False)

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._env is None or not self._ready:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        res = self._env.step(action)
        # TextWorld returns (obs, score, done, infos)
        if len(res) == 4:
            obs, reward, done, infos = res
            terminated, truncated = bool(done), False
        else:  # Fallback if a different API is observed
            obs, reward, terminated, truncated, infos = res

        self._steps += 1
        done_flag = bool(terminated or truncated or (self._steps >= self._max_steps))
        self._last_infos = dict(infos or {})
        return (
            str(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._format_info(obs, infos, reward, done_flag),
        )

    def observation(self) -> str:
        # TextWorld has no explicit observe call; rely on last obs in wrapper logic
        # The server caches the last observation in its info dict.
        # Provided here for completeness if needed.
        desc = self._last_infos.get("description", "")
        inv = self._last_infos.get("inventory", "")
        return f"{desc}\n\n{inv}".strip()

    def close(self):
        try:
            if self._env is not None:
                self._env.close()
        finally:
            self._env = None
            self._env_id = None
            self._steps = 0
            self._ready = False

    # ---------------- internal helpers ---------------- #
    def _format_info(
        self,
        obs: Any,
        infos: Optional[Dict[str, Any]],
        reward: float,
        done: bool,
    ) -> Dict[str, Any]:
        infos = infos or {}
        return {
            "observation": str(obs),
            "available_actions": infos.get("admissible_commands", []),
            "score": infos.get("score", 0),
            "max_score": infos.get("max_score", 0),
            "won": infos.get("won", False),
            "lost": infos.get("lost", False),
            "reward": float(reward),
            "done": bool(done),
        }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Smoke test for TextWorldEnv.")
    parser.add_argument(
        "--game",
        type=str,
        default="data/textworld/games/textworld_1.z8",
        help="Path to a TextWorld game (.ulx/.z8). If a .json is given, the sibling .ulx will be inferred.",
    )
    parser.add_argument("--max_steps", type=int, default=50)
    args = parser.parse_args()

    game_path = args.game
    # If a .json is provided, try to infer the compiled game (.ulx)
    if game_path.endswith(".json"):
        candidate = game_path[:-5] + ".ulx"
        if os.path.exists(candidate):
            game_path = candidate
        else:
            print(f"[ERROR] Expected compiled game next to JSON: {candidate} not found.")
            sys.exit(1)

    env = TextWorldEnv(max_steps=args.max_steps)
    print(f"[INFO] Resetting with game: {game_path}")
    obs, info = env.reset(game_file=game_path)
    print("[RESET] observation:\n", str(obs))
    print("[RESET] info keys:", list(info.keys()))

    def do_step(action: str):
        ob, rew, term, trunc, inf = env.step(action)
        print(f"[STEP] action='{action}' | reward={rew} | done={term or trunc}")
        print("       observation:", str(ob))

    # Try a couple of basic actions
    do_step("look")
    do_step("inventory")

    # Test observation() helper
    print("[OBSERVE]", env.observation()[:200])

    # Close
    env.close()
    print("[CLOSE] done")
