# AgentEnv TextWorld

A FastAPI server wrapping the TextWorld environment for AgentGym.

## Usage

1. Activate your TextWorld Python environment:

```
source /home/v-liyixia/CODES/AgentGym-RL/uv_textworld/bin/activate
```

2. Install this package inside that environment (editable or build):

```
cd AgentGym/agentenv-textworld
pip install -e .
```

3. Launch the environment server:

```
textworld --host 0.0.0.0 --port 36012
```

4. Create + reset from the client side by POSTing to `/create` with `games_dir` or using `/reset` with a specific `game_path`.

## HTTP API

- `POST /create` with JSON `{ "games_dir": "/path/to/games", "max_steps": 50 }`
- `POST /reset` with JSON `{ "id": 0, "data_idx": 0 }` or `{ "id": 0, "game_path": "/path/to/file.ulx" }`
- `POST /step` with JSON `{ "id": 0, "action": "open door" }`
- `GET /observation?id=0`
- `GET /available_actions?id=0`
- `GET /detail?id=0`
- `POST /close` with JSON `{ "id": 0 }`

This mirrors the patterns used by other AgentGym environments (e.g., AlfWorld, TextCraft).

