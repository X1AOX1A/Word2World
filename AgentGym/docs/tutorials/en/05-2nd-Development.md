# Add a Custom Environment to AgentGym

This document explains in detail how to add a custom environment (it is recommended to use `agentenv-textcraft` as the base environment for modification).

## 1. Implementing a Custom Environment

Directory structure:

```
└─AgentGym
    └─agentenv-babyai
        ├── pyproject.toml
        ├── README.md
        │
        └── agentenv_babyai
            ├── environment.py
            ├── launch.py
            ├── model.py
            ├── server.py
            └── __init__.py
```

### 1.1 Environment Class Implementation

Each `agentenv` needs an environment class as its foundation, serving as the actual environment interaction module on the server:

```python
# Using TextCraft Env as an example
class TextCraftEnv(gym.Env[str, str]):
    def __init__(self, crafting_tree, commands, goal):
        # Initialize required data for the environment, such as datasets, etc.
        self.observation = None

    def step(self, action):
        """Execute an action and return the result.

        Args:
            action: Action string
        Preconditions:
            - Must have called reset() and not yet terminated
            - Cannot call after terminated=True (must reset)
        Concurrency:
            - step() and reset() cannot run simultaneously; the server ensures this via serial scheduling or locks
        Returns:
            observation: New observable state
            reward: Immediate reward for this step (not cumulative)
            terminated: Whether completed/failed
            info: Diagnostic info (cumulative score, error reason, debug fields, etc.)
        """
        reward = 0.0  # TODO
        terminated = False  # TODO
        info = {
            # TODO
        }
        # Parse action (can be done client-side; validated again here)
        norm_action = (action or "").strip()
        # Execute action logic
        # Update observation (environment-specific)
        self.observation = f"TODO"
  
        return (self.observation, reward, terminated, info)

    def observation(self) -> Any:  # To align with a client observe name, could rename to observe()
        """Return current environment state."""
        return self.observation

    def reset(self, idx=None):
        """Reset the environment to a specified task index.
        When called:
            - After initial create()
            - After each task ends
        Args:
            idx: (Optional) Dataset entry / level / seed index; None means default task
        Preconditions:
            - Can be called at any time
        Concurrency:
            - Cannot run simultaneously with step(); server ensures via serial scheduling or locks
        Returns:
            Initial observation
        """
        #######################
        # TODO
        # Implement reset logic
        #######################
        return self.observation

    def render(self, mode="human"):
        # Inherit gym.Env method
        pass

    def close(self):
        # Optional: if environment instantiation consumes large memory, implement to reclaim resources
        super().close()
        for attr in ['inventory', 'action_regexes', 'count_regex', 'commands', 'goal']:
            if hasattr(self, attr):
                if isinstance(getattr(self, attr), (dict, list, set)):
                    getattr(self, attr).clear()
                delattr(self, attr)
```

### 1.2 Environment Server Implementation

After implementing or reusing an existing environment class, implement an environment server class to manage it:

- Assign environment instance ids uniformly
- Maintain id -> Env mappings
- Provide thread-safe `create/step/reset/close` operations
- (Optional) Action parsing, caching, etc.

```python
# Using Textcraft Environment as example
class TextCraft_Wrapper:
    """Environment server wrapper (Server-Side Manager).

    Responsibilities:
        - Allocate & manage multiple Env instances
        - Provide thread-safe CRUD operations
        - (Extensible) action pre-processing, debug stats, rate limiting, concurrency quotas

    Concurrency:
        - self._lock protects id allocation
        - `step/reset` do not use a global lock -> assumes same env_id calls are serialized by upper layer. If "concurrent access to same env_id" is needed, add per-env locks.
    """

    def __init__(self, minecraft_dir="agentenv_textcraft/"):
        # Initialize management variables like unique env id, creation lock, etc.
  
        self._max_id: int = 0  # Next available env id (auto-increment). Must modify inside lock.
  
        self.env: Dict[int, TextCraftEnv] = {}  # id -> Env instance mapping
  
        self.info: Dict[int, dict] = {}  # id -> last step/reset summary (optional cache)
   
        self.ls: List[int] = []  # Active id list (for safety checks and unified closing)
  
        self.crafting_tree = CraftingTree(minecraft_dir=minecraft_dir)  # Shared read-only resource to avoid rebuilding expensive structure
  
        self._lock = threading.Lock()  # Protect id allocation and structural writes

    def create(self) -> dict:
        """Create a new environment instance and return its id.
        Called: Client binds environment for first time via create(), instantiates the class from 1.1.
        Concurrency:
            - Lock only around id allocation; other initialization without lock to reduce contention
            - (Extensible) create per-env lock to ensure safe concurrent access
        Returns: {"id": int}
        """
        with self._lock:
            env_id = self._max_id
            self._max_id += 1
        new_env = TextCraftEnv(crafting_tree=self.crafting_tree, commands=None, goal="demo")
        # Initialize once (can pass data_idx=0 or random)
        new_env.reset(idx=0)
        self.ls.append(env_id)
        self.env[env_id] = new_env
        print(f"-------Env {env_id} created--------")

        return {"id": env_id}

    def step(self, env_idx: int, action: str):
        """Dispatch an action to the specified environment.
        Preconditions: env_idx must exist
        Concurrency: Assumes same env_idx calls are serialized by client; can add per-env local lock
        Error handling: If not found raise KeyError or return unified error structure
        """
        env = self.env[env_idx]
        ob, reward, terminated, info = env.step(action)
        self.info[env_idx] = {"observation": ob, "reward": reward, "done": terminated, **info}
        return {
            "observation": ob,
            "reward": reward,
            "done": terminated,
        }

    def reset(self, env_idx: int, data_idx: int):
        """Reset specified environment.
        Called: After each task / after initialization
        Args:
            env_idx: environment id
            data_idx: dataset entry or task id
        Concurrency: Assumes same env_idx calls are serialized; can add per-env local lock if needed
        """
        ob = self.env[env_idx].reset(idx=data_idx)
        self.info[env_idx] = {"observation": ob, "reward": 0.0, "score": 0.0, "done": False}
        return self.info[env_idx]

    def observation(self, env_idx: int):
        """Return cached last observation.
        For 'real-time' state (non-cached) call env.observation() directly.
        """
        if env_idx in self.info:
            return {"observation": self.info[env_idx]["observation"]}
        return {"error": "Env info not initialized"}

    def observe(self, env_idx: int):
        """(Optional) Query Env directly, bypassing cache."""
        env = self.env[env_idx]
        return {"observation": env.observation()}

    def close(self, env_id: int) -> dict:
        """Close and release an environment instance.
        Concurrency: Remove from self.ls first, then call env.close() to avoid concurrent step
        Returns: {"closed": bool}
        """
        try:
            if env_id in self.ls:
                self.ls.remove(env_id)
            env = self.env.pop(env_id)
            env.close()
            self.info.pop(env_id, None)
            print(f"-------Env {env_id} closed--------")

            return {"closed": True}
        except KeyError:
            return {"closed": False, "error": "Env not exist"}
        except Exception as e:
            return {"closed": False, "error": str(e)}

    def __del__(self):
        """Release memory when process exits."""
        for idx in list(self.ls):
            try:
                self.close(idx)
            except Exception:
                pass
```

### 1.3 Environment Client Implementation

The client (`EnvClient`) lives on the Agent process side. It communicates with the server via HTTP, can parse LLM output, extract actions, retry on error, and record local cached state for the current task (e.g., last observation, done flag, etc.).

Storage location: `agentenv/agentenv/envs/myenv.py`.

```python
import re
import requests
from typing import Any, Mapping

class BabyAIEnvClient(BaseEnvClient):
    # Conversation start (Prompt Bootstrapping): set role / rules for LLM
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": 'You are an exploration master that wants to finish every goal you are given...',
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
        """Initialize client and create an environment instance on the server.
        Args:
            env_server_base: Server address, e.g., http://127.0.0.1:8000
            data_len: Dataset size (used for __len__)
            timeout: Timeout for a single HTTP call (seconds)
        Initialization flow:
            1. Save config -> 2. POST /create -> 3. Record env_id
        """
        super().__init__(*args, **kwargs)
        # 1. Save config
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        # 2. POST /create
        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        # 3. Record env_id
        ok = ok.json()
        self.env_id = ok["id"]

    def __len__(self):
        return self.data_len

    # ------------------ Internal HTTP Wrappers ------------------ #
    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """
        Unified POST wrapper, auto-adds env_id.
        Exception policy: assert.
        """
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}", json=data, timeout=self.timeout
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        """
        Unified GET wrapper.
        Exception policy: assert.
        """
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}", timeout=self.timeout
        )
        assert res.status_code == 200
        return res.json()

    # ------------------ Environment Interaction Methods ------------------ #
    def observe(self) -> str:
        """Get current observation.
        Two strategies:
            1. Use local cache self.info["observation"] (faster)
            2. Directly query server `_get("observation")` (real-time, extra HTTP cost)
        Choose as needed.
        """
        if self.info:
            return self.info["observation"]
        response = self._get("observation")
        return response["observation"]

    def step(self, action: str) -> StepOutput:
        """Execute one environment action.
        Processing:
            1. (Optional) Parse `Action:` segment from LLM reply, clean illegal chars (prevent injection/format pollution)
            2. POST /step
            3. (Optional) Update local cache self.info
        Parsing rules:
            * Only one Action allowed; multiple -> return instructive error
        Returns: StepOutput (state=new observation, reward=score (may be cumulative), done=whether finished)
        Note: reward/score semantics may vary by server; here we use score as training reward example.
        """
        # 1. Parse Action
        action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", action, re.DOTALL)
        if len(action_matches) > 1:
            return StepOutput(
                state=(
                    "Error: Only one 'Action' is allowed per response. Please adjust your response."
                ),
                reward=0,
                done=False,
            )
        action = action_matches[-1] if action_matches else ""
        action = re.sub(r"[^A-Za-z0-9, ]+", "", action)  # Filter abnormal symbols (relax as needed)
        action = " ".join(action.split()).strip()
        # 2. POST /step
        response = self._post("step", {"action": action})
        # 3. Update local cache
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],  # Immediate reward
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        """Reset environment to specified task index.
        data_idx: Usually maps to task id in the custom environment
        Returns: Initial observation structure returned by server
        """
        response = self._post("reset", {"data_idx": data_idx})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return response

    def close(self) -> dict[str, Any]:
        """Close remote environment and free resources.
        When to call: end of training / process exit / shard reclamation
        Note: After close, step/reset will fail; instance should be released.
        """
        response = self._post("close", {})
        return response
```

### 1.4 Task Class Implementation

```python
class BabyAITask(BaseTask):
    # Specify the environment client class used by this task
    env_client_cls = BabyAIEnvClient
    # Task name (used for logs, registry, etc.)
    env_name = "BabyAI"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        """Construct the task.
        Args:
            client_args: Dict of parameters passed to EnvClient (e.g., env_server_base, data_len, etc.)
            n_clients: Number of parallel environment clients (!= number of server envs, depends on reuse)
        """
        super().__init__(client_args, n_clients, *args, **kwargs)
```

### 1.5 HTTP Service Implementation (FastAPI-Based)

The Web layer performs parameter validation, calls corresponding server methods, and returns JSON.

```python
# Using BabyAI Server as example
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class StepRequestBody(BaseModel):
    id: int
    action: str

class ResetRequestBody(BaseModel):
    id: int
    data_idx: int = 0

class CloseRequestBody(BaseModel):
    id: int

server = TextCraft_Wrapper()

@app.get("/")
def hello():
    return "This is environment BabyAI."

@app.post("/create")
def create():
    return server.create()

@app.post("/step")
def step(body: StepRequestBody):
    return server.step(body.id, body.action)

@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.id, body.data_idx)

@app.get("/observation")
def get_observation(id: int):
    print(f"Observing environment {id}")
    return server.observe(id)

@app.post("/close")
def close(body: CloseRequestBody):
    return server.close(body.id)
```

### 1.6 Concurrency & Concurrent Request Support

#### 1.6.1 Thread-Safety Mechanisms

- **Global Lock (Server level)**
  - Used only when allocating a new environment ID or modifying the active environment list.
  - Keep the critical section as small as possible to avoid a performance bottleneck caused by a coarse global lock.
- **Local Lock (Env level)**
  - Each environment instance may optionally have its own `threading.Lock()`.
  - Ensures `step/reset` calls for the same `env_id` are executed mutually exclusively.
  - Different environments can execute in parallel to improve throughput.

#### 1.6.2 Concurrency at the FastAPI Layer

- FastAPI is coroutine (async) based by default and can handle high concurrency.
- If the internal environment computation is time‑consuming (e.g., simulation), it is recommended to keep the server layer interfaces synchronous; FastAPI's async scheduling will prevent other requests from being blocked.

#### 1.6.3 Request Isolation & Resource Reclamation

- Each `env_id` corresponds to an independent environment instance; state is fully isolated to avoid cross‑task contamination.
- The optional `close()` method releases memory and any occupied resources.

#### 1.6.4 Common Concurrency Scenarios

- **Multiple clients concurrently calling `create()`**
  - `_lock` ensures unique ID allocation (no duplication).
- **A single client issuing rapid consecutive `step()` calls**
  - Ensure serialization inside the same environment instance (via the per‑env local lock if configured).

## 2. API Interfaces

### 2.1 Main HTTP Endpoints of Environment Server

1. `/create`: Create environment
2. `/observation`: Get current environment observation
3. `/step`: Execute action
4. `/reset`: Reset environment
5. `/close`: (Optional) Clean up environment memory

### 2.2 Main Class Method Interfaces

1. `BaseEnvClient.observe()`: Observe environment
2. `BaseEnvClient.step(action)`: Execute action
3. `BaseEnvClient.reset(idx)`: Reset environment

## 3. Common Code Patterns and Conventions

### 3.1 Environment Client Implementation Pattern

* Each environment client class inherits `BaseEnvClient`
* Must implement: `__init__`, `observe`, `step`, `reset`
* Define `conversation_start` as initial prompt context

### 3.2 Task Implementation Pattern

* Each task class inherits `BaseTask`
* Specify `env_client_cls` and `env_name`
* Extensible: batch rollout, async sampling, etc.

### 3.3 Data and Interaction Format Conventions

* Use ReAct format to organize dialogue and actions
* Interaction records use `ConversationMessage`
