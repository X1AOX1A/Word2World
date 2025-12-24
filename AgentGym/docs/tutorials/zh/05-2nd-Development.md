# AgentGym二次开发添加自定义环境设计文档

本文档将详细介绍如何添加自定义环境（推荐参考agentenv-textcraft作为base环境进行修改）。

## 1. 自定义环境实现

目录结构

```
└─AgentGym
    └─agentenv-babyai
        ├── pyproject.toml
        ├── README.md
        │
        └─agentenv_babyai
            ├─── environment.py
            ├─── launch.py
            ├─── model.py
            ├─── server.py
            └─── __init__.py
```

### 1.1 环境类实现

每个agentenv都需要以一个环境类为基础，作为服务器实际的环境交互模块：

```python
# 以TextCraft Env为例
class TextCraftEnv(gym.Env[str, str]):
    def __init__(self, crafting_tree, commands, goal):
        # 初始化环境需要存储的一些必要数据，如数据集等
	    self.observation = None

    def step(self, action):
        """执行动作并返回结果

        参数:
            action: 动作字符串
        前置条件:
            - 必须已调用过 reset() 且未终止
            - 不可在 terminated=True 后继续调用（需 reset）
        并发问题:
            - step() 与 reset() 不可同时执行；服务端通过串行调度或锁保障
        返回:
            observation: 新的可观察状态
            reward: 本步即时奖励 (非累计)
            terminated: 是否已完成/失败
            info: 诊断信息（累计得分、错误原因、调试字段等）
        """
        reward = 0.0 # TODO
        terminated = False # TODO
        info = {
            # TODO
        }
        # 解析动作（可在客户端完成，这里再次校验）
        norm_action = (action or "").strip()
        # 执行动作逻辑实现
        # 更新 observation（具体环境具体实现）
        self.observation = f"TODO"
  
        return (self.observation, reward, terminated, info)

    def observation(self) -> Any:  # 若想与客户端 observe 对齐，可命名为 observe()
        """返回当前环境状态。"""
        return self.observation

    def reset(self, idx=None):
        """重置环境到指定任务索引。
        调用:
            - 初次调用create()后
            - 每个任务结束后
        参数:
            idx: （可选）数据集条目 / 关卡 / 种子索引；None 表示默认任务
        前置条件:
            - 可以在任意时刻调用
        并发问题:
            - 不可与 step() 同时执行；服务端通过串行调度或锁保障
        返回:
            初始 observation
        """
        #######################
        # TODO
        # 具体实现环境重置逻辑
        #######################
        return self.observation

    def render(self, mode="human"):
	    # 继承gym.Env类成员函数
        pass

    def close(self):
	    # 可选，若实例化环境会占用大量内存，建议实现该函数以回收内存垃圾
        super().close()
        for attr in ['inventory', 'action_regexes', 'count_regex', 'commands', 'goal']:
            if hasattr(self, attr):
                if isinstance(getattr(self, attr), (dict, list, set)):
                    getattr(self, attr).clear()
                delattr(self, attr)
```

### 1.2 环境服务器实现

实现环境类或使用现有的环境类后，需实现一个环境服务器类来管理上述类：

- 统一分配环境实例 id
- 保存 id -> Env 的映射
- 提供线程安全的 `create/step/reset/close` 操作
- （可选）做动作解析、缓存等

```python
# 以Textcraft Environment为例
class TextCraft_Wrapper:
    """环境服务器包装器 (Server-Side Manager)。

    职责:
        - 分配 & 管理多个 Env 实例
        - 提供线程安全的增删查改操作
        - （可扩展）进行动作预处理、调试统计、限流、并发配额等

    并发问题:
        - 通过 self._lock 控制 id 分配
        - `step/reset` 默认不加全局锁 -> 假设同一 env_id 的调用在上层串行。若存在“同一 env_id 并发访问”需求，可为 env 增加内部锁的映射
    """

    def __init__(self, minecraft_dir="agentenv_textcraft/"):
	    # 初始化服务器管理相关变量，如唯一的环境id，创建环境锁等。
  
        self._max_id: int = 0 # 下一个可用的环境 id（自增）。必须在锁内修改
  
        self.env: Dict[int, TextCraftEnv] = {} # id -> Env 实例映射
  
        self.info: Dict[int, dict] = {} # id -> 最近一次 step/reset 的摘要信息（可选缓存）
   
        self.ls: List[int] = []  # 活跃 id 列表（方便安全检查与统一关闭）
  
        self.crafting_tree = CraftingTree(minecraft_dir=minecraft_dir) # 共享只读资源，避免重复构建耗时结构
  
        self._lock = threading.Lock() # 保护 id 分配和结构性写操作

    def create(self) -> dict:
        """创建一个新的环境实例并返回其 id。
        调用: 客户端首次绑定环境时调用create()函数，实例化1.1的环境类。
        并发: 
            - 仅分配 id 时加锁，其余初始化不持锁，降低锁粒度。
            - （可扩展）为每个环境创建一个环境锁，保证并发访问安全
        返回: {"id": int}
        """
        with self._lock:
            env_id = self._max_id
            self._max_id += 1
        new_env = TextCraftEnv(crafting_tree=self.crafting_tree, commands=None, goal="demo")
        # 初始化一次 (可传 data_idx=0 或随机)
        new_env.reset(idx=0)
        self.ls.append(env_id)
        self.env[env_id] = new_env
        print(f"-------Env {env_id} created--------")

        return {"id": env_id}

    def step(self, env_idx: int, action: str):
        """分发动作到指定环境。
        前置条件: env_idx 必须存在
        并发问题: 假设同一 env_idx 的调用由上层客户端串行调度；可加 env 级别局部锁保证并发安全。
        错误处理: 不存在则抛出 KeyError 或返回统一错误结构
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
        """重置指定环境。
        调用: 每个任务结束 / 环境初始化后
        参数:
            env_idx: 环境 id
            data_idx: 对应数据集条目或任务 id
        并发问题: 假设同一 env_idx 的调用由上层客户端串行调度；可加 env 级别局部锁保证并发安全。
        """
        ob = self.env[env_idx].reset(idx=data_idx)
        self.info[env_idx] = {"observation": ob, "reward": 0.0, "score": 0.0, "done": False}
        return self.info[env_idx]

    def observation(self, env_idx: int):
        """返回服务器端保存的最近一次 observation。
        若需“实时”状态（非缓存）可直接调用 env.observation()。
        """
        if env_idx in self.info:
            return {"observation": self.info[env_idx]["observation"]}
        return {"error": "Env info not initialized"}

    def observe(self, env_idx: int):
        """（可选函数）实时向 Env 查询，绕过缓存。"""
        env = self.env[env_idx]
        return {"observation": env.observation()}

    def close(self, env_id: int) -> dict:
        """关闭并释放一个环境实例。
        并发问题: 先在self.ls中移除再调用 env.close()，避免并发 step
        返回: {"closed": bool}
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
        """进程结束时释放内存。"""
        for idx in list(self.ls):
            try:
                self.close(idx)
            except Exception:
                pass
```

### 1.3 环境客户端实现

客户端（EnvClient）位于 **Agent进程侧**，负责通过 HTTP 调用服务器端接口，可做 LLM 输出解析、动作提取、错误重试等，记录当前环境任务的本地缓存状态（如 last observation, done 标志等）

储存位置：`agentenv/agentenv/envs/myenv.py` 。

```python
import re
import requests
from typing import Any, Mapping

class BabyAIEnvClient(BaseEnvClient):
    # 对话起始 (Prompt Bootstrapping)：给 LLM 设定角色 / 规则
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
        """初始化客户端并在服务器上创建一个环境实例。
        参数:
            env_server_base: Server地址，如 http://127.0.0.1:8000
            data_len: 数据集大小（用于 __len__）
            timeout: 单次 HTTP 调用超时时间 (秒)
        初始化流程:
            1. 保存配置 -> 2. POST /create -> 3. 记录 env_id
        """
        super().__init__(*args, **kwargs)
        # 1. 保存配置
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        # 2. POST /create
        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        # 3. 记录 env_id
        ok = ok.json()
        self.env_id = ok["id"]

    def __len__(self):
        return self.data_len

    # ------------------ 内部 HTTP 调用封装 ------------------ #
    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        """
        统一 POST 请求封装，自动补充 env_id。
        异常策略: assert。
        """
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}", json=data, timeout=self.timeout
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        """
        统一 GET 请求封装。
        异常策略: assert。
        """
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}", timeout=self.timeout
        )
        assert res.status_code == 200
        return res.json()

    # ------------------ 环境交互方法 ------------------ #
    def observe(self) -> str:
        """获取当前 observation。
        两种策略：
            1. 使用本地缓存 self.info["observation"]（更快）
            2. 直连服务器 `_get("observation")`（实时，额外 HTTP 开销）
        可根据需求自行选择实现。
        """
        if self.info:
            return self.info["observation"]
        response = self._get("observation")
        return response["observation"]

    def step(self, action: str) -> StepOutput:
        """执行一次环境交互动作。
        处理流程:
            1. （可选）从 LLM 回复文本中解析 `Action:` 段, 清洗非法字符（防止注入/格式污染）
            2. POST /step
            3. （可选）更新本地缓存 self.info
        解析规则:
            * 只允许一个 Action，多个则返回提示性错误
        返回: StepOutput (state=新 observation, reward=score(可为累计), done=是否结束)
        注意: reward/score 的语义视服务器实现可能不同，这里示例用 score 作为训练奖励
        """
        # 1. 解析 Action
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
        action = re.sub(r"[^A-Za-z0-9, ]+", "", action)  # 过滤异常符号（按需放宽）
        action = " ".join(action.split()).strip()
        # 2. POST /step
        response = self._post("step", {"action": action})
        # 3. 更新本地缓存
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],  # 即时 reward
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["reward"],
            done=response["done"],
        )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        """重置环境到指定任务索引。
        data_idx: 通常对应自定义环境中映射的任务 id
        返回: 服务端返回的初始 observation 结构
        """
        response = self._post("reset", {"data_idx": data_idx})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return response

    def close(self) -> dict[str, Any]:
        """关闭远端环境，释放资源。
        调用时机: 训练结束 / 进程退出 / 任务分片回收
        注意: close 后再 step/reset 会失败，应释放该实例
        """
        response = self._post("close", {})
        return response
```

### 1.4 任务类实现

```python
class BabyAITask(BaseTask):
    # 指定此任务使用的环境客户端类
    env_client_cls = BabyAIEnvClient
    # 任务名称（可用于日志、注册表）
    env_name = "BabyAI"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        """构造任务。
        参数:
            client_args: 传递给 EnvClient 的参数字典（如 env_server_base, data_len 等）
            n_clients: 并行环境客户端数量（!= server env 数量，取决于是否复用）
        """
        super().__init__(client_args, n_clients, *args, **kwargs)
```

### 1.5 HTTP 服务实现（基于 FastAPI）

Web 层进行参数校验，调用 server 对应方法，返回 JSON。

```python
# 以BabyAI Server为例
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

### 1.6 并发请求支持

#### 1.6.1 线程安全机制

- **全局锁 (Server 级)**
  - 仅在分配新环境 ID、修改活跃环境列表时使用。
  - 锁粒度应尽量小，避免因全局锁导致性能瓶颈。
- **局部锁 (Env 级)**
  - 每个环境实例可选配一个独立的 `threading.Lock()`。
  - 用于确保同一 `env_id` 上的 `step/reset` 调用互斥执行。
  - 不同环境之间可并行执行，提升吞吐量。

#### 1.6.2 FastAPI 层的并发

- FastAPI 默认基于  **异步协程** ，可高并发处理请求。
- 若环境内部计算耗时较长（如仿真模拟），建议在 `server` 层保持  **同步接口** ，由 FastAPI 异步调度保证请求不会阻塞。

#### 1.6.3 请求隔离与资源回收

- 每个 `env_id` 对应一个独立环境实例，状态完全隔离，避免不同训练任务之间的污染。
- 可选 `close()` 方法释放内存和占用的资源

#### 1.6.4 常见并发场景

- **多个客户端并发调用 `create()`**
  - 通过 `_lock` 控制 ID 分配，确保不会重复。
- **同一客户端快速连续调用 `step()`**
  - 需保证同一环境实例内操作的串行化（局部锁）。

## 2. API接口

### 2.1 环境服务器主要HTTP接口

1. `/create`: 创建环境
2. `/observation`: 获取当前环境观察
3. `/step`: 执行动作
4. `/reset`: 重置环境
5. `/close`:（可选）清理环境内存

### 2.2 主要类方法接口

1. `BaseEnvClient.observe()`: 观察环境
2. `BaseEnvClient.step(action)`: 执行动作
3. `BaseEnvClient.reset(idx)`: 重置环境

## 3. 常见的代码模式和约定

### 3.1 环境客户端实现模式

* 每个环境客户端类继承 `BaseEnvClient`
* 必需实现: `__init__`, `observe`, `step`, `reset`
* 定义 `conversation_start` 作为初始提示词上下文

### 3.2 任务实现模式

* 每个任务类继承 `BaseTask`
* 指定 `env_client_cls` 与 `env_name`
* 可扩展：批量 rollout、异步采样等

### 3.3 数据与交互格式约定

* 使用ReAct格式组织对话和行动
* 交互记录使用 `ConversationMessage`
