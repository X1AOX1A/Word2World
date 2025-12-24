import json
import os
from dataclasses import dataclass, field

import concurrent.futures
import queue
import transformers
from tqdm import tqdm

from agentenv.envs import (
    AcademiaTask,
    AlfWorldTask,
    BabyAITask,
    MazeTask,
    MovieTask,
    SearchQATask,
    SciworldTask,
    SheetTask,
    SqlGymTask,
    TextCraftTask,
    TodoTask,
    WeatherTask,
    WebarenaTask,
    WebshopTask,
    WordleTask,
    TextworldTask,
)


@dataclass
class ExtractArguments:
    inference_file: str = field(metadata={"help": "Dataset file (same as eval)."})
    output_json: str = field(metadata={"help": "Where to write extracted initial messages."})
    task_name: str = field(default="webshop", metadata={"help": "Task name."})
    env_server_base: str = field(default=None, metadata={"help": "AgentGym env server base URL."})
    timeout: int = field(default=2400)
    num_examples: int = field(default=-1, metadata={"help": "Only take first N indices (-1 for all)."})
    action_format: str = field(default="react", metadata={"help": "react | function_calling | code_as_action"})


def _task_class(task_name: str):
    task_classes = {
        "webshop": WebshopTask,
        "alfworld": AlfWorldTask,
        "babyai": BabyAITask,
        "sciworld": SciworldTask,
        "textcraft": TextCraftTask,
        "webarena": WebarenaTask,
        "sqlgym": SqlGymTask,
        "maze": MazeTask,
        "wordle": WordleTask,
        "weather": WeatherTask,
        "todo": TodoTask,
        "movie": MovieTask,
        "sheet": SheetTask,
        "academia": AcademiaTask,
        "searchqa": SearchQATask,
        "textworld": TextworldTask,
    }
    cls = task_classes.get(task_name.lower())
    if cls is None:
        raise ValueError(f"Unsupported task name: {task_name}")
    return cls


def _load_selected_idxs(inference_file: str, num_examples: int):
    with open(inference_file, "r") as f:
        data = json.load(f)

    # Keep consistent with scripts/eval/openai/eval_openai_async.py
    if "alfworld_train" in inference_file:
        all_data_idxs = list(range(len(data)))
    else:
        all_data_idxs = [int(item["item_id"].split("_")[-1]) for item in data]

    if num_examples is not None and int(num_examples) > 0:
        return all_data_idxs[: int(num_examples)], all_data_idxs
    return all_data_idxs, all_data_idxs


def main(args: dict):
    inference_file = args["inference_file"]
    output_json = args["output_json"]
    task_name = args["task_name"]

    selected_idxs, all_data_idxs = _load_selected_idxs(
        inference_file=inference_file,
        num_examples=args.get("num_examples", -1),
    )
    data_len = (max(all_data_idxs) + 1) if len(all_data_idxs) > 0 else 0

    task_cls = _task_class(task_name)
    env_args = {
        "env_server_base": args["env_server_base"],
        "data_len": data_len,
        "timeout": args["timeout"],
        "action_format": args.get("action_format", "react"),
    }
    n_workers = 150
    task = task_cls(client_args=env_args, n_clients=n_workers)
    client_pool: "queue.SimpleQueue[int]" = queue.SimpleQueue()
    for i in range(len(task.clients)):
        client_pool.put(i)

    extracted = []
    def _extract_one(data_idx: int):
        client_i = client_pool.get()
        try:
            client = task.clients[client_i]
            client.reset(data_idx)
            state = client.observe()
            cs = client.conversation_start
            if cs is None or len(cs) < 2:
                raise RuntimeError(
                    f"conversation_start must have >=2 messages, got {None if cs is None else len(cs)} for task {task_name}"
                )

            # Match agentenv/controller/task.py for APIAgent initial payload
            initial_messages = [
                {"role": "user", "content": cs[0]["value"]},
                {"role": "assistant", "content": cs[1]["value"]},
                {"role": "user", "content": state},
            ]
            return data_idx, {
                "item_id": f"{task_name}_{data_idx}",
                "data_idx": data_idx,
                "messages": initial_messages,
            }
        except Exception as e:
            # skip
            print(f"Error extracting agent instruction for data_idx {data_idx}: {e}")
            return data_idx, None
        finally:
            client_pool.put(client_i)

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_extract_one, idx) for idx in selected_idxs]
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="[Extract initial messages]",
        ):
            data_idx, item = fut.result()
            if item:
                results[data_idx] = item

    extracted = [results[idx] for idx in selected_idxs if idx in results]

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(
            extracted,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # "直接返回初始信息"：打印第一条，方便确认
    if extracted:
        print(json.dumps(extracted[0], ensure_ascii=False, indent=2))
    print(f"Saved {len(extracted)} items to {output_json}")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(ExtractArguments)
    (args,) = parser.parse_args_into_dataclasses()
    main(vars(args))