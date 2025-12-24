import asyncio
import json
import os
import time
from dataclasses import dataclass, field

import transformers
from tqdm import tqdm

from agentenv.controller import APIAgent, AzureAPIAgent, Evaluator
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
class EvalArguments:
    api_key: str
    base_url: str
    model: str
    inference_file: str = field(metadata={"help": "Test dataset."})
    output_dir: str
    max_tokens: int = field(default=4096)
    temperature: float = field(default=1)
    top_p: float = field(default=1)
    num_examples: int = field(default=-1)
    max_concurrency: int = field(default=10, metadata={"help": "Max parallel tasks"})
    task_name: str = field(
        default="webshop", metadata={"help": "Task name for evaluation"}
    )

    # conversation rounds
    max_round: int = field(
        default=6,
        metadata={"help": "Interaction rounds between agents and environment"},
    )

    # environment parameters
    env_server_base: str = field(default=None)
    # data_len: int = field(default=200)
    timeout: int = field(default=2400)


def _build_evaluator(args):
    env_args = {
        "env_server_base": args["env_server_base"],
        "data_len": args["data_len"],
        "timeout": args["timeout"],
    }

    # task_name - task dict
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

    task_class = task_classes.get(args["task_name"].lower(), None)
    if task_class is None:
        raise ValueError(f"Unsupported task name: {args['task_name']}")

    api_class = AzureAPIAgent if args["api_key"] == "azure" else APIAgent
    evaluator = Evaluator(
        api_class(
            api_key=args["api_key"],
            base_url=args["base_url"],
            model=args["model"],
            max_tokens=args["max_tokens"],
            temperature=args["temperature"],
            top_p=args["top_p"],
        ),
        [task_class(client_args=env_args, n_clients=1)],
    )
    return evaluator


def _process_single_example_async(args, data_idx: int):
    """Synchronous worker used from asyncio threads.

    Returns: (score, success)
    Raises: propagates any exception from evaluator.eval so caller can count errors.
    """
    evaluator = _build_evaluator(args)

    # Do not infinite-retry; propagate exceptions to caller
    exps = evaluator.eval(
        max_rounds=args["max_round"],
        idxs=[data_idx],
    )

    score = float(exps.score)
    success = float(exps.success)

    cur_experiences = exps.experiences
    exp = cur_experiences[-1]
    conversation = exp.conversation
    reward = exp.reward

    item_id = f"{args['task_name']}_{data_idx}"
    out_path = os.path.join(args["output_dir"], f"{item_id}.json")
    with open(out_path, 'w') as f:
        json.dump(
            {
                "conversations": conversation,
                "item_id": item_id,
                "reward": reward,
                "score": score,
                "success": success,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    return score, success


async def process_single_example(args, data_idx: int, sem: asyncio.Semaphore):
    async with sem:
        return await asyncio.to_thread(_process_single_example_async, args, data_idx)


async def main(args):
    DATA_PATH = args["inference_file"]

    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, "r") as file:
        test_data = json.load(file)

    # Preserve order
    if "alfworld_train" in args["inference_file"]:
        # AgentGym/agentenv-alfworld/configs/mappings_train.json
        all_data_idxs = [i for i in range(len(test_data))]
    else:
        all_data_idxs = [int(item["item_id"].split("_")[-1]) for item in test_data]

    # Infer data_len from dataset indices so env length matches
    args["data_len"] = (max(all_data_idxs) + 1) if len(all_data_idxs) > 0 else 0

    max_n = int(args.get("num_examples", -1))
    if max_n is not None and max_n > 0:
        selected_idxs = all_data_idxs[:max_n]
    else:
        selected_idxs = all_data_idxs

    os.makedirs(args["output_dir"], exist_ok=True)

    # Resume: load completed
    completed = {}
    print("Checking for existing results to resume...")
    for idx in tqdm(selected_idxs):
        out_file = os.path.join(args["output_dir"], f"{args['task_name']}_{idx}.json")
        if os.path.exists(out_file):
            try:
                with open(out_file, "r") as f:
                    item = json.load(f)
                    completed[idx] = (
                        float(item.get("reward", 0.0)),
                        float(item.get("success", 0.0)),
                    )
            except Exception:
                pass

    pending_idxs = [idx for idx in selected_idxs if idx not in completed]

    total_selected = len(selected_idxs)
    remaining = len(pending_idxs)
    processed_count = total_selected - remaining
    if processed_count > 0:
        print(f"Resuming: {remaining} remaining out of {total_selected} examples.")
    else:
        print(f"Total examples to run: {total_selected}.")

    total_score = sum(v[0] for v in completed.values())
    total_success = sum(v[1] for v in completed.values())
    processed_count = processed_count

    start_time = time.time()
    sem = asyncio.Semaphore(int(args.get("max_concurrency", 10)))

    # Create tasks
    tasks = [process_single_example(args, idx, sem) for idx in pending_idxs]

    error_count, new_processed_count = 0, 0
    with tqdm(total=total_selected, initial=processed_count, desc="[Evaluation]") as pbar:
        # Initialize postfix with current accuracy when resuming
        init_acc = round((total_success / processed_count) * 100, 2) if processed_count > 0 else 0
        pbar.set_postfix(acc=f"{init_acc:.2f}%", processed=0, api_errors=0)
        for coro in asyncio.as_completed(tasks):
            try:
                score, success = await coro
                total_score += score
                total_success += success
                processed_count += 1
                new_processed_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error during evaluation: {e}")
            finally:
                pbar.update(1)
                cur_acc = round((total_success / processed_count) * 100, 2) if processed_count > 0 else 0
                pbar.set_postfix(acc=f"{cur_acc:.2f}%", processed=new_processed_count, api_errors=error_count)

    process_time = time.time() - start_time
    Score = round(total_score / processed_count, 2) if processed_count > 0 else 0
    Success = round(total_success / processed_count * 100, 2) if processed_count > 0 else 0
    print("\n\n==== EVALUATION ====\n")
    print(f"Score: {Score}, {total_score}/{processed_count}")
    print(f"Success: {Success}%, {total_success}/{processed_count}")
    print(f"Errors count: {error_count}/{processed_count+error_count}")
    print(f"Time: {process_time} seconds")
    print(f"Saved to {args['output_dir']}")
    metrics = {
        "accuracy": Success,
        "success": total_success,
        "api_errors": error_count,
        "total": processed_count + error_count,
        "time_seconds": process_time,
    }
    metrics_path = os.path.join(args["output_dir"], "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(EvalArguments)
    (args,) = parser.parse_args_into_dataclasses()
    args = vars(args)
    print(args)
    print(json.dumps(args, indent=2, ensure_ascii=False))
    asyncio.run(main(args))
