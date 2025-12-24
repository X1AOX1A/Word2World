from agentenv.envs import AlfWorldEnvClient, SciworldEnvClient, TextworldEnvClient, WebshopEnvClient
from agentenv.controller.types import ActionFormat

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_json(file_name):
    with open(file_name, "r") as f:
        dict_objs = json.load(f)
    return dict_objs

import os
import json
def write_json(dict_objs, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w+", encoding='utf-8') as f:
        json.dump(dict_objs, f, indent=4, ensure_ascii=False)

class AlfWorldEnv:
    def __init__(self, env_addr: str, index: int, *, timeout: int = 300):
        self.client = AlfWorldEnvClient(
            env_server_base=env_addr,
            data_len=-1,
            timeout=timeout,
            action_format=ActionFormat.REACT,
        )
        self.client.reset(game=index)
        # init_observation = self.client.observe()

    def step(self, react_message) -> dict:
        step_output = self.client.step(react_message)
        observation = step_output.state
        reward = step_output.reward
        done = step_output.done
        success = False
        if done:
            success = True if (reward==1 or reward==100) else False
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "success": success,
        }


class SciWorldEnv:
    def __init__(self, env_addr: str, index: int, *, timeout: int = 300):
        self.client = SciworldEnvClient(
            env_server_base=env_addr,
            data_len=-1,
            timeout=timeout,
            action_format=ActionFormat.REACT,
        )
        self.client.reset(data_idx=index)
        # init_observation = self.client.observe()

    def step(self, react_message) -> dict:
        step_output = self.client.step(react_message)
        observation = step_output.state
        reward = step_output.reward
        done = step_output.done
        success = False
        if done:
            success = True if (reward==1 or reward==100) else False
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "success": success,
        }


class TextWorldEnv:
    def __init__(self, env_addr: str, *, index=1, timeout: int = 300):
        self.client = TextworldEnvClient(
            env_server_base=env_addr,
            data_len=-1,
            timeout=timeout,
            action_format=ActionFormat.REACT,
            games_dir="data/textworld/games",
        )

        self.client.reset(data_idx=index, game_path="data/textworld/games")
        # init_observation = self.client.observe()

    def step(self, react_message) -> dict:
        step_output = self.client.step(react_message)
        observation = step_output.state
        reward = step_output.reward
        done = step_output.done
        success = False
        if done:
            success = True if (reward==1 or reward==100) else False
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "success": success,
        }


class WebshopEnv:
    def __init__(self, env_addr: str, index: int, *, timeout: int = 300):
        self.client = WebshopEnvClient(
            env_server_base=env_addr,
            data_len=-1,
            timeout=timeout,
            action_format=ActionFormat.REACT,
        )
        # Reset environment and capture initial state details
        self.client.reset(idx=index)
        # init_observation = self.client.observe()
        # init_instruction_text = self.client.get_instruction_text()
        # init_available_actions = self.client.get_available_actions()
        # self.init_observation = {
        #     "observation": init_observation,
        #     "instruction_text": init_instruction_text,
        #     "available_actions": init_available_actions,
        # }

    def step(self, react_message) -> dict:
        step_output = self.client.step(react_message)
        observation = step_output.state
        reward = step_output.reward
        done = step_output.done
        success = False
        if done:
            success = True if (reward==1 or reward==100) else False
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "success": success,
        }


def process_single_example(json_file, port=36001, verbose=True):
    item = read_json(json_file)
    game_idx = int(item["item_id"].split(TASK+"_")[-1].split("_")[0].split(".json")[0])

    if "alfworld" in TASK:
        env = AlfWorldEnv(env_addr=f"http://localhost:{port}", index=game_idx)
    elif TASK == "sciworld":
        env = SciWorldEnv(env_addr=f"http://localhost:{port}", index=game_idx)
    elif TASK == "textworld":
        env = TextWorldEnv(env_addr=f"http://localhost:{port}", index=game_idx)
    elif TASK == "webshop":
        env = WebshopEnv(env_addr=f"http://localhost:{port}", index=game_idx)

    # save to original folder/valid_on_real_env/ +  json_file
    folder, file_name = os.path.split(json_file)
    output_file = os.path.join(folder, "valid_on_real_env", file_name)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    agent_reacts = []
    world_model_observations = []
    for message in item["conversations"][3:]:
        if message["role"] == "assistant":
            agent_reacts.append(message["content"])
        if message["role"] == "user":
            world_model_observations.append(message["content"])

    assert len(agent_reacts) == len(world_model_observations)

    output = {}
    output["conversation"] = []
    output["item_id"] = item["item_id"]
    output["world_model_success"] = item["success"]
    for react, world_model_obs in zip(agent_reacts, world_model_observations):
        result = env.step(react)
        if verbose:
            print("#"*20)
            print("React Message:")
            print(react)
            print("Observation:")
            print(result["observation"])

        output["conversation"].append({
            "react": react,
            "world_model_observation": world_model_obs,
            "env_observation": result["observation"].split("\nAVAILABLE ACTIONS")[0].strip() if result["observation"] else "",
        })
        if result["done"]:
            output["reward"] = result["reward"]
            output["success"] = result["success"]
            write_json(output, output_file)
            return result["success"]==1

    if verbose: print("Failed")
    output["reward"] = result["reward"]
    output["success"] = 0
    write_json(output, output_file)
    return False

def main(test_file_root, port=36001, max_workers=40):
    import os
    from tqdm import tqdm
    import time

    # Only count per-sample jsons produced by run.py, e.g. "{TASK}_{id}.json"
    # Avoid including aggregated files like "metrics.json" which would double-count.
    json_files = [
        os.path.join(test_file_root, f)
        for f in os.listdir(test_file_root)
        if f.endswith(".json") and f.startswith(f"{TASK}_")
    ]
    total_files = len(json_files)

    # Resume: check existing results under valid_on_real_env
    success_count = 0
    processed = 0
    error_count = 0
    start_time = time.time()

    pending_files = []
    for src_path in json_files:
        folder, file_name = os.path.split(src_path)
        out_path = os.path.join(folder, "valid_on_real_env", file_name)
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                    if isinstance(prev, dict) and "success" in prev:
                        processed += 1
                        success_count += int(prev["success"])  # 1 or 0
                    else:
                        pending_files.append(src_path)
            except Exception:
                # Corrupted or unreadable output; re-run
                pending_files.append(src_path)
        else:
            pending_files.append(src_path)

    # Init progress with previously processed
    progress = tqdm(total=total_files, initial=processed, desc="Processing")
    init_acc = (success_count / processed * 100) if processed > 0 else 0.0
    progress.set_postfix({
        "accuracy": f"{success_count}/{processed} = {init_acc:.2f}",
        "api_errors": error_count,
    })

    def _run_single(path):
        return process_single_example(path, port=port, verbose=False)

    # for file_path in pending_files:
    #     if _run_single(file_path):
    #         success_count += 1

    #     processed += 1
    #     success_rate = success_count / processed * 100 if processed > 0 else 0.0
    #     progress.set_postfix({
    #         "accuracy": f"{success_count}/{processed} = {success_rate:.2f}",
    #         "errors": error_count,
    #     })
    #     progress.update(1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if len(pending_files) > 0:
            future_to_file = {executor.submit(_run_single, file_path): file_path for file_path in pending_files}
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    if future.result():
                        success_count += 1
                except Exception as exc:
                    error_count += 1
                    tqdm.write(f"Error processing {file_path}: {exc}")
                processed += 1
                success_rate = success_count / processed * 100 if processed > 0 else 0.0
                progress.set_postfix({
                    "accuracy": f"{success_count}/{processed} = {success_rate:.2f}",
                    "errors": error_count,
                })
                progress.update(1)

    progress.close()

    print(test_file_root)
    print(f"Success Rate: {success_count}/{total_files-error_count} = {success_count/(total_files-error_count)*100:.2f}")
    print(f"Errors count: {error_count}/{total_files}")
    print(f"Time: {time.time() - start_time:.2f} seconds")
    print(f"Interaction histories saved to {os.path.join(test_file_root, 'valid_on_real_env/')}")
    # save acc json to output root
    acc_output_file = os.path.join(test_file_root, "valid_on_real_env", "_metrics.json")
    metrics = {
        "total": total_files,
        "processed": processed,
        "success": success_count,
        "errors": error_count,
        "accuracy": success_count/(total_files-error_count) if (total_files-error_count)>0 else 0.0,
    }
    write_json(metrics, acc_output_file)
    print(f"Saved overall metrics to {acc_output_file}")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["alfworld", "alfworld_valid_seen", "alfworld_valid_unseen", "sciworld", "textworld", "webshop"], help="The task name.")
    parser.add_argument("--test_file_root", type=str, required=True, help="The folder containing test json files.")
    parser.add_argument("--port", type=int, required=False, default=36001, help="The port of the real environment server.")
    parser.add_argument("--max_workers", type=int, required=False, default=50, help="The max workers for ThreadPoolExecutor.")
    args = parser.parse_args()

    global TASK
    TASK = args.task
    main(args.test_file_root, port=args.port, max_workers=args.max_workers)