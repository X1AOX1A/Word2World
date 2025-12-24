import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from azure_openai import get_client as get_azure_client


def write_json(dict_objs, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w+", encoding='utf-8') as f:
        json.dump(dict_objs, f, indent=4, ensure_ascii=False)

def read_json(file_name):
    with open(file_name, "r") as f:
        dict_objs = json.load(f)
    return dict_objs

class WorldModel:
    def __init__(self, wm_messages, client, model_name="llm_world_model"):
        self.history = wm_messages
        self.client = client
        self.model_name = model_name

    def llm_generate(self, messages):
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=512,
            temperature=0,
            top_p=1,
        )
        return response.choices[0].message.content

    def done(self, observation):
        if " [SUCCESS]" in observation:
            return observation.split(" [SUCCESS]")[0], True
        else:
            return observation, False

    def step(self, action):
        self.history.append({"role": "user", "content": action})
        observation = self.llm_generate(self.history)
        observation, done = self.done(observation)
        self.history.append({"role": "assistant", "content": observation})
        return observation, done


class ReactAgent:
    def __init__(self, agent_messages, agent_model_name, api_key, api_base_url):
        self.history = agent_messages
        self.agent_model_name = agent_model_name
        self.api_key = api_key
        self.api_base_url = api_base_url

    def llm_generate(self, messages):
        if "azure" in self.api_key:
            self.client, self.model_name = get_azure_client(model_name=self.agent_model_name)
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_base_url)
            self.model_name = self.agent_model_name
        if "gpt-5" in self.model_name or "claude" in self.model_name:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
            )
        else:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=4096,
                temperature=1,
                top_p=1,
            )
        return response.choices[0].message.content

    def parse_action(self, text: str):
        # AgentGym/agentenv/agentenv/controller/utils.py:L118
        """
        ReAct format:
        ```
        Thought:
        I think ...

        Action:
        action
        ```
        """
        invalid_format_flg = False
        _split = text.rsplit("Action:", 1)
        if len(_split) == 0:
            _thought, _action = text
            invalid_format_flg = True
        elif len(_split) == 1:
            if "search[" in text or "click[" in text:
                _thought, _action = "", _split[0]
            else:
                _thought, _action = _split[0], ""
            invalid_format_flg = True
        else:
            assert len(_split) == 2
            _thought, _action = _split

        thought = _thought.split("Thought:")
        if len(thought) == 1:
            thought = thought[0]
            invalid_format_flg = True
        else:
            thought = thought[1].strip()
        action = _action.strip()
        if invalid_format_flg:
            print(
                "The text is not in the correct format. Parsing result may not be accurate."
            )
            print("###RAW TEXT:\n", text)
            print("\n###PARSED THOUGHT:\n", thought)
            print("\n###PARSED ACTION:\n", action)
        return thought, action

    def react(self, observation):
        self.history.append({"role": "user", "content": observation})
        max_retries=50
        for i in range(max_retries):
            try:
                react = self.llm_generate(self.history)
                if react is not None:
                    break
            except Exception as e:
                if i == max_retries - 1:
                    raise e
        self.history.append({"role": "assistant", "content": react})
        thought, action = self.parse_action(react)
        return react, thought, action



def process_single_sample(
    agent_messages,
    wm_messages,
    id,
    max_steps=50,
    output_file=".outputs/alfworld_X.json",
    agent_model: str = "gpt-4o",
    wm_port: int = 8000,
    wm_name: str = "llm_world_model",
):
    wm_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{wm_port}/v1")
    world_model = WorldModel(wm_messages, wm_client, model_name=wm_name)
    react_agent = ReactAgent(agent_messages[:-1], agent_model)
    observation = agent_messages[-1]["content"]

    try:
        # run interaction loop
        for _ in range(max_steps):
            react, thought, action = react_agent.react(observation)
            observation, done = world_model.step(action)
            if done:
                break

        # save interaction history
        output_data = {
            "conversations": react_agent.history + [{"role": "user", "content": observation}],
            "item_id": f"{TASK}_{id}",
            "reward": 1.0 if done else 0.0,
            "success": 1 if done else 0,
        }
        write_json(output_data, output_file)
        # print(f"Saved interaction history to {output_file}")
        return 1 if done else 0
    except Exception as e:
        output_data = {
            "item_id": f"{TASK}_{id}",
            "conversations": react_agent.history,
            "error": str(e),
        }
        write_json(output_data, f".debug/error_{TASK}_{id}.json")
        print(f"Error occurred during processing sample {id}. Saved debug info.")
        raise


def main():
    global TASK

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Interact with world model using a React agent")
    task_choices = ["alfworld", "sciworld", "textworld", "webshop"]
    parser.add_argument(
        "--task",
        type=str,
        choices=task_choices,
        default="alfworld",
        help="Task to run: alfworld | sciworld | textworld | webshop",
    )
    parser.add_argument(
        "--model",
        type=str,
        # choices=model_choices,
        default="gpt-4o",
        help="Agent model for Azure OpenAI endpoints",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for Azure OpenAI endpoints",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="",
        help="API base url for Azure OpenAI endpoints",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max interaction steps per item",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=150,
        help="Max concurrent items to process",
    )
    parser.add_argument(
        "--wm-port",
        type=int,
        default=8000,
        help="World model server port (default: 8000)",
    )
    parser.add_argument(
        "--wm-name",
        type=str,
        default="llm_world_model_0",
        help="World model served name (default: llm_world_model)",
    )
    parser.add_argument(
        "--agent-instruct-file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--wm-instruct-file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="./outputs/interact_with_world_model/",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all)",
    )

    args = parser.parse_args()

    TASK = args.task
    agent_model = args.model
    max_steps = args.max_steps
    max_concurrency = args.max_concurrency
    wm_port = args.wm_port
    wm_name = args.wm_name
    agent_instruct_file = args.agent_instruct_file
    wm_instruct_file = args.wm_instruct_file
    output_root = args.output_root

    agent_instruct_data = read_json(agent_instruct_file)
    wm_instruct_data = read_json(wm_instruct_file)

    def merge_data(agent_instruct_data, wm_instruct_data):
        agent_instruct_data = {item["data_idx"]: item for item in agent_instruct_data}
        wm_instruct_data = {item["id"]: item for item in wm_instruct_data}
        merged_data = []
        for id in agent_instruct_data.keys():
            if id in wm_instruct_data:
                merged_data.append({
                    "id": id,
                    "agent_messages": agent_instruct_data[id]["messages"],
                    "wm_messages": wm_instruct_data[id]["messages"],
                })
            else:
                print(f"WARNING: id {id} is not in wm_instruct_data")
        return merged_data

    test_data = merge_data(agent_instruct_data, wm_instruct_data)
    total_items = len(test_data)
    if args.n_samples > 0:
        test_data = test_data[: args.n_samples]
    print(f"Loaded {total_items} items from {agent_instruct_file} and {wm_instruct_file}")

    pending_items = []
    processed_count = 0
    total_success = 0.0
    for item in test_data:
        id = item["id"]
        agent_messages = item["agent_messages"]
        wm_messages = item["wm_messages"]

        # init_observation = item["messages"][0]["content"]  # with env description
        output_file = os.path.join(output_root, f"{TASK}_{id}.json")
        if os.path.exists(output_file):
            existing = read_json(output_file)
            success_val = existing.get("success")
            if success_val is None:
                success_val = 1 if (existing.get("reward", 0)==1 or existing.get("reward", 0)==100) else 0
            success_val = float(success_val)
            total_success += success_val
            processed_count += 1
        else:
            pending_items.append((agent_messages, wm_messages, id, output_file))

    remaining = len(pending_items)
    if processed_count > 0:
        init_acc = (total_success / processed_count) * 100 if processed_count else 0

    if remaining < total_items:
        print(f"Resuming: {remaining} remaining out of {total_items} items.")
    else:
        print(f"Total items to run: {total_items}.")

    if not pending_items:
        print("All interaction histories already exist. Nothing to run.")
        if processed_count > 0:
            final_acc = (total_success / processed_count) * 100 if processed_count else 0
            print(f"Final accuracy: {final_acc:.2f}% ({total_success}/{processed_count}).")
        return

    # # debug with for loop
    # for init_observation, item_id, output_file in pending_items:
    #     process_single_sample(agent_messages, wm_messages, item_id, max_steps, output_file, agent_model, wm_port, wm_name)

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                process_single_sample,
                agent_messages,
                wm_messages,
                item_id,
                max_steps,
                output_file,
                agent_model,
                wm_port,
                wm_name,
            ): (item_id, output_file)
            for agent_messages, wm_messages, item_id, output_file in pending_items
        }

        error_count = 0
        new_processed_count = 0
        with tqdm(
            total=total_items,
            initial=processed_count,
            desc="Running agents",
        ) as pbar:
            init_acc = (total_success / processed_count * 100) if processed_count else 0
            pbar.set_postfix(acc=f"{init_acc:.2f}%", processed=0, api_errors=0)

            for future in as_completed(futures):
                item_id, output_file = futures[future]
                try:
                    success = future.result()
                    total_success += success
                    processed_count += 1
                    new_processed_count += 1
                except Exception as exc:
                    error_count += 1
                    print(f"Error during interaction for id {item_id}: {exc}")
                finally:
                    pbar.update(1)
                    cur_acc = (total_success / processed_count * 100) if processed_count else 0
                    pbar.set_postfix(
                        acc=f"{cur_acc:.2f}%",
                        processed=new_processed_count,
                        api_errors=error_count,
                    )

    final_acc = (total_success / processed_count * 100) if processed_count else 0
    print(f"\nFinal accuracy: {final_acc:.2f}% ({total_success}/{processed_count}). API errors: {error_count}.")
    print(f"Interaction histories saved to {output_root}")
    # save acc json to output root
    acc_output_file = os.path.join(output_root, f"_metrics.json")
    acc_data = {
        "task": TASK,
        "agent_model": agent_model,
        "total_items": total_items,
        "total_success": total_success,
        "processed_items": processed_count,
        "accuracy": final_acc,
        "api_errors": error_count,
    }
    write_json(acc_data, acc_output_file)
    print(f"Saved overall metrics to {acc_output_file}")


if __name__ == "__main__":
    main()