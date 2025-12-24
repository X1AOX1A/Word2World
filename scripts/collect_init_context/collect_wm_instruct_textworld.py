import os
import json
def write_json(dict_objs, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w+", encoding='utf-8') as f:
        json.dump(dict_objs, f, indent=4, ensure_ascii=False)

def read_json(file_name):
    with open(file_name, "r") as f:
        dict_objs = json.load(f)
    return dict_objs

def parse_action(text: str):
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
    _split = text.rsplit("Action:", 1)
    if len(_split) == 0:
        _thought, _action = text
    elif len(_split) == 1:
        if "search[" in text or "click[" in text:
            _thought, _action = "", _split[0]
        else:
            _thought, _action = _split[0], ""
    else:
        assert len(_split) == 2
        _thought, _action = _split
    action = _action.strip()
    return action

def process_item(item):
    messages = [
        {"role": "system", "content": item["messages"][2]["content"]},
    ]
    return messages

def _process_item(item):
    id = item["data_idx"]
    messages = process_item(item)
    return {"id": id, "messages": messages}


from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
if __name__ == "__main__":
    num_examples = -1
    global ACTION_ONLY
    ACTION_ONLY = True
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_instruct", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    agent_instruct = args.agent_instruct
    output_file = args.output_file

    all_data = read_json(agent_instruct)
    if num_examples > 0:
        all_data = all_data[:num_examples]

    conversations = []
    # for file in tqdm(all_files):
    #     conversations.append(_process_file(file))
    with ThreadPoolExecutor(max_workers=100) as executor:
        for result in tqdm(executor.map(_process_item, all_data), total=len(all_data)):
            conversations.append(result)

    print(f"Saving {len(conversations)} conversations to {output_file}")
    write_json(conversations, output_file)