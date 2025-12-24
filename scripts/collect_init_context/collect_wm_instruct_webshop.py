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
    with ThreadPoolExecutor(max_workers=100) as executor:
        for result in tqdm(executor.map(_process_item, all_data), total=len(all_data)):
            conversations.append(result)

    print(f"Saving {len(conversations)} conversations to {output_file}")
    write_json(conversations, output_file)