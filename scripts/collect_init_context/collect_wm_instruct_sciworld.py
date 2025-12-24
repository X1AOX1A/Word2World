import os
import json

import argparse
import sys
from typing import Any, Dict, List, Tuple
from scienceworld import ScienceWorldEnv

EXCEPTIONS = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}


def build_games_order() -> List[Tuple[str, int]]:
    """按服务端相同逻辑构造 (taskName, variationIdx) 列表。"""
    env = ScienceWorldEnv()
    try:
        pairs: List[Tuple[str, int]] = []
        for task_id, task_name in env.tasks.items():  # OrderedDict(ID2TASK)
            if task_id in EXCEPTIONS:
                continue
            max_var = env.get_max_variations(task_name)
            for v in range(max_var):
                pairs.append((task_name, v))
        return pairs
    finally:
        env.close()


def get_full_snapshot(env: ScienceWorldEnv) -> Dict[str, Any]:
    env.step("look around")
    def safe(fn):
        try:
            return fn()
        except Exception:
            return None
    data = {
        "observation": safe(env.look),
        "inventory": safe(env.inventory),
        "task_description": safe(env.get_task_description),
        "goal_progress": safe(env.get_goal_progress),
        "possible_actions": safe(env.get_possible_actions),
        "possible_actions_with_ids": safe(env.get_possible_actions_with_IDs),
        "possible_objects": safe(env.get_possible_objects),
        "valid_action_object_combinations": safe(env.get_valid_action_object_combinations),
        "valid_action_object_combinations_with_templates": safe(env.get_valid_action_object_combinations_with_templates),
        "object_tree": safe(env.getObjectTree),
        "current_moves": safe(env.get_num_moves),
    }
    # Referent LUT（对象指称映射），若可用则加入
    try:
        data["possible_object_referent_lut"] = env.get_possible_object_referent_LUT()
    except Exception:
        data["possible_object_referent_lut"] = None
    return data

def collect_room_views(env: ScienceWorldEnv) -> List[Dict[str, Any]]:
    """使用 teleportAction，逐房间采集 look 文本与 possible_objects。

    要求：env 当前已 load(..., simplificationStr='teleportAction') 并已 reset/step 过。
    返回：[ { 'room': str, 'look': str, 'possible_objects': List[str] }, ... ]
    """
    out: List[Dict[str, Any]] = []
    ot = env.getObjectTree() or {}
    rooms = [r.get('name') for r in (ot.get('contents') or {}).values() if isinstance(r, dict)]
    for room in rooms:
        if not room:
            continue
        env.step(f"teleport {room}")
        look = env.look()
        pos = env.get_possible_objects()
        out.append({
            'room': room,
            'look': look,
            'possible_objects': pos,
        })
    return out


def get_sciworld_description(id):
    games = build_games_order()
    if id < 0 or id >= len(games):
        print(f"[Error] id 超出范围: 0 <= id < {len(games)}")
        sys.exit(1)

    task_name, var_idx = games[id]
    env = ScienceWorldEnv()
    env.load(task_name, var_idx, 'teleportAction')

    env_description = ""

    snapshot = get_full_snapshot(env)

    def section(title: str):
        return f"\n=== {title} ===\n"
    # env_description += section("Task Description")
    # env_description += f"{snapshot.get('task_description') or ''}\n"
    # env_description += section("Observation (look)")
    # env_description += f"{snapshot.get('observation') or ''}\n"
    # env_description += section("Inventory")
    # env_description += f"{snapshot.get('inventory') or ''}\n"
    env_description += section("Goal Progress")
    env_description += f"{snapshot.get('goal_progress') or ''}\n"
    actions = snapshot.get("possible_actions") or []
    env_description += section(f"Possible Actions")
    for a in actions:
        env_description += f" - {a}\n"

    # objs = snapshot.get("possible_objects") or []
    # env_description += section(f"Possible Objects ({len(objs)})")
    # if objs:
    #     env_description += ", ".join(objs) + "\n"

    # combos = snapshot.get("valid_action_object_combinations_with_templates") or []
    # if combos:
    #     env_description += section(f"Valid Action-Object Templates ({len(combos)}) [showing up to 100]")
    #     for item in combos[:100]:
    #         action = item.get("action")
    #         tmpl_id = item.get("template_id")
    #         obj_ids = item.get("obj_ids")
    #         env_description += f" - [{tmpl_id}] {action} :: {obj_ids}\n"

    # ot = snapshot.get("object_tree")
    # if ot is not None:
    #     env_description += section("Object Tree (JSON head)")
    #     text = json.dumps(ot, ensure_ascii=False)
    #     head = text[:2000]
    #     env_description += head + ("..." if len(text) > 2000 else "") + "\n"

    # env_description += section("Compact Summary from Object Tree")
    # env_description += compact_text(snapshot) + "\n"

    # env_description += section("Misc")
    # env_description += f"Moves: {snapshot.get('current_moves')}\n"

    env_description += "\n=== Per-Room Observations ===\n"
    room_views = collect_room_views(env)
    for rv in room_views:
        room = rv['room']
        text = rv['look']
        objs = rv.get('possible_objects') or []
        env_description += f"== Room: {room} ==\n"
        # 直接输出 look 文本原样，包含 In it, you see / You also see（doors）
        env_description += text.strip() + "\n"
        env_description += "Possible Objects:" + (" " + ", ".join(objs) if objs else " (none)") + "\n\n"

    env.close()
    return env_description




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

def process_item(item, id):
    env_description = get_sciworld_description(id)

    system_prompt_template = """# Environment Information (Only visible to Assistant)

{env_description}

# User Environment Information (Displayed to User)

{user_instruction}
"""

    messages = []
    user_instruction = item["messages"][2]["content"]
    system_prompt = system_prompt_template.format(
        env_description=env_description,
        user_instruction=user_instruction
    )
    messages.append({"role": "system", "content": system_prompt})
    return messages

def _process_item(item):
    id = item["data_idx"]
    messages = process_item(item, id)
    return {"id": id, "messages": messages}

def _process_item_safe(item):
    try:
        return _process_item(item)
    except Exception as e:
        return {"__error__": True, "item": item.get("item_id", None), "id": item.get("data_idx", None), "error": str(e)}

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
    err_count = 0
    with ThreadPoolExecutor(max_workers=20) as executor:
        with tqdm(total=len(all_data)) as pbar:
            # 显示初始的 err_count=0
            pbar.set_postfix({"errors": err_count})
            for result in executor.map(_process_item_safe, all_data):
                if isinstance(result, dict) and result.get("__error__"):
                    err_count += 1
                elif result is None:
                    err_count += 1
                else:
                    conversations.append(result)
                # 每次迭代都更新显示当前错误数
                pbar.set_postfix({"errors": err_count})
                pbar.update(1)

    print(f"Saving {len(conversations)} conversations to {output_file}. Errors: {err_count}")
    write_json(conversations, output_file)