import os
import json
from pathlib import Path

def write_json(dict_objs, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w+", encoding='utf-8') as f:
        json.dump(dict_objs, f, indent=4, ensure_ascii=False)

def read_json(file_name):
    with open(file_name, "r") as f:
        dict_objs = json.load(f)
    return dict_objs

#!/usr/bin/env python3
"""输出 ALFWorld 某条轨迹的完整状态描述（语义关系 + 坐标）。"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_IN_RECEP_RE = re.compile(r"\(inReceptacle ([^ )]+) ([^ )]+)\)")
_RECEPTACLE_LOC_RE = re.compile(r"\(receptacleAtLocation ([^ )]+) ([^ )]+)\)")
_OBJECT_AT_LOC_RE = re.compile(r"\(objectAtLocation ([^ )]+) ([^ )]+)\)")


@dataclass
class GameMetadata:
    id_to_label: Dict[str, str]
    base_to_labels: Dict[str, List[str]]
    base_to_ids: Dict[str, List[str]]
    alias_to_base: Dict[str, str]
    object_to_receptacle: Dict[str, str]
    location_to_receptacle: Dict[Tuple[int, ...], List[str]]
    object_to_location: Dict[str, Tuple[int, ...]]

    def humanize_id(self, full_id: Optional[str]) -> Optional[str]:
        if full_id is None:
            return None
        label = self.id_to_label.get(full_id)
        if label:
            return label
        # Some planner ids include additional suffixes (e.g., '|SinkBasin').
        parts = full_id.split('|')
        for end in range(len(parts), 1, -1):
            candidate = '|'.join(parts[:end])
            label = self.id_to_label.get(candidate)
            if label:
                return label
        base = full_id.split('|', 1)[0]
        return self.humanize_base(base)

    def humanize_base(self, base_token: Optional[str]) -> str:
        if not base_token:
            return 'object'
        canonical = self.alias_to_base.get(_normalize_alias(base_token))
        if canonical is None:
            return base_token.replace('_', ' ')
        labels = self.base_to_labels.get(canonical.lower())
        if labels:
            return labels[-1]
        return camel_to_text(canonical)

    def humanize_location(self, loc: str, fallback: Optional[str] = None) -> str:
        loc_tuple = loc_to_tuple(loc)
        candidates = []
        if loc_tuple in self.location_to_receptacle:
            candidates.extend(self.location_to_receptacle[loc_tuple])
        if len(loc_tuple) == 4:
            prefix = loc_tuple[:3]
            for key, ids in self.location_to_receptacle.items():
                if key[:3] == prefix:
                    candidates.extend(ids)
        if not candidates:
            return self.humanize_base(fallback)
        candidates = sorted(set(candidates))
        label = self.humanize_id(candidates[0])
        return label or self.humanize_base(fallback)

    def receptacle_for_object(self, obj_id: Optional[str]) -> Optional[str]:
        if not obj_id:
            return None
        recep = self.object_to_receptacle.get(obj_id)
        if recep:
            return recep
        loc = self.object_to_location.get(obj_id)
        if loc and loc in self.location_to_receptacle:
            return self.location_to_receptacle[loc][0]
        return None


def camel_to_text(name: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return spaced.replace('_', ' ').lower()


def _normalize_alias(text: str) -> str:
    return ''.join(ch for ch in text.lower() if ch.isalnum())


def _format_entity(label: Optional[str]) -> str:
    if not label:
        return 'object'
    parts = label.strip().split()
    if not parts:
        return 'object'
    if parts[-1].isdigit():
        number = parts[-1]
        base_parts = parts[:-1]
        base = ''.join(base_parts).lower() if base_parts else 'object'
        return f"{base} {number}"
    return ''.join(parts).lower()


def loc_to_tuple(loc: str) -> Tuple[int, ...]:
    if not loc.startswith('loc|'):
        return tuple()
    parts = loc.split('|')[1:]
    values = []
    for part in parts:
        if part:
            try:
                values.append(int(float(part)))
            except ValueError:
                pass
    return tuple(values)


def demangle(token: str) -> str:
    text = token
    text = text.replace('_bar_', '|')
    text = text.replace('_minus_', '-')
    text = text.replace('_plus_', '+')
    text = text.replace('_dot_', '.')
    text = text.replace('_comma_', ',')
    return text


def parse_game_metadata(pddl_source: Path) -> GameMetadata:
    if pddl_source.suffix == '.pddl' and not pddl_source.name.endswith('.tw-pddl'):
        pddl = pddl_source.read_text()
    else:
        data = json.loads(pddl_source.read_text())
        pddl = data['pddl_problem']

    base_to_ids_raw: Dict[str, set[str]] = {}
    for token in _TOKEN_RE.findall(pddl):
        if '_bar_' not in token:
            continue
        demangled = demangle(token)
        if demangled.startswith('loc|'):
            continue
        base = demangled.split('|', 1)[0]
        if 'basin' in token.lower():
            base += 'basin'
        base_to_ids_raw.setdefault(base, set()).add(token)

    id_to_label: Dict[str, str] = {}
    base_to_labels: Dict[str, List[str]] = {}
    base_to_ids: Dict[str, List[str]] = {}
    alias_to_base: Dict[str, str] = {}
    for base, encoded_ids in base_to_ids_raw.items():
        sorted_encoded_ids = sorted(encoded_ids)
        sorted_ids = [demangle(enc_id) for enc_id in sorted_encoded_ids]
        labels = []
        num_stack = list(range(1, len(sorted_ids) + 1))
        for full_id in sorted_ids:
            label = f"{camel_to_text(base)} {num_stack.pop()}"
            id_to_label[full_id] = label
            labels.append(label)
        key = base.lower()
        base_to_labels[key] = labels
        base_to_ids[key] = sorted_ids
        alias_to_base[_normalize_alias(base)] = base
        alias_to_base[key] = base
        alias_to_base[_normalize_alias(camel_to_text(base))] = base

    object_to_receptacle: Dict[str, str] = {}
    for obj_token, rec_token in _IN_RECEP_RE.findall(pddl):
        obj_id = demangle(obj_token)
        rec_id = demangle(rec_token)
        if rec_id.startswith('loc|'):
            continue
        object_to_receptacle[obj_id] = rec_id

    location_to_receptacle: Dict[Tuple[int, ...], List[str]] = {}
    for rec_token, loc_token in _RECEPTACLE_LOC_RE.findall(pddl):
        rec_id = demangle(rec_token)
        loc_id = demangle(loc_token)
        loc_tuple = loc_to_tuple(loc_id)
        if not loc_tuple:
            continue
        location_to_receptacle.setdefault(loc_tuple, []).append(rec_id)

    object_to_location: Dict[str, Tuple[int, ...]] = {}
    for obj_token, loc_token in _OBJECT_AT_LOC_RE.findall(pddl):
        obj_id = demangle(obj_token)
        loc_tuple = loc_to_tuple(demangle(loc_token))
        if loc_tuple:
            object_to_location[obj_id] = loc_tuple

    return GameMetadata(id_to_label, base_to_labels, base_to_ids, alias_to_base, object_to_receptacle, location_to_receptacle, object_to_location)

TOKEN_RE = re.compile(r"\(([^()]+)\)")
OBJECT_SECTION_RE = re.compile(r":objects(.*?)(?::init|\Z)", re.S)
TYPED_OBJECT_RE = re.compile(r"([^\s]+)\s*-\s*([A-Za-z0-9_-]+)")


def demangle(token: str) -> str:
    return (
        token.replace("_bar_", "|")
        .replace("_minus_", "-")
        .replace("_plus_", "+")
        .replace("_dot_", ".")
        .replace("_comma_", ",")
    )


def _format_location(loc: str, metadata) -> str:
    label = metadata.humanize_location(loc, None)
    if label and label != "object":
        return _format_entity(label)
    loc_tuple = loc_to_tuple(loc)
    if loc_tuple:
        return "loc" + str(loc_tuple)
    return loc


def _compact_label(label: Optional[str]) -> str:
    if not label:
        return "object"
    parts = label.strip().split()
    if not parts:
        return "object"
    if parts[-1].isdigit():
        return " ".join(["".join(parts[:-1]).lower(), parts[-1]])
    return "".join(parts).lower()


def parse_typed_objects(initial_state: Path, metadata) -> Dict[str, List[Tuple[str, str]]]:
    """Return mapping from type name to list of (full_id, formatted label)."""
    text = initial_state.read_text()
    section = OBJECT_SECTION_RE.search(text)
    if not section:
        return {}
    typed: Dict[str, List[Tuple[str, str]]] = {}
    for token, type_name in TYPED_OBJECT_RE.findall(section.group(1)):
        full_id = demangle(token)
        label = metadata.humanize_id(full_id) or metadata.humanize_base(full_id.split("|", 1)[0])
        formatted = _compact_label(label if label else _format_entity(full_id))
        typed.setdefault(type_name, []).append((full_id, formatted))
    for values in typed.values():
        values.sort(key=lambda item: item[1])
    return typed


def parse_relations(initial_state: Path, metadata) -> Dict[str, List[str]]:
    text = initial_state.read_text()
    location_relations = set()
    location_coords = set()
    open_state_relations = set()
    toggle_state_relations = set()
    pickupable_relations = set()
    openable_ids: Set[str] = set()
    opened_ids: Set[str] = set()
    toggleable_ids: Set[str] = set()
    toggled_ids: Set[str] = set()
    pickupable_ids: Set[str] = set()
    contained_pairs: Set[Tuple[str, str]] = set()
    unhandled_relations = set()

    for raw in TOKEN_RE.findall(text):
        tokens = raw.split()
        if not tokens:
            continue
        pred, *args = tokens
        args = [demangle(arg) for arg in args]
        if any("?" in arg for arg in args):  # goal 中带变量的谓词，跳过
            continue

        if pred == "inReceptacle" and len(args) == 2:
            obj, recep = args
            obj_name = _format_entity(metadata.humanize_id(obj) or obj)
            recep_name = _format_entity(metadata.humanize_id(recep) or recep)
            location_relations.add(f"{obj_name} is at {recep_name}")
            contained_pairs.add((obj, recep))
        elif pred == "objectAtLocation" and len(args) == 2:
            obj, loc = args
            obj_name = _format_entity(metadata.humanize_id(obj) or obj)
            loc_name = _format_location(loc, metadata)
            location_coords.add(f"{obj_name} is at {loc_name}")
        elif pred == "openable" and len(args) == 1:
            openable_ids.add(args[0])
        elif pred == "opened" and len(args) == 1:
            opened_ids.add(args[0])
        elif pred == "toggleable" and len(args) == 1:
            toggleable_ids.add(args[0])
        elif pred == "pickupable" and len(args) == 1:
            pickupable_ids.add(args[0])
        elif pred.lower() == "istoggled" and len(args) == 1:
            toggled_ids.add(args[0])
        else:
            unhandled_relations.add(f"{pred} {' '.join(args)}")

    for full_id in openable_ids:
        label = _format_entity(metadata.humanize_id(full_id) or full_id)
        if full_id in opened_ids:
            open_state_relations.add(f"{label} is open")
        else:
            open_state_relations.add(f"{label} is closed")

    for full_id in toggleable_ids:
        label = _format_entity(metadata.humanize_id(full_id) or full_id)
        if full_id in toggled_ids:
            toggle_state_relations.add(f"{label} is on")
        else:
            toggle_state_relations.add(f"{label} is off")

    for full_id in pickupable_ids:
        label = _format_entity(metadata.humanize_id(full_id) or full_id)
        pickupable_relations.add(f"{label} is pickupable")

    return {
        "location": sorted(location_relations),
        "location_coords": sorted(location_coords),
        "open_state": sorted(open_state_relations),
        "toggle_state": sorted(toggle_state_relations),
        "pickupable": sorted(pickupable_relations),
        "unhandled": sorted(unhandled_relations),
        "_raw": {
            "openable_ids": sorted(openable_ids),
            "opened_ids": sorted(opened_ids),
            "toggleable_ids": sorted(toggleable_ids),
            "toggled_ids": sorted(toggled_ids),
            "pickupable_ids": sorted(pickupable_ids),
            "contained_pairs": sorted(contained_pairs),
        },
    }


def build_available_actions(
    metadata,
    typed_objects: Dict[str, List[Tuple[str, str]]],
    raw_relations: Dict[str, List],
    object_lists: Dict[str, List[str]],
) -> List[str]:
    actions: List[str] = []
    seen: Set[str] = set()

    def add(action: str) -> None:
        if action and action not in seen:
            seen.add(action)
            actions.append(action)

    for base_action in ("look", "inventory", "help"):
        add(base_action)

    for _, label in typed_objects.get("receptacle", []):
        add(f"go to {label}")

    for full_id in raw_relations.get("openable_ids", []):
        label = _compact_label(metadata.humanize_id(full_id) or _format_entity(full_id))
        add(f"open {label}")
        add(f"close {label}")

    for full_id in raw_relations.get("toggleable_ids", []):
        label = _compact_label(metadata.humanize_id(full_id) or _format_entity(full_id))
        add(f"use {label}")

    for labels in object_lists.values():
        for label in labels:
            formatted = _compact_label(label)
            add(f"examine {formatted}")

    for _, label in typed_objects.get("receptacle", []):
        add(f"examine {label}")

    for obj_id, recep_id in raw_relations.get("contained_pairs", []):
        obj_label = _compact_label(metadata.humanize_id(obj_id) or _format_entity(obj_id))
        recep_label = _compact_label(metadata.humanize_id(recep_id) or _format_entity(recep_id))
        add(f"take {obj_label} from {recep_label}")

    for obj_id in raw_relations.get("pickupable_ids", []):
        obj_label = _compact_label(metadata.humanize_id(obj_id) or _format_entity(obj_id))
        add(f"take {obj_label}")

    return actions


def _coords_from_id(full_id: str) -> Optional[Tuple[float, float, float]]:
    pieces = full_id.split("|")[1:]
    coords: List[float] = []
    for piece in pieces:
        try:
            coords.append(float(piece))
        except ValueError:
            break
        if len(coords) == 3:
            break
    if len(coords) == 3:
        return tuple(coords)
    return None


def describe_positions(traj_json: Path, metadata) -> List[str]:
    data = json.loads(traj_json.read_text())

    id_coords: Dict[str, Tuple[float, float, float]] = {}
    for ids in metadata.base_to_ids.values():
        for full_id in ids:
            coords = _coords_from_id(full_id)
            if coords:
                id_coords[full_id] = coords

    used_ids = set()
    lines: List[str] = []

    for obj in data["scene"].get("object_poses", []):
        object_name: str = obj["objectName"]
        base_alias = _normalize_alias(object_name.split("_", 1)[0])
        canonical = metadata.alias_to_base.get(base_alias, object_name.split("_", 1)[0])
        base_key = canonical.lower()

        position = obj["position"]
        rotation = obj["rotation"]

        best_id = None
        best_dist = float("inf")
        for full_id in metadata.base_to_ids.get(base_key, []):
            if full_id in used_ids:
                continue
            coords = id_coords.get(full_id)
            if not coords:
                continue
            dist = sum((coords[i] - position[axis]) ** 2 for i, axis in enumerate(("x", "y", "z")))
            if dist < best_dist:
                best_dist = dist
                best_id = full_id

        if best_id:
            used_ids.add(best_id)
            label = _format_entity(metadata.humanize_id(best_id) or best_id)
        else:
            label = f"{canonical.lower()} ({object_name})"

        lines.append(
            f"{label}: pos=({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f}), "
            f"rot=({rotation['x']:.1f}, {rotation['y']:.1f}, {rotation['z']:.1f})"
        )

    return sorted(lines)


def describe_toggles(traj_json: Path, metadata) -> List[str]:
    data = json.loads(traj_json.read_text())
    toggles = data["scene"].get("object_toggles", [])
    lines: List[str] = []

    for entry in toggles:
        obj_type = entry.get("objectType")
        if not obj_type:
            continue
        is_on = entry.get("isOn")
        alias = _normalize_alias(obj_type)
        canonical = metadata.alias_to_base.get(alias, obj_type)
        base_key = canonical.lower()
        ids = metadata.base_to_ids.get(base_key, [])

        if not ids:
            state = "on" if is_on else "off"
            lines.append(f"{canonical.lower()} is {state}")
            continue

        for full_id in ids:
            label = _format_entity(metadata.humanize_id(full_id) or full_id)
            state = "on" if is_on else "off"
            lines.append(f"{label} is {state}")

    return sorted(lines)


def get_alfworld_description(traj_path: Path):
    init_pddl = traj_path.with_name("initial_state.pddl")
    if not init_pddl.exists():
        raise FileNotFoundError(f"Cannot find initial_state.pddl next to {traj_path}")

    game_file = traj_path.with_name("game.tw-pddl")
    metadata = parse_game_metadata(game_file if game_file.exists() else init_pddl)

    relations = parse_relations(init_pddl, metadata)
    toggles = describe_toggles(traj_path, metadata)
    poses = describe_positions(traj_path, metadata)
    typed_objects = parse_typed_objects(init_pddl, metadata)
    object_lists = metadata.base_to_labels
    raw_relations = relations.get("_raw", {})
    available_actions = build_available_actions(metadata, typed_objects, raw_relations, object_lists)
    label_cache: Dict[str, str] = {}

    def label_for(full_id: str) -> str:
        """Return cached human-readable label for an object id."""
        if full_id not in label_cache:
            label_cache[full_id] = _format_entity(metadata.humanize_id(full_id) or full_id)
        return label_cache[full_id]

    def describe_contents(items: List[str]) -> str:
        """Format contained object labels into a readable list."""
        if not items:
            return ""
        formatted = sorted(items)
        return "a " + ", a ".join(formatted)

    recep_to_objs_by_id: Dict[str, List[str]] = {}
    for obj_id, recep_id in raw_relations.get("contained_pairs", []):
        recep_to_objs_by_id.setdefault(recep_id, []).append(label_for(obj_id))

    openable_ids = set(raw_relations.get("openable_ids", []))
    opened_ids = set(raw_relations.get("opened_ids", []))
    all_recep_ids = set(recep_to_objs_by_id.keys()) | openable_ids | opened_ids

    combined_recep_lines: List[str] = []
    for recep_id in sorted(all_recep_ids, key=label_for):
        recep_label = label_for(recep_id)
        contents = describe_contents(recep_to_objs_by_id.get(recep_id, []))
        if recep_id in opened_ids:
            if contents:
                combined_recep_lines.append(f"{recep_label} is opened, on the {recep_label}, you see {contents}")
            else:
                combined_recep_lines.append(f"{recep_label} is opened, in it, you see nothing.")
        elif recep_id in openable_ids:
            if contents:
                combined_recep_lines.append(f"{recep_label} is closed, if opened, on the {recep_label}, you see {contents}")
            else:
                combined_recep_lines.append(f"{recep_label} is closed, if opened, in it, you see nothing.")
        else:
            if contents:
                combined_recep_lines.append(f"On the {recep_label}, you see {contents}")
            else:
                combined_recep_lines.append(f"On the {recep_label}, you see nothing.")

    relation_sections = [
        # ("=== Unhandled Predicates ===", relations["unhandled"]),
        # ("=== Relative Positions ===", relations["location"]),
        ("=== Objects on Receptacles ===", combined_recep_lines),
        # ("=== Absolute Locations (objectAtLocation) ===", relations["location_coords"]),
        ("=== Toggle States ===", relations["toggle_state"]),
        # ("=== Pickupable Objects ===", relations["pickupable"]),
    ]

    description_lines: List[str] = []

    object_list_lines: List[str] = []
    for base, labels in sorted(object_lists.items()):
        if not labels:
            continue
        display = ", ".join(_compact_label(label) for label in labels)
        object_list_lines.append(f" - {base}: {display}")
    # if object_list_lines:
    #     description_lines.append("=== Object List ===")
    #     description_lines.extend(object_list_lines)

    def add_section(title: str, lines: List[str]) -> None:
        if not lines:
            return
        if description_lines:
            description_lines.append("")
        description_lines.append(title)
        for line in lines:
            description_lines.append(f" - {line}")

    for title, lines in relation_sections:
        add_section(title, lines)

    # if available_actions:
    #     if description_lines:
    #         description_lines.append("")
    #     description_lines.append("=== Available High-Level Actions ===")
    #     for action in available_actions:
    #         description_lines.append(f" - {action}")

    # if toggles:
    #     if description_lines:
    #         description_lines.append("")
    #     description_lines.append("=== Toggle States (from traj_data.json) ===")
    #     for line in toggles:
    #         description_lines.append(f" - {line}")

    # if description_lines or toggles:
    #     description_lines.append("")
    # description_lines.append("=== Object Positions (from traj_data.json) ===")
    # for line in poses:
    #     description_lines.append(f" - {line}")

    return "\n".join(description_lines)


def get_traj_file_from_id(id, split):
    if split == "train":
        mapping_file = "/home/v-liyixia/CODES/AgentGym-RL/AgentGym/agentenv-alfworld/configs/mappings_train.json"
        traj_data_file = "/home/v-liyixia/.cache/alfworld/json_2.1.1/train/{task_type}/{task_id}/traj_data.json"
    elif split == "valid_train":
        mapping_file = "/home/v-liyixia/CODES/AgentGym-RL/AgentGym/agentenv-alfworld/configs/mappings_test.json"
        traj_data_file = "/home/v-liyixia/.cache/alfworld/json_2.1.1/valid_train/{task_type}/{task_id}/traj_data.json"
    elif split == "valid_seen":
        mapping_file = "/home/v-liyixia/CODES/AgentGym-RL/AgentGym/agentenv-alfworld/configs/mappings_valid_seen.json"
        traj_data_file = "/home/v-liyixia/.cache/alfworld/json_2.1.1/valid_seen/{task_type}/{task_id}/traj_data.json"
    elif split == "valid_unseen":
        mapping_file = "/home/v-liyixia/CODES/AgentGym-RL/AgentGym/agentenv-alfworld/configs/mappings_valid_unseen.json"
        traj_data_file = "/home/v-liyixia/.cache/alfworld/json_2.1.1/valid_unseen/{task_type}/{task_id}/traj_data.json"
    else:
        raise ValueError(f"Unknown split: {split}")
    with open(mapping_file) as f:
        mapping = json.load(f)
    for item in mapping:
        if item["item_id"] == id:
            task_type = item["task_type"]
            task_id = item["task_id"]
            break
    traj_path = Path(traj_data_file.format(task_type=task_type, task_id=task_id)).expanduser()
    return traj_path


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
    traj_path = get_traj_file_from_id(id, split=SPLIT)
    env_description = get_alfworld_description(traj_path)

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


from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
if __name__ == "__main__":
    num_examples = -1
    global SPLIT
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_instruct", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--split", type=str, default="valid_train", choices=["train", "valid_train", "valid_seen", "valid_unseen"])
    args = parser.parse_args()
    agent_instruct = args.agent_instruct
    output_file = args.output_file
    SPLIT = args.split

    all_data = read_json(agent_instruct)
    if num_examples > 0:
        all_data = all_data[:num_examples]

    conversations = []
    with ThreadPoolExecutor(max_workers=500) as executor:
        for result in tqdm(executor.map(_process_item, all_data), total=len(all_data)):
            conversations.append(result)

    print(f"Saving {len(conversations)} conversations to {output_file}")
    write_json(conversations, output_file)