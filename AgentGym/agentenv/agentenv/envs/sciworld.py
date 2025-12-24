import json
import re
from typing import Any, Mapping

import requests
from requests.exceptions import RequestException

from agentenv.controller import (
    BaseAdapter,
    BaseEnvClient,
    BaseTask,
    extract_python_code_blocks,
    format_code_as_action_prompt,
    format_function_call_prompt,
    parse_python_code_comments,
)
from agentenv.controller.types import (
    ActionFormat,
    ActionWithTought,
    ConversationMessage,
    StepOutput,
)

# Five actions take two arguments, 16 take one argument, and four actions take zero arguments.
SCIWORLD_FUNCTION_DESCRIPTION = [
    {
        "name": "open", 
        "description": "Opens a container. You may have to give the specific location of the container if necessary(eg.door to kitchen, door to living room).",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The container you want to open."
                }
            },
            "required": ["obj"]
        }
    },
    {
        "name": "close", 
        "description": "Closes a container. You may also have to give the specific location of the container if necessary(eg.door to kitchen, door to living room).",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The container you want to close."
                },
            },
            "required": ["obj"]
        }
    },
    {
        "name": "activate",
        "description": "Activate a container, which means to turn something up. eg. Activate stove in order to heat something.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The container you want to activate."
                },
            },
            "required": ["obj"]
        },
        
    },
    {
        "name": "deactivate",
        "description": "Deactivate a container, which means to shut something down. eg. Deactivate sink in order to prevent water overflow.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The container you want to deactivate."
                },
            },
            "required": ["obj"]
        },
        
    },
    {
        "name": "connect", 
        "description": "Connect electrical components in order to create a working elecrical circuit.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj1":{
                    "type": "string",
                    "description":"The first object you choose to create the elecrical circuit."
                },
                "obj2":{
                    "type": "string",
                    "description": "The second object you choose to create the elecrical circuit."
                }
            },
            "required": ["obj1", "obj2"]
        }
    },
    {
        "name": "disconnect", 
        "description": "Disconnect electrical components.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The object you choose to disconncect from the elecrical circuit."
                }
            },
            "required": ["obj"]
        }
    },
    {
        "name": "use", 
        "description": "Use a device/item as a tool. You may also have to give the other object which your device/item will be used on.",
        "parameters":{
            "type": "object",
            "properties":{
                "tool":{
                    "type": "string",
                    "description":"The device/item you choose to use as a tool."
                },
                "obj":{
                    "type": "string",
                    "description": "The object which you choose to use the device/item on."
                }
            },
            "required": ["tool"]
        }
    },
    {
        "name": "lookaround",
        "description": "Describe the current room.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name": "lookat",
        "description": "Describe an object in detail.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The object you want to get some detailed information."
                }
            },
        "required": ["obj"]
        },
    },
    {
        "name": "lookin",
        "description": "Describe the contents of a container.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The object you want to know its contents."
                }
            },
        "required": ["obj"]
        },
    },
    {
        "name": "read",
        "description": "Read a note or book in order to find out instruction to accomplish a certain task.",
        "parameters":{
            "type": "object",
            "properties":{
                "info":{
                    "type": "string",
                    "description":"The note or book you want to read."
                }
            },
        "required": ["info"]
        },
    },
    {
        "name": "move", 
        "description": "Move an object to a container(except the inventory). Note that if you want to move an object to the inventory, the function you should call is \'pickup\'.If you want to move an object from the inventory to the outside, the function you should call is \'drop\'.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The object you want to move to the container."
                },
                "container":{
                    "type": "string",
                    "description": "The container you choose to put the object in/on."
                }
            },
            "required": ["obj", "container"]
        }
    },
    {
        "name":"pickup",
        "description":"Move an object to the inventory for future actions.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description": "The object you choose to move to the inverntory."
                }
            },
        "required": ["obj"]
        },
    },
    {
        "name": "drop",
        "description": "Move an object from the inventory to the outside for future actions.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description":"The object you want to move from the inventory."
                },
            },
        "required": ["obj"]
        },
    },
    {
        "name": "pour",
        "description": "Pour a liquid into a container.",
        "parameters":{
            "type": "object",
            "properties":{
                "liq":{
                    "type": "string",
                    "description": "The liquid you choose to pour.."
                },
                "container":{
                    "type": "string",
                    "description": "The container you choose to pour the liquid into."
                }
            },
        "required": ["liq", "container"]
        },
    },
    {
        "name": "dunk",
        "description": "Dunk a container into a liquid.",
        "parameters":{
            "type": "object",
            "properties":{
                "liq":{
                    "type": "string",
                    "description": "The liquid you choose to dunk the container in."
                },
                "container":{
                    "type": "string",
                    "description": "The container you choose to be dunk in the chosen liquid."
                }
            },
        "required": ["liq", "container"]
        },
    },
    {
        "name": "mix",
        "description": "Chemically mix the contents of the container.",
        "parameters":{
            "type": "object",
            "properties":{
                "container":{
                    "type": "string",
                    "description": "The container whose contents you want to chemically mix."
                }
            },
        "required": ["container"]
        },
    },
    {
        "name": "goto",
        "description": "Move to a new location.",
        "parameters":{
            "type": "object",
            "properties":{
                "loc":{
                    "type": "string",
                    "description": "The location you want to arrive at."
                }
            },
        "required": ["loc"]
        },
    },
    # the following function is only available in the simplified mode.
    # {
    #     "name": "teleport",
    #     "description": "Activates a container, Like a stove in order to heat something.",
    #     "parameters":{
    #         "type": "object",
    #         "properties":{

    #         },
    #     "additionalProperties": "false",
    #     "required": ["obj"]
    #     },
    # },
    {
        "name": "eat",
        "description": "Eat a food.",
        "parameters":{
            "type": "object",
            "properties":{
                "food":{
                    "type": "string",
                    "description": "The food you choose to eat."
                }
            },
        "required": ["food"]
        },
    },
    {
        "name": "flush",
        "description": "Flush a toilet.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description": "The specific toilet to flush."
                }
            },
        "required": ["obj"]
        },
    },
    {
        "name": "focus",
        "description": "Signal intent on a task object.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "string",
                    "description": "The task object to send signal intent."
                }
            },
        "required": ["obj"]
        },
    },
    {
        "name": "wait",
        "description": "Take no action for some duration. Call this function if you have finished you task or you decided to wait for response from the environment.",
        "parameters":{
            "type": "object",
            "properties":{
                "duration":{
                    "type": "int",
                    "description": "The number of iterations you choose to wait."
                }
            },
        "required": ["obj"]
        },
    },
    {
        "name": "choose",
        "description": "Enter the number for the action you intended based on the previous response from the environment.",
        "parameters":{
            "type": "object",
            "properties":{
                "option":{
                    "type": "int",
                    "description": "The number of the action you intended."
                }
            },
        "required": ["obj"]
        },
    },
    {
        "name": "examine",
        "description": "Provides a description of the objects. You may call this function if you want to know what kind of substance the object is.",
        "parameters":{
            "type": "object",
            "properties":{
                "obj":{
                    "type": "int",
                    "description": "The objects you want to get some description of."
                }
            },
        "required": ["obj"]
        },
    },
    {
        "name": "task",
        "description": "Describe the current task. Call this function if you have to confirm it.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    },
    {
        "name":"inventory",
        "description":"Displays the list of objects currently being carried by you.",
        "parameters": {
            "type": "object",
            "properties": {},
        }
    },
]


class SciWorldAdapter(BaseAdapter):
    conversation_start_dict = {
        ActionFormat.REACT:(
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": 'You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. Here are the actions you may take: [{"action": "open/close OBJ", "description": "open/close a container"}, {"action": "de/activate OBJ", "description": "activate/deactivate a device"}, {"action": "connect OBJ to OBJ", "description": "connect electrical components"}, {"action": "disconnect OBJ", "description": "disconnect electrical components"}, {"action": "use OBJ [on OBJ]", "description": "use a device/item"}, {"action": "look around", "description": "describe the current room"}, {"action": "look at OBJ", "description": "describe an object in detail"}, {"action": "look in OBJ", "description": "describe a container\'s contents"}, {"action": "read OBJ", "description": "read a note or book"}, {"action": "move OBJ to OBJ", "description": "move an object to a container"}, {"action": "pick up OBJ", "description": "move an object to the inventory"}, {"action": "put down OBJ", "description": "drop an inventory item"}, {"action": "pour OBJ into OBJ", "description": "pour a liquid into a container"}, {"action": "dunk OBJ into OBJ", "description": "dunk a container into a liquid"}, {"action": "mix OBJ", "description": "chemically mix a container"}, {"action": "go to LOC", "description": "move to a new location"}, {"action": "eat OBJ", "description": "eat a food"}, {"action": "flush OBJ", "description": "flush a toilet"}, {"action": "focus on OBJ", "description": "signal intent on a task object"}, {"action": "wait", "description": "take no action for 10 iterations"}, {"action": "wait1", "description": "take no action for 1 iteration"}, {"action":"examine OBJ","description":"provides a description of the objects present on or in a receptacle."}, {"action": "task", "description": "describe current task"}, {"action": "inventory", "description": "list your inventory"}]\nYour response should use the following format:\nThought:\nyour thoughts.\n\nAction:\nyour next action',
                }
            ),
            ConversationMessage(
                {
                    "from": "gpt",
                    "loss": False,
                    "value": "OK. I'll follow your instructions and try my best to solve the task.",
                }
            ),
        ),
        ActionFormat.FUNCTION_CALLING: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": f'You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. An action should be done by invoking an function.\n\n {format_function_call_prompt(SCIWORLD_FUNCTION_DESCRIPTION)}\n\n\nAfter your each turn, the environment will give you immediate feedback based on your taken actions. if the envrionment output \"No known action matches that input.\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given functions. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
                }
            ),
            ConversationMessage(
                {
                    "from": "gpt",
                    "loss": False,
                    "value": "OK. I'll follow your instructions and try my best to solve the task.",
                }
            ),
        ),
        ActionFormat.CODE_AS_ACTION: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": f'You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task. You can perform one of these actions by writing python code to invoke a function.\n\n {format_function_call_prompt(SCIWORLD_FUNCTION_DESCRIPTION)}\n\n\nAfter your each turn, the environment will give you immediate feedback based on your taken actions. if the envrionment output \"No known action matches that input.\", that means the previous action is invalid and you should try more options.\n Reminder: \n1. the action must be chosen from the given functions. The objects you choose must exist in the current room. Any actions except provided available actions will be regarded as illegal. \n2. Think when necessary, try to act directly more in the process.',
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
    }

    valid_functions_args = {
        "open": ["obj"],
        "close" : ["obj"],
        "activate": ["obj"],
        "deactivate": ["obj"],
        "connect": ["obj1", "obj2"],
        "disconnect": ["obj"],
        "use": ["tool", "obj"],
        "lookaround": [],
        "lookat": ["obj"],
        "read": ["obj"],
        "move": ["obj", "container"],
        "pickup": ["obj"],
        "drop": ["obj"],
        "pour": ["liq", "container"],
        "dunk": ["container", "liq"],
        "mix": ["container"],
        "goto": ["loc"],
        "eat": ["food"],
        "flush": ["obj"],
        "focus": ["obj"],
        "wait": ["duration"],
        "choose": ["option"],
        "examine": ["obj"],
        "task": [],
        "inventory": []
    }

    function_to_name = {
        "open": "open",
        "close" : "close",
        "activate": "activate",
        "deactivate": "deactivate",
        "connect": "connect",
        "disconnect": "disconnect",
        "use": "use",
        "lookaround": "look around",
        "lookat": "look at",
        "read": "read",
        "move": "move",
        "pickup": "pick up",
        "drop": "drop",  # or put down. Use "drop" based on sciworld_train.json in AgentTraj-L.
        "pour": "pour",
        "dunk": "dunk",
        "mix": "mix",
        "goto": "go to",
        "eat": "eat",
        "flush": "flush",
        "focus": "focus on",
        "choose": "choose",
        "wait": "wait",
        "examine": "examine",
        "task": "task",
        "inventory": "inventory"
    }

    conjunction_words = {
        "connect": "to",
        "use": "on",
        "move": "to",
        "pour": "into",
        "dunk": "in"
    }

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        _fn_call = json.loads(
            "{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}", strict=False
        )
        thought = _fn_call["thought"]
        fn_name = _fn_call["function_name"].strip()
        args = _fn_call["arguments"]

        if fn_name not in SciWorldAdapter.valid_functions_args:
            raise ValueError("Invalid function name.")
        arg_ls = SciWorldAdapter.valid_functions_args[fn_name]
        if len(args) == 1:
            # read recipe
            action_name = SciWorldAdapter.function_to_name[fn_name]
            arg = args[arg_ls[0]]
            if fn_name == "wait":
                action = f'{action_name}{arg}'
            elif fn_name == "choose":
                action = f'{arg}'
            else:
                action = f'{action_name} {arg}'
        elif len(args) == 0:
            # look around
            action = f'{SciWorldAdapter.function_to_name[fn_name]}'
        else:  # two arguments
            # pour milk into mug
            action_name = SciWorldAdapter.function_to_name[fn_name]
            conjunction = SciWorldAdapter.conjunction_words[fn_name]
            action = f'{action_name} {args[arg_ls[0]]} {conjunction} {args[arg_ls[1]]}'
        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        valid_action_flag = False
        fn_name = ''
        action_name = ''
        for k, v in SciWorldAdapter.function_to_name.items():
            if action_with_thought.action.startswith(v):
                valid_action_flag = True
                fn_name = k
                action_name = v
                break
        if action_with_thought.action.isdigit():
            fn_name = 'choose'
        elif not valid_action_flag:
            raise ValueError(f"{action_with_thought.action}: Invalid action.")
        # inventory
        # look at mug/ wait1/ open door to kitchen
        # pour milk into mug
        arg_ls = SciWorldAdapter.valid_functions_args[fn_name]
        str_arg = action_with_thought.action.replace(action_name, '', 1).strip()
        if fn_name in SciWorldAdapter.conjunction_words:
            separator = SciWorldAdapter.conjunction_words[fn_name]
            str_arg_ls = re.split(fr'\s+{separator}\s+', str_arg)
            str_arg_ls = [s.strip() for s in str_arg_ls]
        else:
            str_arg_ls = [str_arg.strip()] if len(str_arg) else []

        if len(str_arg_ls) > len(arg_ls):
            raise TypeError(f"Got unexpected arguments. function {fn_name} expected {len(arg_ls)} but got {len(str_arg_ls)}.")

        if len(str_arg_ls) == 0:
            args = {}
        elif len(str_arg_ls) == 1:
            args = {
                arg_ls[0]: str_arg_ls[0] if fn_name != "wait" else re.findall(r'\d+', str_arg_ls[0])[0]
            }
        else:
            args = {
                arg_ls[0]: str_arg_ls[0],
                arg_ls[1]: str_arg_ls[1]
            }
        return json.dumps(
            {
                "thought": action_with_thought.thought,
                "function_name": fn_name,
                "arguments": args
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        def open(obj: str):
            action_name = SciWorldAdapter.function_to_name["open"]
            return f"{action_name} {obj}"

        def close(obj: str):
            action_name = SciWorldAdapter.function_to_name["close"]
            return f"{action_name} {obj}"

        def activate(obj: str):
            action_name = SciWorldAdapter.function_to_name["activate"]
            return f"{action_name} {obj}"

        def deactivate(obj: str):
            action_name = SciWorldAdapter.function_to_name["deactivate"]
            return f"{action_name} {obj}"

        def connect(obj1: str, obj2: str):
            action_name = SciWorldAdapter.function_to_name["connect"]
            conjuction = SciWorldAdapter.conjunction_words["connect"]
            return f"{action_name} {obj1} {conjuction} {obj2}"

        def disconnect(obj: str):
            action_name = SciWorldAdapter.function_to_name["disconnect"]
            return f"{action_name} {obj}"

        def use(tool: str, obj: str=''):
            action_name = SciWorldAdapter.function_to_name["use"]
            conjuction = SciWorldAdapter.conjunction_words["use"]
            return f"{action_name} {tool} {conjuction} {obj}" if obj else f"{action_name} {tool}"

        def lookaround():
            action_name = SciWorldAdapter.function_to_name["lookaround"]
            return f"{action_name}"

        def lookat(obj: str):
            action_name = SciWorldAdapter.function_to_name["lookat"]
            return f"{action_name} {obj}"

        def read(obj: str):
            action_name = SciWorldAdapter.function_to_name["read"]
            return f"{action_name} {obj}"

        def move(obj: str, container: str):
            action_name = SciWorldAdapter.function_to_name["move"]
            conjuction = SciWorldAdapter.conjunction_words["move"]
            return f"{action_name} {obj} {conjuction} {container}"

        def pickup(obj: str):
            action_name = SciWorldAdapter.function_to_name["pickup"]
            return f"{action_name} {obj}" 

        def drop(obj: str):
            action_name = SciWorldAdapter.function_to_name["drop"]
            return f"{action_name} {obj}" 

        def pour(liq: str, container: str):
            action_name = SciWorldAdapter.function_to_name["pour"]
            conjuction = SciWorldAdapter.conjunction_words["pour"]
            return f"{action_name} {liq} {conjuction} {container}" 

        def dunk(container: str, liq: str):
            action_name = SciWorldAdapter.function_to_name["dunk"]
            conjuction = SciWorldAdapter.conjunction_words["dunk"]
            return f"{action_name} {container} {conjuction} {liq}" 

        def mix(container: str):
            action_name = SciWorldAdapter.function_to_name["mix"]
            return f"{action_name} {container}" 

        def goto(loc: str):
            action_name = SciWorldAdapter.function_to_name["goto"]
            return f"{action_name} {loc}"  

        def eat(food: str):
            action_name = SciWorldAdapter.function_to_name["eat"]
            return f"{action_name} {food}" 

        def flush(obj: str):
            action_name = SciWorldAdapter.function_to_name["flush"]
            return f"{action_name} {obj}" 

        def focus(obj: str):
            action_name = SciWorldAdapter.function_to_name["focus"]
            return f"{action_name} {obj}"

        def wait(duration: str):
            action_name = SciWorldAdapter.function_to_name["wait"]
            return f"{action_name}{duration}"

        def choose(option: str):
            return f"{option}"

        def examine(obj: str):
            action_name = SciWorldAdapter.function_to_name["examine"]
            return f"{action_name} {obj}"

        def task():
            return f"{SciWorldAdapter.function_to_name['task']}"

        def inventory():
            return f"{SciWorldAdapter.function_to_name['inventory']}"

        code = extract_python_code_blocks(text)
        try:
            action = eval(code, {
                "open": open,
                "close" : close,
                "activate": activate,
                "deactivate": deactivate,
                "connect": connect,
                "disconnect": disconnect,
                "use": use,
                "lookaround": lookaround,
                "lookat": lookat,
                "read": read,
                "move": move,
                "pickup": pickup,
                "drop": drop,
                "pour": pour,
                "dunk": dunk,
                "mix": mix,
                "goto": goto,
                "eat": eat,
                "flush": flush,
                "focus": focus,
                "wait": wait,
                "choose": choose,
                "examine": examine,
                "task": task,
                "inventory": inventory
            })
        except Exception as e:
            raise ValueError(f"Invalid action:{code}") from e  
        thought = parse_python_code_comments(code)
        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def to_code_as_action(action_with_thought: ActionWithTought) -> str:
        text = f"```python\n#{action_with_thought.thought}\n"
        valid_action_flag = False
        fn_name = ''
        action_name = ''
        for k, v in SciWorldAdapter.function_to_name.items():
            if action_with_thought.action.startswith(v):
                valid_action_flag = True
                fn_name = k
                action_name = v
                break
        if action_with_thought.action.isdigit():
            fn_name = 'choose'
        elif not valid_action_flag:
            raise ValueError(f"{action_with_thought.action}: Invalid action.")
        # inventory
        # look at mug/ wait1/ open door to kitchen
        # pour milk into mug
        fn_name = fn_name.strip()
        arg_ls = SciWorldAdapter.valid_functions_args[fn_name]
        str_arg = action_with_thought.action.replace(action_name, '', 1).strip()
        if fn_name in SciWorldAdapter.conjunction_words:
            separator = SciWorldAdapter.conjunction_words[fn_name]
            str_arg_ls = re.split(fr'\s+{separator}\s+', str_arg)
            str_arg_ls = [s.strip() for s in str_arg_ls]
        else:
            str_arg_ls = [str_arg.strip()] if len(str_arg) else []

        if len(str_arg_ls) > len(arg_ls):
            raise TypeError(f"Got unexpected arguments. function {fn_name} expected {len(arg_ls)} but got {len(str_arg_ls)}.")

        if len(str_arg_ls) == 0:
            text += f"{fn_name}()"
        elif len(str_arg_ls) == 1:
            arg_list = (
                repr(f"{str_arg_ls[0]}")
                if fn_name != "wait"
                else repr(re.findall(r"\d+", str_arg_ls[0])[0])
            )
            text += f"{fn_name}({arg_list})"
        else:
            text += f"{fn_name}({repr(f'{str_arg_ls[0]}')},{repr(f'{str_arg_ls[1]}')})"
        text += "\n```"
        return text

class SciworldEnvClient(BaseEnvClient):
    adapter_cls = SciWorldAdapter

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")
        self.conversation_start = self.adapter_cls.conversation_start_dict[
            self.action_format
        ]
        ok = ok.json()
        self.env_id = ok["id"]

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        if action.endswith("</s>"):
            action = action[:-5]
        try:
            action = self.adapter_cls.action_parser(action, self.action_format)
        except Exception as e:
            print(e, action)
            return StepOutput(
                state="Invalid Action.\n\n" + self.observe(), reward=0.0, done=False
            )
        response = self._post("step", {"action": action})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "score": response["score"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["score"],
            done=response["done"],
        )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": data_idx})
        self.info = {
            "observation": response["task_description"] + '\n' + response["observation"],
            "reward": 0,
            "score": 0,
            "done": False,
        }
        return response

    def close(self):
        response = self._post("close",{})
        return response

class SciworldTask(BaseTask):
    env_client_cls = SciworldEnvClient
    env_name = "SciWorld"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
