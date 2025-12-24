"""
Async Flask (Quart) Server
"""

from quart import Quart, jsonify, request
from quart_cors import cors
import subprocess
import json
import glob
import os
import time
import datetime
import asyncio
from .environment import webarena_env_server

app = Quart(__name__)

VISUAL = os.environ.get("VISUAL", "false").lower() == "true"
if VISUAL:
    print("Running in VISUAL mode")
    app = cors(
        app, 
        allow_origin="*",
    )
_max_id=0
_max_id_lock=asyncio.Lock()

@app.route("/", methods=["GET"])
async def generate_ok():
    """Test connectivity"""
    return "ok"

@app.route("/list_envs", methods=["GET"])
async def list_envs():
    """List all environments. """
    return jsonify(list(webarena_env_server.env.keys()))

@app.route("/create", methods=["POST"])
async def create():
    """Create a new environment"""
    global _max_id
    async with _max_id_lock:
        env_idx = _max_id
        _max_id += 1
    env = await asyncio.to_thread(webarena_env_server.create, env_idx)
    return jsonify({"env_idx": env})

@app.route("/step", methods=["POST"])
async def step():
    """
    Make an action
    """
    step_query = await request.get_json()
    step_data = await asyncio.to_thread(
        webarena_env_server.step, step_query["env_idx"], step_query["action"]
    )
    step_response = {
        "observation": step_data[0],
        "reward": step_data[1],
        "terminated": step_data[2],
        "truncated": step_data[3],
        "info": step_data[4],
    }
    return jsonify(step_response)

@app.route("/observation", methods=["GET"])
async def get_observation():
    """
    current observation
    """
    env_idx = request.args.get("env_idx", type=int)
    obs = await asyncio.to_thread(webarena_env_server.observation, env_idx)
    return jsonify(obs)

@app.route("/observation_metadata", methods=["GET"])
async def get_obsmetadata():
    """
    current observation metadata
    """
    env_idx = request.args.get("env_idx", type=int)
    obs_meta = await asyncio.to_thread(webarena_env_server.observation_metadata, env_idx)
    return jsonify(obs_meta)

def check_cookies_expiration():
    relogin_tolerance = 2700 # 1 hours in seconds
    now = datetime.datetime.now().timestamp()
    earliest_expiry = float('inf')
    
    auth_dir = ".auth"
    cookie_files = glob.glob(os.path.join(auth_dir, "*.json"))
    
    if not cookie_files:
        return True
        
    for cookie_file in cookie_files:
        try:
            with open(cookie_file, 'r') as f:
                data = json.load(f)
                
            if not data.get('cookies'):
                continue
                
            for cookie in data['cookies']:
                if 'expires' in cookie:
                    expires = float(cookie['expires'])
                    if expires <= 0:
                        continue
                    earliest_expiry = min(earliest_expiry, expires)
        except Exception as e:
            print(f"Reading {cookie_file} Error: {e}")
    
    if earliest_expiry != float('inf'):
        time_to_expiry = earliest_expiry - now
        # print(f"oldest cookie will be expired in {time_to_expiry/3600:.2f} hours")
        return time_to_expiry < relogin_tolerance
    
    return True  

@app.route("/reset", methods=["POST"])
async def reset():
    """
    reset the environment
    """
    reset_query = await request.get_json()
    reset_query["options"] = {
        "config_file": f"./config_files/{reset_query['idx']}.json"
    }
    try:
        if check_cookies_expiration():
            print("cookie will be expired in one hour, executeing auto_login...")
            subprocess.run(
                ["python", "browser_env/auto_login.py"]
            )  # This will take time.
            
        obs, info, sites,object = await asyncio.to_thread(
            webarena_env_server.reset,
            reset_query["env_idx"], reset_query["seed"], reset_query["options"]
        )
    except Exception as e:
        print(e)
        obs={"text": "TimeoutError"}
        sites = "Error"
        object = "Error"
    reset_response = {"observation": obs["text"], "sites": sites, "object": object}
    return jsonify(reset_response)

@app.route("/close", methods=["POST"])
async def close():
    close_query = await request.get_json()
    try:
        await asyncio.wait_for(
            asyncio.to_thread(webarena_env_server.close, close_query["env_idx"]),
            timeout=30.0
        )
        close_response = {"closed":"closed"}
    except asyncio.TimeoutError:
        print(f"Close Env {close_query['env_idx']} time out.")
        close_response = {"closed":f"Close Env {close_query['env_idx']} time out."}
    except Exception as e:
        close_response = {"closed":f"Close Env {close_query['env_idx']} Error: {e}"}
    return jsonify(close_response)

def handle_exit(signum, frame):
    print("\n[INFO] Shutting down server gracefully...")
    for env_idx in list(webarena_env_server.envs.keys()):
        print(f"Closing environment {env_idx}...")
        try:
            webarena_env_server.close(env_idx)
        except Exception as e:
            print(f"Error closing environment {env_idx}: {e}")
    try:
        auth_dir = "./.auth"
        print(f"[INFO] Cleaning up {auth_dir} folder...")
        if os.path.exists(auth_dir) and os.path.isdir(auth_dir):
            auth_files = glob.glob(os.path.join(auth_dir, "*.json"))
            for file_path in auth_files:
                try:
                    os.remove(file_path)
                    print(f"  Removed {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"  Error removing {os.path.basename(file_path)}: {e}")
            print(f"[INFO] Removed {len(auth_files)} auth files")
        else:
            print(f"[INFO] Auth directory {auth_dir} not found or not a directory")
    except Exception as e:
        print(f"[ERROR] Failed to clean up auth folder: {e}")

    print("[INFO] Cleanup complete. Exiting now.")
    os._exit(0)

# 捕获 SIGINT（CTRL+C）
import signal
signal.signal(signal.SIGINT, handle_exit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)