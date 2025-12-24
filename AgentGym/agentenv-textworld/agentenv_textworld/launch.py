"""
Entrypoint for the TextWorld agent environment.
"""

import argparse
import uvicorn


def launch():
    """entrypoint for `textworld` command"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run("agentenv_textworld:app", host=args.host, port=args.port)

