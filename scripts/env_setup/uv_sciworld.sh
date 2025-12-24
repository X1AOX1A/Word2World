# !/bin/bash
set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_sciworld --python 3.8
source uv_sciworld/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

# Ref: AgentGym/agentenv-sciworld/README.md
cd AgentGym/agentenv-sciworld
uv pip install -e .
uv pip list

sudo apt-get update
sudo apt-get install -y openjdk-17-jre-headless