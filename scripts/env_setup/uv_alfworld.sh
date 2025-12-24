# !/bin/bash
set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_alfworld --python 3.9
source uv_alfworld/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

# Ref: AgentGym/agentenv-alfworld/setup.sh
cd AgentGym/agentenv-alfworld
uv pip install alfworld==0.3.3
uv pip uninstall opencv-python
uv pip install -e .

## export ALFWORLD_DATA=~/.cache/alfworld
## alfworld-download
## We modify the put to move, and add help action
# download from https://huggingface.co/datasets/X1AOX1A/LLMasWorldModels/blob/main/alfworld.zip
# and unzip data/alfworld.zip -d ~/.cache
# this will be downloaded with scripts/download_data/download_data.py
uv pip list