# !/bin/bash
set -ex

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv uv_agentgym_rl --python 3.10
source uv_agentgym_rl/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

uv pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# install flash-atten
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
uv pip install $FLASH_ATTENTION_NAME
rm -f $FLASH_ATTENTION_NAME

# for RL
cd AgentGym-RL
uv pip install -e .

# for agentgym
echo "Preparing environment for agentenv..."
cd ../AgentGym/agentenv
uv pip install -e .
uv pip install transformers==4.51.3

uv pip install openai azure.identity
uv pip install vllm==0.6.3
uv pip install outlines==0.1.8
uv pip install matplotlib
uv pip uninstall ray && uv pip install ray[default]
uv pip install "click==8.2.1"
uv pip list
