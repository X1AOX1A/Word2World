#!/usr/bin/env bash
set -euo pipefail

# Create and prepare a dedicated venv for TextWorld env server
# Mirrors scripts/env_setup/uv_sciworld.sh

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

export PATH="$HOME/.local/bin:$PATH"
uv venv uv_textworld --python 3.12 || true
source uv_textworld/bin/activate
uv pip install --upgrade pip
export UV_LINK_MODE=copy

# Install local TextWorld (from clone) and the AgentGym TextWorld server
cd AgentGym/agentenv-textworld

# Prefer local clone if present
if [ -d "textworld" ]; then
  echo "Installing local TextWorld clone..."
  uv pip install -e ./textworld
fi

echo "Installing agentenv_textworld server..."
uv pip install -e .
uv pip list | grep -E "(textworld|fastapi|uvicorn|agentenv_textworld)" || true

echo "Done. Activate via: source uv_textworld/bin/activate"

