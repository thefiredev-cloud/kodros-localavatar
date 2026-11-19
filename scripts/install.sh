#!/usr/bin/env bash
# Two venvs: the CLI + Kokoro (numpy 2) and the SadTalker engine (numpy 1.23, torch 2.1 cu121).
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv ~/.venvs/localavatar --python 3.12
VIRTUAL_ENV=~/.venvs/localavatar uv pip install -e '.[test]'

uv venv ~/.venvs/localavatar-engine --python 3.10
VIRTUAL_ENV=~/.venvs/localavatar-engine uv pip install \
  --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
VIRTUAL_ENV=~/.venvs/localavatar-engine uv pip install -r engine-requirements.txt

# Pinned SadTalker source + weights into ~/.local/share/localavatar (mode 0700).
~/.venvs/localavatar/bin/localavatar setup

mkdir -p ~/.local/bin
ln -sf ~/.venvs/localavatar/bin/localavatar ~/.local/bin/localavatar
echo "ok: run 'localavatar doctor'"
