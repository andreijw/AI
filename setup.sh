#!/usr/bin/env bash
# Create a virtual environment for this project on Linux or macOS (Windows: setup.ps1).
#
# Profiles:
#   cpu (default)  .venv      .[dev,video]         No PyTorch, like CI: the default environment for every tool.
#   gpu            .venv-gpu  .[dev,video,torch]   Adds PyTorch for the optional DDPG backend (backend: torch)
#                                                  and its tests; from PyTorch's CUDA 13.2 index, which drives
#                                                  the Jetson's GPU (docs/pytorch_jetson_timing.md). About 5 GiB.
#
# Usage: ./setup.sh [--profile cpu|gpu] [--skip-checks]
# An existing environment is reused and its packages upgraded. Then the CI checks run (unless --skip-checks).
set -euo pipefail

profile="cpu"
checks=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) profile="${2:-}"; shift 2 ;;
        --skip-checks) checks=0; shift ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

case "$profile" in
    cpu) venv=".venv"; extras="dev,video"; index_args=() ;;
    gpu)
        venv=".venv-gpu"; extras="dev,video,torch"
        index_args=(--index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple)
        ;;
    *) echo "--profile must be cpu or gpu, got '$profile'" >&2; exit 2 ;;
esac

cd "$(dirname "$0")"
python="${PYTHON:-python3.12}"
command -v "$python" >/dev/null || python="python3"
version="$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[[ "$version" == "3.12" ]] || echo "Warning: $python is Python $version; CI uses 3.12." >&2

[[ -x "$venv/bin/python" ]] || "$python" -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip
"$venv/bin/python" -m pip install --upgrade -e ".[$extras]" "${index_args[@]}"

if [[ "$profile" == "gpu" ]]; then
    "$venv/bin/python" -c "import torch; print('PyTorch', torch.__version__, '- CUDA available:', torch.cuda.is_available())"
fi

if [[ $checks -eq 1 ]]; then
    "$venv/bin/python" -m ruff check .
    "$venv/bin/python" -m ruff format --check .
    "$venv/bin/python" -m mypy src
    "$venv/bin/python" -m pytest -q
fi
echo "Ready: $venv ($profile profile). Run tools as $venv/bin/python -m ..."
