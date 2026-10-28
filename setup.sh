#!/usr/bin/env bash
# Create a virtual environment for this project on Linux or macOS (Windows: setup.ps1).
#
# Profiles:
#   cpu (default)  .venv      .[dev,video,box2d]         No PyTorch: the default environment for every tool.
#   gpu            .venv-gpu  .[dev,video,torch,box2d]   Adds PyTorch for the optional DDPG and TD3 backend
#                                                        (backend: torch) and its tests; from PyTorch's CUDA 13.2
#                                                        index, which drives the Jetson's GPU
#                                                        (docs/pytorch_jetson_timing.md). About 5 GiB.
# Both build Box2D for LunarLander first (box2d-py from source with swig, about 90 s; needs a C++ compiler and the
# Python headers). CI installs neither PyTorch nor Box2D, and skips their tests.
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
    cpu) venv=".venv"; extras="dev,video,box2d"; index_args=() ;;
    gpu)
        venv=".venv-gpu"; extras="dev,video,torch,box2d"
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
# Box2D before the extras: Gymnasium's box2d extra pins box2d==2.3.10, which has no aarch64 build, so the
# box2d extra names box2d-py, which builds from source with swig; pip's isolated build cannot see swig.
# Build tools at the versions box2d-py 2.3.5 (a 2018 setup.py) was verified with on 2026-10-27; --no-binary keeps
# it on the reviewed source release.
"$venv/bin/python" -m pip install swig==4.5.1 "setuptools<85" "wheel<0.49"
if ! PATH="$PWD/$venv/bin:$PATH" "$venv/bin/python" -m pip install --no-build-isolation --no-binary box2d-py \
        box2d-py==2.3.5; then
    echo "Building box2d-py failed: it needs a C++ compiler and the Python 3.12 headers" \
        "(Debian/Ubuntu: sudo apt install build-essential python3.12-dev)." >&2
    exit 1
fi
"$venv/bin/python" -m pip install --upgrade -e ".[$extras]" "${index_args[@]}"
"$venv/bin/python" -c "import Box2D; print('Box2D', Box2D.__version__)"

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
