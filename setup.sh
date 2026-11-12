#!/usr/bin/env bash
# Create a virtual environment for this project on Linux or macOS (Windows: setup.ps1).
#
# Both install requirements-dev.txt (the three distributions under packages/, editable, and the dev tools) plus:
#   cpu (default)  .venv      rl-core[video,onnx], rl-challenges[box2d], rl-navigation[video]
#                             No PyTorch: the default environment for every tool.
#   gpu            .venv-gpu  the same and rl-core[torch]: PyTorch for the optional DDPG, TD3 and SAC backend
#                             (backend: torch) and its tests, from PyTorch's CUDA 13.2 index, which drives the
#                             Jetson's GPU (docs/pytorch_jetson_timing.md). About 5 GiB.
# Both build Box2D for LunarLander and BipedalWalker first (box2d-py from source with swig, about 95 s; needs a
# C++ compiler and the Python headers). CI installs neither PyTorch nor Box2D, and skips their tests; it installs
# the onnx extra.
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
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

case "$profile" in
    cpu) venv=".venv"; core_extras="video,onnx"; index_args=() ;;
    gpu)
        venv=".venv-gpu"; core_extras="video,onnx,torch"
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
# Build tools at the versions box2d-py 2.3.8 was verified with on 2026-11-06 (2.3.5 on 2026-10-27); --no-binary keeps
# it on the reviewed source release.
"$venv/bin/python" -m pip install swig==4.5.1 "setuptools<85" "wheel<0.49"
if ! PATH="$PWD/$venv/bin:$PATH" "$venv/bin/python" -m pip install --no-build-isolation --no-binary box2d-py \
        box2d-py==2.3.8; then
    echo "Building box2d-py failed: it needs a C++ compiler and the Python 3.12 headers" \
        "(Debian/Ubuntu: sudo apt install build-essential python3.12-dev)." >&2
    exit 1
fi
# The single rl-cartpole distribution came before the split into packages/ (2026-11-11); its editable .pth
# points at src/, which no longer holds the code.
if "$venv/bin/python" -m pip show --quiet rl-cartpole > /dev/null 2>&1; then
    "$venv/bin/python" -m pip uninstall -y rl-cartpole
fi
"$venv/bin/python" -m pip install --upgrade -r requirements-dev.txt -e "./packages/rl-core[$core_extras]" \
    -e "./packages/rl-navigation[gym,challenge,video]" -e "./packages/rl-challenges[box2d]" "${index_args[@]}"
if [[ -d src ]] && [[ -z "$(git ls-files src)" ]]; then
    echo "src/ holds no tracked file since the split into packages/: review \`git clean -ndX src/\`," \
        "then remove it with \`rm -rf src/\`." >&2
fi
"$venv/bin/python" -c "import Box2D; print('Box2D', Box2D.__version__)"

if [[ "$profile" == "gpu" ]]; then
    "$venv/bin/python" -c "import torch; print('PyTorch', torch.__version__, '- CUDA available:', torch.cuda.is_available())"
fi

if [[ $checks -eq 1 ]]; then
    "$venv/bin/python" -m ruff check .
    "$venv/bin/python" -m ruff format --check .
    "$venv/bin/python" -m mypy
    "$venv/bin/python" -m pytest -q
fi
echo "Ready: $venv ($profile profile). Run tools as $venv/bin/python -m ..."
