#!/usr/bin/env bash
# Give a git worktree its own virtual environment, layered on the main checkout's.
#
# The main checkout's environment (.venv, or .venv-gpu with --base .venv-gpu) holds an editable install
# of the main checkout, so a worktree that used it would import the main checkout's code and read its
# challenge entry points. This makes <worktree>/<base>: a venv whose site-packages sees the main
# environment's packages (one .pth line; the main checkout's own editable .pth is not processed from
# there, so its code stays off the path) and holds an editable install of the worktree itself. No
# network: the install uses --no-deps --no-index and the main environment's setuptools.
#
# Usage: scripts/worktree_venv.sh [<worktree>] [--base .venv|.venv-gpu]
# Then run every tool as <worktree>/<base>/bin/python -m ...  Never `pip install -e` a worktree into the
# main checkout's environment: that redirects every other agent's imports to the worktree.
set -euo pipefail

worktree=""
base=".venv"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base)
            [[ $# -ge 2 ]] || { echo "--base needs a value (.venv or .venv-gpu)" >&2; exit 2; }
            base="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) worktree="$1"; shift ;;
    esac
done
worktree="$(cd "${worktree:-$(dirname "$0")/..}" && pwd)"
# The first entry of `git worktree list` is the main checkout.
main="$(git -C "$worktree" worktree list --porcelain | sed -n '1s/^worktree //p')"
if [[ "$main" == "$worktree" ]]; then
    echo "$worktree is the main checkout: use its own $base (./setup.sh)." >&2
    exit 2
fi
main_python="$main/$base/bin/python"
[[ -x "$main_python" ]] || { echo "No $main_python: run ./setup.sh in $main first." >&2; exit 2; }

"$main_python" -m venv "$worktree/$base"
python="$worktree/$base/bin/python"
main_site="$("$main_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
site="$("$python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
echo "$main_site" > "$site/_main_checkout.pth"

projects=("$worktree")
if [[ -d "$worktree/packages" ]]; then
    # Only when this re-runs into a venv made before the split into packages/: it holds the old single
    # distribution, whose .pth points at src/. A fresh venv never has it.
    if compgen -G "$site/rl_cartpole-*.dist-info" > /dev/null; then
        "$python" -m pip uninstall --quiet -y rl-cartpole
    fi
    projects=()
    for pyproject in "$worktree"/packages/*/pyproject.toml; do
        projects+=("$(dirname "$pyproject")")
    done
fi
editables=()
for project in "${projects[@]}"; do
    editables+=(-e "$project")
done
"$python" -m pip install --quiet --no-deps --no-index --no-build-isolation "${editables[@]}"
"$python" -c "import rl_core; print('rl_core from', rl_core.__file__)"
echo "Ready: run tools as $python -m ..."
