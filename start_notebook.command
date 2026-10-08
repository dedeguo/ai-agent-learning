#!/bin/bash

PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)" || exit 1
cd -- "$PROJECT_DIR" || exit 1

fail() {
    printf '[ERROR] %s\n' "$1" >&2
    if [ -t 0 ]; then
        read -r -p "Press Enter to close this window..." _
    fi
    exit 1
}

PYTHON="$PROJECT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    printf '%s\n' \
        'Project virtual environment was not found.' \
        'Run these commands in the project directory first:' \
        '  python3 -m venv .venv' \
        '  .venv/bin/python -m pip install -r requirement.txt' >&2
    fail 'Cannot use the project Python.'
fi

if ! "$PYTHON" -c 'import jupyterlab' >/dev/null 2>&1; then
    printf '%s\n' \
        'Check the virtual environment and install the dependencies:' \
        '  .venv/bin/python -m pip install -r requirement.txt' >&2
    fail 'Cannot load JupyterLab using the project Python.'
fi

printf '%s\n' \
    'Starting JupyterLab in the project directory...' \
    'Keep this window open. Press Ctrl+C to stop the server.'
exec "$PYTHON" -m jupyterlab --ServerApp.ip=127.0.0.1 "$@"
