#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-}"

if [ -z "${PYTHON_BIN}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "Python is required but was not found on PATH." >&2
    exit 1
  fi
fi

VENV_DIR=".venv"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  echo "Created virtual environment at ${VENV_DIR}"
fi

if [ -f "${VENV_DIR}/Scripts/activate" ]; then
  # Windows layout
  # shellcheck disable=SC1091
  source "${VENV_DIR}/Scripts/activate"
else
  # POSIX layout
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

"${PYTHON_BIN}" -m pip install --upgrade pip
pip install -r requirements.txt

echo "Dependencies installed in the virtual environment."
