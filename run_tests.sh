#!/usr/bin/env bash
# Script para executar toda a suíte de testes usando pytest

if command -v pytest >/dev/null 2>&1; then
    pytest -v "$@"
elif [ -f "$HOME/.local/bin/pytest" ]; then
    "$HOME/.local/bin/pytest" -v "$@"
elif command -v wsl.exe >/dev/null 2>&1; then
    wsl.exe bash -c "cd /mnt/c/Users/preto/Repo/3AIER_Luiz && ~/.local/bin/pytest -v $*"
elif command -v wsl >/dev/null 2>&1; then
    wsl bash -c "cd /mnt/c/Users/preto/Repo/3AIER_Luiz && ~/.local/bin/pytest -v $*"
else
    python3 -m pytest -v "$@" || python -m pytest -v "$@"
fi
