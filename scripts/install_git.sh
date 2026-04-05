#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y git gh git-lfs openssh-client ncurses-term ca-certificates curl ripgrep

echo
echo "Installing Python runtime dependencies into the conda env: easynco"
if command -v conda >/dev/null 2>&1; then
  conda run -n easynco python -m pip install numpy pandas matplotlib pyyaml scipy
  if ! conda run -n easynco python -c "import torch" >/dev/null 2>&1; then
    conda run -n easynco python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  fi
else
  echo "conda not found, skipped Python package installation for easynco" >&2
fi

git lfs install --skip-repo || true

echo
echo "Git, GitHub CLI, Git LFS, and the Python experiment dependencies are installed."
echo "If GitHub authentication is not configured yet, run:"
echo "  gh auth login --web --git-protocol https"
