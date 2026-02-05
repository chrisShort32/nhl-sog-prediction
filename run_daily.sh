#!/usr/bin/env bash
set -euo pipefail

# Ensure correct working directory
cd /home/blumpkin/pipeline

# Load env vars (API keys)
source /home/blumpkin/.bashrc

# Activate venv
source /home/blumpkin/venv/bin/activate

# Run pipeline
python  -u daily_run.py
