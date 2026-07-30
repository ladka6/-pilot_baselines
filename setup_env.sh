#!/bin/bash
# One-time env setup for the PILOT baselines. The TOSCA repo's venv canNOT be
# reused: its timm 1.x breaks PILOT's prompt backbones (vit_l2p etc. rely on
# timm 0.6.12 internals). timm 0.6.12 also can't run on Python 3.11+ (a
# dataclasses check tightened in 3.11 rejects timm's maxxvit module), so this
# deliberately uses the cluster's system Python 3.9 rather than a loaded
# module toolchain (Snellius's 2023/2024 toolchains only offer 3.11/3.12).
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
