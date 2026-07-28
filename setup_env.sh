#!/bin/bash
# One-time env setup for the PILOT baselines. The TOSCA repo's venv canNOT be
# reused: its timm 1.x breaks PILOT's prompt backbones (vit_l2p etc. rely on
# timm 0.6.12 internals).
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
