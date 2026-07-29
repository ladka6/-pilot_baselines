#!/bin/bash
#SBATCH --job-name=pilot-mos-cub
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=16:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "$HOME/pilot_baselines"
# PILOT needs its OWN venv (python 3.10 + timm 0.6.12; the TOSCA repo's
# timm 1.x breaks the prompt backbones). Create it once with ./setup_env.sh
source .venv/bin/activate
# Datasets are shared with the TOSCA repo.
[ -e data ] || ln -s "$HOME/continious_learning/data" data
mkdir -p logs

SEED=$((1993 + SLURM_ARRAY_TASK_ID))
python main.py --config exps/bench/mos_cub.json --seed "$SEED"
