#!/bin/bash
# Submits all 6 dataset jobs for dualprompt (eval_shuffle=true ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch slurm_shuffled/run_dualprompt_cifar224.sh
sbatch slurm_shuffled/run_dualprompt_cub.sh
sbatch slurm_shuffled/run_dualprompt_imageneta.sh
sbatch slurm_shuffled/run_dualprompt_imagenetr.sh
sbatch slurm_shuffled/run_dualprompt_omnibenchmark.sh
sbatch slurm_shuffled/run_dualprompt_vtab.sh
