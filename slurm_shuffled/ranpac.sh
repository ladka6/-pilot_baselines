#!/bin/bash
# Submits all 6 dataset jobs for ranpac (eval_shuffle=true ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch slurm_shuffled/run_ranpac_cifar224.sh
sbatch slurm_shuffled/run_ranpac_cub.sh
sbatch slurm_shuffled/run_ranpac_imageneta.sh
sbatch slurm_shuffled/run_ranpac_imagenetr.sh
sbatch slurm_shuffled/run_ranpac_omnibenchmark.sh
sbatch slurm_shuffled/run_ranpac_vtab.sh
