#!/bin/bash
# Submits all 6 dataset jobs for simplecil (eval_shuffle=true ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch slurm_shuffled/run_simplecil_cifar224.sh
sbatch slurm_shuffled/run_simplecil_cub.sh
sbatch slurm_shuffled/run_simplecil_imageneta.sh
sbatch slurm_shuffled/run_simplecil_imagenetr.sh
sbatch slurm_shuffled/run_simplecil_omnibenchmark.sh
sbatch slurm_shuffled/run_simplecil_vtab.sh
