#!/bin/bash
# Submits all 6 dataset jobs for mos (eval_shuffle=true ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch slurm_shuffled/run_mos_cifar224.sh
sbatch slurm_shuffled/run_mos_cub.sh
sbatch slurm_shuffled/run_mos_imageneta.sh
sbatch slurm_shuffled/run_mos_imagenetr.sh
sbatch slurm_shuffled/run_mos_omnibenchmark.sh
sbatch slurm_shuffled/run_mos_vtab.sh
