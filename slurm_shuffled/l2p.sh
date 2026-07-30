#!/bin/bash
# Submits all 6 dataset jobs for l2p (eval_shuffle=true ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch slurm_shuffled/run_l2p_cifar224.sh
sbatch slurm_shuffled/run_l2p_cub.sh
sbatch slurm_shuffled/run_l2p_imageneta.sh
sbatch slurm_shuffled/run_l2p_imagenetr.sh
sbatch slurm_shuffled/run_l2p_omnibenchmark.sh
sbatch slurm_shuffled/run_l2p_vtab.sh
