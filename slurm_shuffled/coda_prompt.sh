#!/bin/bash
# Submits all 6 dataset jobs for coda_prompt (eval_shuffle=true ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch slurm_shuffled/run_coda_prompt_cifar224.sh
sbatch slurm_shuffled/run_coda_prompt_cub.sh
sbatch slurm_shuffled/run_coda_prompt_imageneta.sh
sbatch slurm_shuffled/run_coda_prompt_imagenetr.sh
sbatch slurm_shuffled/run_coda_prompt_omnibenchmark.sh
sbatch slurm_shuffled/run_coda_prompt_vtab.sh
