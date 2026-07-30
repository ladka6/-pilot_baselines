#!/bin/bash
# Submits all 6 dataset jobs for aper_adapter (eval_shuffle=true ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch slurm_shuffled/run_aper_adapter_cifar224.sh
sbatch slurm_shuffled/run_aper_adapter_cub.sh
sbatch slurm_shuffled/run_aper_adapter_imageneta.sh
sbatch slurm_shuffled/run_aper_adapter_imagenetr.sh
sbatch slurm_shuffled/run_aper_adapter_omnibenchmark.sh
sbatch slurm_shuffled/run_aper_adapter_vtab.sh
