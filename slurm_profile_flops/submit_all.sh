#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch slurm_profile_flops/run_simplecil_cifar224.sh
sbatch slurm_profile_flops/run_ranpac_cifar224.sh
sbatch slurm_profile_flops/run_l2p_cifar224.sh
sbatch slurm_profile_flops/run_dualprompt_cifar224.sh
sbatch slurm_profile_flops/run_coda_prompt_cifar224.sh
sbatch slurm_profile_flops/run_aper_adapter_cifar224.sh
sbatch slurm_profile_flops/run_ease_cifar224.sh
sbatch slurm_profile_flops/run_mos_cifar224.sh
