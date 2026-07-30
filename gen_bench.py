"""Generate the baseline benchmark grid: 8 methods x 6 datasets, matching the
TOSCA repo's protocol (same class splits, shuffle=true, seeds 1993-1997).

Per (method, dataset) this writes one config under exps/bench/ and one SLURM
array script under slurm/ (--array=0-4 -> seeds 1993..1997, one seed per job
so a crashed seed never takes the others down; main.py --seed overrides the
config's seed list).

Hyperparameters come from the method's own reference config: the *_inr.json
variant for imagenetr, the default (cifar) variant for everything else --
only dataset/split/shuffle/seed/prefix are overridden.

Usage:  python gen_bench.py            # writes files, prints sbatch commands
"""
import json
import os

METHODS = {
    # method -> (default config, inr config)
    "simplecil": ("exps/simplecil.json", "exps/simplecil_inr.json"),
    "ranpac": ("exps/ranpac.json", "exps/ranpac_inr.json"),
    "l2p": ("exps/l2p.json", "exps/l2p_inr.json"),
    "dualprompt": ("exps/dualprompt.json", "exps/dualprompt_inr.json"),
    "coda_prompt": ("exps/coda_prompt.json", "exps/coda_prompt_inr.json"),
    "aper_adapter": ("exps/aper_aperpter.json", "exps/aper_aperpter_inr.json"),
    "ease": ("exps/ease.json", "exps/ease_inr.json"),
    "mos": ("exps/mos.json", "exps/mos_inr.json"),
}

# PILOT's reference configs default to ViT-B/16-IN1K. Our own TOSCA configs
# (and the TOSCA paper's main Table 1) use ViT-B/16-IN21K -- override to the
# IN21K variant per method so the baselines are comparable to our numbers.
# (l2p/dualprompt/coda_prompt in21k variants were added to their backbone/
# files; the others already ship an in21k-registered name in utils/inc_net.py.)
BACKBONE_IN21K = {
    "simplecil": "pretrained_vit_b16_224_in21k",
    "ranpac": "pretrained_vit_b16_224_in21k_adapter",
    "aper_adapter": "pretrained_vit_b16_224_in21k_adapter",
    "ease": "vit_base_patch16_224_in21k_ease",
    "mos": "vit_base_patch16_224_in21k_mos",
    "l2p": "vit_base_patch16_224_in21k_l2p",
    "dualprompt": "vit_base_patch16_224_in21k_dualprompt",
    "coda_prompt": "vit_base_patch16_224_in21k_coda_prompt",
}

# dataset -> (init_cls, increment), identical to the TOSCA repo's configs
DATASETS = {
    "cifar224": (5, 5),
    "cub": (10, 10),
    "imageneta": (20, 20),
    "imagenetr": (20, 20),
    "omnibenchmark": (30, 30),
    "vtab": (10, 10),
}

SEEDS = [1993, 1994, 1995, 1996, 1997]

# Light methods finish a single seed comfortably within a day; prompt-based
# and expert-based methods get the long limit.
TIME_LIGHT = "06:00:00"
# 8h, not 16h: a single 30-job wave at 16h (61,440 SBU) exceeds this
# account's entire balance (~39k SBU) by itself, guaranteeing cancellation
# regardless of what else is queued. 8h (30,720 SBU) fits with headroom,
# and real usage is expected to be far below even that (RanPAC finished in
# minutes against a 6h reservation).
TIME_HEAVY = "08:00:00"
HEAVY = {"l2p", "dualprompt", "coda_prompt", "ease", "mos"}

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=pilot-{method}-{dataset}
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time={time}
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
python main.py --config {config} --seed "$SEED"
"""


def main():
    os.makedirs("exps/bench", exist_ok=True)
    os.makedirs("slurm", exist_ok=True)

    sbatch_cmds = []
    for method, (default_cfg, inr_cfg) in METHODS.items():
        for dataset, (init_cls, increment) in DATASETS.items():
            base_path = inr_cfg if dataset == "imagenetr" else default_cfg
            with open(base_path) as f:
                config = json.load(f)

            config["prefix"] = "bench"
            config["dataset"] = dataset
            config["init_cls"] = init_cls
            config["increment"] = increment
            config["shuffle"] = True
            config["seed"] = SEEDS
            config["backbone_type"] = BACKBONE_IN21K[method]

            config_path = f"exps/bench/{method}_{dataset}.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                f.write("\n")

            script_path = f"slurm/run_{method}_{dataset}.sh"
            with open(script_path, "w") as f:
                f.write(
                    SBATCH_TEMPLATE.format(
                        method=method,
                        dataset=dataset,
                        time=TIME_HEAVY if method in HEAVY else TIME_LIGHT,
                        config=config_path,
                    )
                )
            os.chmod(script_path, 0o755)
            sbatch_cmds.append(f"sbatch {script_path}")

    print(f"{len(sbatch_cmds)} (method, dataset) pairs x 5 seeds generated.")
    print("Nothing submitted. To launch:")
    for cmd in sbatch_cmds:
        print(f"  {cmd}")


if __name__ == "__main__":
    main()
