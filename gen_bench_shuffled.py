"""Generate the eval_shuffle=true ablation grid: ALL 8 methods x 6 datasets.

Same protocol as gen_bench.py (same splits/seeds/backbones) but with
eval_shuffle=true forced on every config, regardless of method. Non-prompt-pool
methods (simplecil/ranpac/aper_adapter/ease/mos) and coda_prompt are expected
to be batch-order invariant at eval time (confirmed via code inspection: no
cross-sample aggregation in their forward pass) -- running them here too turns
that into an empirical control, not just an assumption. l2p/dualprompt use
batchwise_prompt (majority-vote prompt selection across the batch), so they
are expected to show a real accuracy delta vs. the eval_shuffle=false runs in
exps/bench/.

Time limit: 3h flat for every job. The heaviest observed run so far (prompt/
adapter methods, original bench sweep) finished within 1.5-2h; 3h leaves
headroom while enabling more concurrent jobs per SBU wave than the old 6h/8h
split allowed.

Usage:  python gen_bench_shuffled.py
"""
import json
import os

METHODS = {
    "simplecil": ("exps/simplecil.json", "exps/simplecil_inr.json"),
    "ranpac": ("exps/ranpac.json", "exps/ranpac_inr.json"),
    "l2p": ("exps/l2p.json", "exps/l2p_inr.json"),
    "dualprompt": ("exps/dualprompt.json", "exps/dualprompt_inr.json"),
    "coda_prompt": ("exps/coda_prompt.json", "exps/coda_prompt_inr.json"),
    "aper_adapter": ("exps/aper_aperpter.json", "exps/aper_aperpter_inr.json"),
    "ease": ("exps/ease.json", "exps/ease_inr.json"),
    "mos": ("exps/mos.json", "exps/mos_inr.json"),
}

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

DATASETS = {
    "cifar224": (5, 5),
    "cub": (10, 10),
    "imageneta": (20, 20),
    "imagenetr": (20, 20),
    "omnibenchmark": (30, 30),
    "vtab": (10, 10),
}

SEEDS = [1993, 1994, 1995, 1996, 1997]

# Flat 3h for every job: highest observed runtime in the original (non-shuffled)
# sweep was ~1.5-2h, so 3h leaves headroom while letting more jobs run
# concurrently per SBU wave than the old 6h/8h split.
TIME_LIMIT = "03:00:00"

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=pilot-{method}-shuf-{dataset}
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
source .venv/bin/activate
[ -e data ] || ln -s "$HOME/continious_learning/data" data
mkdir -p logs

SEED=$((1993 + SLURM_ARRAY_TASK_ID))
python main.py --config {config} --seed "$SEED"
"""


def main():
    os.makedirs("exps/bench_shuffled", exist_ok=True)
    os.makedirs("slurm_shuffled", exist_ok=True)

    sbatch_cmds = []
    for method, (default_cfg, inr_cfg) in METHODS.items():
        for dataset, (init_cls, increment) in DATASETS.items():
            base_path = inr_cfg if dataset == "imagenetr" else default_cfg
            with open(base_path) as f:
                config = json.load(f)

            config["prefix"] = "benchshuf"
            config["dataset"] = dataset
            config["init_cls"] = init_cls
            config["increment"] = increment
            config["shuffle"] = True
            config["seed"] = SEEDS
            config["backbone_type"] = BACKBONE_IN21K[method]
            config["eval_shuffle"] = True

            config_path = f"exps/bench_shuffled/{method}_{dataset}.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                f.write("\n")

            script_path = f"slurm_shuffled/run_{method}_{dataset}.sh"
            with open(script_path, "w") as f:
                f.write(
                    SBATCH_TEMPLATE.format(
                        method=method,
                        dataset=dataset,
                        time=TIME_LIMIT,
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
