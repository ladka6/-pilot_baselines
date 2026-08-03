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

Time limits are per-(method, dataset), not per-dataset -- actual cost varies
enormously by method (e.g. coda_prompt takes ~85x longer than simplecil on
the same dataset; a single flat-per-dataset number either wastes budget on
fast methods or times out slow ones). See TIME_LIMIT_FOR below.

Usage:  python gen_bench_shuffled.py
"""
import json
import math
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

# Measured train_seconds_total (wall clock, from aggregate_results.py's
# efficiency table) for (method, dataset) pairs that have actually
# completed a run. Ground truth -- everything else here is derived or
# guessed from this.
OBSERVED_TRAIN_SECONDS = {
    ("simplecil", "cifar224"): 130.10, ("simplecil", "cub"): 55.93,
    ("simplecil", "imageneta"): 27.46, ("simplecil", "imagenetr"): 71.67,
    ("simplecil", "omnibenchmark"): 232.56, ("simplecil", "vtab"): 8.53,
    ("ranpac", "cifar224"): 736.55, ("ranpac", "cub"): 405.05,
    ("ranpac", "imageneta"): 261.29, ("ranpac", "imagenetr"): 398.25,
    ("ranpac", "omnibenchmark"): 1383.50, ("ranpac", "vtab"): 211.90,
    ("aper_adapter", "cifar224"): 556.56, ("aper_adapter", "cub"): 142.30,
    ("aper_adapter", "imageneta"): 134.33, ("aper_adapter", "imagenetr"): 311.51,
    ("aper_adapter", "omnibenchmark"): 1412.45, ("aper_adapter", "vtab"): 150.70,
    ("ease", "cifar224"): 6836.74, ("ease", "cub"): 1711.22,
    ("ease", "imageneta"): 914.27, ("ease", "imagenetr"): 3202.72,
    ("ease", "omnibenchmark"): 10673.48, ("ease", "vtab"): 290.34,
    ("mos", "cifar224"): 6011.11, ("mos", "cub"): 1655.85,
    ("mos", "imageneta"): 940.16, ("mos", "imagenetr"): 1848.65,
    ("mos", "omnibenchmark"): 10736.30, ("mos", "vtab"): 279.90,
    ("coda_prompt", "cifar224"): 11117.14, ("coda_prompt", "cub"): 2353.39,
    ("coda_prompt", "imageneta"): 1418.64, ("coda_prompt", "vtab"): 833.17,
    ("l2p", "vtab"): 226.48,
    ("dualprompt", "vtab"): 209.02,
}

# coda_prompt on imagenetr/omnibenchmark TIMED OUT (no completed
# measurement to derive from) -- these are reasoned from partial-progress
# pacing instead:
#  - imagenetr: TIMEOUT'd at exactly 3h across all 5 seeds; its _inr config
#    sets tuned_epoch=50 vs the default 20, implying ~3.86h real need.
#  - omnibenchmark: paced ~29min/task across 10 tasks (job 25180737, seen
#    live via log tail), ~4.8h projected total.
MANUAL_TIME_OVERRIDES = {
    ("coda_prompt", "imagenetr"): "06:00:00",
    ("coda_prompt", "omnibenchmark"): "08:00:00",
}

# l2p/dualprompt have never been run beyond the standalone VTAB ablation --
# no observed data for their other 5 datasets. Both are prompt-pool methods
# structurally similar to coda_prompt, but with much smaller tuned_epoch (5
# vs coda_prompt's 20 by default, 10 vs 50 for the imagenetr _inr variant
# specifically) -- derive a conservative estimate by scaling coda_prompt's
# own known/estimated time by that epoch ratio, rather than guessing blind
# or reusing an unrelated per-dataset default. Unverified until real data
# lands; watch for TIMEOUTs and adjust.
EPOCH_RATIO = {"default": 5 / 20, "imagenetr": 10 / 50}


def _hms(seconds):
    seconds = max(seconds, 3600)  # 1h floor
    seconds = math.ceil(seconds / 1800) * 1800  # round up to nearest 30 min
    h, r = divmod(int(seconds), 3600)
    m = r // 60
    return f"{h:02d}:{m:02d}:00"


def _hms_to_seconds(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


SAFETY_FACTOR = 1.8  # headroom over the single observed data point


def time_limit_for(method, dataset):
    if (method, dataset) in MANUAL_TIME_OVERRIDES:
        return MANUAL_TIME_OVERRIDES[(method, dataset)]
    observed = OBSERVED_TRAIN_SECONDS.get((method, dataset))
    if observed is not None:
        return _hms(observed * SAFETY_FACTOR)
    if method in ("l2p", "dualprompt"):
        ratio = EPOCH_RATIO["imagenetr"] if dataset == "imagenetr" else EPOCH_RATIO["default"]
        coda_seconds = (
            _hms_to_seconds(MANUAL_TIME_OVERRIDES[("coda_prompt", dataset)])
            if ("coda_prompt", dataset) in MANUAL_TIME_OVERRIDES
            else _hms_to_seconds(_hms(OBSERVED_TRAIN_SECONDS[("coda_prompt", dataset)] * SAFETY_FACTOR))
        )
        return _hms(coda_seconds * ratio)
    raise KeyError(f"No time estimate for ({method}, {dataset}) -- add data or an override.")


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


# Plain bash launcher (not itself an sbatch script): submits the per-dataset
# array job for every dataset of one method in a single invocation, e.g.
# `./slurm_shuffled/ranpac.sh` submits all 6 of that method's dataset jobs.
LAUNCHER_TEMPLATE = """#!/bin/bash
# Submits all {n_datasets} dataset jobs for {method} (eval_shuffle=true ablation).
set -euo pipefail
cd "$(dirname "$0")/.."
{sbatch_lines}
"""


def main():
    os.makedirs("exps/bench_shuffled", exist_ok=True)
    os.makedirs("slurm_shuffled", exist_ok=True)

    sbatch_cmds = []
    for method, (default_cfg, inr_cfg) in METHODS.items():
        method_sbatch_lines = []
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
                        time=time_limit_for(method, dataset),
                        config=config_path,
                    )
                )
            os.chmod(script_path, 0o755)
            sbatch_cmds.append(f"sbatch {script_path}")
            method_sbatch_lines.append(f"sbatch slurm_shuffled/run_{method}_{dataset}.sh")

        launcher_path = f"slurm_shuffled/{method}.sh"
        with open(launcher_path, "w") as f:
            f.write(
                LAUNCHER_TEMPLATE.format(
                    method=method,
                    n_datasets=len(DATASETS),
                    sbatch_lines="\n".join(method_sbatch_lines),
                )
            )
        os.chmod(launcher_path, 0o755)

    print(f"{len(sbatch_cmds)} (method, dataset) pairs x 5 seeds generated.")
    print(f"Plus {len(METHODS)} per-method launchers (slurm_shuffled/<method>.sh).")
    print("Nothing submitted. To launch one method's full dataset sweep:")
    print("  ./slurm_shuffled/<method>.sh")


if __name__ == "__main__":
    main()
