"""Generate dedicated FLOPs-profiling configs: 8 methods x 1 dataset (cifar224
by default), single seed, reduced epochs, config flag profile_train_flops=true.

cifar224 has the most tasks (20) of any dataset here, making it the most
informative single choice for distinguishing task-0 vs. later-task training
behavior (the exact nuance this profiling run exists to capture) -- no need
to burn budget profiling all 6 datasets when one representative one suffices
for a correction factor.

Measures REAL forward+backward FLOPs per task via torch.profiler (see
trainer.py's profile_train_flops path), replacing the analytic
train_gflops_est heuristic used by aggregate_results.py. Real profiling
naturally captures per-method training nuances (only task 0 trains for
ranpac/aper_adapter, only the current adapter -- not the eval-time ensemble
-- trains for ease/mos) without per-method special-casing.

Epochs reduced to 1 (from up to 20) across every epoch-related config key
(epochs/tuned_epoch/init_epochs/later_epochs) to keep profiler overhead
bounded -- accuracy from these runs is meaningless and must never be mixed
into the accuracy tables. aggregate_results.py enforces this by filename
prefix ("profileflops_"), loaded via a separate path that only ever feeds
the measured FLOPs into the efficiency table.

Usage:  python gen_profile_flops.py
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
}

SEED = [1993]  # single seed, this is a representative profiling run, not accuracy

# Every epoch-related key any of the 8 methods reads, forced to 1 so a
# whole-dataset profiling run finishes fast regardless of which key a given
# method actually uses (extras are harmless no-ops for methods that don't
# read them).
EPOCH_KEYS = ["epochs", "tuned_epoch", "init_epochs", "later_epochs"]

# 1h flat: even the heaviest full accuracy run (cifar224, up to 20 epochs x
# 20 tasks) finished within ~3h; at 1 epoch/task these should be roughly
# 1/20th that, with generous headroom for torch.profiler overhead.
TIME_LIMIT = "01:00:00"

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=pilot-{method}-flops-{dataset}
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time={time}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/pilot_baselines"
source .venv/bin/activate
[ -e data ] || ln -s "$HOME/continious_learning/data" data
mkdir -p logs

python main.py --config {config} --seed 1993
"""


def main():
    os.makedirs("exps/profile_flops", exist_ok=True)
    os.makedirs("slurm_profile_flops", exist_ok=True)

    launcher_lines = []
    for method, (default_cfg, inr_cfg) in METHODS.items():
        for dataset, (init_cls, increment) in DATASETS.items():
            base_path = inr_cfg if dataset == "imagenetr" else default_cfg
            with open(base_path) as f:
                config = json.load(f)

            config["prefix"] = "profileflops"
            config["dataset"] = dataset
            config["init_cls"] = init_cls
            config["increment"] = increment
            config["shuffle"] = True
            config["seed"] = SEED
            config["backbone_type"] = BACKBONE_IN21K[method]
            config["eval_shuffle"] = True  # match the canonical eval protocol
            config["profile_train_flops"] = True
            for k in EPOCH_KEYS:
                if k in config:
                    config[k] = 1

            config_path = f"exps/profile_flops/{method}_{dataset}.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                f.write("\n")

            script_path = f"slurm_profile_flops/run_{method}_{dataset}.sh"
            with open(script_path, "w") as f:
                f.write(
                    SBATCH_TEMPLATE.format(
                        method=method, dataset=dataset,
                        time=TIME_LIMIT, config=config_path,
                    )
                )
            os.chmod(script_path, 0o755)
            launcher_lines.append(f"sbatch {script_path}")

    with open("slurm_profile_flops/submit_all.sh", "w") as f:
        f.write("#!/bin/bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/..\"\n")
        f.write("\n".join(launcher_lines) + "\n")
    os.chmod("slurm_profile_flops/submit_all.sh", 0o755)

    print(f"{len(launcher_lines)} (method, dataset) profiling configs generated "
          f"(1 seed, 1 epoch each).")
    print("Nothing submitted. To launch all: ./slurm_profile_flops/submit_all.sh")
    print("Or submit individually via the printed sbatch commands, in waves.")


if __name__ == "__main__":
    main()
