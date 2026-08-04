"""Generate dedicated FLOPs-profiling configs: 8 methods x 1 dataset (cifar224
by default), single seed, REDUCED epoch count (1 instead of the real
tuned_epoch/epochs/init_epochs/later_epochs, up to 50), config flag
profile_train_flops=true.

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

Epochs reduced to 1: a full-epoch-count attempt (8h/14h time limits) OOM'd
on all 8 methods, some (ranpac, aper_adapter) before even completing task
0. Root cause: torch.profiler(with_flops=True) records a permanent,
never-discarded metadata entry (op name, tensor shapes, dtypes, FLOPs) for
EVERY tensor operation executed while active -- unlike normal training,
whose memory is flat/steady-state per step, this profiling record grows
for as long as the profiler context is open. Profiling a full multi-epoch
task means tens of thousands of batches' worth of operation records held
in memory simultaneously, which exceeded 40GB even for a single task.

aggregate_results.py's per_seed_summary already scales
measured_flops x (real_epochs / profiled_epochs) per task using each
run's logged profiled_epochs field, so profiling 1 epoch and extrapolating
to the real epoch count needs no further code change here. This
extrapolation is exact, not approximate, for FLOPs specifically (unlike
wall-clock time): FLOPs depend only on tensor shapes, and none of these 8
methods (nor TOSCA) change their per-task computational graph's shapes
between epochs (no early stopping, no epoch-dependent architecture
changes) -- verified by reading every method's training loop.

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
# single profiling run finishes in a bounded, small number of batches
# regardless of which key a given method actually uses (extras are
# harmless no-ops for methods that don't read them).
EPOCH_KEYS = ["epochs", "tuned_epoch", "init_epochs", "later_epochs"]

# 1h flat: even the heaviest full accuracy run finished within ~3h; at 1
# epoch/task these should be a fraction of that, with generous headroom
# for torch.profiler's per-op overhead (much smaller now that it's bounded
# to 1 epoch instead of accumulating across the whole training loop).
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
