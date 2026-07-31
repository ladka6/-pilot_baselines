"""Generate dedicated FLOPs-profiling configs: 8 methods x 1 dataset (cifar224
by default), single seed, FULL epoch count (same as the real accuracy run),
config flag profile_train_flops=true.

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

Full epoch count (not reduced) so the measured FLOPs are ground truth for
the actual training run, not an extrapolation from a 1-epoch sample --
removes any assumption that per-epoch cost is uniform across epochs. This
means the profiler runs across the whole training loop, so time limits
match the real accuracy sweep's per-dataset limits, not a fast reduced-scale
probe. Accuracy from these runs still isn't meant to be reported (single
seed, not 5), so they stay filtered out of the accuracy tables by filename
prefix ("profileflops_") regardless.

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

# cifar224's own real per-dataset time limit from gen_bench_shuffled.py
# (TIME_BY_DATASET["cifar224"]) was 6h for the accuracy sweep; give the
# profiler run extra headroom on top of that since torch.profiler's
# with_flops instrumentation adds real per-op overhead across the full
# training loop (not just one probe step).
TIME_LIMIT = "08:00:00"

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
            # Epoch counts left as-is (full, not reduced) -- see docstring.

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
          f"(1 seed, full epoch count).")
    print("Nothing submitted. To launch all: ./slurm_profile_flops/submit_all.sh")
    print("Or submit individually via the printed sbatch commands, in waves.")


if __name__ == "__main__":
    main()
