"""Report accuracy for the eval_shuffle=true runs (prefix "benchshuf"), our
canonical evaluation protocol for the PILOT baselines going forward.

Reuses the *_metrics.json files aggregate_results.py reads, but filters to
prefix="benchshuf" only (recovered from the metrics filename, since
MetricsLogger's meta dict doesn't carry it) -- ignores any leftover
eval_shuffle=false ("bench") runs.

Usage:
    python aggregate_shuffle_ablation.py [--roots logs]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_runs(roots):
    """-> {(model, dataset): {seed: tasks}}"""
    runs = defaultdict(dict)
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*_metrics.json"), recursive=True):
            if not os.path.basename(path).startswith("benchshuf_"):
                continue
            try:
                with open(path) as f:
                    run = json.load(f)
            except (OSError, json.JSONDecodeError):
                print(f"skipping unreadable {path}")
                continue
            meta, tasks = run.get("meta", {}), run.get("tasks", [])
            if not tasks:
                continue
            key = (str(meta.get("model_name")), str(meta.get("dataset")))
            seed = meta.get("seed")
            prev = runs[key].get(seed)
            if prev is None or len(tasks) > len(prev):
                runs[key][seed] = tasks
    return runs


def final_top1(tasks):
    curve = [t.get("cnn_top1") for t in tasks if t.get("cnn_top1") is not None]
    return curve[-1] if curve else None


def avg_inc_acc(tasks):
    curve = [t.get("cnn_top1") for t in tasks if t.get("cnn_top1") is not None]
    return float(np.mean(curve)) if curve else None


def summarize(by_seed):
    n_tasks = [len(t) for t in by_seed.values()]
    expected = max(n_tasks) if n_tasks else 0
    complete = {s: t for s, t in by_seed.items() if len(t) == expected}
    finals = [f for f in (final_top1(t) for t in complete.values()) if f is not None]
    avgs = [a for a in (avg_inc_acc(t) for t in complete.values()) if a is not None]
    return {
        "n_seeds": len(complete),
        "n_partial": len(by_seed) - len(complete),
        "final_mean": float(np.mean(finals)) if finals else None,
        "final_std": float(np.std(finals)) if finals else None,
        "avg_mean": float(np.mean(avgs)) if avgs else None,
        "avg_std": float(np.std(avgs)) if avgs else None,
    }


def fmt(mean, std):
    if mean is None:
        return "-"
    return f"{mean:.2f}±{std:.2f}" if std is not None else f"{mean:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=["logs"])
    cli = parser.parse_args()

    runs = load_runs([r for r in cli.roots if os.path.isdir(r)])
    if not runs:
        print("No benchshuf_*_metrics.json found under: " + ", ".join(cli.roots))
        return

    hdr = f"{'method':<16}{'dataset':<16}{'seeds':<7}{'final top1':<16}{'avg inc acc':<16}"
    print(hdr)
    print("-" * len(hdr))
    for (model, dataset), by_seed in sorted(runs.items()):
        s = summarize(by_seed)
        flag = f"  [{s['n_partial']} partial]" if s["n_partial"] else ""
        print(
            f"{model:<16}{dataset:<16}{s['n_seeds']:<7}"
            f"{fmt(s['final_mean'], s['final_std']):<16}"
            f"{fmt(s['avg_mean'], s['avg_std']):<16}{flag}"
        )


if __name__ == "__main__":
    main()
