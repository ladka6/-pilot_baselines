"""Compare eval_shuffle=false (prefix bench) vs eval_shuffle=true (prefix
benchshuf) accuracy for every (method, dataset) that has results under both.

Reuses the same *_metrics.json files aggregate_results.py reads, but that
script keys runs by (model, dataset) alone, which silently merges bench and
benchshuf runs of the same method/dataset. This script keys by
(model, dataset, prefix) instead -- prefix is recovered from the metrics
filename (`<prefix>_<seed>_<backbone>_metrics.json`), since MetricsLogger's
meta dict doesn't carry it.

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
    """-> {(model, dataset, prefix): {seed: tasks}}"""
    runs = defaultdict(dict)
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*_metrics.json"), recursive=True):
            try:
                with open(path) as f:
                    run = json.load(f)
            except (OSError, json.JSONDecodeError):
                print(f"skipping unreadable {path}")
                continue
            meta, tasks = run.get("meta", {}), run.get("tasks", [])
            if not tasks:
                continue
            prefix = os.path.basename(path).split("_")[0]
            key = (str(meta.get("model_name")), str(meta.get("dataset")), prefix)
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
    finals = [final_top1(t) for t in complete.values()]
    avgs = [avg_inc_acc(t) for t in complete.values()]
    finals = [f for f in finals if f is not None]
    avgs = [a for a in avgs if a is not None]
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
        print("No *_metrics.json found under: " + ", ".join(cli.roots))
        return

    pairs = defaultdict(dict)
    for (model, dataset, prefix), by_seed in runs.items():
        pairs[(model, dataset)][prefix] = summarize(by_seed)

    hdr = f"{'method':<16}{'dataset':<16}{'orig (n)':<16}{'shuf (n)':<16}{'delta (final)':<14}"
    print(hdr)
    print("-" * len(hdr))
    for (model, dataset), by_prefix in sorted(pairs.items()):
        orig = by_prefix.get("bench")
        shuf = by_prefix.get("benchshuf")
        if orig is None and shuf is None:
            continue
        orig_s = f"{fmt(orig['final_mean'], orig['final_std'])} (n={orig['n_seeds']})" if orig else "-"
        shuf_s = f"{fmt(shuf['final_mean'], shuf['final_std'])} (n={shuf['n_seeds']})" if shuf else "-"
        delta = "-"
        if orig and shuf and orig["final_mean"] is not None and shuf["final_mean"] is not None:
            delta = f"{shuf['final_mean'] - orig['final_mean']:+.2f}"
        flags = []
        if orig and orig["n_partial"]:
            flags.append(f"{orig['n_partial']} orig partial")
        if shuf and shuf["n_partial"]:
            flags.append(f"{shuf['n_partial']} shuf partial")
        flag_s = f"  [{', '.join(flags)}]" if flags else ""
        print(f"{model:<16}{dataset:<16}{orig_s:<16}{shuf_s:<16}{delta:<14}{flag_s}")


if __name__ == "__main__":
    main()
