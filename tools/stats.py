"""Statistics for the benchmark (NumPy only, no torch needed).

Reads every results.csv under the given folders, plus the per-example test
predictions (preds_*.npz) that qvb/train.py saves next to them.

    # summary table: mean, std and 95% CI per (task, model, lr)
    python -m tools.stats results/

    # learning-rate selection: best lr per (task, model) by mean VALIDATION accuracy
    python -m tools.stats results/ --select-lr

    # paired comparison of two models on the same test examples
    python -m tools.stats results/ --task pair_same --compare unlu gaussian --margin 1.0

Confidence intervals use a two-level bootstrap: resample seeds, then resample
test examples, so both sources of randomness are included. Paired comparisons
resample the SAME seeds and examples for both models, which cancels the
difficulty of individual test examples. "Equivalent" means the 90% CI of the
difference lies inside +/- margin points (TOST, alpha = 0.05 each side).
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

B = 2000
RNG = np.random.default_rng(12345)


def load_rows(folders):
    rows = []
    for folder in folders:
        for path in glob.glob(os.path.join(folder, "**", "results.csv"), recursive=True):
            base = os.path.dirname(path)
            for r in csv.DictReader(open(path)):
                if r.get("quick", "False") == "True":
                    continue
                r["_dir"] = base
                rows.append(r)
    return rows


def tag(r):
    return (f"{r['model']}_{r.get('task', 'digits')}_{r.get('dataset', 'mnist')}{r.get('img', 14)}_"
            f"{r.get('patch', '2x14')}_lr{float(r.get('lr', 1e-3)):g}_s{r['seed']}")


def correctness(r):
    """Per-example 0/1 correctness for one run, or None if no predictions were saved."""
    path = os.path.join(r["_dir"], f"preds_{tag(r)}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return (d["pred"] == d["label"]).astype(np.float32)


def key(r):
    return (r.get("task", "digits"), r["model"], float(r.get("lr", 1e-3)))


def boot_mean(mat):
    """mat: (seeds, examples) 0/1 matrix -> bootstrap distribution of the mean."""
    s, n = mat.shape
    out = np.empty(B)
    for b in range(B):
        si = RNG.integers(0, s, s)
        ei = RNG.integers(0, n, n)
        out[b] = mat[np.ix_(si, ei)].mean()
    return out


def summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    print("| task | model | lr | seeds | params | test mean ± std | 95% CI | val mean |")
    print("|---|---|---|---|---|---|---|---|")
    for (task, model, lr), rs in sorted(groups.items()):
        acc = np.array([float(r["test_acc"]) for r in rs]) * 100
        val = np.mean([float(r["best_val"]) for r in rs]) * 100
        mats = [correctness(r) for r in rs]
        if all(m is not None for m in mats) and len({len(m) for m in mats}) == 1:
            dist = boot_mean(np.stack(mats)) * 100
            ci = f"[{np.percentile(dist, 2.5):.1f}, {np.percentile(dist, 97.5):.1f}]"
        else:
            ci = "n/a (no per-example preds)"
        sd = acc.std(ddof=1) if len(acc) > 1 else 0.0
        print(f"| {task} | {model} | {lr:g} | {len(rs)} | {rs[0]['params']} | "
              f"{acc.mean():.1f} ± {sd:.1f} | {ci} | {val:.1f} |")


def select_lr(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        task, model, lr = key(r)
        groups[(task, model)][lr].append(float(r["best_val"]))
    print("| task | model | chosen lr (best mean val) | mean val per lr |")
    print("|---|---|---|---|")
    for (task, model), by_lr in sorted(groups.items()):
        means = {lr: np.mean(v) for lr, v in by_lr.items()}
        best = max(means, key=means.get)
        detail = ", ".join(f"{lr:g}: {100 * m:.1f} (n={len(by_lr[lr])})" for lr, m in sorted(means.items()))
        print(f"| {task} | {model} | **{best:g}** | {detail} |")


def compare(rows, task, a, b, margin, lr_a=None, lr_b=None):
    def runs(model, lr):
        rs = [r for r in rows if r.get("task", "digits") == task and r["model"] == model
              and (lr is None or float(r.get("lr", 1e-3)) == lr)]
        if lr is None and rs:   # default: the lr with the most seeds, then highest mean val
            by = defaultdict(list)
            for r in rs:
                by[float(r.get("lr", 1e-3))].append(r)
            best = max(by, key=lambda k: (len(by[k]), np.mean([float(r["best_val"]) for r in by[k]])))
            rs = by[best]
        return {r["seed"]: correctness(r) for r in rs}

    ra, rb = runs(a, lr_a), runs(b, lr_b)
    seeds = sorted(set(ra) & set(rb), key=int)
    seeds = [s for s in seeds if ra[s] is not None and rb[s] is not None]
    if not seeds:
        sys.exit(f"no common seeds with per-example predictions for {a} vs {b} on {task}")
    da = np.stack([ra[s] for s in seeds])
    db = np.stack([rb[s] for s in seeds])
    diff = (da - db) * 100                                   # (seeds, examples), in points
    s, n = diff.shape
    dist = np.empty(B)
    for i in range(B):
        si = RNG.integers(0, s, s)
        ei = RNG.integers(0, n, n)
        dist[i] = diff[np.ix_(si, ei)].mean()
    est = diff.mean()
    lo95, hi95 = np.percentile(dist, [2.5, 97.5])
    lo90, hi90 = np.percentile(dist, [5, 95])
    p = min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean()))
    equiv = lo90 > -margin and hi90 < margin
    print(f"task={task}: {a} − {b} = {est:+.2f} points over {len(seeds)} seeds × {n} test examples")
    print(f"  95% CI [{lo95:+.2f}, {hi95:+.2f}]   bootstrap p ≈ {p:.3f}")
    print(f"  90% CI [{lo90:+.2f}, {hi90:+.2f}]   equivalent within ±{margin:g}? {'YES' if equiv else 'no'}")
    if lo95 > 0:
        print(f"  -> {a} is better")
    elif hi95 < 0:
        print(f"  -> {b} is better")
    else:
        print("  -> no significant difference")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    p.add_argument("--select-lr", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("MODEL_A", "MODEL_B"))
    p.add_argument("--task", default="digits")
    p.add_argument("--margin", type=float, default=1.0, help="equivalence margin in points")
    p.add_argument("--lr-a", type=float)
    p.add_argument("--lr-b", type=float)
    a = p.parse_args()
    rows = load_rows(a.folders)
    if not rows:
        sys.exit("no results found")
    if a.select_lr:
        select_lr(rows)
    elif a.compare:
        compare(rows, a.task, *a.compare, a.margin, a.lr_a, a.lr_b)
    else:
        summary(rows)


if __name__ == "__main__":
    main()
