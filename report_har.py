"""Collect results.json files and compare them against the paper's HAR chart.

The HAR accuracies exist only as a bar chart in the paper -- there is no table --
so PAPER below is read off fig/HAR.png of the arXiv source (2111.03976, Figure 10;
Figure 11 in the accepted paper) and is
accurate to about +/-0.5. Treat a gap under ~1 point as agreement.

    python report_har.py results/
"""
import argparse
import glob
import json
import os

# model -> {arm: (in-set, out-of-set)}
PAPER = {
    "rt_2dcnn":         {"dft": (50.4, 28.4), "cubelearn": (79.8, 34.1)},
    "dt_2dcnn":         {"dft": (72.0, 69.4), "cubelearn": (93.9, 80.2)},
    "at_2dcnn":         {"dft": (44.2, 27.0), "cubelearn": (72.3, 29.1)},
    "rdt_2dcnn_lstm":   {"dft": (97.5, 71.6), "cubelearn": (98.7, 72.1)},
    "rdt_3dcnn":        {"dft": (95.5, 70.0), "cubelearn": (99.1, 77.2)},
    "rat_2dcnn_lstm":   {"dft": (80.4, 33.4), "cubelearn": (85.2, 36.8)},
    "rat_3dcnn":        {"dft": (80.8, 37.0), "cubelearn": (88.4, 43.2)},
    "dat_2dcnn_lstm":   {"dft": (98.8, 86.9), "cubelearn": (99.4, 91.1)},
    "dat_3dcnn":        {"dft": (90.6, 81.5), "cubelearn": (98.7, 92.4)},
    "rdat_3dcnn_lstm":  {"dft": (98.8, 45.8), "cubelearn": (98.8, 45.3)},
}


def fmt(ours, paper):
    if paper is None:
        return f"{ours:6.2f}    —        —"
    d = ours - paper
    flag = "ok" if abs(d) <= 1.0 else ("HIGH" if d > 0 else "LOW")
    return f"{ours:6.2f}  {paper:6.1f}  {d:+6.2f}  {flag}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="results")
    args = p.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.root, "*", "*", "*", "results.json"))):
        with open(path) as f:
            r = json.load(f)
        arm = "dft" if r["lpp_lr"] == 0 else "cubelearn"
        rows.append((r["model"], arm, r, path))

    if not rows:
        print(f"no results.json under {args.root}/")
        return

    print(f"{'model':18s} {'arm':10s} {'seed':>4s} {'split':9s} "
          f"{'ours':>6s}  {'paper':>6s}  {'delta':>6s}")
    print("-" * 74)
    for model, arm, r, path in rows:
        target = PAPER.get(model, {}).get(arm)
        for i, split in enumerate(("test_in", "test_out")):
            acc = r[split]["acc"] * 100
            paper = target[i] if target else None
            head = f"{model:18s} {arm:10s} {r['seed']:>4d}" if i == 0 else " " * 34
            print(f"{head} {split:9s} {fmt(acc, paper)}")
        print(f"{'':34s} {'val':9s} {r['val_acc'] * 100:6.2f}"
              f"    (best epoch {r['best_epoch']}/{r['epochs']}, {r['train_seconds']:.0f}s)")

    # The paper's central claim is the CubeLearn - DFT delta, so state it directly.
    # Group by model, epochs, seed, batch size and classifier learning rate:
    # would silently compare unlike runs.
    groups = {}
    for model, arm, r, _ in rows:
        groups.setdefault((model, r["epochs"], r["seed"], r["batch_size"], r["lr"]), {})[arm] = r
    print("\nCubeLearn − DFT (the paper's claim):")
    for (model, epochs, seed, batch, lr), arms in sorted(groups.items()):
        tag = f"{model} @{epochs}ep seed{seed} bs{batch} lr{lr}"
        if {"dft", "cubelearn"} <= set(arms):
            t = PAPER.get(model, {})
            for split, idx in (("test_in", 0), ("test_out", 1)):
                ours = (arms["cubelearn"][split]["acc"] - arms["dft"][split]["acc"]) * 100
                exp = (t["cubelearn"][idx] - t["dft"][idx]) if t else None
                extra = f"  (paper {exp:+.1f})" if exp is not None else ""
                print(f"  {tag:34s} {split:9s} {ours:+6.2f}{extra}")
        else:
            print(f"  {tag:34s} incomplete — have {sorted(arms)}")


if __name__ == "__main__":
    main()
