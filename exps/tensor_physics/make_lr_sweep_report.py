"""
make_lr_sweep_report.py — Aggregate lr-sweep training logs into a Markdown report.

Reads SLURM stdout logs that the training scripts produce, extracts loss curves
keyed on (step, epoch), and writes a Markdown report plus loss-curve plots.

Each log line of interest matches:
    step=XXXXXXX  epoch=YY  loss=Z.ZZZZ  elapsed=H.HHHh  mem=...GB

Usage:
    python make_lr_sweep_report.py
    python make_lr_sweep_report.py --out /tmp/report.md
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SWEEPS = {
    "karman": {
        "logs_dir": "/anvil/projects/x-eng260004/factor_diffusion/our_method_results/karman_vortex_2d_lr_sweep/logs",
        "outdir":   "/anvil/projects/x-eng260004/factor_diffusion/our_method_results/karman_vortex_2d_lr_sweep",
        "lrs":      ["1e-2", "5e-3", "1e-3", "5e-4", "5e-5"],
        "time_limit_h": 10.0,
    },
    "burgers": {
        "logs_dir": "/anvil/projects/x-eng260004/factor_diffusion/our_method_results/burgers_2d_lr_sweep/logs",
        "outdir":   "/anvil/projects/x-eng260004/factor_diffusion/our_method_results/burgers_2d_lr_sweep",
        "lrs":      ["1e-2", "5e-3", "1e-3", "5e-4"],
        "time_limit_h": 5.0,
    },
}

LOSS_RE = re.compile(
    r"step=(\d+)\s+epoch=(\d+)\s+loss=([\-\d\.eE\+]+)\s+elapsed=([\-\d\.eE\+]+)h"
)


def parse_log(path):
    steps, epochs, losses, elapsed = [], [], [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = LOSS_RE.search(line)
            if not m:
                continue
            try:
                s = int(m.group(1))
                e = int(m.group(2))
                l = float(m.group(3))
                t = float(m.group(4))
            except ValueError:
                continue
            steps.append(s)
            epochs.append(e)
            losses.append(l)
            elapsed.append(t)
    return steps, epochs, losses, elapsed


def find_log_for_lr(logs_dir, lr_tag):
    candidates = sorted(glob.glob(os.path.join(logs_dir, f"lr{lr_tag}_*.out")))
    return candidates[-1] if candidates else None


def list_checkpoints(outdir, lr_tag):
    pattern = os.path.join(outdir, f"lr_{lr_tag}", "checkpoints", "epoch*_step*.pt")
    return sorted(glob.glob(pattern))


def summarize(steps, epochs, losses, elapsed):
    if not losses:
        return None
    last_window = losses[-min(20, len(losses)):]
    return {
        "n_log_points":     len(losses),
        "max_step":         steps[-1],
        "max_epoch":        epochs[-1],
        "final_loss":       losses[-1],
        "tail_avg_loss":    sum(last_window) / len(last_window),
        "min_loss":         min(losses),
        "min_loss_step":    steps[losses.index(min(losses))],
        "elapsed_hours":    elapsed[-1],
    }


def plot_curves(sweep_name, lr_to_curve, out_png, ylog=True):
    plt.figure(figsize=(9, 5))
    for lr_tag, (steps, _epochs, losses, _elapsed) in lr_to_curve.items():
        if losses:
            plt.plot(steps, losses, label=f"lr={lr_tag}", linewidth=1.2, alpha=0.85)
    plt.xlabel("step")
    plt.ylabel("loss" + (" (log)" if ylog else ""))
    if ylog:
        plt.yscale("log")
    plt.title(f"{sweep_name} — lr sweep loss curves")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close()


def render_report(results, out_md):
    lines = ["# LR Sweep Report", ""]
    for sweep_name, info in results.items():
        cfg = SWEEPS[sweep_name]
        lines.append(f"## {sweep_name}")
        lines.append("")
        lines.append(f"- Time budget per run: **{cfg['time_limit_h']:.1f} h**")
        lines.append(f"- Target epochs: **500**")
        lines.append(f"- Output dir: `{cfg['outdir']}`")
        lines.append(f"- Loss curves: ![{sweep_name} curves]({sweep_name}_curves.png)")
        lines.append("")
        lines.append(
            "| lr | epochs reached | last step | final loss | tail-avg loss | best loss | best @ step | elapsed (h) | log file |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        per_lr = info["per_lr"]
        for lr_tag in cfg["lrs"]:
            row = per_lr.get(lr_tag)
            if row is None:
                lines.append(
                    f"| {lr_tag} | – | – | – | – | – | – | – | _no log found_ |"
                )
                continue
            summary = row["summary"]
            log_path = row["log"]
            if summary is None:
                lines.append(
                    f"| {lr_tag} | – | – | – | – | – | – | – | `{os.path.basename(log_path)}` (no loss lines yet) |"
                )
                continue
            lines.append(
                f"| {lr_tag} "
                f"| {summary['max_epoch']} "
                f"| {summary['max_step']} "
                f"| {summary['final_loss']:.4f} "
                f"| {summary['tail_avg_loss']:.4f} "
                f"| {summary['min_loss']:.4f} "
                f"| {summary['min_loss_step']} "
                f"| {summary['elapsed_hours']:.2f} "
                f"| `{os.path.basename(log_path)}` |"
            )
        lines.append("")
        # Best lr by tail-avg loss
        scored = [
            (lr, per_lr[lr]["summary"]["tail_avg_loss"])
            for lr in cfg["lrs"]
            if per_lr.get(lr) and per_lr[lr]["summary"] is not None
        ]
        if scored:
            best_lr, best_val = min(scored, key=lambda kv: kv[1])
            lines.append(f"**Best lr (lowest tail-avg loss): `{best_lr}` → {best_val:.4f}**")
            lines.append("")
        # Checkpoints
        lines.append("**Checkpoints saved**")
        lines.append("")
        for lr_tag in cfg["lrs"]:
            ckpts = list_checkpoints(cfg["outdir"], lr_tag)
            names = [os.path.basename(p) for p in ckpts]
            lines.append(f"- lr={lr_tag}: " + (", ".join(names) if names else "_none_"))
        lines.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote report → {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/anvil/projects/x-eng260004/factor_diffusion/our_method_results/lr_sweep_report",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for sweep_name, cfg in SWEEPS.items():
        per_lr = {}
        lr_to_curve = {}
        for lr_tag in cfg["lrs"]:
            log = find_log_for_lr(cfg["logs_dir"], lr_tag)
            if log is None:
                per_lr[lr_tag] = None
                continue
            steps, epochs, losses, elapsed = parse_log(log)
            per_lr[lr_tag] = {
                "log":     log,
                "summary": summarize(steps, epochs, losses, elapsed),
            }
            lr_to_curve[lr_tag] = (steps, epochs, losses, elapsed)
        out_png = os.path.join(args.out_dir, f"{sweep_name}_curves.png")
        plot_curves(sweep_name, lr_to_curve, out_png)
        results[sweep_name] = {"per_lr": per_lr}

    render_report(results, os.path.join(args.out_dir, "report.md"))


if __name__ == "__main__":
    main()
