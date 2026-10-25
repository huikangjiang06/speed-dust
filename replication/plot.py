"""Plot the completed, user-selected 1M-token subset of DUST Figures 1 and 2.

Reads saved results only. It never loads a model or starts a training process.
"""

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (42, 43, 44)
SETTINGS = (
    ("backprop", None, "Backprop", "#202124"),
    ("dust", 256, "DUST, K = 256", "#a17bd1"),
    ("dust", 1024, "DUST, K = 1,024", "#7543b5"),
    ("dust", 4096, "DUST, K = 4,096", "#3e176d"),
)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "figures" / "1m_selected").resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(root / "cache" / "matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import FixedLocator, FuncFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#858b94",
        "axes.labelcolor": "#252932",
        "xtick.color": "#464c56",
        "ytick.color": "#464c56",
        "legend.fontsize": 10,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })

    curves, summaries, per_seed, input_files, groups = [], [], [], [], []
    shared_data = None
    shared_source = None
    for method, population, label, color in SETTINGS:
        losses, test_losses, best_vals, best_steps = [], [], [], []
        steps = None
        for seed in SEEDS:
            name = f"{method}-1M" + (f"-p{population}" if population else "") + f"-s{seed}"
            run = root / "runs" / name
            result_path, config_path = run / "result.json", run / "config.json"
            result = json.loads(result_path.read_text())
            config = json.loads(config_path.read_text())
            for item in (result, config):
                assert (item["method"], item["tokens"], item["population"], item["seed"]) == (
                    method, "1M", population, seed
                ), f"Unexpected identity: {name}"
            assert result["complete"] is True, f"Incomplete run: {name}"
            assert config["total_steps"] == 61 and config["training_tokens"] == 999424
            assert config["training_tokens"] == config["total_steps"] * 8 * config["model"]["sequence_len"]
            if shared_data is None:
                shared_data, shared_source = config["data_sha256"], config["source_sha256"]
            assert config["data_sha256"] == shared_data, "Mismatched input data"
            assert config["source_sha256"] == shared_source, "Mismatched training implementation"
            history = [row for row in result["history"] if "val_loss" in row]
            run_steps = np.array([row["step"] for row in history])
            assert list(run_steps) == [0, 1, *range(2, 61, 2), 61], f"Incomplete evaluation history: {name}"
            if steps is None:
                steps = run_steps
            assert np.array_equal(steps, run_steps), "Cannot aggregate different evaluation grids"
            values = np.array([row["val_loss"] for row in history], dtype=float)
            assert np.isfinite(values).all() and np.isfinite(result["test_at_best_val"])
            best_index = int(np.argmin(values))
            assert np.isclose(values[best_index], result["best_val"], atol=1e-12, rtol=0)
            losses.append(values)
            test_losses.append(result["test_at_best_val"])
            best_vals.append(result["best_val"])
            best_steps.append(int(steps[best_index]))
            per_seed.append(dict(method=method, population=population, seed=seed,
                                 best_validation_step=int(steps[best_index]),
                                 best_validation_loss=result["best_val"],
                                 test_at_best_validation=result["test_at_best_val"]))
            for path in (result_path, config_path):
                input_files.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        losses = np.array(losses)
        test_losses = np.array(test_losses)
        mean, sd = losses.mean(axis=0), losses.std(axis=0, ddof=1)
        fraction = steps / config["total_steps"]
        for i, step in enumerate(steps):
            curves.append(dict(method=method, population=population, step=int(step),
                               training_fraction=float(fraction[i]), tokens_seen=int(step * 8 * 2048),
                               validation_mean=float(mean[i]), validation_sample_sd=float(sd[i]),
                               **{f"validation_seed_{seed}": float(losses[j, i]) for j, seed in enumerate(SEEDS)}))
        summary = dict(method=method, population=population, seeds=len(SEEDS),
                       test_mean=float(test_losses.mean()), test_sample_sd=float(test_losses.std(ddof=1)),
                       best_validation_mean=float(np.mean(best_vals)),
                       best_validation_sample_sd=float(np.std(best_vals, ddof=1)))
        summaries.append(summary)
        groups.append(dict(method=method, population=population, label=label, color=color,
                           fraction=fraction, mean=mean, sd=sd, test_losses=test_losses,
                           **{key: summary[key] for key in ("test_mean", "test_sample_sd")}))

    def decorate(ax):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e2e5ea", linewidth=0.7)
        ax.tick_params(length=3.5)

    def figure_one(ax):
        decorate(ax)
        for group in groups:
            is_bp = group["method"] == "backprop"
            ax.fill_between(group["fraction"], group["mean"] - group["sd"], group["mean"] + group["sd"],
                            color=group["color"], alpha=0.075 if is_bp else 0.13, linewidth=0)
            ax.plot(group["fraction"], group["mean"], color=group["color"],
                    linestyle=(0, (4, 2.4)) if is_bp else "-", linewidth=2.0,
                    label=group["label"], zorder=5 if is_bp else 3)
        ax.set(xlim=(0, 1.01), ylim=(5.82, 8.43), xlabel="Training fraction",
               ylabel="Validation loss (nats/token)", title="Figure 1  |  Validation over training")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.legend(loc="upper right", handlelength=2.8)

    def figure_two(ax):
        decorate(ax)
        bp = groups[0]
        dust = groups[1:]
        populations = np.array([group["population"] for group in dust])
        means = np.array([group["test_mean"] for group in dust])
        sds = np.array([group["test_sample_sd"] for group in dust])
        ax.axhspan(bp["test_mean"] - bp["test_sample_sd"], bp["test_mean"] + bp["test_sample_sd"],
                   color=bp["color"], alpha=0.065, linewidth=0)
        bp_line = ax.axhline(bp["test_mean"], color=bp["color"], linewidth=1.8, linestyle=(0, (1.3, 2.2)),
                             label=f"Backprop: {bp['test_mean']:.4f}")
        dust_line = ax.errorbar(populations, means, yerr=sds, color="#7543b5", marker="o", markersize=7,
                               linewidth=1.9, capsize=5, capthick=1.3, elinewidth=1.3,
                               label="DUST: mean ± 1 SD", zorder=4)
        for group in dust:
            ax.scatter([group["population"]] * len(SEEDS), group["test_losses"],
                       s=18, facecolors="white", edgecolors="#7543b5", linewidths=0.9, zorder=5)
        for i, (population, mean, sd) in enumerate(zip(populations, means, sds)):
            ax.annotate(f"{mean:.4f}", (population, mean + sd if i == 0 else mean - sd),
                        xytext=(0, 9 if i == 0 else -15), textcoords="offset points", ha="center",
                        va="bottom" if i == 0 else "top", fontsize=10, color="#58258b")
        ax.set_xscale("log", base=4)
        ax.set_xlim(170, 6200)
        ax.set_ylim(5.865, 6.14)
        ax.xaxis.set_major_locator(FixedLocator(populations))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.set(xlabel="DUST population K (draws per update)",
               ylabel="Test loss (nats/token)", title="Figure 2  |  Test loss vs. population")
        ax.legend(handles=[dust_line, bp_line], loc="upper right", handlelength=2.8)

    footer = "1M-token replication · 999,424 training tokens · seeds 42, 43, 44"
    figures = []
    for name, draw, description in (
        ("figure1_1m", figure_one, "Lines: mean across 3 seeds. Shading: ±1 sample SD."),
        ("figure2_1m", figure_two, "Test at each seed's best validation checkpoint. Bars/band: ±1 sample SD. Small circles: seeds."),
    ):
        fig, ax = plt.subplots(figsize=(8.3, 5.7))
        fig.subplots_adjust(left=0.115, right=0.965, bottom=0.21, top=0.88)
        draw(ax)
        fig.text(0.115, 0.94, footer, color="#555e6b", fontsize=10)
        fig.text(0.115, 0.075, description, color="#555e6b", fontsize=8.6)
        if name == "figure2_1m":
            fig.text(0.115, 0.04, "The connecting line joins measured means; no power-law fit or extrapolation is shown.",
                     color="#555e6b", fontsize=8.6)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"{name}.{extension}", dpi=240)
        figures.append(fig)
    with PdfPages(output / "figures_1_and_2_1m.pdf") as pdf:
        for fig in figures:
            pdf.savefig(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.9))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.24, top=0.82, wspace=0.25)
    figure_one(axes[0])
    figure_two(axes[1])
    fig.suptitle("DUST replication · 1M training tokens", x=0.065, y=0.965, ha="left", fontsize=17, weight="semibold")
    fig.text(0.065, 0.905, "Backprop and DUST K = 256, 1,024, 4,096 · three seeds per setting", fontsize=11, color="#555e6b")
    fig.text(0.065, 0.115, "Figure 1: mean validation loss; shaded bands show ±1 sample SD.", fontsize=9, color="#555e6b")
    fig.text(0.065, 0.075, "Figure 2: test loss at each seed's best validation checkpoint; bars/band show ±1 sample SD; small circles are seeds.",
             fontsize=9, color="#555e6b")
    fig.text(0.065, 0.035, "999,424 training tokens · seeds 42, 43, 44 · measured results only; no power-law fit or extrapolation", fontsize=9, color="#555e6b")
    fig.savefig(output / "figures_1_and_2_1m.png", dpi=220)
    plt.close("all")

    write_csv(output / "figure1_validation_curves.csv", curves)
    write_csv(output / "figure2_test_summary.csv", summaries)
    write_csv(output / "figure2_per_seed.csv", per_seed)
    provenance = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        scope="Only 1M-token Backprop and DUST K=256,1024,4096; three complete seeds each.",
        paper="https://qlabs.sh/research/dust/",
        figure1="Pointwise arithmetic mean of validation losses; x=step/61 (tokens_seen/999424); no smoothing.",
        figure2="Mean test loss evaluated at each seed's minimum-validation checkpoint. Lines connect observed means.",
        variability="Sample standard deviation across seeds 42,43,44 (ddof=1); not a confidence interval or standard error.",
        adaptations="1M panel only; requested K=256,1024,4096. Seed variability added. No EGGROLL, power-law fit, or extrapolation.",
        training_tokens=999424, seeds=list(SEEDS), selected_runs=len(per_seed),
        data_sha256=shared_data, training_source_sha256=shared_source,
        plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__, matplotlib_version=matplotlib.__version__,
        summaries=summaries, inputs=input_files,
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "plot.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(output=str(output), selected_runs=len(per_seed), summaries=summaries), indent=2))


if __name__ == "__main__":
    main()
