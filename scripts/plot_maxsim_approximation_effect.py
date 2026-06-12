#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def load_rows(root: Path, mode: str, dataset: str) -> list[dict[str, str]]:
    path = root / mode / f"{dataset}_exrabitq_compressed_ids_hnsw-quantized-complete_k.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_high_dim_rows(root: Path, dataset: str) -> list[dict[str, str]]:
    path = root / dataset / f"{dataset}_exrabitq_compressed_ids_hnsw-quantized-complete_k.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(rows: list[dict[str, str]], tk: int, k: int) -> dict[str, str]:
    for row in rows:
        if int(row["tk"]) == tk and int(row["k"]) == k:
            return row
    raise RuntimeError(f"Missing row for tk={tk}, k={k}")


def pct_change(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0
    return (new_value / old_value - 1.0) * 100.0


def plot_for_k(
    phase_root: Path,
    no_algorithm_root: Path,
    algorithm8_root: Path,
    output_dir: Path,
    k_value: int,
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 0.8,
            "hatch.linewidth": 0.7,
        }
    )

    datasets = [
        ("clerc-med-multi", "(a) CLERC-768"),
        ("clerc-large-multi", "(b) CLERC-1024"),
    ]
    tk_values = [50, 100, 200]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(8.4, 3.75),
        sharey=True,
        constrained_layout=False,
    )
    colors = {
        "without": "#f7d77b",
        "with": "#e1a93a",
        "graph": "#8b80d8",
    }
    hatch_without = "..."
    bar_width = 0.145
    group_spacing = 0.42
    x = np.arange(len(tk_values)) * group_spacing

    for subplot_idx, (ax, (dataset, dataset_title)) in enumerate(zip(axes, datasets)):
        phase_rows = load_rows(phase_root, "without", dataset)
        no_algorithm_rows = load_high_dim_rows(no_algorithm_root, dataset)
        algorithm8_rows = load_high_dim_rows(algorithm8_root, dataset)

        index_search_time = []
        rerank_without = []
        rerank_with = []
        for tk in tk_values:
            phase_row = find_row(phase_rows, tk, k_value)
            row_without = find_row(no_algorithm_rows, tk, k_value)
            row_with = find_row(algorithm8_rows, tk, k_value)
            graph = float(phase_row["GraphTraversalSeconds"])
            qps_no_algo = float(row_without["QPS"])
            qps_algo8 = float(row_with["QPS"])
            total_without = 1000.0 / qps_no_algo
            total_with = 1000.0 / qps_algo8
            rerank_without_time = max(0.0, total_without - graph)
            rerank_with_time = max(0.0, total_with - graph)

            index_search_time.append(graph)
            rerank_without.append(rerank_without_time)
            rerank_with.append(rerank_with_time)

        ax.bar(
            x - 0.46 * bar_width,
            index_search_time,
            bar_width,
            label="Index-based search",
            color=colors["graph"],
            edgecolor="black",
            hatch=hatch_without,
            linewidth=0.65,
        )
        ax.bar(
            x + 0.46 * bar_width,
            index_search_time,
            bar_width,
            color=colors["graph"],
            edgecolor="black",
            linewidth=0.65,
        )
        ax.bar(
            x - 0.46 * bar_width,
            rerank_without,
            bar_width,
            label="Reranking w/o Algorithm 8",
            bottom=index_search_time,
            color=colors["without"],
            edgecolor="black",
            hatch=hatch_without,
            linewidth=0.65,
        )
        ax.bar(
            x + 0.46 * bar_width,
            rerank_with,
            bar_width,
            label="Reranking with Algorithm 8",
            bottom=index_search_time,
            color=colors["with"],
            edgecolor="black",
            linewidth=0.65,
        )

        for idx, tk in enumerate(tk_values):
            total_without = index_search_time[idx] + rerank_without[idx]
            total_with = index_search_time[idx] + rerank_with[idx]
            runtime_reduction = (1.0 - total_with / total_without) * 100.0 if total_without > 0.0 else 0.0
            y_high = max(total_without, total_with)
            if y_high <= 0:
                continue
            x_old = x[idx] - 0.46 * bar_width
            x_new = x[idx] + 0.46 * bar_width
            bracket_scale = 1.13 if tk == 200 else 1.075
            text_scale = 1.035 if tk == 200 else 1.055
            bracket_y = y_high * bracket_scale
            ax.plot(
                [x_old, x_old, x_new],
                [total_without, bracket_y, bracket_y],
                color="#b42318",
                lw=1.45,
                solid_capstyle="butt",
            )
            ax.annotate(
                "",
                xy=(x_new, total_with),
                xytext=(x_new, bracket_y),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#b42318",
                    lw=1.45,
                    shrinkA=0,
                    shrinkB=0,
                ),
            )
            ax.text(
                (x_old + x_new) / 2.0,
                bracket_y * text_scale,
                f"-{runtime_reduction:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#b42318",
                fontweight="bold",
            )

        if subplot_idx == 0:
            ax.set_ylabel("Runtime (s)", fontsize=12.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.22, linewidth=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=10.5, width=0.8)
        ax.margins(y=0.16)
        ax.set_xlim(x[0] - 0.32, x[-1] + 0.32)
        ax.text(
            0.5,
            -0.18,
            dataset_title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12.0,
            fontweight="bold",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([str(tk) for tk in tk_values], fontsize=11.0, fontweight="bold")
        ax.set_xlabel(r"$\tau$", fontsize=13.0, fontweight="bold", labelpad=1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        prop={"weight": "bold", "size": 11.0},
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.0,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.82, wspace=0.09)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"clerc_maxsim_approximation_qps_breakdown_k{k_value}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=Path("/home/ali/hnsw-skipping-construction-quantizer/results/Aprroximation effect"),
    )
    parser.add_argument(
        "--no-algorithm-root",
        type=Path,
        default=Path("/home/ali/hnsw-skipping-construction-quantizer/MultiHNSW/high-dim"),
    )
    parser.add_argument(
        "--algorithm8-root",
        type=Path,
        default=Path("/home/ali/hnsw-skipping-construction-quantizer/MultiHNSW + Algorithm 8/high-dim"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/ali/hnsw-skipping-construction-quantizer/results/Aprroximation effect/plots"),
    )
    args = parser.parse_args()

    for k_value in [10, 100]:
        output_path = plot_for_k(
            args.phase_root,
            args.no_algorithm_root,
            args.algorithm8_root,
            args.output_dir,
            k_value,
        )
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
