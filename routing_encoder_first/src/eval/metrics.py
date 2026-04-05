from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_records_and_artifacts(
    records: List[Dict[str, object]],
    train_distribution: str,
    train_problem_size: int,
    output_dir: str | Path,
    examples: List[Dict[str, object]],
) -> None:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figs_dir = output_dir / "figs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    if df.empty:
        return

    df.to_csv(tables_dir / "all_eval_records.csv", index=False)

    main_results = df[(df["distribution"] == train_distribution) & (df["problem_size"] == train_problem_size)]
    main_results.to_csv(tables_dir / "main_results.csv", index=False)

    efficiency_profile = df[
        [
            "distribution",
            "problem_size",
            "method",
            "candidate_count",
            "avg_cost",
            "wall_time_sec",
            "instances_per_sec",
            "candidates_per_sec",
            "peak_memory_mb",
        ]
    ]
    efficiency_profile.to_csv(tables_dir / "efficiency_profile.csv", index=False)

    size_generalization = df[df["distribution"] == train_distribution]
    size_generalization.to_csv(tables_dir / "size_generalization.csv", index=False)

    distribution_shift = df[df["problem_size"] == train_problem_size]
    distribution_shift.to_csv(tables_dir / "distribution_shift.csv", index=False)

    candidate_diversity = df[
        [
            "distribution",
            "problem_size",
            "method",
            "avg_edge_disagreement",
            "avg_best_minus_greedy",
        ]
    ]
    candidate_diversity.to_csv(tables_dir / "candidate_diversity.csv", index=False)

    ablation_projector = df[df["method"].str.contains("edge_field|order_sort", regex=True)]
    ablation_projector.to_csv(tables_dir / "ablation_projector.csv", index=False)

    ablation_loss = pd.DataFrame(
        [
            {"objective": "instance_relative_listwise_kl", "status": "implemented"},
            {"objective": "raw_cost_regression", "status": "placeholder_only"},
            {"objective": "pairwise_ranking", "status": "placeholder_only"},
        ]
    )
    ablation_loss.to_csv(tables_dir / "ablation_loss.csv", index=False)

    _plot_quality_vs_time(main_results, figs_dir / "quality_vs_time.png")
    _plot_peak_memory(main_results, figs_dir / "peak_memory_comparison.png")
    _plot_size_generalization(size_generalization, figs_dir / "size_generalization_curve.png")
    _plot_distribution_shift(distribution_shift, figs_dir / "distribution_shift_curve.png")
    _plot_diversity_vs_quality(candidate_diversity, figs_dir / "candidate_diversity_vs_quality.png")
    _plot_giant_tour_examples(examples, figs_dir / "giant_tour_examples.png")
    _plot_split_examples(examples, figs_dir / "split_examples.png")


def save_learning_curves(
    train_history: List[Dict[str, object]],
    validation_history: List[Dict[str, object]],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    figs_dir = output_dir / "figs"
    tables_dir = output_dir / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(train_history)
    val_df = pd.DataFrame(validation_history)

    if not train_df.empty:
        train_df.to_csv(tables_dir / "training_history.csv", index=False)
        _plot_train_loss(train_df, figs_dir / "train_loss_curve.png")
        _plot_train_costs(train_df, figs_dir / "train_cost_curve.png")
        _plot_train_speed(train_df, figs_dir / "train_step_time_curve.png")

    if not val_df.empty:
        val_df.to_csv(tables_dir / "validation_history.csv", index=False)
        _plot_validation_costs(val_df, figs_dir / "validation_cost_curve.png")
        _plot_validation_feasibility(val_df, figs_dir / "validation_feasibility_curve.png")
        _plot_validation_runtime(val_df, figs_dir / "validation_runtime_curve.png")


def _plot_quality_vs_time(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for _, row in df.iterrows():
        if not np.isfinite(row["avg_cost"]):
            continue
        x = row["wall_time_sec"] / max(row.get("candidate_count", 1), 1)
        plt.scatter(x, row["avg_cost"], label=row["method"])
        plt.text(x, row["avg_cost"], row["method"], fontsize=8)
    plt.xlabel("Wall Time Per Candidate (s)")
    plt.ylabel("Average Cost")
    plt.title("Quality vs Time")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_train_loss(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df["step"], df["loss"], label="total_loss")
    if "listwise_loss" in df:
        plt.plot(df["step"], df["listwise_loss"], label="listwise_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_train_costs(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df["step"], df["greedy_cost"], label="greedy_cost")
    plt.plot(df["step"], df["best_cost"], label="best_cost")
    plt.xlabel("Step")
    plt.ylabel("Cost")
    plt.title("Training Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_train_speed(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df["step"], df["elapsed_sec"], label="step_time_sec")
    if "peak_memory_mb" in df:
        plt.plot(df["step"], df["peak_memory_mb"], label="peak_memory_mb")
    plt.xlabel("Step")
    plt.title("Training Runtime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_validation_costs(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    for (scope, method), group in df.groupby(["scope", "method"]):
        ordered = group.sort_values("step")
        plt.plot(ordered["step"], ordered["avg_cost"], marker="o", label=f"{scope}:{method}")
    plt.xlabel("Step")
    plt.ylabel("Average Cost")
    plt.title("Validation Cost")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_validation_feasibility(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    for (scope, method), group in df.groupby(["scope", "method"]):
        ordered = group.sort_values("step")
        plt.plot(ordered["step"], ordered["feasibility_rate"], marker="o", label=f"{scope}:{method}")
    plt.xlabel("Step")
    plt.ylabel("Feasibility Rate")
    plt.title("Validation Feasibility")
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_validation_runtime(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    for (scope, method), group in df.groupby(["scope", "method"]):
        ordered = group.sort_values("step")
        plt.plot(ordered["step"], ordered["wall_time_sec"], marker="o", label=f"{scope}:{method}")
    plt.xlabel("Step")
    plt.ylabel("Wall Time (s)")
    plt.title("Validation Runtime")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_peak_memory(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    finite = df[np.isfinite(df["peak_memory_mb"])]
    plt.bar(finite["method"], finite["peak_memory_mb"])
    plt.ylabel("Peak GPU Memory (MB)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_size_generalization(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for method, group in df.groupby("method"):
        ordered = group.sort_values("problem_size")
        ordered = ordered[np.isfinite(ordered["avg_cost"])]
        if ordered.empty:
            continue
        plt.plot(ordered["problem_size"], ordered["avg_cost"], marker="o", label=method)
    plt.xlabel("Problem Size")
    plt.ylabel("Average Cost")
    plt.title("Size Generalization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_distribution_shift(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    distributions = list(dict.fromkeys(df["distribution"].tolist()))
    x_positions = range(len(distributions))
    width = 0.8 / max(len(df["method"].unique()), 1)
    for offset, (method, group) in enumerate(df.groupby("method")):
        y = []
        for distribution in distributions:
            subset = group[group["distribution"] == distribution]
            y.append(float(subset["avg_cost"].iloc[0]) if not subset.empty else float("nan"))
        shifted = [x + offset * width for x in x_positions]
        plt.bar(shifted, y, width=width, label=method)
    plt.xticks([x + width for x in x_positions], distributions)
    plt.ylabel("Average Cost")
    plt.title("Distribution Shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_diversity_vs_quality(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    for _, row in df.iterrows():
        if not np.isfinite(row["avg_best_minus_greedy"]):
            continue
        plt.scatter(row["avg_edge_disagreement"], row["avg_best_minus_greedy"], label=row["method"])
    plt.xlabel("Average Edge Disagreement")
    plt.ylabel("Best - Greedy Improvement")
    plt.title("Candidate Diversity vs Quality")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_giant_tour_examples(examples: List[Dict[str, object]], path: Path) -> None:
    if not examples:
        return
    num_examples = min(len(examples), 4)
    figure, axes = plt.subplots(1, num_examples, figsize=(4 * num_examples, 4))
    axes = np.atleast_1d(axes)
    for axis, example in zip(axes, examples[:4]):
        coords = example["coords"]
        depot = example["depot"]
        order = example["order"]
        axis.scatter(coords[:, 0], coords[:, 1], s=12, c="tab:blue")
        axis.scatter(depot[0], depot[1], s=80, c="tab:red", marker="*")
        ordered_xy = coords[order]
        axis.plot(ordered_xy[:, 0], ordered_xy[:, 1], c="tab:orange", linewidth=1.2)
        axis.plot([ordered_xy[-1, 0], ordered_xy[0, 0]], [ordered_xy[-1, 1], ordered_xy[0, 1]], c="tab:orange", linewidth=1.2)
        axis.set_title(example["title"])
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_split_examples(examples: List[Dict[str, object]], path: Path) -> None:
    if not examples:
        return
    num_examples = min(len(examples), 4)
    figure, axes = plt.subplots(1, num_examples, figsize=(4 * num_examples, 4))
    axes = np.atleast_1d(axes)
    palette = plt.cm.get_cmap("tab20")
    for axis, example in zip(axes, examples[:4]):
        coords = example["coords"]
        depot = example["depot"]
        routes = example["routes"]
        axis.scatter(coords[:, 0], coords[:, 1], s=12, c="tab:blue")
        axis.scatter(depot[0], depot[1], s=80, c="tab:red", marker="*")
        for route_idx, route in enumerate(routes):
            route_xy = coords[route]
            color = palette(route_idx % 20)
            axis.plot([depot[0], route_xy[0, 0]], [depot[1], route_xy[0, 1]], c=color, linewidth=1.2)
            axis.plot(route_xy[:, 0], route_xy[:, 1], c=color, linewidth=1.2)
            axis.plot([route_xy[-1, 0], depot[0]], [route_xy[-1, 1], depot[1]], c=color, linewidth=1.2)
        axis.set_title(example["title"])
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
