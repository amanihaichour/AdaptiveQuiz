"""plots.py

Plotting functions for evaluation analysis.
All visualization code is centralized here for easy maintenance and customization.
"""

import os
from typing import Any, Dict

import pandas as pd


def plot_accuracy_vs_questions(curve_df: pd.DataFrame, output_path: str) -> None:
    """Plot accuracy vs number of questions asked (main result)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to generate plots.") from exc

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for strategy in curve_df["strategy"].unique():
        sub = curve_df[curve_df["strategy"] == strategy]
        plt.plot(sub["question_number"], sub["accuracy_pct"], marker="o", label=strategy)

    plt.xlabel("Questions Asked")
    plt.ylabel("Level-Estimate Accuracy (%)")
    plt.title("Mock Step-1: Accuracy vs Questions")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(conf_df: pd.DataFrame, output_path: str, levels: tuple = ("weak", "medium", "strong"), strategies: list = None) -> None:
    """Plot confusion matrices (rows=true level, cols=predicted level) for each strategy."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("matplotlib and seaborn required.") from exc

    if strategies is None:
        strategies = ["uncertainty", "entropy", "qbc", "random"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Confusion Matrices by Strategy", fontsize=14)
    axes = axes.flatten()

    for idx, strategy in enumerate(strategies):
        ax = axes[idx]
        strat_df = conf_df[conf_df["strategy"] == strategy]
        cm = pd.crosstab(strat_df["true_level"], strat_df["predicted_level"])
        cm = cm.reindex(levels, axis=0, fill_value=0).reindex(levels, axis=1, fill_value=0)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"Strategy: {strategy}")
        ax.set_xlabel("Predicted Level")
        ax.set_ylabel("True Level")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_precision_recall(pr_df: pd.DataFrame, output_path: str, levels: tuple = ("weak", "medium", "strong")) -> None:
    """Plot precision and recall per level and strategy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required.") from exc

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric_idx, metric in enumerate(["precision", "recall"]):
        ax = axes[metric_idx]
        for level in levels:
            sub = pr_df[pr_df["level"] == level]
            ax.plot(sub["strategy"], sub[metric], marker="o", label=level)
        ax.set_xlabel("Strategy")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} per Level")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confidence_over_time(conf_df: pd.DataFrame, output_path: str) -> None:
    """Plot average system confidence at each question step."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required.") from exc

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(conf_df["question_number"], conf_df["avg_confidence"], marker="o", color="green", linewidth=2)
    plt.xlabel("Question Number")
    plt.ylabel("Average Confidence")
    plt.title("System Confidence Over Time")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_robustness(robust_df: pd.DataFrame, output_path: str) -> None:
    """Plot accuracy drop vs noise level (random answer flipping)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required.") from exc

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(robust_df["noise_level_pct"], robust_df["avg_accuracy"], marker="o", linewidth=2, label="avg_accuracy")
    plt.fill_between(
        robust_df["noise_level_pct"],
        robust_df["avg_accuracy"] - robust_df["std_accuracy"],
        robust_df["avg_accuracy"] + robust_df["std_accuracy"],
        alpha=0.2,
    )
    plt.xlabel("Noise Level (%)")
    plt.ylabel("Accuracy")
    plt.title("Robustness: Accuracy vs Response Noise")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_sensitivity(sens_results: Dict[str, pd.DataFrame], output_path: str) -> None:
    """Plot accuracy curves under different difficulty distributions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required.") from exc

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sensitivity Analysis: Difficulty Distribution", fontsize=14)
    axes = axes.flatten()

    for idx, (variant_name, curve_df) in enumerate(sens_results.items()):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break
        for strategy in curve_df["strategy"].unique():
            sub = curve_df[curve_df["strategy"] == strategy]
            ax.plot(sub["question_number"], sub["accuracy_pct"], marker="o", label=strategy)
        ax.set_xlabel("Questions Asked")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Variant: {variant_name}")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
