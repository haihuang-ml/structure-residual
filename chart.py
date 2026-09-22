from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


BBC_DEPTH_SMALL = {
    r"$K=4$": {
        0: {"mean": 4.118, "standard_error": 0.005},
        2: {"mean": 4.116, "standard_error": 0.002},
        4: {"mean": 4.122, "standard_error": 0.005},
        6: {"mean": 4.131, "standard_error": 0.002},
        8: {"mean": 4.148, "standard_error": 0.004},
        10: {"mean": 4.159, "standard_error": 0.005},
    },
    r"$K=8$": {
        0: {"mean": 4.110, "standard_error": 0.008},
        2: {"mean": 4.099, "standard_error": 0.004},
        4: {"mean": 4.112, "standard_error": 0.002},
        6: {"mean": 4.128, "standard_error": 0.002},
        8: {"mean": 4.148, "standard_error": 0.002},
        10: {"mean": 4.171, "standard_error": 0.002},
    },
    r"$K=16$": {
        0: {"mean": 4.107, "standard_error": 0.002},
        2: {"mean": 4.102, "standard_error": 0.001},
        4: {"mean": 4.109, "standard_error": 0.005},
        6: {"mean": 4.128, "standard_error": 0.003},
        8: {"mean": 4.155, "standard_error": 0.005},
        10: {"mean": 4.183, "standard_error": 0.002},
    },
}


BBC_DEPTH_MEDIUM = {
    r"$K=4$": {
        0: {"mean": 4.086, "standard_error": 0.004},
        2: {"mean": 4.095, "standard_error": 0.004},
        4: {"mean": 4.085, "standard_error": 0.005},
        6: {"mean": 4.089, "standard_error": 0.003},
        8: {"mean": 4.088, "standard_error": 0.002},
        10: {"mean": 4.092, "standard_error": 0.009},
        12: {"mean": 4.094, "standard_error": 0.003},
        14: {"mean": 4.103, "standard_error": 0.003},
        16: {"mean": 4.105, "standard_error": 0.006},
        18: {"mean": 4.115, "standard_error": 0.006},
        20: {"mean": 4.112, "standard_error": 0.005},
        22: {"mean": 4.108, "standard_error": 0.003},
    },
    r"$K=8$": {
        0: {"mean": 4.088, "standard_error": 0.008},
        2: {"mean": 4.069, "standard_error": 0.007},
        4: {"mean": 4.072, "standard_error": 0.004},
        6: {"mean": 4.073, "standard_error": 0.002},
        8: {"mean": 4.082, "standard_error": 0.008},
        10: {"mean": 4.090, "standard_error": 0.005},
        12: {"mean": 4.096, "standard_error": 0.003},
        14: {"mean": 4.102, "standard_error": 0.004},
        16: {"mean": 4.113, "standard_error": 0.005},
        18: {"mean": 4.120, "standard_error": 0.003},
        20: {"mean": 4.130, "standard_error": 0.004},
        22: {"mean": 4.144, "standard_error": 0.004},
    },
    r"$K=16$": {
        0: {"mean": 4.091, "standard_error": 0.008},
        2: {"mean": 4.066, "standard_error": 0.002},
        4: {"mean": 4.067, "standard_error": 0.003},
        6: {"mean": 4.071, "standard_error": 0.001},
        8: {"mean": 4.075, "standard_error": 0.005},
        10: {"mean": 4.084, "standard_error": 0.005},
        12: {"mean": 4.090, "standard_error": 0.006},
        14: {"mean": 4.101, "standard_error": 0.004},
        16: {"mean": 4.118, "standard_error": 0.004},
        18: {"mean": 4.127, "standard_error": 0.005},
        20: {"mean": 4.144, "standard_error": 0.002},
        22: {"mean": 4.163, "standard_error": 0.001},
    },
}


CC_DEPTH_SMALL = {
    r"$\beta$=5e-3": {
        0: {"mean": 4.149, "standard_error": 0.001},
        2: {"mean": 4.137, "standard_error": 0.001},
        4: {"mean": 4.129, "standard_error": 0.004},
        6: {"mean": 4.121, "standard_error": 0.004},
        8: {"mean": 4.115, "standard_error": 0.003},
        10: {"mean": 4.113, "standard_error": 0.002},
    },
    r"$\beta$=2.5e-3": {
        0: {"mean": 4.143, "standard_error": 0.002},
        2: {"mean": 4.142, "standard_error": 0.003},
        4: {"mean": 4.130, "standard_error": 0.003},
        6: {"mean": 4.122, "standard_error": 0.003},
        8: {"mean": 4.113, "standard_error": 0.005},
        10: {"mean": 4.106, "standard_error": 0.002},
    },
    r"$\beta$=1.25e-3": {
        0: {"mean": 4.143, "standard_error": 0.001},
        2: {"mean": 4.146, "standard_error": 0.002},
        4: {"mean": 4.141, "standard_error": 0.003},
        6: {"mean": 4.122, "standard_error": 0.000},
        8: {"mean": 4.117, "standard_error": 0.001},
        10: {"mean": 4.112, "standard_error": 0.002},
    },
}


CC_DEPTH_MEDIUM = {
    r"$\beta$=5e-3": {
        0: {"mean": 4.103, "standard_error": 0.002},
        2: {"mean": 4.094, "standard_error": 0.003},
        4: {"mean": 4.093, "standard_error": 0.004},
        6: {"mean": 4.083, "standard_error": 0.002},
        8: {"mean": 4.072, "standard_error": 0.001},
        10: {"mean": 4.068, "standard_error": 0.004},
        12: {"mean": 4.058, "standard_error": 0.002},
        14: {"mean": 4.059, "standard_error": 0.002},
        16: {"mean": 4.053, "standard_error": 0.006},
        18: {"mean": 4.046, "standard_error": 0.000},
        20: {"mean": 4.049, "standard_error": 0.003},
        22: {"mean": 4.049, "standard_error": 0.000},
    },
    r"$\beta$=2.5e-3": {
        0: {"mean": 4.102, "standard_error": 0.003},
        2: {"mean": 4.103, "standard_error": 0.001},
        4: {"mean": 4.099, "standard_error": 0.002},
        6: {"mean": 4.099, "standard_error": 0.002},
        8: {"mean": 4.087, "standard_error": 0.003},
        10: {"mean": 4.071, "standard_error": 0.003},
        12: {"mean": 4.063, "standard_error": 0.002},
        14: {"mean": 4.060, "standard_error": 0.002},
        16: {"mean": 4.049, "standard_error": 0.002},
        18: {"mean": 4.042, "standard_error": 0.001},
        20: {"mean": 4.045, "standard_error": 0.001},
        22: {"mean": 4.044, "standard_error": 0.004},
    },
    r"$\beta$=1.25e-3": {
        0: {"mean": 4.104, "standard_error": 0.003},
        2: {"mean": 4.109, "standard_error": 0.001},
        4: {"mean": 4.113, "standard_error": 0.001},
        6: {"mean": 4.116, "standard_error": 0.002},
        8: {"mean": 4.104, "standard_error": 0.005},
        10: {"mean": 4.093, "standard_error": 0.008},
        12: {"mean": 4.078, "standard_error": 0.005},
        14: {"mean": 4.069, "standard_error": 0.006},
        16: {"mean": 4.063, "standard_error": 0.006},
        18: {"mean": 4.052, "standard_error": 0.006},
        20: {"mean": 4.045, "standard_error": 0.001},
        22: {"mean": 4.040, "standard_error": 0.002},
    },
}


CONFIG_NLL_SMALL = {
    "Vanilla": {"mean": 3.722, "standard_error": 0.002},
    "CC": {"mean": 3.712, "standard_error": 0.001},
    "BBC": {"mean": 3.685, "standard_error": 0.002},
    "BBC+CC": {"mean": 3.673, "standard_error": 0.004},
}


CONFIG_NLL_MEDIUM = {
    "Vanilla": {"mean": 3.615, "standard_error": 0.002},
    "CC": {"mean": 3.600, "standard_error": 0.003},
    "BBC": {"mean": 3.575, "standard_error": 0.002},
    "BBC+CC": {"mean": 3.563, "standard_error": 0.003},
}


ABL_100M_NLL = {
    "BBC": {
        "Vanilla": {"mean": 4.142, "standard_error": 0.002},
        "Sliding": {"mean": 4.143, "standard_error": 0.001},
        "Rand. sparse": {"mean": 4.150, "standard_error": 0.002},
        "BBC": {"mean": 4.099, "standard_error": 0.004},
    },
    "CC": {
        # "Vanilla": {"mean": 4.142, "standard_error": 0.002},
        "Cosine": {"mean": 4.142, "standard_error": 0.004},
        "Normalized": {"mean": 4.141, "standard_error": 0.001},
        "CC": {"mean": 4.106, "standard_error": 0.002},
    },
}


ABL_300M_NLL = {
    "BBC": {
        "Rand. BBC": {"mean": 3.702, "standard_error": 0.003},
        "BBC": {"mean": 3.684, "standard_error": 0.001},
    },
    "CC": {
        "Scale": {"mean": 3.718, "standard_error": 0.001},
        "CC-2": {"mean": 3.718, "standard_error": 0.003},
        "CC": {"mean": 3.712, "standard_error": 0.001},
    },
}


BBC_IMPROVEMENT_SMALL = {
    "BBC": {
        0.3: {"mean": 0.99, "standard_error": 0.07},
        0.6: {"mean": 1.01, "standard_error": 0.03},
        1.0: {"mean": 1.04, "standard_error": 0.06},
    },
    "BBC+CC": {
        0.3: {"mean": 1.30, "standard_error": 0.15},
        0.6: {"mean": 1.24, "standard_error": 0.13},
        1.0: {"mean": 1.36, "standard_error": 0.10},
    },
}


BBC_IMPROVEMENT_MEDIUM = {
    "BBC": {
        0.3: {"mean": 1.11, "standard_error": 0.02},
        0.6: {"mean": 1.19, "standard_error": 0.04},
        1.0: {"mean": 1.25, "standard_error": 0.04},
    },
    "BBC+CC": {
        0.3: {"mean": 1.44, "standard_error": 0.09},
        0.6: {"mean": 1.40, "standard_error": 0.03},
        1.0: {"mean": 1.51, "standard_error": 0.02},
    },
}


_IMPROVEMENT_20B_VALUES = {
    "BBC": [1.04, 0.90, 0.98, 1.09, 0.93, 1.01, 1.01, 1.00, 0.86, 0.88,
            0.95, 0.96, 1.04, 1.06, 1.06, 1.09, 1.13, 1.13, 1.13, 1.14],
    "BBC+CC": [1.06, 0.94, 1.01, 0.97, 0.80, 0.85, 0.88, 0.97, 0.86, 0.90,
               0.95, 0.98, 1.07, 1.10, 1.09, 1.13, 1.15, 1.18, 1.21, 1.20],
    "BBC+CC Adapt": [1.11, 0.99, 1.09, 1.10, 1.01, 1.08, 1.11, 1.13, 1.01, 1.05,
                      1.09, 1.08, 1.14, 1.20, 1.16, 1.23, 1.23, 1.25, 1.29, 1.28],
}
IMPROVEMENT_20B = {
    name: {
        float(tokens): {"mean": improvement, "standard_error": 0.0}
        for tokens, improvement in enumerate(improvements, start=1)
    }
    for name, improvements in _IMPROVEMENT_20B_VALUES.items()
}


SPECTRAL_RESIDUAL_100M = {
    "Vanilla": {0.05: 0.528, 0.10: 0.468, 0.20: 0.381, 0.30: 0.314, 0.50: 0.208},
    "CC": {0.05: 0.413, 0.10: 0.359, 0.20: 0.285, 0.30: 0.232, 0.50: 0.150},
}


ZERO_POSITION_CONTRAST = {
    0: {
        "Vanilla": {-1: -0.0035, -3: -0.0171, -5: -0.0145, -7: -0.2552},
        "BBC": {-1: 0.0080, -3: 0.0014, -5: -0.0407, -7: -0.2264},
    },
    4: {
        "Vanilla": {-1: 0.0007, -3: -0.0019, -5: -0.0044, -7: -0.0110},
        "BBC": {-1: 0.0066, -3: 0.0045, -5: 0.0024, -7: 0.0039},
    },
}


def plot_depth(
    path: Path,
    summary: dict[str, dict[int, dict[str, float]]],
    x_label: str,
    y_label: str,
) -> None:
    """Plot mean NLL by depth, with standard-error bars for every series."""
    plt.rcParams.update({"font.size": 30})
    plt.figure(figsize=(10, 5.6))

    for name, values in summary.items():
        points = sorted(values.items())
        depths = [depth for depth, _ in points]
        means = [metrics["mean"] for _, metrics in points]
        standard_errors = [metrics["standard_error"] for _, metrics in points]
        plt.errorbar(depths, means, yerr=standard_errors, marker="o", capsize=5, label=name)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(fontsize=26, handlelength=1.2, labelspacing=0.2, borderpad=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_improvement(
    path: Path,
    summaries: dict[str, dict[str, dict[float, dict[str, float]]]],
    x_label: str,
    y_label: str,
) -> None:
    """Plot BBC/BBC+CC improvement for small and medium models together."""
    if not summaries:
        raise ValueError("At least one model-size summary is required")
    plt.rcParams.update({"font.size": 30})
    plt.figure(figsize=(10, 5.6))

    token_budgets: list[float] | None = None
    colors = {"BBC": "C0", "BBC+CC": "C1"}
    line_styles = {"Small": "--", "Medium": "-"}
    for model_size, summary in summaries.items():
        for name, values in summary.items():
            points = sorted(values.items())
            budgets = [budget for budget, _ in points]
            means = [metrics["mean"] for _, metrics in points]
            standard_errors = [metrics["standard_error"] for _, metrics in points]
            plt.errorbar(
                budgets,
                means,
                yerr=standard_errors,
                color=colors.get(name),
                linestyle=line_styles.get(model_size, "-"),
                capsize=5,
                label="_nolegend_",
            )
            if token_budgets is None:
                token_budgets = budgets

    if token_budgets is not None:
        plt.xticks(token_budgets, [f"{budget:g}B" for budget in token_budgets])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(bottom=0.75)
    legend_handles = [
        Line2D([], [], color="C0", label="BBC"),
        Line2D([], [], color="C1", label="BBC+CC"),
        Line2D([], [], color="black", linestyle="--", label="Small"),
        Line2D([], [], color="black", linestyle="-", label="Medium"),
    ]
    plt.legend(
        handles=legend_handles,
        fontsize=26,
        ncol=4,
        loc="lower center",
        handlelength=1.0,
        handletextpad=0.25,
        columnspacing=0.55,
        labelspacing=0.0,
        borderpad=0.1,
    )
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_20b_improvement(
    path: Path,
    summary: dict[str, dict[float, dict[str, float]]],
    x_label: str | None,
    y_label: str,
) -> None:
    """Plot 1B--20B NLL improvement curves without reported uncertainty."""
    if not summary:
        raise ValueError("At least one improvement series is required")

    plt.rcParams.update({"font.size": 30})
    plt.figure(figsize=(10, 5.6))
    colors = {"BBC": "C0", "BBC+CC": "C1", "BBC+CC Adapt": "C2"}
    line_styles = {"BBC": "-", "BBC+CC": "--", "BBC+CC Adapt": "-"}
    line_widths = {"BBC": 1.5, "BBC+CC": 1.5, "BBC+CC Adapt": 3.0}
    token_budgets: list[float] | None = None
    for name, values in summary.items():
        points = sorted(values.items())
        budgets = [budget for budget, _ in points]
        means = [metrics["mean"] for _, metrics in points]
        plt.plot(
            budgets,
            means,
            color=colors.get(name),
            linestyle=line_styles.get(name, "-"),
            linewidth=line_widths.get(name, 1.5),
            label=name,
        )
        if token_budgets is None:
            token_budgets = budgets

    if token_budgets is not None:
        tick_budgets = [budget for budget in token_budgets if budget in {1, 5, 10, 15, 20}]
        plt.xticks(tick_budgets, [f"{budget:g}" for budget in tick_budgets])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(
        fontsize=24,
        ncol=1,
        loc="lower right",
        handlelength=1.0,
        labelspacing=0.0,
        borderpad=0.1,
    )
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_zero_position(
    path: Path,
    summaries: dict[int, dict[str, dict[int, float]]],
    x_label: str | None = "Position relative to boundary",
    y_label: str = r"$\Delta_b - \Delta_i$",
) -> None:
    """Plot boundary-versus-interior ablation contrast by interior position."""
    if not summaries:
        raise ValueError("At least one warm-up summary is required")

    plt.rcParams.update({"font.size": 30})
    plt.figure(figsize=(10, 5.6))
    colors = {"Vanilla": "C0", "BBC": "C1"}
    line_styles = {0: "--", 4: "-"}
    markers = {0: "x", 4: "o"}
    for warmup_segments, summary in summaries.items():
        for model, values in summary.items():
            points = sorted(values.items())
            positions = [position for position, _ in points]
            contrasts = [contrast for _, contrast in points]
            plt.plot(
                positions,
                contrasts,
                color=colors.get(model),
                linestyle=line_styles.get(warmup_segments, "-"),
                marker=markers.get(warmup_segments, "o"),
                label=f"{model}, warmup={warmup_segments}",
            )

    positions = sorted({position for summary in summaries.values() for values in summary.values() for position in values})
    plt.xticks(positions)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(fontsize=24, ncol=1, loc="lower right", handlelength=1.0, labelspacing=0.0, borderpad=0.1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_spectral_residual(
    path: Path,
    summary: dict[str, dict[float, float]],
    x_label: str = r"$m / N$",
    y_label: str = r"$\Vert H - H_m \Vert_F^2 / \Vert H \Vert_F^2$",
) -> None:
    """Plot normalized spectral residuals by retained positional-mode fraction."""
    if not summary:
        raise ValueError("At least one spectral-residual series is required")

    plt.rcParams.update({"font.size": 30})
    plt.figure(figsize=(10, 5.6))
    for name, values in summary.items():
        points = sorted(values.items())
        fractions = [fraction for fraction, _ in points]
        residuals = [residual for _, residual in points]
        plt.plot(fractions, residuals, marker="o", label=name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(fontsize=26, handlelength=1.2, labelspacing=0.2, borderpad=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_300m_nll(
    path: Path,
    summaries: dict[str, dict[str, dict[str, float]]],
    x_label: str,
    y_label: str,
    y_min: float = 2.0,
) -> None:
    """Plot configurations within GPT-2 small/medium groups, with error bars."""
    if not summaries:
        raise ValueError("At least one model-size summary is required")
    configurations = list(next(iter(summaries.values())))
    if any(list(summary) != configurations for summary in summaries.values()):
        raise ValueError("All model-size summaries must use the same configuration order")

    plt.rcParams.update({"font.size": 30})
    figure, axis = plt.subplots(figsize=(10, 5.6))
    model_sizes = list(summaries)
    positions = np.arange(len(model_sizes))
    bar_width = 0.4 / len(configurations)
    offset = -(len(configurations) - 1) * bar_width / 2

    for index, configuration in enumerate(configurations):
        means = [summaries[model_size][configuration]["mean"] for model_size in model_sizes]
        standard_errors = [summaries[model_size][configuration]["standard_error"] for model_size in model_sizes]
        axis.bar(
            positions + offset + index * bar_width,
            means,
            bar_width,
            yerr=standard_errors,
            capsize=5,
            label=configuration,
        )

    axis.set_xticks(positions, model_sizes)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_ylim(bottom=y_min)
    axis.legend(fontsize=26, ncol=2, handlelength=1.2, labelspacing=0.2, borderpad=0.3)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_abl_100m(
    path: Path,
    summaries: dict[str, dict[str, dict[str, float]]],
    x_label: str | None,
    y_label: str,
    y_min: float = 4.0,
    y_max: float | None = None,
    legend_columns: list[list[str]] | None = None,
) -> None:
    """Plot BBC and CC ablations with standard-error bars."""
    if not summaries:
        raise ValueError("At least one ablation summary is required")

    plt.rcParams.update({"font.size": 30})
    figure, axis = plt.subplots(figsize=(10, 5.6))
    ablations = list(summaries)
    configurations = [list(summary) for summary in summaries.values()]
    positions = np.arange(len(ablations))
    max_configuration_count = max(map(len, configurations))
    bar_width = 0.4 / max_configuration_count
    color_offset = 0
    for ablation_index, (ablation, summary) in enumerate(summaries.items()):
        offset = -(len(summary) - 1) * bar_width / 2
        for configuration_index, (configuration, metrics) in enumerate(summary.items()):
            axis.bar(
                positions[ablation_index] + offset + configuration_index * bar_width,
                metrics["mean"],
                bar_width,
                yerr=metrics["standard_error"],
                capsize=5,
                color=f"C{color_offset + configuration_index}",
                label=configuration,
            )
        color_offset += len(summary)

    axis.set_xticks(positions, ablations)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_ylim(bottom=y_min, top=y_max)
    handles, labels = axis.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    if legend_columns is not None:
        max_rows = max(map(len, legend_columns))
        legend_handles = []
        legend_labels = []
        for column in legend_columns:
            for label in column:
                legend_handles.append(unique_labels[label])
                legend_labels.append(label)
            for _ in range(max_rows - len(column)):
                legend_handles.append(Line2D([], [], color="none"))
                legend_labels.append("")
    else:
        legend_handles = unique_labels.values()
        legend_labels = unique_labels.keys()
    axis.legend(
        legend_handles,
        legend_labels,
        fontsize=26,
        ncol=2,
        handlelength=1.2,
        labelspacing=0.2,
        borderpad=0.3,
    )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_abl_300m(
    path: Path,
    summaries: dict[str, dict[str, dict[str, float]]],
    x_label: str | None,
    y_label: str,
    y_min: float = 3.65,
    y_max: float | None = None,
) -> None:
    """Plot 300M BBC and CC ablations with standard-error bars."""
    plot_abl_100m(
        path,
        summaries,
        x_label,
        y_label,
        y_min,
        y_max,
        legend_columns=[list(summary) for summary in summaries.values()],
    )


if __name__ == "__main__":
    # plot_depth("images/bbc_depth_small.pdf", BBC_DEPTH_SMALL, "Depth", "NLL")
    # plot_depth("images/bbc_depth_medium.pdf", BBC_DEPTH_MEDIUM, "Depth", "NLL")
    # plot_depth("images/cc_depth_small.pdf", CC_DEPTH_SMALL, "Depth", "NLL")
    # plot_depth("images/cc_depth_medium.pdf", CC_DEPTH_MEDIUM, "Depth", "NLL")
    # plot_improvement(
    #     "images/bbc_cc_improvement.pdf",
    #     {"Small": BBC_IMPROVEMENT_SMALL, "Medium": BBC_IMPROVEMENT_MEDIUM},
    #     None,
    #     "Improvement (%)",
    # )
    # plot_300m_nll(
    #     "images/300m_nll.pdf",
    #     {"GPT-2 small": CONFIG_NLL_SMALL, "GPT-2 medium": CONFIG_NLL_MEDIUM},
    #     None,
    #     "NLL",
    #     y_min=3.5
    # )
    # plot_abl_100m("images/abl_100m.pdf", ABL_100M_NLL, None, "NLL", y_min=4.0)
    # plot_abl_300m("images/abl_300m.pdf", ABL_300M_NLL, None, "NLL", y_min=3.65, y_max=3.8)
    # plot_20b_improvement("images/20b_improvement.pdf", IMPROVEMENT_20B, None, "Improvement (%)")
    plot_zero_position("images/zero_position.pdf", ZERO_POSITION_CONTRAST)
    # plot_spectral_residual("images/cc_spectral_residual.pdf", SPECTRAL_RESIDUAL_100M)
