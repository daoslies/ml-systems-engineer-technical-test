"""
plot_results.py — Signapse inference profiler visualiser
=========================================================
Reads a benchmark CSV produced by profiler.py and generates a
two-figure output:

  Figure 1 — Bar charts: p50 + p99 latency, CPU and GPU split into
             separate panels so they share a meaningful y-axis scale.

  Figure 2 — Summary table with per-column heatmap colouring on all
             numeric metric columns so the best/worst conditions are
             immediately visible.

Usage:
    python plot_results.py
    python plot_results.py --csv my_results.csv
    python plot_results.py --csv results.csv --out chart.png
             (table is saved alongside as <stem>_table.png)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from profiler.py CSV output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, default="benchmark_results.csv")
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading + grouping
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[plot] Error: CSV not found: {path}", file=sys.stderr)
        sys.exit(1)
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "label":      row["label"],
                "device":     row["device"],
                "batch_size": int(row["batch_size"]),
                "p50":        float(row["p50"]),
                "p95":        float(row.get("p95", row.get("p99", 0))),
                "p99":        float(row["p99"]),
                "mean":       float(row["mean"]),
                "std":        float(row["std"]),
                "outliers":   int(row["outliers"]),
                "runs":       int(row["runs"]),
                "cpu_util":   float(row.get("cpu_util", 0)),
                "gpu_util":   float(row.get("gpu_util", 0)),
            })
    if not rows:
        print("[plot] Error: CSV is empty.", file=sys.stderr)
        sys.exit(1)
    return rows


def group_rows(rows: list[dict]) -> dict:
    """
    Returns nested dict: grouped[label][config] -> averaged stats.
    config = "{device} bs{batch_size}"  e.g. "cuda bs1"
    Stats are averaged across repeated runs of the same configuration.
    """
    buckets: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        config = f"{row['device']} bs{row['batch_size']}"
        for key in ("p50", "p95", "p99", "mean", "std", "outliers",
                    "cpu_util", "gpu_util"):
            buckets[(row["label"], config)][key].append(row[key])

    grouped: dict = defaultdict(dict)
    for (label, config), vals in buckets.items():
        grouped[label][config] = {k: sum(v) / len(v) for k, v in vals.items()}
    return grouped


def config_sort_key(c: str) -> tuple:
    dev, bs = c.split(" bs")
    return ({"cpu": 0, "cuda": 1}.get(dev, 99), int(bs))


# ---------------------------------------------------------------------------
# Palette + style constants
# ---------------------------------------------------------------------------

PYTORCH_COLOUR = "#2563EB"
ONNX_COLOUR    = "#EA580C"

LABEL_COLOURS = {
    "PyTorch": PYTORCH_COLOUR,
    "ONNX":    ONNX_COLOUR,
}



def apply_base_style(fig, axes_flat):
    fig.patch.set_facecolor("#FFFFFF")
    for ax in axes_flat:
        ax.set_facecolor("#FAFAFA")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#E5E7EB")
        ax.tick_params(axis="both", which="both", length=0,
                       labelsize=9, colors="#6B7280")
        ax.yaxis.grid(True, color="#E5E7EB", linewidth=0.6, linestyle="--")
        ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

def annotate_callout(ax, x, y, text, xytext,
                     color="#1F2937", arrow_color="#9CA3AF"):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        fontsize=8,
        color=color,
        ha="center",
        va="bottom",
        arrowprops=dict(
            arrowstyle="-",
            color=arrow_color,
            lw=1.0,
            connectionstyle="arc3,rad=0.0",
        ),
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="#E5E7EB",
            linewidth=0.8,
        ),
    )


# ---------------------------------------------------------------------------
# Figure 1 — Bar charts (CPU and GPU in separate panels)
# ---------------------------------------------------------------------------

def draw_bar_panel(ax, grouped, labels, configs, stat_key, alpha=0.85):
    """Draw a grouped-bar panel for *configs* (all same device)."""
    import numpy as np

    n_configs = len(configs)
    n_labels  = len(labels)
    bar_w     = 0.28
    gap       = 0.18
    group_w   = n_labels * bar_w + gap
    xs        = np.arange(n_configs) * group_w

    bar_positions = {}   # (label, config) -> x centre

    for li, label in enumerate(labels):
        colour    = LABEL_COLOURS.get(label, "#6B7280")
        x_offsets = xs + li * bar_w - (n_labels - 1) * bar_w / 2

        for ci, config in enumerate(configs):
            if config not in grouped[label]:
                continue
            stats = grouped[label][config]
            val   = stats[stat_key]
            std   = stats["std"]
            x     = x_offsets[ci]
            bar_positions[(label, config)] = x

            ax.bar(x, val, width=bar_w * 0.84,
                   color=colour, alpha=alpha,
                   zorder=3, linewidth=0)

            ax.errorbar(x, val, yerr=std,
                        fmt="none", ecolor="#374151",
                        elinewidth=1.1, capsize=3.5, capthick=1.1,
                        zorder=4)

            ax.text(x, val + std + ax.get_ylim()[1] * 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color="#374151", fontweight="500")


    ax.set_xticks(xs)
    # Always show device and batch size as 'cpu bs=1', 'cuda bs=16', etc.
    def format_config_label(config):
        try:
            dev, bs = config.split(" bs")
            return f"{dev} bs={bs}"
        except Exception:
            return config
    ax.set_xticklabels(
        [format_config_label(c) for c in configs],
        fontsize=10, color="#374151",
    )
    return bar_positions, xs, group_w


def plot_bars(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("[plot] matplotlib not installed: pip install matplotlib",
              file=sys.stderr)
        sys.exit(1)

    grouped = group_rows(rows)
    labels  = sorted(grouped.keys(),
                     key=lambda l: (0 if "PyTorch" in l else 1))
    all_configs = sorted(
        {c for ldata in grouped.values() for c in ldata},
        key=config_sort_key,
    )
    cpu_configs  = [c for c in all_configs if c.startswith("cpu")]
    cuda_configs = [c for c in all_configs if c.startswith("cuda")]

    # Grid: 2 rows (p50 top, p99 bottom) × 2 cols (CPU left, CUDA right)
    # CPU panel is narrower if fewer configs
    cpu_w  = len(cpu_configs)
    cuda_w = len(cuda_configs)
    width_ratios = [cpu_w, cuda_w] if cpu_w and cuda_w else [1, 1]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
        gridspec_kw={
            "hspace": 0.52,
            "wspace": 0.28,
            "width_ratios": width_ratios,
            "left": 0.07, "right": 0.97,
            "top": 0.83, "bottom": 0.09,
        },
    )
    apply_base_style(fig, axes.flat)

    panel_specs = [
        (axes[0, 0], cpu_configs,  "p50", 0.85, "CPU",      "#9CA3AF"),
        (axes[0, 1], cuda_configs, "p50", 0.85, "GPU (CUDA)", "#9CA3AF"),
        (axes[1, 0], cpu_configs,  "p99", 0.70, "CPU",      "#9CA3AF"),
        (axes[1, 1], cuda_configs, "p99", 0.70, "GPU (CUDA)", "#9CA3AF"),
    ]

    all_bar_pos = {}  # keyed by (row, col)
    for idx, (ax, configs, stat, alpha, device_label, _) in enumerate(panel_specs):
        if not configs:
            ax.set_visible(False)
            continue
        draw_bar_panel(ax, grouped, labels, configs, stat, alpha)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.40)
        # Re-draw with correct ylim so value labels sit properly
        ax.cla()
        apply_base_style(fig, [ax])
        positions, xs, group_w = draw_bar_panel(ax, grouped, labels, configs, stat, alpha)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.40)
        all_bar_pos[(idx // 2, idx % 2)] = (positions, xs, group_w)

        ax.set_ylabel("latency (ms)", fontsize=9, color="#6B7280", labelpad=8)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

        ymax = ax.get_ylim()[1]
        # Single, moderately bold and sized subplot title
        ax.text(0.5, 1.06, device_label,
            transform=ax.transAxes,
            ha="center", fontsize=12, fontweight="semibold",
            color="#374151")

    # ── Row labels (left edge) ───────────────────────────────────────────────
    for row_i, row_label in enumerate(["p50 latency — median performance",
                                        "p99 latency — tail behaviour"]):
        axes[row_i, 0].text(
            -0.14, 1.14, row_label,
            transform=axes[row_i, 0].transAxes,
            fontsize=11 if row_i == 0 else 9,
            fontweight="bold",
            color="#111827" if row_i == 0 else "#6B7280",
        )


    # ── Legend ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=LABEL_COLOURS.get(l, "#6B7280"), label=l, alpha=0.85)
        for l in labels
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper center",
        ncol=len(labels),
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.97),
        handlelength=1.2,
        handleheight=0.9,
    )

    # ── Title + footnote ─────────────────────────────────────────────────────
    fig.text(
        0.5, 0.995,
        "Signapse inference profiler — ResNet18 benchmark",
        ha="center", va="top",
        fontsize=14, fontweight="bold", color="#111827",
    )
    fig.text(
        0.5, 0.005,
        "Error bars = \u00b11 std dev  \u00b7  \u25b3N = outlier count "
        "(runs > mean + 2\u03c3)  \u00b7  GPU util polling underreports on fast models",
        ha="center", va="bottom",
        fontsize=7.5, color="#9CA3AF",
    )

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[plot] Bar chart saved to: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Heatmap summary table
# ---------------------------------------------------------------------------

def _heatmap_colour(val: float, col_min: float, col_max: float,
                    lower_is_better: bool = True, log_scale: bool = False) -> str:
    """
    Map a value within [col_min, col_max] to a hex colour.
    Green = best, red = worst (or reversed when lower_is_better=False).
    Returns a light pastel so dark text stays readable.
    If log_scale is True, use log10 scaling for value mapping.
    """
    import math
    if col_max == col_min:
        t = 0.5
    else:
        if log_scale and col_min > 0 and val > 0:
            log_min = math.log10(col_min)
            log_max = math.log10(col_max)
            log_val = math.log10(val)
            t = (log_val - log_min) / (log_max - log_min)
        else:
            t = (val - col_min) / (col_max - col_min)
    # For lower_is_better, t=0 (min) should be green, t=1 (max) should be red
    # For higher_is_better, t=0 (min) should be red, t=1 (max) should be green
    if not lower_is_better:
        t = 1 - t
    # Interpolate green (#D1FAE5) → amber (#FEF3C7) → red (#FEE2E2)
    if t <= 0.5:
        s = t * 2
        r = int(0xD1 + (0xFE - 0xD1) * s)
        g = int(0xFA + (0xF3 - 0xFA) * s)
        b = int(0xE5 + (0xC7 - 0xE5) * s)
    else:
        s = (t - 0.5) * 2
        r = int(0xFE + (0xFE - 0xFE) * s)
        g = int(0xF3 + (0xE2 - 0xF3) * s)
        b = int(0xC7 + (0xE2 - 0xC7) * s)
    return f"#{r:02X}{g:02X}{b:02X}"


def plot_table(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import to_rgba
        import numpy as np
    except ImportError:
        print("[plot] matplotlib not installed: pip install matplotlib",
              file=sys.stderr)
        sys.exit(1)

    # ── Build one row per unique (label, device, batch_size) ─────────────────
    # Average repeated runs
    buckets: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["label"], row["device"], row["batch_size"])
        for field in ("p50", "p95", "p99", "mean", "std",
                      "outliers", "runs", "cpu_util", "gpu_util"):
            buckets[key][field].append(row[field])

    table_rows = []
    for (label, device, bs), vals in sorted(
        buckets.items(), key=lambda x: (x[0][1], x[0][2], x[0][0])
    ):
        avg = {k: sum(v) / len(v) for k, v in vals.items()}
        throughput = bs / (avg["mean"] / 1000)     # items / second
        cv         = avg["std"] / avg["mean"] * 100

        # Simple bottleneck heuristic
        if device == "cpu":
            bottleneck = "CPU" if avg["cpu_util"] > 40 else "Memory/IO"
        else:
            if avg["gpu_util"] > 40:
                bottleneck = "GPU"
            elif avg["cpu_util"] > 40:
                bottleneck = "Data loading"
            else:
                bottleneck = "Underutilised"

        table_rows.append({
            "Framework":   label,
            "Device":      device.upper(),
            "Batch":       bs,
            "Mean (ms)":   avg["mean"],
            "P50 (ms)":    avg["p50"],
            "P95 (ms)":    avg["p95"],
            "P99 (ms)":    avg["p99"],
            "Std (ms)":    avg["std"],
            "CV (%)":      cv,
            "Throughput":  throughput,
            "CPU util":    avg["cpu_util"],
            "GPU util":    avg["gpu_util"],
            "Outliers":    int(avg["outliers"]),
            "Bottleneck":  bottleneck,
        })

    # Sort by mean latency (fastest first)
    table_rows.sort(key=lambda r: r["Mean (ms)"])

    # ── Column metadata ───────────────────────────────────────────────────────
    # (column_name, fmt_fn, lower_is_better, apply_heatmap)
    
    COLS = [
        ("Framework",  str,                  None,  False, False),
        ("Device",     str,                  None,  False, False),
        ("Batch",      lambda v: str(int(v)), None,  False, False),
        ("Mean (ms)",  lambda v: f"{v:.3f}", True,  True,  True),
        ("P50 (ms)",   lambda v: f"{v:.3f}", True,  True,  True),
        ("P95 (ms)",   lambda v: f"{v:.3f}", True,  True,  True),
        ("P99 (ms)",   lambda v: f"{v:.3f}", True,  True,  True),
        ("Std (ms)",   lambda v: f"{v:.3f}", True,  True,  True),
        ("CV (%)",     lambda v: f"{v:.1f}", True,  True,  False),
        ("Throughput", lambda v: f"{v:.0f}", False, True,  True),
        ("CPU util",   lambda v: f"{v:.1f}%", None, False, False),
        ("GPU util",   lambda v: f"{v:.1f}%", None, False, False),
        ("Outliers",   lambda v: str(int(v)), True,  True,  False),
        ("Bottleneck", str,                  None,  False, False),
    ]

    col_names = [c[0] for c in COLS]
    n_cols    = len(col_names)
    n_rows    = len(table_rows)

    # Pre-compute per-column numeric ranges for heatmap
    col_ranges = {}
    for col_name, fmt_fn, lib, apply_hm, log_scale in COLS:
        if not apply_hm:
            continue
        vals = [r[col_name] for r in table_rows]
        col_ranges[col_name] = (min(vals), max(vals), lib, log_scale)

    # ── Figure ────────────────────────────────────────────────────────────────
    row_h   = 0.52    # inches per data row
    head_h  = 0.65    # header row
    fig_h   = head_h + n_rows * row_h + 1.6   # +padding for title/footnote
    fig_w   = 17

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.axis("off")

    # Column widths (fractions summing to 1, then scaled to fig_w)
    col_widths = [
        0.075,  # Framework
        0.050,  # Device
        0.038,  # Batch
        0.065,  # Mean
        0.065,  # P50
        0.065,  # P95
        0.065,  # P99
        0.065,  # Std
        0.055,  # CV
        0.075,  # Throughput
        0.060,  # CPU util
        0.060,  # GPU util
        0.055,  # Outliers
        0.082,  # Bottleneck
    ]
    assert len(col_widths) == n_cols

    # Normalise to [0,1] and compute cumulative x positions
    total = sum(col_widths)
    col_widths = [w / total for w in col_widths]
    col_x = [sum(col_widths[:i]) for i in range(n_cols)]

    # Draw area: [0,1] x [0,1] in axes coords
    # Map row index → y (top-down)
    total_rows  = n_rows + 1   # data rows + header
    row_heights = [head_h / (head_h + n_rows * row_h)] + \
                  [row_h  / (head_h + n_rows * row_h)] * n_rows

    def row_y(i: int) -> float:
        """Bottom y of row i in [0,1] axes coords (header = 0)."""
        return 1 - sum(row_heights[: i + 1])

    def draw_cell(row_i, col_i, text, bg="#FFFFFF",
                  bold=False, align="center", text_color="#1F2937"):
        x  = col_x[col_i]
        w  = col_widths[col_i]
        y  = row_y(row_i)
        h  = row_heights[row_i]
        pad = 0.004

        rect = plt.Rectangle(
            (x + pad, y + pad), w - 2 * pad, h - 2 * pad,
            transform=ax.transAxes,
            facecolor=bg, edgecolor="#E5E7EB", linewidth=0.4,
            clip_on=False,
        )
        ax.add_patch(rect)

        tx = x + w * (0.08 if align == "left" else 0.5)
        ty = y + h / 2
        ax.text(
            tx, ty, text,
            transform=ax.transAxes,
            ha=align, va="center",
            fontsize=8.5,
            fontweight="bold" if bold else "normal",
            color=text_color,
        )

    # Header row
    HEADER_BG   = "#1E3A5F"
    HEADER_TEXT = "#FFFFFF"
    # Friendly display names for header (newline for two-line headers)
    HEADER_LABELS = {
        "Mean (ms)":  "Mean\n(ms)",
        "P50 (ms)":   "P50\n(ms)",
        "P95 (ms)":   "P95\n(ms)",
        "P99 (ms)":   "P99\n(ms)",
        "Std (ms)":   "Std\n(ms)",
        "CV (%)":     "CV\n(%)",
        "Throughput": "Throughput\n(items/s)",
        "CPU util":   "CPU\nutil",
        "GPU util":   "GPU\nutil",
    }
    for ci, col_name in enumerate(col_names):
        draw_cell(0, ci,
                  HEADER_LABELS.get(col_name, col_name),
                  bg=HEADER_BG, bold=True,
                  text_color=HEADER_TEXT)

    # Data rows
    FRAMEWORK_STRIPE = {
        "PyTorch": "#EFF6FF",
        "ONNX":    "#FFF7ED",
    }
    for ri, row_data in enumerate(table_rows, start=1):
        framework = row_data["Framework"]
        row_stripe = FRAMEWORK_STRIPE.get(framework, "#FFFFFF")

        for ci, (col_name, fmt_fn, lib, apply_hm, log_scale) in enumerate(COLS):
            val = row_data[col_name]
            text = fmt_fn(val)
            if apply_hm and col_name in col_ranges:
                cmin, cmax, lower_better, log_col = col_ranges[col_name]
                bg = _heatmap_colour(float(val), cmin, cmax, lower_better, log_col)
            else:
                bg = row_stripe
            align = "left" if col_name in ("Framework", "Bottleneck") else "center"
            bold  = col_name == "Mean (ms)"
            draw_cell(ri, ci, text, bg=bg, bold=bold, align=align)

    # ── Heatmap legend ────────────────────────────────────────────────────────
    import numpy as np
    legend_ax = fig.add_axes([0.72, 0.01, 0.20, 0.025])
    gradient  = np.linspace(0, 1, 256).reshape(1, -1)
    colours   = np.array([
        [int(_heatmap_colour(t, 0, 1, True)[i:i+2], 16) / 255
         for i in (1, 3, 5)]
        for t in np.linspace(0, 1, 256)
    ])
    legend_ax.imshow(
        colours[np.newaxis, :, :],
        aspect="auto", extent=[0, 1, 0, 1],
    )
    legend_ax.set_yticks([])
    legend_ax.set_xticks([0, 0.5, 1])
    legend_ax.set_xticklabels(["best", "mid", "worst"],
                               fontsize=7.5, color="#6B7280")
    legend_ax.tick_params(length=0)
    for sp in legend_ax.spines.values():
        sp.set_visible(False)
    fig.text(0.72, 0.045, "Heatmap colour scale (metric columns)",
             fontsize=7.5, color="#9CA3AF", ha="left")

    # Framework stripe legend
    fw_patches = [
        mpatches.Patch(facecolor=v, edgecolor="#D1D5DB",
                       linewidth=0.5, label=k)
        for k, v in FRAMEWORK_STRIPE.items()
    ]


    # ── Title + footnote ─────────────────────────────────────────────────────
    fig.text(
        0.5, 0.995,
        "Comprehensive performance benchmark — ResNet18 inference",
        ha="center", va="top",
        fontsize=13, fontweight="bold", color="#111827",
    )
    fig.text(
        0.5, 0.965,
        "PyTorch (eager) vs ONNX Runtime  ·  sorted by mean latency (fastest first)",
        ha="center", va="top",
        fontsize=9, color="#6B7280",
    )
    footnote = (
        "CV = coefficient of variation (std / mean).  "
        "Throughput = batch size / mean latency \u00d7 1000.  "
        "Bottleneck determined by utilisation thresholds (>40%).  "
        "Values averaged across repeated runs."
    )
    fig.text(0.01, 0.005, footnote,
            ha="left", va="bottom",
            fontsize=7.5, color="#9CA3AF")

    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[plot] Table saved to:     {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    csv_path = Path(args.csv)

    if args.out:
        bar_path   = Path(args.out)
        table_path = bar_path.with_name(bar_path.stem + "_table" + bar_path.suffix)
    else:
        bar_path   = csv_path.with_name(csv_path.stem + "_plot.png")
        table_path = csv_path.with_name(csv_path.stem + "_table.png")

    print(f"[plot] Reading {csv_path} ...")
    rows = load_csv(csv_path)
    print(f"[plot] Loaded {len(rows)} result rows.\n")

    plot_bars(rows, bar_path)
    plot_table(rows, table_path)


if __name__ == "__main__":
    main()