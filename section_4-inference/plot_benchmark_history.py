import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys


def plot_benchmark_history(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print("CSV columns:", list(df.columns))

    # ── column classification ──────────────────────────────────────────────
    timing_cols = [c for c in df.columns if c.startswith('timing_') and c.endswith('_s')]
    # exclude std_ms (spread metric, not a trend line) and handle throughput separately
    latency_cols   = [c for c in df.columns if c.endswith('_ms') and c != 'std_ms']
    throughput_cols = [c for c in df.columns if c == 'throughput_rps']

    print("Latency columns  :", latency_cols)
    print("Throughput columns:", throughput_cols)
    print("Timing columns   :", timing_cols)

    if not latency_cols and not timing_cols:
        print("No plottable columns found.")
        return

    # ── shared x-axis ──────────────────────────────────────────────────────
    # Use a simple integer index as x so tick positions always align,
    # then apply commit+time strings as labels.
    x = list(range(len(df)))
    xlabel = ''
    if 'timestamp' in df.columns and 'commit_message' in df.columns:
        xtick_labels = [
            f"{row['commit_message']}\n({pd.to_datetime(row['timestamp']).strftime('%H:%M')})"
            for _, row in df.iterrows()
        ]
    elif 'timestamp' in df.columns:
        xtick_labels = [pd.to_datetime(row['timestamp']).strftime('%H:%M')
                        for _, row in df.iterrows()]
    else:
        xtick_labels = [str(i) for i in x]

    # ── colour palettes ────────────────────────────────────────────────────
    latency_colors = {
        'p50_ms':  '#e63946',
        'p95_ms':  '#f4a261',
        'p99_ms':  '#e9c46a',
        'mean_ms': '#457b9d',
    }
    timing_colors = {
        'timing_total_s':            '#1d3557',
        'timing_inference_s':        '#e63946',
        'timing_preprocessing_s':    '#2a9d8f',
        'timing_postprocessing_s':   '#f4a261',
        'timing_batching_s':         '#a8dadc',
        'timing_resize_s':           '#c77dff',
        'timing_cuda_sync_before_s': '#6c757d',
        # New GPU Event timing columns
        'timing_gpu_active_s':       '#e63946',  # same red family as inference (it's a subset)
        'timing_gpu_idle_before_s':  '#adb5bd',  # muted grey — idle time is waste
        'timing_launch_overhead_s':  '#ffe066',  # yellow — overhead to watch
    }
    throughput_color = '#2a9d8f'

    # ── detect large outlier that would dwarf everything else ──────────────
    # Use p50 (or first latency col) as the row-level signal for outlier rows,
    # so that p95/p99 from the same bad run don't pollute the threshold calculation.
    OUTLIER_RATIO = 5
    has_broken_axis = False
    outlier_threshold = None   # upper bound for the main (zoomed) panel
    outlier_value = None       # representative value for the outlier strip
    anchor_col = 'p50_ms' if 'p50_ms' in latency_cols else (latency_cols[0] if latency_cols else None)
    if anchor_col:
        anchor_vals = df[anchor_col].dropna().sort_values()
        if len(anchor_vals) >= 2 and anchor_vals.iloc[-1] > OUTLIER_RATIO * anchor_vals.iloc[-2]:
            has_broken_axis = True
            # Everything below the second-highest p50 is "normal"
            normal_p50_max    = anchor_vals.iloc[-2]
            outlier_threshold = normal_p50_max * 1.5   # top of the zoomed panel
            outlier_value     = anchor_vals.iloc[-1]   # bottom of the outlier strip

    # ── figure layout ──────────────────────────────────────────────────────
    has_timing     = bool(timing_cols) and df[timing_cols].notna().any().any()
    has_throughput = bool(throughput_cols) and df[throughput_cols].notna().any().any()

    if has_broken_axis:
        if has_timing:
            fig = plt.figure(figsize=(13, 13))
            # rows: outlier strip | main latency | timing
            # Use nested gridspecs: top two rows share tight hspace, timing is separated
            gs = fig.add_gridspec(3, 1, height_ratios=[1, 3, 3],
                                  hspace=0,        # will override per-pair below
                                  top=0.95, bottom=0.12)
            gs.update(hspace=0)
            ax_top  = fig.add_subplot(gs[0])
            ax_main = fig.add_subplot(gs[1])
            ax_tim  = fig.add_subplot(gs[2])
            # Tighten gap between outlier strip and main, leave room above timing
            fig.subplots_adjust(hspace=0)
            pos_top  = ax_top.get_position()
            pos_main = ax_main.get_position()
            pos_tim  = ax_tim.get_position()
            # nudge: collapse top/main gap, open main/timing gap
            gap = 0.02   # small gap between broken-axis pair
            sep = 0.06   # larger separation before timing panel
            total_h = pos_top.height + pos_main.height
            ax_top.set_position( [pos_top.x0,  pos_tim.y0 + pos_tim.height + sep + pos_main.height + gap,
                                   pos_top.width, pos_top.height])
            ax_main.set_position([pos_main.x0, pos_tim.y0 + pos_tim.height + sep,
                                  pos_main.width, pos_main.height])
        else:
            fig = plt.figure(figsize=(13, 8))
            gs  = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.04)
            ax_top  = fig.add_subplot(gs[0])
            ax_main = fig.add_subplot(gs[1])
            ax_tim  = None
    else:
        ax_top = None
        if has_timing:
            fig, (ax_main, ax_tim) = plt.subplots(2, 1, figsize=(13, 10),
                                                   gridspec_kw={'hspace': 0.15})
        else:
            fig, ax_main = plt.subplots(figsize=(13, 6))
            ax_tim = None

    # ── helper ─────────────────────────────────────────────────────────────
    def plot_latency(ax, cols):
        for col in cols:
            if col not in df:
                continue
            ax.plot(x, df[col],
                    label=col,
                    color=latency_colors.get(col),
                    linewidth=2, marker='o', markersize=4)

    def add_break_marks_bottom(ax):
        """Diagonal slash marks on the bottom edge of ax."""
        d = 0.012
        kw = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
        ax.plot((-d, +d), (-d, +d), **kw)
        ax.plot((1-d, 1+d), (-d, +d), **kw)

    def add_break_marks_top(ax):
        """Diagonal slash marks on the top edge of ax."""
        d = 0.012
        kw = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1)
        ax.plot((-d, +d), (1-d, 1+d), **kw)
        ax.plot((1-d, 1+d), (1-d, 1+d), **kw)

    # ── outlier strip ──────────────────────────────────────────────────────
    if has_broken_axis and ax_top is not None:
        plot_latency(ax_top, latency_cols)
        ax_top.set_yscale('log')
        ax_top.set_ylim(outlier_value * 0.85, outlier_value * 1.10)
        ax_top.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax_top.set_xticks([])
        ax_top.set_ylabel('ms')
        ax_top.set_title('Benchmark History', fontsize=13, fontweight='bold', pad=10)
        add_break_marks_bottom(ax_top)

    # ── main latency panel ─────────────────────────────────────────────────
    plot_latency(ax_main, latency_cols)
    ax_main.set_yscale('log')

    if has_broken_axis:
        # Zoom Y floor using range-based padding so small post-drop changes are visible
        post_drop_vals = pd.concat([df[c] for c in latency_cols if c in df]).dropna()
        post_drop_vals = post_drop_vals[post_drop_vals < outlier_threshold]
        if len(post_drop_vals):
            data_range = post_drop_vals.max() - post_drop_vals.min()
            pad = max(data_range * 0.15, 1.0)   # at least 1ms padding
            y_floor = max(1.0, post_drop_vals.min() - pad)  # must be >0 for log scale
            y_ceil  = post_drop_vals.max() + pad
        else:
            y_floor, y_ceil = 1.0, outlier_threshold
        ax_main.set_ylim(y_floor, y_ceil)
        ax_main.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        add_break_marks_top(ax_main)
    else:
        ax_main.set_title('Benchmark History', fontsize=13, fontweight='bold')

    # throughput on twin axis
    if has_throughput:
        ax_rps = ax_main.twinx()
        ax_rps.plot(x, df['throughput_rps'],
                    label='throughput_rps',
                    color=throughput_color,
                    linewidth=2, linestyle=':', marker='s', markersize=4)
        ax_rps.set_ylabel('Throughput (req/s)', color=throughput_color)
        ax_rps.tick_params(axis='y', labelcolor=throughput_color)
        lines2, labs2 = ax_rps.get_legend_handles_labels()
    else:
        lines2, labs2 = [], []

    ax_main.set_ylabel('Latency (ms)')
    ax_main.set_xlabel(xlabel if ax_tim is None else '')
    ax_main.grid(axis='y', linestyle='--', alpha=0.4)

    lines1, labs1 = ax_main.get_legend_handles_labels()
    ax_main.legend(lines1 + lines2, labs1 + labs2,
                   loc='upper left', fontsize=8, framealpha=0.8)

    def apply_xtick_labels(ax, show_labels=True):
        ax.set_xticks(x)
        if show_labels:
            ax.set_xticklabels(xtick_labels, rotation=35, ha='right', fontsize=7)
        else:
            ax.set_xticklabels([])

    if ax_top is not None:
        apply_xtick_labels(ax_top, show_labels=False)
    # Always show labels on ax_main (bottom panel of the broken-axis pair, or only panel)
    apply_xtick_labels(ax_main, show_labels=(ax_tim is None))

    # ── timing breakdown panel ─────────────────────────────────────────────
    if ax_tim is not None:
        # Only plot rows that have at least one non-NaN timing value
        timing_mask = df[timing_cols].notna().any(axis=1)
        x_tim = [i for i, m in enumerate(timing_mask) if m]

        for col in timing_cols:
            lw = 2.5 if col == 'timing_total_s' else 1.5
            ls = '-'  if col == 'timing_total_s' else '--'
            # Human-friendly label for legend
            label_map = {
                'timing_gpu_active_s': 'gpu_active',
                'timing_gpu_idle_before_s': 'gpu_idle_before',
                'timing_launch_overhead_s': 'launch_overhead',
            }
            label = label_map.get(col, col.replace('timing_', '').replace('_s', ''))
            y_tim = df.loc[timing_mask, col].values
            ax_tim.plot(x_tim, y_tim,
                        label=label,
                        color=timing_colors.get(col),
                        linewidth=lw, linestyle=ls,
                        marker='o', markersize=4)

        ax_tim.set_ylabel('Time (s)')
        ax_tim.set_xlabel(xlabel)
        ax_tim.set_title('Timing Breakdown (per-request averages)', fontsize=11)
        ax_tim.grid(axis='y', linestyle='--', alpha=0.4)

        # Zoom Y to the actual data range with range-based padding
        tim_vals = df.loc[timing_mask, timing_cols].values.flatten()
        tim_vals = tim_vals[~pd.isna(tim_vals)]
        if len(tim_vals):
            tim_range = tim_vals.max() - tim_vals.min()
            tim_pad   = max(tim_range * 0.15, 1e-4)
            ax_tim.set_ylim(tim_vals.min() - tim_pad, tim_vals.max() + tim_pad)

        ax_tim.legend(loc='upper left', fontsize=8, framealpha=0.8)
        apply_xtick_labels(ax_tim, show_labels=True)

    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    csv_path    = sys.argv[1] if len(sys.argv) > 1 else "benchmark_history.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "benchmark_history.png"
    plot_benchmark_history(csv_path, output_path)