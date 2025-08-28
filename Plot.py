#!/usr/bin/env python3
# ASCII-only plotting script for NWM CSV

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR_DEFAULT = "/media/12TB/Sujan/NWM/Plots"

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot NWM time series from the consolidated CSV. "
            "Choose which ensembles (mem1..mem4), which lead times (0/6/12/18), "
            "and whether to include per-lead means and/or overall mean."
        )
    )
    p.add_argument(
        "--csv",
        required=True,
        help="Path to the CSV (e.g. 5092616_20250101_mem1-4_shift.csv)",
    )
    p.add_argument(
        "--outdir",
        default=PLOTS_DIR_DEFAULT,
        help="Output directory for PNG (default: %s)" % PLOTS_DIR_DEFAULT,
    )
    # Allow zero or more ensembles. If you want only means, omit this flag
    # or pass --ensembles with nothing after it.
    p.add_argument(
        "--ensembles",
        nargs="*",
        choices=["mem1", "mem2", "mem3", "mem4"],
        default=["mem1", "mem2", "mem3", "mem4"],
        help="Which ensembles to plot (default: all). Example: --ensembles mem1 mem3",
    )
    p.add_argument(
        "--leads",
        nargs="+",
        type=int,
        choices=[0, 6, 12, 18],
        default=[0, 6, 12, 18],
        help="Which lead columns to plot (default: 0 6 12 18)",
    )
    p.add_argument(
        "--mean-leads",
        action="store_true",
        help='Also plot per-lead ensemble means: "mean 0hrs", "mean 6hrs", ...',
    )
    p.add_argument(
        "--mean-all",
        action="store_true",
        help='Also plot "mean all" line.',
    )
    p.add_argument(
        "--title",
        default=None,
        help="Custom plot title (optional)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="PNG DPI (default: 140)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window (in addition to saving PNG)",
    )
    return p.parse_args()

def make_output_name(csv_path, ensembles, leads, mean_leads, mean_all):
    base = os.path.splitext(os.path.basename(csv_path))[0]
    ens = "none" if not ensembles else "-".join(ensembles)
    ld = "-".join(str(x) for x in leads) if leads else "none"
    ml = "ml1" if mean_leads else "ml0"
    ma = "ma1" if mean_all else "ma0"
    return "%s__%s__leads-%s__%s_%s.png" % (base, ens, ld, ml, ma)

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv)
    if "time" not in df.columns:
        raise RuntimeError("CSV must have a 'time' column.")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Build list of series to plot (column name -> label)
    wanted_cols = []

    # Ensemble lead columns
    if args.ensembles:
        for mem in args.ensembles:
            for lead in args.leads:
                col = "%s %shrs" % (mem, lead)
                if col in df.columns:
                    wanted_cols.append((col, col))
                else:
                    print("Warning: column not found in CSV: %s" % col)

    # Per-lead means
    if args.mean_leads:
        for lead in args.leads:
            col = "mean %shrs" % lead
            if col in df.columns:
                wanted_cols.append((col, col))
            else:
                print("Warning: column not found in CSV: %s" % col)

    # Overall mean
    if args.mean_all:
        if "mean all" in df.columns:
            wanted_cols.append(("mean all", "mean all"))
        else:
            print('Warning: column not found in CSV: "mean all"')

    if not wanted_cols:
        raise RuntimeError("No matching columns to plot -- check --ensembles/--leads/--mean-* and your CSV columns.")

    # Title (try to infer feature/date from filename if not provided)
    if args.title:
        title = args.title
    else:
        base = os.path.basename(args.csv)
        # example: 5092616_20250101_mem1-4_shift.csv
        m = re.match(r"(?P<feat>\d{4,})_(?P<date>\d{8})_", base)
        if m:
            title = "Feature %s - %s" % (m.group("feat"), m.group("date"))
        else:
            title = base

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label in wanted_cols:
        ax.plot(df["time"], df[col], label=label)

    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Discharge (m^3/s)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=9, frameon=False)

    # Save
    out_name = make_output_name(args.csv, args.ensembles, args.leads, args.mean_leads, args.mean_all)
    out_path = os.path.join(args.outdir, out_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    print("Saved plot: %s" % out_path)

    if args.show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
