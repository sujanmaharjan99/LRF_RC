#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

CSV_ROOT_DEFAULT = "/media/12TB/Sujan/NWM/Csv"
PLOT_OUT_DEFAULT = "/media/12TB/Sujan/NWM/Plots"

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot mean time series for multiple stations from NWM CSVs on one figure"
    )
    p.add_argument(
        "stations",
        nargs="+",
        help="One or more station IDs (feature_id) whose CSVs are in the CSV directory"
    )
    p.add_argument(
        "--csv-dir",
        default=CSV_ROOT_DEFAULT,
        help=f"Directory where the CSVs are stored (default: {CSV_ROOT_DEFAULT})"
    )
    p.add_argument(
        "--outdir",
        default=PLOT_OUT_DEFAULT,
        help=f"Directory to write the PNG plot (default: {PLOT_OUT_DEFAULT})"
    )
    p.add_argument(
        "--outfile",
        default=None,
        help="Output filename for the PNG, e.g., multi_mean.png. Defaults to stations_joined_mean.png"
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional plot title"
    )
    p.add_argument(
        "--ylabel",
        default="streamflow",
        help="Y axis label (default: streamflow)"
    )
    p.add_argument(
        "--start",
        default=None,
        help="Optional start datetime filter, e.g., 2025-01-01 or 2025-01-01T12:00"
    )
    p.add_argument(
        "--end",
        default=None,
        help="Optional end datetime filter, e.g., 2025-01-10 or 2025-01-10T12:00"
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI (default: 150)"
    )
    p.add_argument(
        "--pick",
        choices=["latest", "earliest"],
        default="latest",
        help="If multiple CSVs exist per station, pick the latest (by mtime) or earliest (default: latest)"
    )
    return p.parse_args()

'''python Plot.py 5092616 6013072 \
  --pick earliest \
  --start 2025-01-02 \
  --end 2025-01-08'''

def find_csv_for_station(station, csv_dir, pick="latest"):
    # Expected naming from extractor: <station>_<YYYYMMDD>_<YYYYMMDD>.csv
    pattern = os.path.join(csv_dir, f"{station}_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        return None
    # Choose by file modification time to avoid strict filename parsing
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=(pick == "latest"))
    return matches[0]

def read_and_filter(csv_path, start=None, end=None):
    df = pd.read_csv(csv_path)
    if "time" not in df.columns or "mean" not in df.columns:
        raise ValueError(f"CSV missing required columns 'time' and 'mean': {csv_path}")
    t = pd.to_datetime(df["time"])
    mask = pd.Series(True, index=df.index)
    if start:
        t_start = pd.to_datetime(start)
        mask &= t >= t_start
    if end:
        t_end = pd.to_datetime(end)
        mask &= t <= t_end
    return t[mask], df.loc[mask, "mean"]

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    series_list = []
    labels = []
    used_files = []

    for station in args.stations:
        csv_path = find_csv_for_station(station, args.csv_dir, pick=args.pick)
        if csv_path is None:
            print(f"Warning: no CSV found for station {station} in {args.csv_dir}")
            continue
        try:
            t, y = read_and_filter(csv_path, start=args.start, end=args.end)
        except Exception as e:
            print(f"Skipping {station}: {e}")
            continue
        if t.empty:
            print(f"Warning: no data after filters for station {station} ({csv_path})")
            continue
        series_list.append((t, y))
        labels.append(str(station))
        used_files.append(csv_path)

    if not series_list:
        print("No data to plot. Check station IDs, CSV directory, and filters.")
        sys.exit(1)

    # Build title
    if args.title:
        title = args.title
    else:
        stations_str = ",".join(labels)
        title = f"Mean time series for stations {stations_str}"

    # Plot all on one figure
    plt.figure(figsize=(12, 6))
    for (t, y), label in zip(series_list, labels):
        plt.plot(t, y, label=label)
    plt.xlabel("Time")
    plt.ylabel(args.ylabel)
    plt.title(title)
    plt.legend(title="Station")
    plt.tight_layout()

    # Output filename
    if args.outfile:
        out_name = args.outfile
    else:
        base = "_".join(labels[:5]) + ("_etc" if len(labels) > 5 else "")
        out_name = f"{base}_mean.png"
    out_path = os.path.join(args.outdir, out_name)

    plt.savefig(out_path, dpi=args.dpi)
    plt.close()
    print("Saved plot:", out_path)
    # Optional: show which files were used
    for lab, f in zip(labels, used_files):
        print(f"  {lab}: {f}")

if __name__ == "__main__":
    main()
