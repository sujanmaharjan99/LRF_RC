#!/usr/bin/env python3
"""
Plot NWM forecast(s) vs USGS observations for specified stations.

Directory/layout assumptions:
- NWM CSVs live at: /media/12TB/Sujan/NWM/Csv/
  Filename pattern: fid<NWM_ID>_YYYYMMDD_YYYYMMDD.csv
  Columns include: init_time_utc, Ensemble_Mean_6h, Ensemble_Mean_12h, ...

- USGS CSVs live at: /media/12TB/Sujan/NWM/USGS_data/
  Filename pattern: <USGS_ID>.csv
  Columns include: timestamp, discharge_cfs, discharge_cumecs

You can override directories with --nwm-dir and --usgs-dir.

Examples:
  python Plot.py --fhrs 6 12 24 48
  python Plot.py --stations 880478 3624735 --fhrs 6 30 54 --start 2025-01-01 --end 2025-05-15
  python Plot.py --output-dir ./plots --fhrs 6 12 18 24
"""

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


# Default directories
NWM_DIR_DEFAULT = Path("/media/12TB/Sujan/NWM/Csv/")
USGS_DIR_DEFAULT = Path("/media/12TB/Sujan/NWM/USGS_data/")
PLOTS_DIR_DEFAULT = Path("/media/12TB/Sujan/NWM/Plots")


# NWM feature id -> (USGS id, Station name)
STATION_MAP: Dict[int, Tuple[str, str]] = {
    880478: ("05587450", "Mississippi River at Grafton, IL"),
    5092616: ("07022000", "Mississippi River at Thebes, IL"),
    3624735: ("07010000", "Mississippi River at St. Louis, MO"),
    5089904: ("07020500", "Mississippi River at Chester, IL"),
    5166621: ("06909000", "Missouri River at Boonville, MO"),
    6013072: ("06934500", "Missouri River at Hermann, MO"),
    4391417: ("06893000", "Missouri River at Kansas City, MO"),
    6010106: ("06935965", "Missouri River at St. Charles, MO"),
    2252949: ("06818000", "Missouri River at St. Joseph, MO"),
    3702540: ("06813500", "Missouri River at Rulo, NE"),
    4388401: ("06895500", "Missouri River at Waverly, MO"),
}



def _parse_datetime_series(s: pd.Series) -> pd.Series:
    """Parse timestamps robustly. Convert tz-aware to UTC, keep naive if no tz info."""
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.isna().any():
        dt2 = pd.to_datetime(s, errors="coerce")
        if hasattr(dt2.dt, "tz") and dt2.dt.tz is not None:
            dt2 = dt2.dt.tz_convert("UTC")
        dt = dt.fillna(dt2)
    return dt


def _parse_optional_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.isna(dt):
        dt = pd.to_datetime(s, errors="coerce")
        if hasattr(dt, "tz") and dt.tz is not None:
            dt = dt.tz_convert("UTC")
    return dt


def find_nwm_files(nwm_dir: Path, nwm_id: int) -> List[Path]:
    """Find all NWM CSVs for this feature id: fid<NWM_ID>.csv."""
    pattern = f"fid{nwm_id}.csv"
    return sorted(
        list(nwm_dir.glob(f"fid{nwm_id}.csv")) +
        list(nwm_dir.glob(f"fid{nwm_id}_*.csv"))
    )


def load_usgs(usgs_dir: Path, usgs_id: int) -> Optional[pd.DataFrame]:
    """Load USGS observations. Returns ['timestamp', 'discharge_cumecs']."""
    path = usgs_dir / f"{usgs_id}.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    # Find timestamp column
    ts_col_candidates = [c for c in df.columns if c.lower().startswith("timestamp")]
    ts_col = ts_col_candidates[0] if ts_col_candidates else "timestamp"

    # Ensure cumecs column exists, try to fix common variants
    if "discharge_cumecs" not in df.columns:
        for c in df.columns:
            if "cumec" in c.lower():
                df.rename(columns={c: "discharge_cumecs"}, inplace=True)
                break
    if "discharge_cumecs" not in df.columns:
        raise ValueError(
            f"USGS file {path} lacks 'discharge_cumecs' column. Found: {df.columns.tolist()}"
        )

    df["timestamp"] = _parse_datetime_series(df[ts_col])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df[["timestamp", "discharge_cumecs"]].copy()
    return df


def load_nwm_long(nwm_dir: Path, nwm_id: int, fhrs: List[int]) -> Optional[pd.DataFrame]:
    """
    Load NWM CSVs for the feature id and reshape to long format for requested fhrs.
    For each fhr, valid_time = init_time_utc + fhr hours, value = Ensemble_Mean_{fhr}h.
    Returns ['valid_time', 'fhr', 'value'].
    """
    files = find_nwm_files(nwm_dir, nwm_id)
    if not files:
        return None

    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Warning: could not read {fp}: {e}", file=sys.stderr)
            continue

        df.columns = [str(c).strip() for c in df.columns]
        if "init_time_utc" not in df.columns:
            candidates = [c for c in df.columns if "init" in c.lower() and "time" in c.lower() and "utc" in c.lower()]
            if candidates:
                df.rename(columns={candidates[0]: "init_time_utc"}, inplace=True)
            else:
                print(f"Warning: {fp} missing init_time_utc. Skipping.", file=sys.stderr)
                continue

        df["init_time_utc"] = _parse_datetime_series(df["init_time_utc"])
        df = df.dropna(subset=["init_time_utc"])

        for fhr in fhrs:
            col = f"Ensemble_Mean_{fhr}h"
            if col not in df.columns:
                # try to match variants like 'Ensemble_Mean_6h '
                found = None
                for c in df.columns:
                    if re.fullmatch(rf"Ensemble_Mean_{fhr}h\s*", str(c)):
                        found = c
                        break
                if found:
                    col = found
                else:
                    print(f"Note: {fp} has no column for {fhr}h. Skipping that fhr for this file.", file=sys.stderr)
                    continue

            valid_time = df["init_time_utc"] + pd.to_timedelta(fhr, unit="h")
            frames.append(pd.DataFrame({
                "valid_time": valid_time,
                "fhr": fhr,
                "value": pd.to_numeric(df[col], errors="coerce")
            }))

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["valid_time", "value"]).sort_values(["valid_time", "fhr"])
    return out


def plot_station(
    nwm_id: int,
    usgs_id: int,
    station_name: str,
    nwm_dir: Path,
    usgs_dir: Path,
    output_dir: Path,
    fhrs: List[int],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Optional[Path]:
    """Create and save the plot for one station. Returns path to PNG or None."""
    usgs_df = load_usgs(usgs_dir, usgs_id)
    nwm_df = load_nwm_long(nwm_dir, nwm_id, fhrs)

    if usgs_df is None and nwm_df is None:
        print(f"[{station_name}] No USGS or NWM data found, skipping.", file=sys.stderr)
        return None

    # Filter by date if provided
    if start is not None:
        if usgs_df is not None:
            usgs_df = usgs_df[usgs_df["timestamp"] >= start]
        if nwm_df is not None:
            nwm_df = nwm_df[nwm_df["valid_time"] >= start]

    if end is not None:
        if usgs_df is not None:
            usgs_df = usgs_df[usgs_df["timestamp"] <= end]
        if nwm_df is not None:
            nwm_df = nwm_df[nwm_df["valid_time"] <= end]

    # Plot
    plt.figure(figsize=(12, 6))

    if usgs_df is not None and not usgs_df.empty:
        plt.plot(usgs_df["timestamp"], usgs_df["discharge_cumecs"], label="USGS (cumecs)", linewidth=2)

    if nwm_df is not None and not nwm_df.empty:
        for fhr in sorted(nwm_df["fhr"].unique()):
            sub = nwm_df[nwm_df["fhr"] == fhr]
            if sub.empty:
                continue
            plt.plot(sub["valid_time"], sub["value"], label=f"NWM Ensemble Mean, {fhr}h")

    plt.title(f"{station_name} (NWM vs USGS)")
    plt.xlabel("Time")
    plt.ylabel("Discharge (m³/s)")
    plt.grid(True, which="both", axis="both")
    plt.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", station_name)
    out_path = output_dir / f"{nwm_id}_{usgs_id}_{safe_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Plot NWM forecasts vs USGS observations.")
    p.add_argument(
        "--nwm-dir",
        type=Path,
        default=NWM_DIR_DEFAULT,
        help="Directory with NWM CSV files (fid<NWMID>_start_end.csv).",
    )
    p.add_argument(
        "--usgs-dir",
        type=Path,
        default=USGS_DIR_DEFAULT,
        help="Directory with USGS CSV files (<USGSID>.csv).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PLOTS_DIR_DEFAULT,
        help="Directory to write PNG plots.",
    )
    p.add_argument(
        "--stations",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of NWM station IDs to plot. Defaults to all in mapping.",
    )
    p.add_argument(
        "--fhrs",
        type=int,
        nargs="+",
        required=True,
        help="Forecast hour(s) to plot, for example: --fhrs 6 12 24 48.",
    )
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start datetime, for example 2025-01-01 or 2025-01-01T00:00.",
    )
    p.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end datetime, for example 2025-01-05.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Validate fhrs
    fhrs: List[int] = []
    for v in args.fhrs:
        try:
            iv = int(v)
            if iv <= 0:
                raise ValueError
            fhrs.append(iv)
        except Exception:
            print(f"Invalid forecast hour: {v}. Must be a positive integer.", file=sys.stderr)
            sys.exit(2)

    # Which stations
    if args.stations:
        stations = [int(s) for s in args.stations]
    else:
        stations = list(STATION_MAP.keys())

    # Parse time filters
    start = _parse_optional_dt(args.start)
    end = _parse_optional_dt(args.end)

    nwm_dir = Path(args.nwm_dir)
    usgs_dir = Path(args.usgs_dir)
    output_dir = Path(args.output_dir)

    output_paths: List[Path] = []
    for nwm_id in stations:
        if nwm_id not in STATION_MAP:
            print(f"Unknown station NWM id {nwm_id}. Skipping.", file=sys.stderr)
            continue
        usgs_id, name = STATION_MAP[nwm_id]
        try:
            out = plot_station(
                nwm_id, usgs_id, name, nwm_dir, usgs_dir, output_dir, fhrs, start, end
            )
            if out is not None:
                output_paths.append(out)
                print(f"Saved: {out}")
            else:
                print(f"No plot for {nwm_id} ({name}).", file=sys.stderr)
        except Exception as e:
            print(f"Error plotting {nwm_id} ({name}): {e}", file=sys.stderr)

    if not output_paths:
        print("No plots were created. Check file paths, station IDs, and forecast hours.", file=sys.stderr)


if __name__ == "__main__":
    main()
