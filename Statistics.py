#!/usr/bin/env python3
import argparse, sys, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union

# -------------------- Defaults --------------------
NWM_DIR_DEFAULT = Path("/media/12TB/Sujan/NWM/Csv/mean/")
USGS_DIR_DEFAULT = Path("/media/12TB/Sujan/NWM/USGS_data/")
OUTPUT_CSV_DEFAULT = Path("/media/12TB/Sujan/NWM/Plots/Plots_Statistics/nwm_usgs_stats.csv")
PLOTS_DIR_DEFAULT = Path("/media/12TB/Sujan/NWM/Plots/Plots_Statistics")

# NWM feature id -> (USGS id as STRING, Station name)
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

# -------------------- Helpers --------------------
def load_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Read either CSV or XLSX with stripped column names.
    """
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=nrows)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        # reads first sheet by default; change sheet_name=... if you need a specific sheet
        df = pd.read_excel(path, nrows=nrows)  # requires openpyxl
    else:
        raise ValueError(f"Unsupported file type: {path}")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _parse_optional_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None: return None
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt): return None
    return dt.tz_localize("UTC") if getattr(dt, "tz", None) is None else dt.tz_convert("UTC")

def _force_utc_series(x: pd.Series) -> pd.Series:
    v = pd.to_datetime(x, errors="coerce")
    return v.dt.tz_localize("UTC") if getattr(v.dtype, "tz", None) is None else v.dt.tz_convert("UTC")

def _parse_usgs_timestamp(s: pd.Series, usgs_naive_tz: str = "US/Central") -> pd.Series:
    raw = s.astype(str)
    fast = pd.to_datetime(raw, errors="coerce", utc=True)
    mask = fast.isna()
    if mask.any():
        sub = raw[mask]
        cm = sub.str.contains(r"\bCST\b|\bCDT\b", regex=True, na=False)
        if cm.any():
            stripped = sub[cm].str.replace(r"\s*(CST|CDT)$", "", regex=True)
            tmp = pd.to_datetime(stripped, errors="coerce")
            tmp = tmp.dt.tz_localize("US/Central", nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
            fast.loc[tmp.index] = tmp
        rem = fast.isna()
        if rem.any():
            tmp2 = pd.to_datetime(sub[rem], errors="coerce")
            good = tmp2.notna()
            if good.any():
                tmp2 = tmp2[good].dt.tz_localize(usgs_naive_tz, nonexistent="shift_forward", ambiguous="NaT").dt.tz_convert("UTC")
                fast.loc[tmp2.index] = tmp2
    return fast

def find_nwm_files(nwm_dir: Path, nwm_id: int) -> List[Path]:
    """
    Find NWM files for this feature id.
    Supports:
      - fid<N>.csv, fid<N>_*.csv
      - station_<N>.csv, station_<N>_*.csv
      - station_<N>.xlsx, station_<N>_*.xlsx
    """
    return sorted(
        list(nwm_dir.glob(f"fid{nwm_id}.csv")) +
        list(nwm_dir.glob(f"fid{nwm_id}_*.csv")) +
        list(nwm_dir.glob(f"station_{nwm_id}.csv")) +
        list(nwm_dir.glob(f"station_{nwm_id}_*.csv")) +
        list(nwm_dir.glob(f"station_{nwm_id}.xlsx")) +
        list(nwm_dir.glob(f"station_{nwm_id}_*.xlsx")) +
        list(nwm_dir.glob(f"{nwm_id}.csv")) +
        list(nwm_dir.glob(f"{nwm_id}_*.csv")) +
        list(nwm_dir.glob(f"{nwm_id}.xlsx"))
    )

def detect_all_fhrs_from_file(path: Path) -> List[int]:
    df = load_table(path, nrows=1)
    fhrs = set()
    for c in df.columns:
        name = str(c).strip()
        m1 = re.fullmatch(r"Ensemble_Mean_(\d+)h\s*", name)
        if m1:
            fhrs.add(int(m1.group(1)))
            continue
        m2 = re.fullmatch(r"(?i)f0*(\d{1,3})\s*", name)  # f006, f06, f6
        if m2:
            fhrs.add(int(m2.group(1)))
    return sorted(fhrs)



# -------------------- Loading --------------------
def load_usgs(usgs_dir: Path, usgs_id: str, usgs_naive_tz: str = "US/Central") -> Optional[pd.DataFrame]:
    path = usgs_dir / f"{usgs_id}.csv"
    if not path.exists():
        print(f"[USGS] Missing: {path}", file=sys.stderr); return None
    df = pd.read_csv(path); df.columns = [str(c).strip() for c in df.columns]
    ts_col = next((c for c in df.columns if c.lower().startswith("timestamp")), "timestamp")
    if "discharge_cumecs" not in df.columns:
        for c in df.columns:
            if "cumec" in c.lower(): df.rename(columns={c:"discharge_cumecs"}, inplace=True); break
    if "discharge_cumecs" not in df.columns:
        raise ValueError(f"[USGS] {path} lacks 'discharge_cumecs'. Found: {df.columns.tolist()}")
    ts = _parse_usgs_timestamp(df[ts_col], usgs_naive_tz=usgs_naive_tz)
    df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
    return df[["timestamp","discharge_cumecs"]].copy()

def load_nwm_long(nwm_dir: Path, nwm_id: int, fhrs: List[int]) -> Optional[pd.DataFrame]:
    files = find_nwm_files(nwm_dir, nwm_id)
    if not files:
        print(f"[NWM] No files for {nwm_id}", file=sys.stderr); return None
    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = load_table(fp)
        except Exception as e:
            print(f"[NWM] Could not read {fp}: {e}", file=sys.stderr); continue
        df.columns = [str(c).strip() for c in df.columns]
        init_col = "init_time_utc"
        if init_col not in df.columns:
            cand = [c for c in df.columns if "init" in c.lower() and "time" in c.lower() and "utc" in c.lower()]
            if cand: df.rename(columns={cand[0]: init_col}, inplace=True)
            else: print(f"[NWM] {fp} missing init_time_utc. Skipping.", file=sys.stderr); continue
        init = _force_utc_series(df[init_col]); df[init_col] = init
        df = df.dropna(subset=[init_col])
        for fhr in fhrs:
            # Try common variants: f006 / f6 / Ensemble_Mean_6h
            candidates = [
                f"f{fhr:03d}", f"F{fhr:03d}",
                f"f{fhr}",     f"F{fhr}",
                f"Ensemble_Mean_{fhr}h",
            ]
            col = None
            cols_stripped = {str(c).strip(): c for c in df.columns}
            # direct match first
            for want in candidates:
                if want in cols_stripped:
                    col = cols_stripped[want]
                    break
            # regex fallback
            if col is None:
                col = next(
                    (orig for s, orig in cols_stripped.items()
                     if re.fullmatch(rf"(?i)f0*{fhr}\s*", s) or
                        re.fullmatch(rf"Ensemble_Mean_{fhr}h\s*", s)),
                    None
                )
            if col is None:
                continue  # this file doesn't have this lead

            frames.append(pd.DataFrame({
                "valid_time": df[init_col] + pd.to_timedelta(fhr, unit="h"),
                "fhr": fhr,
                "value": pd.to_numeric(df[col], errors="coerce")
            }))

    if not frames: return None
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["valid_time", "value"])
    out = out.drop_duplicates(subset=["valid_time", "fhr"]).sort_values(["valid_time", "fhr"])
    return out

# -------------------- Metrics --------------------
def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    m = np.nanmean(obs); denom = np.nansum((obs - m) ** 2)
    if denom == 0 or np.isnan(denom): return np.nan
    return 1.0 - np.nansum((sim - obs) ** 2) / denom

def nnse(obs, sim): 
    v = nse(obs, sim); return np.nan if np.isnan(v) else 1.0 / (2.0 - v)

def mse(obs, sim): return np.nanmean((sim - obs) ** 2)
def mae(obs, sim): return np.nanmean(np.abs(sim - obs))

def kge(obs, sim) -> float:
    obs_m, sim_m = np.nanmean(obs), np.nanmean(sim)
    obs_s, sim_s = np.nanstd(obs, ddof=0), np.nanstd(sim, ddof=0)
    valid = np.isfinite(obs) & np.isfinite(sim)
    if valid.sum() < 2 or obs_s == 0 or np.isnan(obs_m) or np.isnan(sim_m): 
        return np.nan
    r = np.corrcoef(obs[valid], sim[valid])[0, 1]
    beta = np.nan if obs_m == 0 else sim_m / obs_m
    gamma = np.nan if (sim_m == 0 or obs_m == 0 or obs_s == 0) else (sim_s / sim_m) / (obs_s / obs_m)
    if np.isnan(r) or np.isnan(beta) or np.isnan(gamma): return np.nan
    return 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

METRIC_FUNS = {"NSE": nse, "NNSE": nnse, "MSE": mse, "MAE": mae, "KGE": kge}

# -------------------- Alignment --------------------
def align_series(usgs: pd.DataFrame, nwm_long: pd.DataFrame, fhr: int, tolerance_minutes: int = 60) -> pd.DataFrame:
    sim = nwm_long[nwm_long["fhr"] == fhr].copy()
    if sim.empty: return pd.DataFrame(columns=["time","obs","sim"])
    usgs = usgs.copy()
    usgs["timestamp"] = _force_utc_series(usgs["timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None)
    sim["valid_time"] = _force_utc_series(sim["valid_time"]).dt.tz_convert("UTC").dt.tz_localize(None)
    usgs = usgs.sort_values("timestamp"); sim = sim.sort_values("valid_time")
    out = pd.merge_asof(
        sim.rename(columns={"valid_time": "time"}),
        usgs.rename(columns={"timestamp": "time_obs", "discharge_cumecs": "obs"}),
        left_on="time", right_on="time_obs",
        direction="nearest", tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    out = out.dropna(subset=["obs"])
    return out[["time", "obs", "value"]].rename(columns={"value":"sim"})

# -------------------- Compute --------------------
def compute_metrics_for_station(nwm_id:int, usgs_id:str, station_name:str, nwm_dir:Path, usgs_dir:Path,
                                fhrs:List[int], start:Optional[pd.Timestamp], end:Optional[pd.Timestamp],
                                metrics:List[str], tolerance_minutes:int=60, usgs_naive_tz:str="US/Central"):
    rows: List[Dict[str, object]] = []
    usgs_df = load_usgs(usgs_dir, usgs_id, usgs_naive_tz=usgs_naive_tz)
    nwm_df  = load_nwm_long(nwm_dir, nwm_id, fhrs)
    if usgs_df is None or nwm_df is None: return rows
    if start is not None:
        usgs_df = usgs_df[usgs_df["timestamp"] >= start]; nwm_df = nwm_df[nwm_df["valid_time"] >= start]
    if end is not None:
        usgs_df = usgs_df[usgs_df["timestamp"] <= end];   nwm_df = nwm_df[nwm_df["valid_time"] <= end]
    for fhr in sorted(set(fhrs)):
        aligned = align_series(usgs_df, nwm_df, fhr, tolerance_minutes=tolerance_minutes)
        if aligned.empty: continue
        obs = aligned["obs"].to_numpy(dtype=float); sim = aligned["sim"].to_numpy(dtype=float)
        finite = np.isfinite(obs) & np.isfinite(sim)
        pairs = int(finite.sum())
        for m in metrics:
            func = METRIC_FUNS[m]
            val = func(obs, sim)
            val = float(val) if np.isscalar(val) or isinstance(val, (np.floating,)) else np.nan
            rows.append({"nwm_id": nwm_id, "usgs_id": usgs_id, "station": station_name,
                         "fhr": fhr, "pairs": pairs, "metric": m, "value": val})
    return rows

# -------------------- Plotting of metrics (per station) --------------------
def plot_station_metrics(df: pd.DataFrame, station: str, out_dir: Path, metrics: List[str]):
    """
    Make a single figure per station: metrics vs lead time (fhr).
    Assumes df has columns: ['station','fhr','metric','value'] filtered to this station.
    """
    if df.empty: 
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for metric, grp in df.groupby("metric"):
        grp = grp.sort_values("fhr")
        plt.plot(grp["fhr"], grp["value"], marker="o", label=metric)
    plt.title(f"{station} — Metrics vs Lead Time")
    plt.xlabel("Forecast hour (h)")
    plt.ylabel("Score")
    plt.grid(True, which="both", axis="both")
    plt.legend()
    plt.tight_layout()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", station)
    out_path = out_dir / f"{safe}_{'_'.join(metrics)}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# -------------------- CLI --------------------

'''# Compute NSE and KGE for all hours found in files, Jan–May 2019, and plot per-station profiles
python Statistics.py --all-fhrs --metrics NSE KGE --start 2019-01-01 --end 2019-05-15 --plot

# Same but only specific stations
python Statistics.py --stations 880478 3624735 --all-fhrs --metrics NSE NNSE MSE MAE KGE --plot

# Explicit hours, custom output and plot directory
python Statistics.py --fhrs 6 30 54 --metrics NSE KGE \
  --start 2019-01-01 --end 2019-05-15 \
  --output /media/12TB/Sujan/NWM/Plots/stats_2019.csv \
  --plot --plot-dir /media/12TB/Sujan/NWM/Plots'''
  
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compute NWM vs USGS statistics and (optionally) plot per-station profiles.")
    p.add_argument("--nwm-dir", type=Path, default=NWM_DIR_DEFAULT)
    p.add_argument("--usgs-dir", type=Path, default=USGS_DIR_DEFAULT)
    p.add_argument("--output", type=Path, default=OUTPUT_CSV_DEFAULT, help="Output CSV path.")
    p.add_argument("--stations", type=int, nargs="*", default=None)
    p.add_argument("--fhrs", type=int, nargs="*", default=None, help="e.g., 6 12 24 48")
    p.add_argument("--all-fhrs", action="store_true", help="Use all Ensemble_Mean_<h>h columns found.")
    p.add_argument("--metrics", type=str, nargs="+", required=True, choices=sorted(METRIC_FUNS.keys()))
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--tolerance-minutes", type=int, default=60)
    p.add_argument("--usgs-naive-tz", type=str, default="US/Central")
    p.add_argument("--plot", action="store_true", help="Also make a PNG per station of metrics vs lead time.")
    p.add_argument("--plot-dir", type=Path, default=PLOTS_DIR_DEFAULT, help="Directory for the metric plots.")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    stations = args.stations if args.stations else list(STATION_MAP.keys())
    start = _parse_optional_dt(args.start); end = _parse_optional_dt(args.end)
    # determine fhrs
    explicit = None
    if args.all_fhrs:
        explicit = None
    else:
        if not args.fhrs:
            print("[Error] Provide --fhrs ... or --all-fhrs", file=sys.stderr); sys.exit(2)
        explicit = [int(v) for v in args.fhrs if int(v) > 0]
    metrics = [m.upper() for m in args.metrics]

    all_rows: List[Dict[str, object]] = []
    for nwm_id in stations:
        if nwm_id not in STATION_MAP:
            print(f"[Warn] Unknown NWM id {nwm_id}. Skipping.", file=sys.stderr); continue
        usgs_id, name = STATION_MAP[nwm_id]
        # detect fhrs if needed
        fhrs = explicit
        if fhrs is None:
            files = find_nwm_files(args.nwm_dir, nwm_id)
            if not files:
                print(f"[NWM] No files for {nwm_id}", file=sys.stderr); continue
            fhrs = detect_all_fhrs_from_file(files[0])
            if not fhrs:
                print(f"[NWM] No Ensemble_Mean_*h columns in {files[0]}", file=sys.stderr); continue
        try:
            rows = compute_metrics_for_station(nwm_id, usgs_id, name, args.nwm_dir, args.usgs_dir,
                                               fhrs, start, end, metrics,
                                               tolerance_minutes=args.tolerance_minutes,
                                               usgs_naive_tz=args.usgs_naive_tz)
            all_rows.extend(rows)
            print(f"[OK] {name}: {len(rows)} rows")
        except Exception as e:
            print(f"[Error] {nwm_id} ({name}): {e}", file=sys.stderr)

    if not all_rows:
        print("[Warn] No metrics computed.", file=sys.stderr); sys.exit(1)

    out_df = pd.DataFrame(all_rows).sort_values(["station","fhr","metric"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"[Saved] {args.output}")

    if args.plot:
        print("[Info] Creating per-station metric plots...")
        for station, sdf in out_df.groupby("station"):
            out_path = plot_station_metrics(sdf, station, args.plot_dir, metrics)
            if out_path:
                print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
