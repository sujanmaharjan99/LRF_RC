#!/usr/bin/env python3
import os
import re
import glob
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr

DATA_ROOT_DEFAULT = "/media/12TB/Sujan/NWM/Data"
CSV_OUT_DEFAULT   = "/media/12TB/Sujan/NWM/Csv"
VAR_NAME_DEFAULT  = "streamflow"
FEAT_NAME_DEFAULT = "feature_id"

LEAD_MIN, LEAD_MAX, LEAD_STEP = 6, 720, 6  # 6-hourly through 30 days
CYCLES = [("t00z", 0), ("t06z", 1), ("t12z", 2), ("t18z", 3)]  # (cycle_tag, row_shift)

FNAME_RE = re.compile(r"^nwm\.t(\d{2})z\.long_range\.channel_rt_(\d)\.f(\d{3})\.conus\.nc$")

def parse_args():
    p = argparse.ArgumentParser(
        description="From one folder day (mem1..mem4), build a shifted table:\n"
                    "time, mem1 0hrs/6hrs/12hrs/18hrs, ..., mem4 0/6/12/18, mean 0/6/12/18, mean all."
    )
    p.add_argument("--date", required=True, help="Folder date YYYYMMDD, e.g. 20250101")
    p.add_argument("-f", "--feature", required=True, type=int, help="feature_id to extract")
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT, help="Root data dir")
    p.add_argument("--csv-out", default=CSV_OUT_DEFAULT, help="CSV output dir")
    p.add_argument("--var-name", default=VAR_NAME_DEFAULT, help="Variable name in NetCDF")
    p.add_argument("--feat-name", default=FEAT_NAME_DEFAULT, help="Feature id dim/coord name")
    p.add_argument("--engine", choices=["netcdf4", "h5netcdf"], default="netcdf4", help="xarray engine")
    return p.parse_args()

def list_cycle_files(mem_dir, cycle_tag):
    """Return list of (lead_hours, filepath) for this member/cycle, sorted by lead."""
    patt = os.path.join(mem_dir, f"nwm.{cycle_tag}.long_range.channel_rt_*.f*.conus.nc")

    found = []
    for fn in glob.glob(patt):
        m = FNAME_RE.match(os.path.basename(fn))
        if not m:
            continue
        # m.group(1) = cycle hour, m.group(2) = routing member (1..4), m.group(3) = lead hours
        fh = int(m.group(3))
        if fh < LEAD_MIN or fh > LEAD_MAX or fh % LEAD_STEP != 0:
            continue
        found.append((fh, fn))
    found.sort(key=lambda x: x[0])
    return found

def read_feature_value(fn, var_name, feat_name, feature_id, engine):
    """Read a single value for the feature_id from a file; return NaN if not available."""
    try:
        ds = xr.open_dataset(fn, engine=engine)
    except Exception:
        return np.nan
    try:
        if var_name not in ds:
            return np.nan
        da = ds[var_name]
        if (feat_name in da.coords) or (feat_name in ds.dims):
            da = da.sel({feat_name: int(feature_id)}, drop=False)
        else:
            return np.nan
        arr = np.asarray(da.values)
        if arr.size == 0:
            return np.nan
        return float(arr.reshape(-1)[0])
    except Exception:
        return np.nan
    finally:
        try:
            ds.close()
        except Exception:
            pass

def series_for_cycle_by_position(mem_dir, cycle_tag, feature_id, var_name, feat_name, engine, leads_grid):
    """
    Build a Series indexed by integer position (0..N-1) along the uniform leads grid.
    We fill by lead-order from files (f006,f012,...), not by timestamps.
    """
    files = list_cycle_files(mem_dir, cycle_tag)
    vals_by_lead = {}
    for fh, fn in files:
        vals_by_lead[fh] = read_feature_value(fn, var_name, feat_name, feature_id, engine)

    vals = [vals_by_lead.get(fh, np.nan) for fh in leads_grid]
    return pd.Series(vals, index=range(len(leads_grid)))

def build_member_columns(mem_dir, member_label, feature_id, var_name, feat_name, engine, leads_grid):
    """
    Returns a DataFrame with columns:
    f"{member_label} 0hrs", f"{member_label} 6hrs", f"{member_label} 12hrs", f"{member_label} 18hrs"
    aligned by position, with shifts 0/1/2/3 rows respectively.
    """
    # one "all-leads" positional series per cycle
    s_by_cycle = {}
    for cyc_tag, shift_rows in CYCLES:
        s_pos = series_for_cycle_by_position(mem_dir, cyc_tag, feature_id, var_name, feat_name, engine, leads_grid)
        if shift_rows:
            s_pos = s_pos.shift(shift_rows)
        s_by_cycle[cyc_tag] = s_pos

    # assemble columns for this member
    df = pd.DataFrame({
        f"{member_label} 0hrs":  s_by_cycle["t00z"],
        f"{member_label} 6hrs":  s_by_cycle["t06z"],
        f"{member_label} 12hrs": s_by_cycle["t12z"],
        f"{member_label} 18hrs": s_by_cycle["t18z"],
    })
    return df

def main():
    args = parse_args()
    try:
        base_dt = datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        print("Invalid --date. Use YYYYMMDD.")
        return

    # uniform 6-hour lead grid [6..720]
    leads_grid = list(range(LEAD_MIN, LEAD_MAX + 1, LEAD_STEP))
    n = len(leads_grid)

    # Build member frames
    member_frames = []
    for m in [1, 2, 3, 4]:
        mem_dir = os.path.join(args.data_root, args.date, f"long_range_mem{m}")
        member_label = f"mem{m}"
        if not os.path.isdir(mem_dir):
            # create empty columns if folder missing
            member_frames.append(pd.DataFrame({
                f"{member_label} 0hrs":  pd.Series([np.nan]*n),
                f"{member_label} 6hrs":  pd.Series([np.nan]*n),
                f"{member_label} 12hrs": pd.Series([np.nan]*n),
                f"{member_label} 18hrs": pd.Series([np.nan]*n),
            }))
            continue

        df_mem = build_member_columns(
            mem_dir=mem_dir,
            member_label=member_label,
            feature_id=args.feature,
            var_name=args.var_name,
            feat_name=args.feat_name,
            engine=args.engine,
            leads_grid=leads_grid
        )
        member_frames.append(df_mem)

    # Concatenate all members side-by-side by position index
    wide = pd.concat(member_frames, axis=1)

    # Time column from base date + lead_hours
    times = [base_dt + timedelta(hours=h) for h in leads_grid]
    wide.insert(0, "time", pd.to_datetime(times))

    # Per-lead ensemble means (across mem1..mem4 for each lead set)
    mean_cols = {}
    for lead_name in ["0hrs", "6hrs", "12hrs", "18hrs"]:
        cols = [c for c in wide.columns if c.startswith("mem") and c.endswith(lead_name)]
        if cols:
            mean_cols[f"mean {lead_name}"] = wide[cols].astype(float).mean(axis=1, skipna=True)

    if mean_cols:
        means_df = pd.DataFrame(mean_cols, index=wide.index)
        wide = pd.concat([wide, means_df], axis=1, copy=False)

    # Mean across ALL mem columns (all four leads x four members) at each time
    all_mem_cols = [c for c in wide.columns if c.startswith("mem")]
    if all_mem_cols:
        wide["mean all"] = wide[all_mem_cols].astype(float).mean(axis=1, skipna=True)
    else:
        wide["mean all"] = np.nan

    # Column order: time, mem1 0/6/12/18, mem2..., mem3..., mem4..., mean 0/6/12/18, mean all
    ordered = ["time"]
    for m in [1, 2, 3, 4]:
        for lead_name in ["0hrs", "6hrs", "12hrs", "18hrs"]:
            col = f"mem{m} {lead_name}"
            if col in wide.columns:
                ordered.append(col)
    for lead_name in ["0hrs", "6hrs", "12hrs", "18hrs"]:
        col = f"mean {lead_name}"
        if col in wide.columns:
            ordered.append(col)
    if "mean all" in wide.columns:
        ordered.append("mean all")

    wide = wide[[c for c in ordered if c in wide.columns]].copy()
    # Make time tz-naive for Excel
    wide["time"] = pd.to_datetime(wide["time"]).dt.tz_localize(None)

    # Save
    os.makedirs(args.csv_out, exist_ok=True)
    out_path = os.path.join(args.csv_out, f"{args.feature}_{args.date}_mem1-4_shift.csv")
    wide.to_csv(out_path, index=False, float_format="%.6f")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
