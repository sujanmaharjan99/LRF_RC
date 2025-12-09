#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import re
import unicodedata
import sys

import pandas as pd
import dataretrieval.nwis as nwis
VELOCITY_MPS_TO_FPS = 3.280839895013123  # 1 m = 3.2808 ft
FT_TO_M = 0.3048
CFS_TO_CUMECS = 0.028316846592
TS_FMT = "%Y-%m-%d %H:%M:%S"

VELOCITY_PARAMS_MPS = {"81380"}  # discharge velocity, m/s
DISCHARGE_PARAMS_CFS = {"00060", "00061"}  # extend if needed

def slugify_name(name: str) -> str:
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    s = re.sub(r"_+", "_", s)
    return s[:80]

def fetch_site_names(sites):
    """Return {site_no: station_name} using NWIS site metadata."""
    try:
        meta = nwis.get_info(sites=",".join(sites))
        if meta is None or meta.empty:
            return {s: "" for s in sites}
        meta = meta.copy()
        if "site_no" not in meta.columns:
            return {s: "" for s in sites}
        meta["site_no"] = meta["site_no"].astype(str)
        name_col = "station_nm"
        if name_col not in meta.columns:
            for c in meta.columns:
                if c.lower().endswith("station_nm"):
                    name_col = c
                    break
        if name_col not in meta.columns:
            return {s: "" for s in sites}
        meta = meta.drop_duplicates(subset=["site_no"])
        return {row["site_no"]: (str(row[name_col]) if pd.notna(row[name_col]) else "")
                for _, row in meta.iterrows()}
    except Exception as e:
        print(f"[names] Warning: could not fetch site names: {e}", file=sys.stderr)
        return {s: "" for s in sites}

def parse_args():
    p = argparse.ArgumentParser(
        description="USGS IV discharge (00060) and stage (00065/00072) to CSVs at native resolution, one file per site."
    )
    p.add_argument("--sites", required=True,
                   help="Comma-separated USGS site numbers, e.g. 03339000 or 03339000,01646500")
    p.add_argument("--start-date", required=True, help="YYYYMMDD")
    p.add_argument("--end-date", required=True, help="YYYYMMDD (inclusive)")
    p.add_argument("--tz", default="America/Chicago",
                   help="Timezone to convert timestamps before writing (default America/Chicago)")
    p.add_argument("--parameter", default="00060",
                   help="Discharge parameter code (default 00060)")
    p.add_argument("--stage-code", default="00065",
                   help="Stage parameter code: 00065=gauge height (ft), 00072=stage (m). Default 00065.")
    p.add_argument("--out", required=True,
                   help="Output directory (e.g., /media/.../USGS_data/). One CSV per site will be created.")
    return p.parse_args()

def pick_col_for_code(df, code: str):
    for c in df.columns:
        if code in c or str(c).endswith(code):
            return c
    return None

def main():
    args = parse_args()
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dates for NWIS
    start_str = pd.to_datetime(args.start_date, format="%Y%m%d").strftime("%Y-%m-%d")
    end_str   = pd.to_datetime(args.end_date,   format="%Y%m%d").strftime("%Y-%m-%d")

    # Local window edges for optional trimming (inclusive)
    start_tz = pd.to_datetime(args.start_date, format="%Y%m%d").tz_localize(args.tz)
    end_tz   = (pd.to_datetime(args.end_date,  format="%Y%m%d") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_localize(args.tz)

    name_map = fetch_site_names(sites)

    for site in sites:
        try:
            df = nwis.get_record(
                sites=site,
                service="iv",
                start=start_str,
                end=end_str,
                parameterCd=[args.parameter, args.stage_code]
            )
        except Exception as e:
            print(f"[{site}] Error fetching data: {e}")
            continue

        if df is None or df.empty:
            print(f"[{site}] No data returned in window.")
            continue

        # Ensure tz-aware index in UTC, then convert to requested tz for output
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()
        df_local = df.tz_convert(args.tz)

        # Optional hard trim in local tz to exact start/end
        df_local = df_local.loc[(df_local.index >= start_tz) & (df_local.index <= end_tz)]

        # Identify series columns
        q_col = pick_col_for_code(df_local, args.parameter)
        st_col = pick_col_for_code(df_local, args.stage_code)

        if q_col is None and st_col is None:
            print(f"[{site}] Neither {args.parameter} nor {args.stage_code} found in data.")
            continue

        # Build rows at native resolution
                # Build rows at native resolution
        rows = []
        idx = df_local.index
        for ts in idx:
            discharge_cfs = None
            discharge_cumecs = None
            stage_val = None
            stage_ft = stage_m = None
            velocity_mps = velocity_fps = None

            # Main parameter value
            if q_col is not None:
                val = df_local.at[ts, q_col]
                if pd.notna(val):
                    val = float(val)
                    if args.parameter in DISCHARGE_PARAMS_CFS:
                        # Discharge in cfs
                        discharge_cfs = val
                        discharge_cumecs = discharge_cfs * CFS_TO_CUMECS
                    elif args.parameter in VELOCITY_PARAMS_MPS:
                        # Velocity in m/s
                        velocity_mps = val
                        velocity_fps = val * VELOCITY_MPS_TO_FPS
                    else:
                        # Fallback: treat as generic series if you want
                        discharge_cfs = val  # or drop this entirely

            # Stage
            if st_col is not None:
                val = df_local.at[ts, st_col]
                if pd.notna(val):
                    stage_val = float(val)

            if stage_val is not None:
                if args.stage_code == "00065":   # feet
                    stage_ft = stage_val
                    stage_m  = stage_val * FT_TO_M
                elif args.stage_code == "00072": # meters
                    stage_m  = stage_val
                    stage_ft = stage_val / FT_TO_M

            # Only write a row if at least one series has a value
            if any(v is not None for v in (discharge_cfs, discharge_cumecs,
                                           velocity_mps, stage_val)):
                rows.append({
                    "timestamp": ts.strftime(TS_FMT),
                    "site_no": site,
                    "site_name": name_map.get(site, ""),
                    "discharge_cfs": f"{discharge_cfs:.3f}" if discharge_cfs is not None else "",
                    "discharge_cumecs": f"{discharge_cumecs:.6f}" if discharge_cumecs is not None else "",
                    "velocity_mps": f"{velocity_mps:.3f}" if velocity_mps is not None else "",
                    "velocity_fps": f"{velocity_fps:.3f}" if velocity_fps is not None else "",
                    "stage_ft": f"{stage_ft:.3f}" if stage_ft is not None else "",
                    "stage_m": f"{stage_m:.3f}" if stage_m is not None else "",
                })


        if not rows:
            print(f"[{site}] No valid observations after filtering.")
            continue

        # Decide filename
        site_name = name_map.get(site, "")
        slug = slugify_name(site_name)
        out_path = out_dir / (f"{site}_{slug}.csv" if slug else f"{site}.csv")

        # Write CSV
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp","site_no","site_name",
                    "discharge_cfs","discharge_cumecs",
                    "velocity_mps","velocity_fps",
                    "stage_ft","stage_m"
                ]
            )

            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"[{site}] Wrote {len(rows)} rows -> {out_path}")

if __name__ == "__main__":
    pd.options.display.width = 140
    main()
