#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import numpy as np

# ---------- Defaults for your setup ----------
DATA_ROOT_DEFAULT = "/media/12TB/Sujan/NWM/Data"   # YYYYMMDD/long_range_memX
CSV_OUT_DEFAULT   = "/media/12TB/Sujan/NWM/Csv"
ENSEMBLES_DEFAULT = ["long_range_mem1", "long_range_mem2", "long_range_mem3", "long_range_mem4"]
VAR_NAME_DEFAULT  = "streamflow"
TIME_NAME_DEFAULT = "time"
FEAT_NAME_DEFAULT = "feature_id"
# ---------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract NWM ensemble time series to CSV with columns: time, mem1..mem4, mean"
    )
    p.add_argument("--start", required=True, help="Start date YYYYMMDD")
    p.add_argument("--end",   required=True, help="End date YYYYMMDD")
    p.add_argument("--feature", "-f", action="append", required=True,
                   help="Feature ID to extract. Repeat for multiple. Example: -f 5092616 -f 6013072")
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT,
                   help=f"Root data directory with dated folders (default: {DATA_ROOT_DEFAULT})")
    p.add_argument("--csv-out", default=CSV_OUT_DEFAULT,
                   help=f"Directory to write CSV files (default: {CSV_OUT_DEFAULT})")
    p.add_argument("--var-name", default=VAR_NAME_DEFAULT,
                   help=f"NetCDF variable name (default: {VAR_NAME_DEFAULT})")
    p.add_argument("--time-name", default=TIME_NAME_DEFAULT,
                   help=f"Time coordinate name (default: {TIME_NAME_DEFAULT})")
    p.add_argument("--feat-name", default=FEAT_NAME_DEFAULT,
                   help=f"Feature id dimension or coordinate name (default: {FEAT_NAME_DEFAULT})")
    p.add_argument("--ensembles", nargs="+", default=ENSEMBLES_DEFAULT,
                   help="List of ensemble folder names under each date. Default: long_range_mem1..4")
    p.add_argument("--engine", choices=["netcdf4", "h5netcdf"], default="netcdf4",
                   help="xarray engine to read NetCDF files (default: netcdf4)")
    p.add_argument("--pattern", default="*.nc",
                   help="Filename pattern inside each ensemble folder (default: *.nc)")
    return p.parse_args()


'''python Extract_csv.py --start 20250101 --end 20250102 -f 5092616 -f 880478 -f 5092616 -f 3624735 -f 5089904 -f 5166621 -f 6013072 -f 4391417 -f 6010106 -f 2252949 -f 3702540 -f 4388401

python extract_nwm_csv.py --start 20250101 --end 20250115 \
  -f 5092616 -f 6013072 -f 880478 \
  --data-root /media/12TB/Sujan/NWM/Data \
  --csv-out /media/12TB/Sujan/NWM/Csv
  
  '''

def parse_date(s):
    try:
        return datetime.strptime(s, "%Y%m%d")
    except ValueError:
        print("Invalid date. Use YYYYMMDD.")
        sys.exit(1)

def daterange(start_dt, end_dt):
    d = start_dt
    while d <= end_dt:
        yield d
        d += timedelta(days=1)

def collect_files_for_member(data_root, dates, member_name, pattern):
    files = []
    for d in dates:
        day_dir = os.path.join(data_root, d.strftime("%Y%m%d"), member_name)
        if not os.path.isdir(day_dir):
            continue
        files.extend(sorted(glob.glob(os.path.join(day_dir, pattern))))
    return files

def extract_series_from_files(nc_files, feature_id, var_name, time_name, feat_name, engine):
    """
    Return a pandas Series indexed by time for one ensemble member.
    Works even when time is not attached to the variable, or is only in attributes.
    """
    import re
    import pandas as pd
    import numpy as np
    import xarray as xr
    from datetime import datetime, timedelta

    def parse_fhour_from_name(path):
        # expects ...f006..., ...f042..., etc.
        m = re.search(r"\bf(\d{3})\b", os.path.basename(path))
        return int(m.group(1)) if m else None

    def coerce_datetime_scalar(val):
        try:
            return pd.to_datetime(np.asarray(val).item())
        except Exception:
            try:
                vv = np.asarray(val)
                if vv.size == 1:
                    return pd.to_datetime(vv.reshape(()).item(), errors="coerce")
            except Exception:
                pass
        return pd.to_datetime(val, errors="coerce")

    def get_time_index(ds, da, tname_candidates, filename):
        # 1) If time is attached to the DataArray as dim or coord
        for cand in tname_candidates:
            if (cand in da.dims) or (cand in da.coords):
                tvals = pd.to_datetime(da[cand].values, errors="coerce")
                return cand, tvals

        # 2) If time exists as a dataset variable, use that
        for cand in tname_candidates:
            if cand in ds.variables:
                tvals = pd.to_datetime(ds[cand].values, errors="coerce")
                return cand, tvals

        # 3) Try CF decoding at dataset level to surface a decodable time
        try:
            ds_dec = xr.decode_cf(ds)
            for cand in tname_candidates:
                if cand in ds_dec.variables:
                    tvals = pd.to_datetime(ds_dec[cand].values, errors="coerce")
                    return cand, tvals
        except Exception:
            pass

        # 4) Build a single timestamp from attributes and filename offset
        # Common attrs: model_initialization_time, model_output_valid_time, time_coverage_start
        fh = parse_fhour_from_name(filename)  # hours of lead time
        # try valid time directly
        for attr_key in ["model_output_valid_time", "time_coverage_start", "time_coverage_end"]:
            if attr_key in ds.attrs:
                t0 = coerce_datetime_scalar(ds.attrs[attr_key])
                if pd.notna(t0):
                    return "derived_time", pd.to_datetime([t0])
        # try init time + fh
        for init_key in ["model_initialization_time", "forecast_reference_time", "analysis_time"]:
            if init_key in ds.attrs and fh is not None:
                tinit = coerce_datetime_scalar(ds.attrs[init_key])
                if pd.notna(tinit):
                    return "derived_time", pd.to_datetime([tinit + timedelta(hours=fh)])

        # Give up
        return None, None

    pieces = []
    time_candidates = [time_name, "time", "valid_time", "forecast_time", "Time"]

    for fn in nc_files:
        try:
            ds = xr.open_dataset(fn, engine=engine)  # decode_times=True by default
        except Exception as e:
            print(f"Skipping unreadable file: {fn} ({e})")
            continue

        try:
            if var_name not in ds.variables:
                print(f"Variable {var_name} not in {fn}, skipping.")
                continue

            da = ds[var_name]

            # Select feature, keep dimension
            if (feat_name in da.coords) or (feat_name in ds.dims):
                da = da.sel({feat_name: int(feature_id)}, drop=False)
            else:
                print(f"{feat_name} not found in {fn}, skipping.")
                continue

            # Find or derive time index
            tname, tvals = get_time_index(ds, da, time_candidates, fn)
            if tvals is None is not None:
                pass  # just to avoid linter complaint
            if tvals is None or len(np.atleast_1d(tvals)) == 0 or pd.isna(tvals).all():
                print(f"time not found or undecodable in {fn}, skipping.")
                continue

            # Collapse all non-time dims to first slice, but keep time intact if it is a dim
            if tname in da.dims:
                other_dims = [d for d in da.dims if d != tname]
                if other_dims:
                    da1d = da.isel(**{d: 0 for d in other_dims})
                else:
                    da1d = da
                yvals = np.asarray(da1d.values).reshape(-1)
                # Align lengths, some files may have scalar but dim exists with length 1
                if yvals.size != len(tvals):
                    # fallback: make both scalar
                    yvals = np.atleast_1d(yvals).reshape(-1)
                    tvals = np.atleast_1d(tvals)
                    if yvals.size == 1 and tvals.size >= 1:
                        yvals = np.repeat(yvals[0], tvals.size)
                s = pd.Series(yvals, index=pd.to_datetime(tvals))
            else:
                # time is not a dim on da, treat as single point
                # reduce data to scalar
                y = np.asarray(da.values)
                if y.size > 1:
                    # take first element consistently
                    y = y.reshape(-1)[0]
                else:
                    y = y.item() if y.shape == () else y.reshape(-1)[0]
                # if tvals has multiple entries, repeat scalar y
                tvals = np.atleast_1d(tvals)
                s = pd.Series(np.repeat(y, tvals.size), index=pd.to_datetime(tvals))

            # Clean and append
            s = s[~pd.isna(s.index)]
            if not s.empty:
                s = s.sort_index()
                pieces.append(s)

        except Exception as e:
            print(f"Error in {fn}: {e}")
        finally:
            try:
                ds.close()
            except Exception:
                pass

    if not pieces:
        return pd.Series(dtype=float)

    out = pd.concat(pieces).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out



def process_feature(args, feature_id, dates):
    mem_cols = [f"mem{i+1}" for i in range(len(args.ensembles))]
    df = pd.DataFrame()

    for i, member in enumerate(args.ensembles):
        files = collect_files_for_member(args.data_root, dates, member, args.pattern)
        if not files:
            print(f"No files for {member} in range. Feature {feature_id}")
            ser = pd.Series(dtype=float)
        else:
            print(f"Feature {feature_id}: {member} files: {len(files)}. Extracting series.")
            ser = extract_series_from_files(
                files,
                feature_id=feature_id,
                var_name=args.var_name,
                time_name=args.time_name,
                feat_name=args.feat_name,
                engine=args.engine
            )
        df[mem_cols[i]] = ser

    df = df.sort_index().dropna(how="all")
    if df.empty:
        print(f"Feature {feature_id}: no data extracted.")
        return None, None

    df["mean"] = df[mem_cols].mean(axis=1, skipna=True)

    # Assemble output CSV path
    os.makedirs(args.csv_out, exist_ok=True)
    csv_name = f"{feature_id}_{dates[0].strftime('%Y%m%d')}_{dates[-1].strftime('%Y%m%d')}.csv"
    csv_path = os.path.join(args.csv_out, csv_name)

    # Write CSV with explicit time column
    df_out = df.copy()
    try:
        df_out.insert(0, "time", df_out.index.tz_convert(None))
    except Exception:
        df_out.insert(0, "time", df_out.index)
    df_out.to_csv(csv_path, index=False, float_format="%.6f")

    return csv_path, df_out

def main():
    args = parse_args()

    start_dt = parse_date(args.start)
    end_dt   = parse_date(args.end)
    if start_dt > end_dt:
        print("Start date must be on or before end date.")
        sys.exit(1)

    # validate features
    features = []
    for f in args.feature:
        try:
            features.append(int(f))
        except ValueError:
            print(f"Feature must be an integer: {f}")
            sys.exit(1)

    dates = list(daterange(start_dt, end_dt))
    if not dates:
        print("Empty date range.")
        sys.exit(1)

    written = []
    for feat in features:
        csv_path, _ = process_feature(args, feat, dates)
        if csv_path:
            print(f"Wrote CSV: {csv_path}")
            written.append(csv_path)

    if not written:
        print("No CSV files were written. Check inputs and folder structure.")
        sys.exit(1)

if __name__ == "__main__":
    main()
