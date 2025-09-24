#!/usr/bin/env python3
import argparse, os, re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import xarray as xr
import tempfile
import traceback
import fcntl
import os, fcntl, tempfile
import pandas as pd
# Sensible defaults to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# If you rely on GCP creds, keep or change this path as needed
os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/media/12TB/Sujan/NWM/Codes/Key/steady-library-470316-r2-5ea9851180c5.json",
)

VALID_CYCLES = ["00", "06", "12", "18"]
ENSEMBLES = ["long_range_mem1", "long_range_mem2", "long_range_mem3", "long_range_mem4"]
FNAME_RE = re.compile(
    r"^nwm\.t(?P<hh>\d{2})z\.long_range\.channel_rt_(?P<mem>\d)\.f(?P<fhr>\d{3})\.conus\.nc$"
)

# NetCDF/HDF5 magic headers
MAGIC_HDF5 = b"\x89HDF\r\n\x1a\n"
MAGIC_NETCDF = b"CDF"

# -------------------------
# Arg parsing
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build wide tables from local NWM Long Range channel_rt files. "
            "Rows are init times (t00, t06, t12, t18). "
            "Columns are Ensemble_Mean_{fhr}h and optionally mem1..mem4."
        )
    )
    p.add_argument("--start", required=True, help="YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYYMMDD")
    p.add_argument(
        "--feature-ids",
        required=True,
        help="Comma-separated feature_ids or a text file with one id per line",
    )
    p.add_argument(
        "--input-dir",
        default="/media/12TB/Sujan/NWM/Data",
        help="Root folder that already has downloaded files",
    )
    p.add_argument(
        "--output-dir",
        default="/media/12TB/Sujan/NWM/Csv",
        help="Folder to write CSV outputs",
    )
    p.add_argument(
        "--horizons",
        default="6-720:6",
        help="Range spec start-end:step, e.g. 6-720:6",
    )
    p.add_argument(
        "--mean-only",
        action="store_true",
        help="Only output ensemble means, omit per-member columns",
    )
    p.add_argument(
        "--engine-fallback",
        action="store_true",
        help="Try h5netcdf if netcdf4 fails to open a file",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Compute means from available members when some are missing or corrupt",
    )
    p.add_argument(
        "--min-members",
        type=int,
        default=3,
        help="Minimum members required to compute a mean when --allow-partial is set",
    )
    p.add_argument(
        "--log-missing",
        default="",
        help="CSV path to log missing or corrupt member files",
    )
    p.add_argument("--verbose", action="store_true", help="Print progress")
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a line every N horizons while filling an init",
    )

    # Parallelization
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    p.add_argument(
        "--parallel-by",
        choices=["day", "init", "feature"],
        default="day",
        help="Parallelization strategy (implemented: day)",
    )
    p.add_argument(
        "--io-threads",
        type=int,
        default=1,
        help="Number of I/O threads per worker (1 recommended for HDF5)",
    )
    p.add_argument(
        "--executor",
        choices=["process"],
        default="process",
        help="Day-level executor type. Use 'process' for HDF5 safety.",
    )

    return p.parse_args()

# -------------------------
# Helpers
# -------------------------
def ensure_dir(pth: str) -> None:
    os.makedirs(pth, exist_ok=True)

def build_rows_for_single_day(day_dt: datetime) -> List[datetime]:
    rows = []
    day = day_dt.strftime("%Y%m%d")
    for hh in VALID_CYCLES:
        rows.append(datetime.strptime(day + hh, "%Y%m%d%H"))
    return rows

def parse_horizons(spec: str) -> List[int]:
    try:
        rng, step = spec.split(":")
        a, b = rng.split("-")
        a, b, step = int(a), int(b), int(step)
        if a > b or step <= 0:
            raise ValueError
        return list(range(a, b + 1, step))
    except Exception:
        raise SystemExit("Invalid --horizons. Use start-end:step, for example 6-720:6")

def load_feature_ids(spec: str) -> List[int]:
    if os.path.isfile(spec):
        with open(spec) as f:
            ids = [int(s.strip()) for s in f if s.strip()]
    else:
        ids = [int(s) for s in spec.split(",") if s.strip()]
    if not ids:
        raise SystemExit("No feature_ids provided")
    return ids

def daterange(start_dt: datetime, end_dt: datetime):
    d = start_dt
    while d <= end_dt:
        yield d
        d += timedelta(days=1)

def is_valid_netcdf(path: str) -> bool:
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 512:
            return False
        with open(path, "rb") as f:
            head = f.read(8)
        return head.startswith(MAGIC_HDF5) or head[:3] == MAGIC_NETCDF
    except Exception:
        return False

def parse_fname(name: str):
    m = FNAME_RE.match(name)
    if not m:
        return None
    return m.group("hh"), int(m.group("fhr")), m.group("mem")

def index_local_for_day(input_dir: str, day: str, horizons_set: set) -> Dict[Tuple[str, int], Dict[str, str]]:
    idx: Dict[Tuple[str, int], Dict[str, str]] = {}
    for mem in ENSEMBLES:
        mem_dir = os.path.join(input_dir, day, mem)
        if not os.path.isdir(mem_dir):
            continue
        for base in os.listdir(mem_dir):
            parsed = parse_fname(base)
            if not parsed:
                continue
            hh, fhr, _ = parsed
            if hh not in VALID_CYCLES or fhr not in horizons_set:
                continue
            fp = os.path.join(mem_dir, base)
            idx.setdefault((hh, fhr), {})[mem] = fp
    return idx

def build_columns(horizons: List[int], mean_only: bool) -> List[str]:
    cols = []
    for fhr in horizons:
        cols.append(f"Ensemble_Mean_{fhr}h")
        if not mean_only:
            cols += [f"mem{i}_{fhr}h" for i in range(1, 5)]
    return cols

def feature_csv_path(output_dir: str, fid: int, span_tag: str) -> str:
    return os.path.join(output_dir, f"fid{fid}_{span_tag}.csv")

# ------------- locking -------------
class LockedAppend:
    def __init__(self, path: str):
        self.path = path
        self.fh = None
    def __enter__(self):
        dirn = os.path.dirname(self.path)
        if dirn:                           # <- add this guard
            os.makedirs(dirn, exist_ok=True)
        self.fh = open(self.path, "a+")
        fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        return self.fh
    def __exit__(self, exc_type, exc, tb):
        try:
            self.fh.flush()
            os.fsync(self.fh.fileno())
        finally:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            self.fh.close()

def append_df_locked(df: pd.DataFrame, out_csv: str):
    """Append DataFrame rows to CSV with a header if the file is empty."""
    with LockedAppend(out_csv) as fh:
        # Determine if file is empty while holding the lock
        fh.seek(0, os.SEEK_END)
        empty = fh.tell() == 0
        df.to_csv(fh, index=False, header=empty)

def log_missing_locked(rows: List[Dict], log_csv: str):
    if not rows or not log_csv:
        return
    df = pd.DataFrame(rows)
    with LockedAppend(log_csv) as fh:
        write_header = fh.tell() == 0
        df.to_csv(fh, index=False, header=write_header)

# -------------------------
# File reading
# -------------------------
def read_netcdf_file(file_path: str, feature_ids: List[int], engine_fallback: bool):
    try:
        try:
            ds = xr.open_dataset(file_path, engine="netcdf4")
        except Exception as e1:
            if engine_fallback:
                ds = xr.open_dataset(file_path, engine="h5netcdf")
            else:
                raise e1
        feat = xr.DataArray(feature_ids, dims=["feature_id"], name="feature_id")
        streamflow_data = ds.sel(feature_id=feat, drop=True)["streamflow"].values
        ds.close()
        return np.asarray(streamflow_data, dtype="float64")
    except Exception:
        return None  # treat as missing/corrupt

# -------------------------
# Worker-side logic
# -------------------------
def process_day_and_write(args_tuple):
    """
    Worker entrypoint: processes one day and writes outputs directly to CSVs.
    Returns ('ok', day) or ('err', traceback_text).
    """
    try:
        (day_dt, feature_ids, horizons, horizons_set, cols, args_dict, verbose, span_tag) = args_tuple

        day = day_dt.strftime("%Y%m%d")
        input_dir = args_dict["input_dir"]
        output_dir = args_dict["output_dir"]
        mean_only = args_dict["mean_only"]
        engine_fallback = args_dict["engine_fallback"]
        allow_partial = args_dict["allow_partial"]
        min_members = args_dict["min_members"]
        log_missing = args_dict["log_missing"]
        io_threads = args_dict["io_threads"]  # kept for CLI compatibility; we read sequentially

        # Build empty per-feature tables for the day
        day_rows = [datetime.strptime(day + hh, "%Y%m%d%H") for hh in VALID_CYCLES]
        per_fid_tables = {
            fid: pd.DataFrame(index=day_rows, columns=cols, dtype="float64") for fid in feature_ids
        }
        for fid in feature_ids:
            per_fid_tables[fid].index.name = "init_time_utc"

        idx = index_local_for_day(input_dir, day, horizons_set)

        missing_rows = []

        for hh in VALID_CYCLES:
            init_dt = datetime.strptime(day + hh, "%Y%m%d%H")

            # Build list of files that should exist for this init
            submissions = []
            for fhr in horizons:
                key = (hh, fhr)
                paths = idx.get(key, {})
                for mem in ENSEMBLES:
                    fp = paths.get(mem, "")
                    if fp and is_valid_netcdf(fp):
                        submissions.append((fhr, mem, fp))
                # If anything is missing for this fhr, we will log later

            # Sequential, safest for HDF5
            horizon_data: Dict[int, Dict[str, np.ndarray]] = {}
            for (fhr, mem, fp) in submissions:
                arr = read_netcdf_file(fp, feature_ids, engine_fallback)
                if arr is not None:
                    horizon_data.setdefault(fhr, {})[mem] = arr

            # Compute and fill
            for fhr in horizons:
                member_data = horizon_data.get(fhr, {})
                have = len(member_data)
                expected = 4
                if allow_partial:
                    if have < min_members:
                        # log insufficient members
                        if log_missing:
                            missing = [m for m in ENSEMBLES if m not in member_data]
                            missing_rows.append(
                                {
                                    "day": day,
                                    "hh": hh,
                                    "fhr": fhr,
                                    "have_members": have,
                                    "expected": expected,
                                    "missing": ";".join(missing),
                                }
                            )
                        continue
                else:
                    if have < expected:
                        if log_missing:
                            missing = [m for m in ENSEMBLES if m not in member_data]
                            missing_rows.append(
                                {
                                    "day": day,
                                    "hh": hh,
                                    "fhr": fhr,
                                    "have_members": have,
                                    "expected": expected,
                                    "missing": ";".join(missing),
                                }
                            )
                        continue

                # mean across available members
                member_arrays = list(member_data.values())
                mean_vals = np.mean(member_arrays, axis=0)

                for j, fid in enumerate(feature_ids):
                    per_fid_tables[fid].at[init_dt, f"Ensemble_Mean_{fhr}h"] = float(mean_vals[j])
                    if not mean_only:
                        for i, mem in enumerate(ENSEMBLES, 1):
                            col = f"mem{i}_{fhr}h"
                            if mem in member_data:
                                per_fid_tables[fid].at[init_dt, col] = float(member_data[mem][j])

            if verbose:
                print(f"[{day} {hh}] filled")

        # Write this day's rows for each feature to its CSV, with locks
        for fid, df_fid_day in per_fid_tables.items():
            out_csv = feature_csv_path(output_dir, fid, span_tag)
            df_out = df_fid_day.reset_index().sort_values("init_time_utc")
            append_df_locked(df_out, out_csv)

        # Write missing log, if any
        if missing_rows and log_missing:
            log_missing_locked(missing_rows, log_missing)

        if verbose:
            print(f"[{day}] written")
        return ("ok", day)
    except Exception as e:
        tb = traceback.format_exc()
        return ("err", f"{e}\n{tb}")

# -------------------------
# MAIN
# -------------------------
def main():
    args = parse_args()

    try:
        start_dt = datetime.strptime(args.start, "%Y%m%d")
        end_dt   = datetime.strptime(args.end,   "%Y%m%d")
    except ValueError:
        raise SystemExit("Dates must be YYYYMMDD")
    if start_dt > end_dt:
        raise SystemExit("Start must not exceed end")

    feature_ids = load_feature_ids(args.feature_ids)
    horizons    = parse_horizons(args.horizons)
    horizons_set = set(horizons)

    ensure_dir(args.input_dir)
    ensure_dir(args.output_dir)

    cols = build_columns(horizons, args.mean_only)
    span_tag = f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"

    n_workers = args.n_workers or mp.cpu_count() or 4
    print(f"Using {n_workers} processs with {args.parallel_by} parallelization")
    if args.parallel_by != "day":
        print("Only --parallel-by day is implemented. Using day.")

    days = list(daterange(start_dt, end_dt))
    print(f"Processing {len(days)} days in parallel...")

    args_dict = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "mean_only": args.mean_only,
        "engine_fallback": args.engine_fallback,
        "allow_partial": args.allow_partial,
        "min_members": args.min_members,
        "log_missing": args.log_missing,
        "io_threads": args.io_threads,  # kept for CLI compatibility
    }
    day_args = [
        (day_dt, feature_ids, horizons, horizons_set, cols, args_dict, args.verbose, span_tag)
        for day_dt in days
    ]

    ok = 0
    errors = []

    # Spawn context avoids HDF5/netCDF issues
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futs = [ex.submit(process_day_and_write, a) for a in day_args]
        for fut in as_completed(futs):
            try:
                status, payload = fut.result()
            except Exception as e:
                errors.append(f"top-level exception: {e}")
                continue
            if status == "ok":
                ok += 1
            else:
                errors.append(payload)

    print(f"Completed {ok}/{len(days)} days.")
    if errors:
        print("Some days failed. First few errors:")
        for e in errors[:5]:
            print(e)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
