#!/usr/bin/env python3
import argparse, os, re
from datetime import datetime
from typing import Iterable, List, Tuple, Optional
import multiprocessing as mp

import numpy as np
import pandas as pd

# ---------- filename pattern ----------
FNAME_RE = re.compile(
    r"^nwm\.t(?P<hh>\d{2})z\.long_range\.channel_rt_(?P<mem>\d)\.f(?P<fhr>\d{3})\.conus\.csv$"
)

MEM1_FOLDER = "long_range_mem1"

# ---------- args ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract mem1 for given station(s) across all folders and write a CSV per station."
    )
    p.add_argument("--root", required=True,
                   help="Root with date subfolders, e.g. /media/12TB/Fari-troute/NWM/output1")
    p.add_argument("--out", required=True, help="Output base folder")
    p.add_argument("--feature-ids", required=True,
                   help="Single feature_id, comma list, or a file with one id per line")
    p.add_argument("--cycles", default="00,06,12,18",
                   help="Comma separated cycles to include, default 00,06,12,18")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers to speed up CSV scanning, default 1")
    p.add_argument("--debug", action="store_true",
                   help="Print detailed info about matches/misses and first rows")
    p.add_argument("--verbose", action="store_true", help="Print progress")
    return p.parse_args()

# ---------- helpers ----------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_feature_ids(spec: str) -> List[int]:
    if os.path.isfile(spec):
        with open(spec) as f:
            ids = [int(s.strip()) for s in f if s.strip()]
    else:
        ids = [int(tok) for tok in spec.split(",") if tok.strip()]
    if not ids:
        raise SystemExit("No feature_ids provided")
    return ids

def vprint(flag: bool, *args):
    if flag:
        print(*args)

def find_date_dirs(root: str) -> List[str]:
    out = []
    for name in os.listdir(root):
        if len(name) == 8 and name.isdigit():
            if os.path.isdir(os.path.join(root, name)):
                out.append(name)
    return sorted(out)

def list_mem1_files(root: str, day: str, cycles: List[str]) -> List[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    mem_dir = os.path.join(root, day, MEM1_FOLDER)
    if not os.path.isdir(mem_dir):
        return out
    for base in os.listdir(mem_dir):
        m = FNAME_RE.match(base)
        if not m:
            continue
        if m.group("mem") != "1":
            continue
        hh = m.group("hh")
        if hh not in cycles:
            continue
        fhr = int(m.group("fhr"))
        out.append((os.path.join(mem_dir, base), hh, fhr))
    return out

# robust column resolver
def _resolve_cols(cols):
    norm = {c.strip().lower(): c for c in cols}
    # try common variants
    fid_key = None
    for cand in ("feature_id", "featureid", "station_id", "stationid"):
        if cand in norm:
            fid_key = norm[cand]
            break
    flow_key = None
    for cand in ("streamflow", "flow", "discharge", "q_out", "qout"):
        if cand in norm:
            flow_key = norm[cand]
            break
    return fid_key, flow_key

def read_value_for_feature(csv_path: str, feature_id: int, debug: bool = False,
                           chunksize: int = 200_000) -> Optional[float]:
    """
    Scan a large CSV in chunks and return the streamflow for the given feature_id.
    Returns None when missing or null.
    """
    try:
        # peek header only to resolve column names
        peek = pd.read_csv(csv_path, nrows=0)
        fid_col, flow_col = _resolve_cols(peek.columns)
        if fid_col is None or flow_col is None:
            if debug:
                print(f"[WARN] {os.path.basename(csv_path)} missing expected columns. "
                      f"Have: {list(peek.columns)}")
            return None

        for chunk in pd.read_csv(csv_path, usecols=[fid_col, flow_col], chunksize=chunksize):
            # normalize column names
            chunk = chunk.rename(columns={fid_col: "feature_id", flow_col: "streamflow"})
            # robust numeric cast
            chunk["feature_id"] = pd.to_numeric(chunk["feature_id"], errors="coerce").astype("Int64")
            sub = chunk[chunk["feature_id"] == feature_id]
            if not sub.empty:
                val = pd.to_numeric(sub["streamflow"], errors="coerce").iloc[0]
                if pd.isna(val):
                    return None
                return float(val)
        return None
    except Exception as e:
        if debug:
            print(f"[ERROR] Failed reading {csv_path}: {e}")
        return None

def _worker_process(args):
    csv_path, day, hh, fhr, feature_id, debug = args
    init_dt = datetime.strptime(day + hh, "%Y%m%d%H")
    val = read_value_for_feature(csv_path, feature_id, debug=debug)
    return (init_dt, fhr, val, csv_path)

# ---------- core ----------
def extract_mem1_for_station(root: str, out_dir: str, feature_id: int,
                             cycles: List[str], workers: int, verbose: bool, debug: bool):
    target_dir = os.path.join(out_dir, "ensemble_1")
    ensure_dir(target_dir)
    out_path = os.path.join(target_dir, f"station_{feature_id}.csv")

    date_dirs = find_date_dirs(root)
    vprint(verbose, f"Found {len(date_dirs)} date folders under {root}")

    tasks = []
    for day in date_dirs:
        files = list_mem1_files(root, day, cycles)
        for csv_path, hh, fhr in files:
            tasks.append((csv_path, day, hh, fhr, feature_id, debug))

    if not tasks:
        pd.DataFrame(columns=["init_time_utc"]).to_csv(out_path, index=False)
        vprint(verbose, f"No mem1 files found. Wrote empty CSV to {out_path}")
        return

    vprint(verbose, f"Scanning {len(tasks)} mem1 CSV files for feature_id {feature_id} with {workers} worker(s)")

    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(_worker_process, tasks, chunksize=64))
    else:
        results = [_worker_process(t) for t in tasks]

    # Count hits for quick diagnosis
    hits = [(dt, fhr, val, p) for (dt, fhr, val, p) in results if val is not None]
    if debug:
        print(f"[DEBUG] Hits: {len(hits)} / {len(results)}")
        # Show up to 5 examples of where we found values or missed
        for tup in hits[:5]:
            print(f"[DEBUG] hit: init={tup[0]} f{tup[1]:03d} from {os.path.basename(tup[3])} -> {tup[2]}")
        if len(hits) == 0:
            # print one example miss to help inspect columns
            any_path = results[0][3]
            try:
                peek = pd.read_csv(any_path, nrows=5)
                print(f"[DEBUG] sample head from {os.path.basename(any_path)}:\n{peek.head()}")
            except Exception as e:
                print(f"[DEBUG] could not sample {any_path}: {e}")

    if not results:
        pd.DataFrame(columns=["init_time_utc"]).to_csv(out_path, index=False)
        vprint(verbose, f"No values found for feature_id {feature_id}. Wrote empty CSV to {out_path}")
        return

    rows = []
    for init_dt, fhr, val, _ in results:
        rows.append((init_dt, fhr, np.nan if val is None else val))

    df_long = pd.DataFrame(rows, columns=["init_time_utc", "fhr", "streamflow"])
    df_long.sort_values(["init_time_utc", "fhr"], inplace=True)
    df_long = df_long.drop_duplicates(["init_time_utc", "fhr"], keep="first")

    df_wide = df_long.pivot(index="init_time_utc", columns="fhr", values="streamflow")
    df_wide.sort_index(inplace=True)
    df_wide.columns = [f"f{int(c):03d}" for c in df_wide.columns]
    df_wide.index.name = "init_time_utc"
    df_wide = df_wide.astype("float64")

    df_wide.reset_index().to_csv(out_path, index=False, na_rep="")
    vprint(verbose, f"Wrote {out_path} with {df_wide.shape[0]} rows and {df_wide.shape[1]} horizons")

def main():
    args = parse_args()
    ensure_dir(args.out)
    cycles = [c.strip() for c in args.cycles.split(",") if c.strip()]
    feature_ids = load_feature_ids(args.feature_ids)

    for fid in feature_ids:
        extract_mem1_for_station(
            root=args.root,
            out_dir=args.out,
            feature_id=fid,
            cycles=cycles,
            workers=max(1, args.workers),
            verbose=args.verbose,
            debug=args.debug,
        )

if __name__ == "__main__":
    main()
