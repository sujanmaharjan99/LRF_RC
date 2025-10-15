#!/usr/bin/env python3
import argparse, os, re, time, threading
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/media/12TB/Sujan/NWM/Codes/Key/steady-library-470316-r2-5ea9851180c5.json"

# -------------------------
# Constants
# -------------------------
BUCKET = "national-water-model"
VALID_CYCLES = ["00", "06", "12", "18"]
ENSEMBLES = ["long_range_mem1", "long_range_mem2", "long_range_mem3", "long_range_mem4"]
FNAME_RE = re.compile(r"^nwm\.t(?P<hh>\d{2})z\.long_range\.channel_rt_(?P<mem>\d)\.f(?P<fhr>\d{3})\.conus\.nc$")

MAGIC_HDF5 = b"\x89HDF\r\n\x1a\n"
MAGIC_NETCDF = b"CDF"

# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Download NWM LR channel_rt mem1-4 files from GCS to local, with validation and skip-if-exists."
    )
    p.add_argument("--start", required=True, help="YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYYMMDD")
    p.add_argument("--input-dir", default="/media/12TB/Sujan/NWM/Data", help="Local root download directory")
    p.add_argument("--cycles", default="00,06,12,18", help="Comma-separated cycles to keep, default 00,06,12,18")
    p.add_argument("--horizons", default="6-720:6",
                   help="Range spec start-end:step. Default 6-720:6. Example 0-720:6 to include f000")
    p.add_argument("--force-redownload", action="store_true", help="Redownload even if file exists")
    p.add_argument("--revalidate-existing", action="store_true",
                   help="Validate existing files and redownload if invalid")
    p.add_argument("--max-retries", type=int, default=3, help="Retries per file if invalid")
    p.add_argument("--workers", type=int, default=16, help="Concurrent downloads (threads). Good values: 8–32.")
    p.add_argument("--verbose", action="store_true", help="Print progress")
    return p.parse_args()

# -------------------------
# Helpers
# -------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def daterange(start_dt: datetime, end_dt: datetime):
    d = start_dt
    while d <= end_dt:
        yield d
        d += timedelta(days=1)

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

def valid_file_name(name: str, cycles_set, horizons_set) -> bool:
    m = FNAME_RE.match(name)
    if not m:
        return False
    hh = m.group("hh")
    fhr = int(m.group("fhr"))
    return (hh in cycles_set) and (fhr in horizons_set)

def is_valid_netcdf(path: str) -> bool:
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 512:
            return False
        with open(path, "rb") as f:
            head = f.read(8)
        return head.startswith(MAGIC_HDF5) or head[:3] == MAGIC_NETCDF
    except Exception:
        return False

# -------------------------
# Thread-local GCS client
# -------------------------
_thread_local = threading.local()

def get_bucket() -> storage.Bucket:
    if not hasattr(_thread_local, "client"):
        _thread_local.client = storage.Client()
        _thread_local.bucket = _thread_local.client.bucket(BUCKET)
    return _thread_local.bucket

# -------------------------
# Download worker
# -------------------------
def download_one(blob_name: str,
                 local_path: str,
                 retries: int,
                 verbose: bool) -> Tuple[str, bool]:
    """
    Returns (action, ok)
      action ∈ {'downloaded','redownloaded'}
      ok ∈ {True, False}
    """
    bucket = get_bucket()
    blob = bucket.blob(blob_name)

    tmp_path = local_path + ".part"

    for attempt in range(1, retries + 1):
        try:
            # Ensure parent dir exists
            ensure_dir(os.path.dirname(local_path))
            # Stream directly to a temp file
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except Exception: pass
            blob.download_to_filename(tmp_path)

            if is_valid_netcdf(tmp_path):
                # Atomic replace
                os.replace(tmp_path, local_path)
                if verbose:
                    print(f"[ok] {os.path.basename(local_path)}")
                # Decide action based on whether file existed
                action = "redownloaded" if os.path.exists(local_path) else "downloaded"
                return (action, True)
            else:
                if verbose:
                    print(f"[warn] invalid after download (attempt {attempt}): {os.path.basename(local_path)}")
                try: os.remove(tmp_path)
                except Exception: pass
                time.sleep(min(5, 1.5 ** attempt))
        except Exception as e:
            if verbose:
                print(f"[err] {os.path.basename(local_path)} attempt {attempt}: {e}")
            try: os.remove(tmp_path)
            except Exception: pass
            time.sleep(min(5, 1.5 ** attempt))

    return ("downloaded", False)  # semantics: a fresh fetch that failed

# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    try:
        start_dt = datetime.strptime(args.start, "%Y%m%d")
        end_dt = datetime.strptime(args.end, "%Y%m%d")
    except ValueError:
        raise SystemExit("Dates must be YYYYMMDD")
    if start_dt > end_dt:
        raise SystemExit("Start must not exceed end")

    cycles = [c.strip() for c in args.cycles.split(",") if c.strip()]
    cycles_set = set(cycles)
    horizons = parse_horizons(args.horizons)
    horizons_set = set(horizons)

    ensure_dir(args.input_dir)

    # A single client for listing in the main thread
    list_client = storage.Client()
    bucket = list_client.bucket(BUCKET)

    total = 0
    scheduled = 0
    downloaded = 0
    redownloaded = 0
    skipped = 0
    failed = 0

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for day_dt in daterange(start_dt, end_dt):
            day = day_dt.strftime("%Y%m%d")
            if args.verbose:
                print(f"== {day} ==")
            for mem in ENSEMBLES:
                prefix = f"nwm.{day}/{mem}/"
                # list_blobs returns an iterator; filter by name pattern
                for b in bucket.list_blobs(prefix=prefix):
                    if b.name.endswith("/"):
                        continue
                    base = os.path.basename(b.name)
                    if not valid_file_name(base, cycles_set, horizons_set):
                        continue

                    total += 1
                    local_dir = os.path.join(args.input_dir, day, mem)
                    local_path = os.path.join(local_dir, base)

                    # Decide whether to schedule a task
                    if os.path.exists(local_path):
                        if args.force_redownload:
                            futures.append(pool.submit(download_one, b.name, local_path, args.max_retries, args.verbose))
                            scheduled += 1
                        elif args.revalidate_existing and not is_valid_netcdf(local_path):
                            if args.verbose:
                                print(f"[redo] {base}")
                            futures.append(pool.submit(download_one, b.name, local_path, args.max_retries, args.verbose))
                            scheduled += 1
                        else:
                            if args.verbose:
                                print(f"[skip] {base}")
                            skipped += 1
                    else:
                        futures.append(pool.submit(download_one, b.name, local_path, args.max_retries, args.verbose))
                        scheduled += 1

        # Collect results as they finish
        for fut in as_completed(futures):
            try:
                action, ok = fut.result()
                if ok:
                    if action == "redownloaded":
                        redownloaded += 1
                    else:
                        downloaded += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

    print(f"Done. total candidates: {total}, scheduled: {scheduled}, downloaded: {downloaded}, "
          f"redownloaded: {redownloaded}, skipped: {skipped}, failed: {failed}")

if __name__ == "__main__":
    main()
