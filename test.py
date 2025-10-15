#!/usr/bin/env python3
import argparse, os, re
from datetime import datetime
from typing import List, Tuple
import multiprocessing as mp
import pandas as pd

FNAME_RE = re.compile(r"^nwm\.t(?P<hh>\d{2})z\.long_range\.channel_rt_(?P<mem>\d)\.f(?P<fhr>\d{3})\.conus\.csv$")
MEM1_FOLDER = "long_range_mem1"

def parse_args():
    p = argparse.ArgumentParser(description="Find where given NWM feature_ids occur in mem1 CSVs.")
    p.add_argument("--root", required=True, help="Root with date subfolders")
    p.add_argument("--feature-ids", required=True, help="Comma list or file with one id per line")
    p.add_argument("--cycles", default="00,06,12,18", help="Cycles to include")
    p.add_argument("--limit", type=int, default=10, help="Max hits to print per feature_id")
    p.add_argument("--workers", type=int, default=8, help="Parallel workers")
    return p.parse_args()

def load_ids(spec: str) -> List[int]:
    if os.path.isfile(spec):
        with open(spec) as f: return [int(s.strip()) for s in f if s.strip()]
    return [int(tok) for tok in spec.split(",") if tok.strip()]

def find_date_dirs(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if len(d)==8 and d.isdigit() and os.path.isdir(os.path.join(root,d))])

def list_mem1_files(root: str, day: str, cycles: List[str]) -> List[Tuple[str,str,int]]:
    out=[]; mem_dir=os.path.join(root,day,MEM1_FOLDER)
    if not os.path.isdir(mem_dir): return out
    for base in os.listdir(mem_dir):
        m=FNAME_RE.match(base)
        if m and m.group("mem")=="1" and m.group("hh") in cycles:
            out.append((os.path.join(mem_dir,base), m.group("hh"), int(m.group("fhr"))))
    return out

def scan_file(args):
    csv_path, fid = args
    try:
        for chunk in pd.read_csv(csv_path, usecols=["feature_id"], dtype={"feature_id":"int64"}, chunksize=200_000):
            if (chunk["feature_id"] == fid).any():
                return csv_path
    except Exception:
        pass
    return None

def main():
    a = parse_args()
    ids = load_ids(a.feature_ids)
    cycles = [c.strip() for c in a.cycles.split(",") if c.strip()]
    days = find_date_dirs(a.root)

    # Build file list once
    files=[]
    for day in days:
        files.extend([p for (p,_,_) in list_mem1_files(a.root, day, cycles)])
    print(f"Scanning {len(files)} files across {len(days)} days...")

    for fid in ids:
        print(f"\n=== Searching for feature_id {fid} ===")
        hits=[]
        with mp.Pool(processes=max(1,a.workers)) as pool:
            for res in pool.imap_unordered(scan_file, [(p,fid) for p in files], chunksize=64):
                if res:
                    hits.append(res)
                    print(f"hit: {res}")
                    if len(hits) >= a.limit:
                        break
        if not hits:
            print("no occurrences found")

if __name__ == "__main__":
    main()
