"""
# All 4 members, no ensemble, 1 month, one GPKG layer
python cfs_gpkg_forcing.py \
  --gpkg /path/to/basin.gpkg \
  --gpkg-layer WBDHU2 \
  --start-date 2018-11-30 \
  --end-date 2018-12-31 \
  --out-root /mnt/12TB/Sujan/CFS_GPKG_TS \
  --members all

# Only member 2, cycles 06 and 18
python cfs_gpkg_forcing.py \
  --gpkg /path/to/basin.gpkg \
  --gpkg-layer WBDHU2 \
  --start-date 2018-11-30 \
  --end-date 2018-12-31 \
  --cycles 06 18 \
  --members 2

# All members + ensemble mean, custom max lead and workers
python cfs_gpkg_forcing.py \
  --gpkg /path/to/basin.gpkg \
  --gpkg-layer WBDHU2 \
  --start-date 2018-11-30 \
  --end-date 2018-12-31 \
  --members all \
  --ensemble-mean \
  --max-lead-hours 720 \
  --max-workers 32
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download CFS 6-hourly precip, clip to a basin from a GPKG, and
write basin-mean mm/6h time series per lead time and member.

Outputs:
  OUT_ROOT/memXX/timeseries_006.csv
  ...
  and optionally OUT_ROOT/mem_ensmean/timeseries_006.csv (ensemble mean)
"""

import argparse
import os
import re
import tempfile
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import boto3
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from tqdm.auto import tqdm
import warnings
import logging

# ------------------------
# Warnings / logging noise
# ------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")

logging.getLogger("cfgrib").setLevel(logging.ERROR)
logging.getLogger("eccodes").setLevel(logging.ERROR)

os.environ["GRIB_LOG_LEVEL"] = "error"

# ------------------------
# S3 client (main process)
# ------------------------
BUCKET = "noaa-cfs-pds"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

# ------------------------
# Filename regex & helpers
# ------------------------
FNAME_RE = re.compile(
    r"flxf(?P<valid>\d{10})\.(?P<mem>\d{2})\.(?P<init>\d{10})\.grb2$"
)


def parse_valid_init_from_key(key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    m = FNAME_RE.search(key)
    if not m:
        raise ValueError(f"cannot parse times from {key}")
    valid = pd.to_datetime(m.group("valid"), format="%Y%m%d%H", utc=True)
    init = pd.to_datetime(m.group("init"), format="%Y%m%d%H", utc=True)
    return valid, init


def s3_prefix(yyyymmdd: str, cycle: str, member: int) -> str:
    return f"cfs.{yyyymmdd}/{cycle}/6hrly_grib_{member:02d}/"


# ------------------------
# cfgrib helpers
# ------------------------
def _open_cfgrib_from_bytes(grib_bytes: bytes, fkeys: dict | None = None) -> xr.Dataset:
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tf:
        tf.write(grib_bytes)
        tf.flush()
        tmp_path = tf.name
    try:
        backend_kwargs = {"indexpath": ":memory:"}
        if fkeys:
            backend_kwargs["filter_by_keys"] = fkeys
        with xr.open_dataset(tmp_path, engine="cfgrib", backend_kwargs=backend_kwargs) as ds:
            ds = ds.load()
        if not ds.data_vars:
            raise RuntimeError(f"Empty dataset for {fkeys}")
        return ds
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _wrap_lon_to_180(da: xr.DataArray) -> xr.DataArray:
    if "lon" in da.coords:
        lon = da["lon"].values
        if np.nanmax(lon) > 180.0:
            lon2 = ((lon + 180.0) % 360.0) - 180.0
            da = da.assign_coords(lon=xr.DataArray(lon2, dims=da["lon"].dims))
            da = da.sortby("lon")
    return da


def open_cfs_precip_6h(grib_bytes: bytes) -> xr.DataArray:
    """Return precip in mm/6h on (lat, lon)."""

    if not (len(grib_bytes) >= 4 and grib_bytes[:4] == b"GRIB"):
        raise ValueError("Not a GRIB file")

    # PRATE, stepType=avg (common in flxf)
    try:
        ds = _open_cfgrib_from_bytes(
            grib_bytes,
            {"shortName": "prate", "typeOfLevel": "surface", "stepType": "avg"},
        )
        da = ds["prate"] * (6 * 3600.0)  # kg/m2/s -> mm/6h
        if "latitude" in da.coords:
            da = da.rename({"latitude": "lat", "longitude": "lon"})
        da = da.rename("cfs_prcp_6h")
        da.attrs["units"] = "mm/6h"
        return _wrap_lon_to_180(da)
    except Exception:
        pass

    # PRATE, any stepType
    try:
        ds = _open_cfgrib_from_bytes(grib_bytes, {"shortName": "prate"})
        da = ds["prate"]
        factor = 6 * 3600.0
        if "step" in da.coords:
            try:
                step_sec = (
                    pd.to_timedelta(da["step"].values)
                    .astype("timedelta64[s]")
                    .astype(int)
                )
                if np.ndim(step_sec) > 0:
                    step_sec = int(np.array(step_sec).ravel()[0])
                factor = float(step_sec)
            except Exception:
                pass
        da = da * factor
        if "latitude" in da.coords:
            da = da.rename({"latitude": "lat", "longitude": "lon"})
        da = da.rename("cfs_prcp_6h")
        da.attrs["units"] = "mm/6h"
        return _wrap_lon_to_180(da)
    except Exception:
        pass

    # APCP/TP accumulated
    for sn in ("apcp", "tp"):
        try:
            ds = _open_cfgrib_from_bytes(
                grib_bytes,
                {"shortName": sn, "typeOfLevel": "surface", "stepType": "accum"},
            )
        except Exception:
            try:
                ds = _open_cfgrib_from_bytes(grib_bytes, {"shortName": sn})
            except Exception:
                ds = None
        if ds is not None and ds.data_vars:
            v = sn if sn in ds.data_vars else list(ds.data_vars)[0]
            da = ds[v]
            if da.attrs.get("units", "").lower() in (
                "m",
                "meter",
                "metre",
                "meters",
                "metres",
            ):
                da = da * 1000.0  # m -> mm
            if "latitude" in da.coords:
                da = da.rename({"latitude": "lat", "longitude": "lon"})
            da = da.rename("cfs_prcp_6h")
            da.attrs["units"] = "mm/6h"
            return _wrap_lon_to_180(da)

    anyds = _open_cfgrib_from_bytes(grib_bytes, None)
    raise RuntimeError(f"Precip not found. Vars present: {list(anyds.data_vars)}")


# ------------------------
# Clipping / basin mean
# ------------------------
def mask_to_basin(da: xr.DataArray, basin: gpd.GeoSeries) -> xr.DataArray:
    """Clip precip field to basin geometry using rioxarray."""
    # Normalize to lat/lon names
    ren = {}
    if "latitude" in da.coords:
        ren["latitude"] = "lat"
    if "longitude" in da.coords:
        ren["longitude"] = "lon"
    if ren:
        da = da.rename(ren)

    da = da.rio.write_crs("EPSG:4326", inplace=False)

    if "lat" in da.dims and "lon" in da.dims:
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        if list(da.dims[-2:]) != ["lat", "lon"]:
            da = da.transpose(..., "lat", "lon")
    elif "y" in da.dims and "x" in da.dims:
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        if list(da.dims[-2:]) != ["y", "x"]:
            da = da.transpose(..., "y", "x")
    else:
        raise RuntimeError(f"Cannot determine spatial dims from {da.dims}")

    basin = basin.to_crs("EPSG:4326")
    return da.rio.clip(basin.geometry.apply(mapping), basin.crs, drop=False)


def basin_mean_mm(masked: xr.DataArray) -> float:
    arr = masked.where(np.isfinite(masked))
    if "lat" not in arr.coords or "lon" not in arr.coords:
        return float("nan")
    w = np.cos(np.deg2rad(arr["lat"])).broadcast_like(arr)
    num = (arr * w).sum(dim=("lat", "lon"), skipna=True)
    den = w.sum(dim=("lat", "lon"), skipna=True)
    val = (num / den).values
    return float(val) if np.isfinite(val) else float("nan")


# ------------------------
# GPKG basin loading
# ------------------------
def load_basin_from_gpkg(path: str, layer: str | None = None) -> gpd.GeoSeries:
    """
    Load a basin polygon from local GPKG and return dissolved geometry
    in EPSG:4326.
    """
    if layer:
        gdf = gpd.read_file(path, layer=layer)
    else:
        # default layer (first) – if there are many, user should supply --gpkg-layer
        gdf = gpd.read_file(path)

    if gdf.empty:
        raise RuntimeError(f"No features found in {path} (layer={layer})")

    gdf = gdf.to_crs("EPSG:4326")
    geom = unary_union(gdf.geometry)
    return gpd.GeoSeries([geom], crs="EPSG:4326")


def _serialize_basins(basins_dict):
    """Turn basins into lightweight payload for workers."""
    payload = []
    for bname, gs in basins_dict.items():
        g = gs.to_crs("EPSG:4326").geometry.iloc[0]
        payload.append((bname, mapping(g), 4326))
    return payload


# ------------------------
# Output helpers
# ------------------------
def out_csv_path(out_root: Path, member_label: str, lead_h: int) -> Path:
    """
    member_label: 'mem01', 'mem02', ..., or 'mem_ensmean'
    """
    p = out_root / member_label / f"timeseries_{lead_h:03d}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def buffer_rows_init():
    """Return a dict: (member, lead_h) -> list[dict rows]."""
    return defaultdict(list)


def buffer_rows_add(buf, member: int, lead_h: int, valid_ts, init_ts, cycle, basin_name, mm_val):
    buf[(member, lead_h)].append(
        {
            "time": valid_ts,  # valid time (ISO string)
            "init_utc": init_ts,
            "cycle": cycle,
            "basin": basin_name,
            "mm_6h": float(mm_val),
        }
    )


def flush_buffer(
    buf,
    out_root: Path,
    members_for_ens: list[int] | None = None,
    write_ensemble: bool = False,
):
    """
    Write per-member CSVs, and optionally ensemble mean over members_for_ens.
    """
    # 1) write per-member CSVs
    for (member, lead_h), rows in buf.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.sort_values("time", inplace=True)
        member_label = f"mem{member:02d}"
        path = out_csv_path(out_root, member_label, lead_h)
        write_header = not path.exists()
        df.to_csv(path, mode="a", header=write_header, index=False)
        print(f"  appended {len(df):4d} rows -> {path}")

    if not write_ensemble or not members_for_ens:
        return

    # 2) ensemble mean over selected members
    records = []
    for (member, lead_h), rows in buf.items():
        for r in rows:
            rr = dict(r)
            rr["member"] = member
            rr["lead_h"] = lead_h
            records.append(rr)

    if not records:
        return

    df_all = pd.DataFrame(records)
    ens_df = df_all[df_all["member"].isin(members_for_ens)].copy()
    if ens_df.empty:
        return

    ens_rows_by_lead = defaultdict(list)

    # group by lead_h, time, basin
    for (lead_h, time, basin), sub in ens_df.groupby(["lead_h", "time", "basin"]):
        mm_mean = sub["mm_6h"].mean()
        # init_utc and cycle should be identical across members for the same key
        init_utc = sub["init_utc"].iloc[0]
        cycle = sub["cycle"].iloc[0]
        ens_rows_by_lead[lead_h].append(
            {
                "time": time,
                "init_utc": init_utc,
                "cycle": cycle,
                "basin": basin,
                "mm_6h": float(mm_mean),
            }
        )

    for lead_h, rows in ens_rows_by_lead.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.sort_values("time", inplace=True)
        path = out_csv_path(out_root, "mem_ensmean", lead_h)
        write_header = not path.exists()
        df.to_csv(path, mode="a", header=write_header, index=False)
        print(f"  appended {len(df):4d} rows -> {path} (ensemble)")


# ------------------------
# Multiprocessing worker
# ------------------------
_S3 = None  # per-process S3


def _worker_init(max_conn: int):
    global _S3
    _S3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, max_pool_connections=max_conn),
    )


def _process_one_key(args):
    """
    args = (key, basins_payload, leads_set)
    Returns list of rows dicts for each basin.
    """
    key, basins_payload, leads_set = args
    rows = []
    try:
        valid, init = parse_valid_init_from_key(key)
        lead_h = int((valid - init).total_seconds() // 3600)
        if lead_h not in leads_set:
            return rows

        # download GRIB
        with tempfile.NamedTemporaryFile(suffix=".grb2", delete=False) as tf:
            tmp_path = tf.name
        try:
            _S3.download_file(BUCKET, key, tmp_path)
            with open(tmp_path, "rb") as f:
                grib = f.read()
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        sixh = open_cfs_precip_6h(grib)

        # reconstruct basins locally
        for bname, geojson, crs_epsg in basins_payload:
            g = gpd.GeoSeries([shape(geojson)], crs=f"EPSG:{crs_epsg}")
            masked = mask_to_basin(sixh, g)
            mm = basin_mean_mm(masked)
            rows.append(
                {
                    "lead_h": lead_h,
                    "valid": valid.isoformat(),
                    "init": init.isoformat(),
                    "cycle": key.split("/")[1],
                    "member": int(key.split("/")[2][-2:]),
                    "basin": bname,
                    "mm_6h": float(mm),
                }
            )
        return rows
    except Exception:
        # swallow worker errors; they just contribute no rows
        return rows


# ------------------------
# Drivers
# ------------------------
def each_date(start_date: str, end_date: str):
    d0 = pd.to_datetime(start_date).date()
    d1 = pd.to_datetime(end_date).date()
    cur = d0
    while cur <= d1:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def list_keys_for(prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = obj["Key"]
            if "/flxf" in k and k.endswith(".grb2"):
                yield k


def parse_members_arg(members_str: str) -> list[int]:
    members_str = members_str.strip().lower()
    if members_str == "all":
        return [1, 2, 3, 4]
    parts = [p.strip() for p in members_str.split(",") if p.strip()]
    out = sorted({int(p) for p in parts})
    for m in out:
        if m not in (1, 2, 3, 4):
            raise ValueError("members must be in {1,2,3,4} or 'all'")
    return out


def run(
    start_date: str,
    end_date: str,
    cycles: list[str],
    members: list[int],
    out_root: Path,
    max_lead_hours: int,
    max_workers: int,
    basins_payload,
    write_ensemble: bool,
):
    out_root.mkdir(parents=True, exist_ok=True)

    # compute allowed lead times based on date span (same logic as before)
    d0 = pd.to_datetime(start_date).tz_localize("UTC")
    d1 = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).tz_localize("UTC")
    total_hours = min(int((d1 - d0).total_seconds() // 3600), max_lead_hours)
    leads_h = list(range(6, total_hours + 1, 6))
    leads_set = set(leads_h)
    print("Will write leads (hours):", leads_h)

    for ymd in each_date(start_date, end_date):
        print(f"\nDate {ymd}")
        buf = buffer_rows_init()

        for cyc in cycles:
            for mem in members:
                prefix = s3_prefix(ymd, cyc, mem)
                keys = sorted(list(list_keys_for(prefix)))
                if not keys:
                    continue

                ctx = get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=ctx,
                    initializer=_worker_init,
                    initargs=(32,),  # S3 max connections per worker
                ) as ex:
                    futures = [
                        ex.submit(_process_one_key, (k, basins_payload, leads_set))
                        for k in keys
                    ]
                    for fut in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        leave=False,
                        desc=f"{ymd} {cyc} mem{mem:02d}",
                    ):
                        rows = fut.result()
                        for r in rows:
                            buffer_rows_add(
                                buf,
                                member=r["member"],
                                lead_h=r["lead_h"],
                                valid_ts=r["valid"],
                                init_ts=r["init"],
                                cycle=r["cycle"],
                                basin_name=r["basin"],
                                mm_val=r["mm_6h"],
                            )

        flush_buffer(
            buf,
            out_root=out_root,
            members_for_ens=members,
            write_ensemble=write_ensemble,
        )

    print("Done.")


# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract CFS precip forcing for a basin from GPKG."
    )
    parser.add_argument("--gpkg", required=True, help="Path to basin GPKG")
    parser.add_argument(
        "--gpkg-layer",
        default=None,
        help="Layer name in GPKG (if omitted, first layer is used)",
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD UTC")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD UTC")
    parser.add_argument(
        "--cycles",
        nargs="+",
        default=["00", "06", "12", "18"],
        help="List of CFS cycles to use, e.g. --cycles 06 18",
    )
    parser.add_argument(
        "--members",
        default="all",
        help="'all' or comma-separated list, e.g. '1,2,4'",
    )
    parser.add_argument(
        "--ensemble-mean",
        action="store_true",
        help="Also write ensemble mean timeseries over selected members",
    )
    parser.add_argument(
        "--out-root",
        default="out_cfs_gpkg_ts",
        help="Output root directory",
    )
    parser.add_argument(
        "--max-lead-hours",
        type=int,
        default=720,
        help="Maximum forecast lead in hours to keep (default 720)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Max worker processes for decoding GRIB (default 16)",
    )

    args = parser.parse_args()

    # Logging to file if you want
    logging.basicConfig(
        filename="cfs_gpkg_forcing.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    members = parse_members_arg(args.members)
    cycles = [c.zfill(2) for c in args.cycles]

    print(f"Using members: {members}")
    print(f"Using cycles: {cycles}")

    # Load basin once from GPKG
    basin = load_basin_from_gpkg(args.gpkg, layer=args.gpkg_layer)
    basins = {"BASIN": basin}
    basins_payload = _serialize_basins(basins)

    out_root = Path(args.out_root)

    run(
        start_date=args.start_date,
        end_date=args.end_date,
        cycles=cycles,
        members=members,
        out_root=out_root,
        max_lead_hours=args.max_lead_hours,
        max_workers=args.max_workers,
        basins_payload=basins_payload,
        write_ensemble=args.ensemble_mean,
    )


if __name__ == "__main__":
    main()
