# !pip install boto3 botocore xarray cfgrib==0.9.14.1 eccodes pandas numpy geopandas shapely pyproj rioxarray requests tqdm

import os, re, tempfile
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa
import geopandas as gpd
import requests
from shapely.geometry import shape,mapping
from shapely.ops import unary_union
from tqdm.auto import tqdm
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import warnings, logging, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

warnings.filterwarnings("ignore", category=FutureWarning, module="cfgrib")
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")

# Silence cfgrib/eccodes logging chatter
logging.getLogger("cfgrib").setLevel(logging.ERROR)
logging.getLogger("eccodes").setLevel(logging.ERROR)

# Some eccodes builds honor this env var for C-level logging
os.environ["GRIB_LOG_LEVEL"] = "error"   # safe no-op if unsupported
# ------------------------
# User inputs
# ------------------------
START_DATE = "2019-01-01"          # inclusive, UTC calendar date
END_DATE   = "2019-03-31"          # inclusive, UTC calendar date
CYCLES     = ("00","06","12","18") # which cycles to scan
MEMBERS    = (1,2,3,4)             # 1..4
OUT_ROOT   = Path("out_cfs_lead_ts")
OUT_ROOT = Path("out_cfs_lead_ts")
MAX_LEAD_HOURS = 720
d0 = pd.to_datetime(START_DATE).tz_localize("UTC")
d1 = (pd.to_datetime(END_DATE) + pd.Timedelta(days=1)).tz_localize("UTC")  # inclusive end
total_hours = min(int((d1 - d0).total_seconds() // 3600), MAX_LEAD_HOURS)
LEADS_H = list(range(6, total_hours + 1, 6))
LEADS_SET = set(LEADS_H)
MAX_WORKERS = 32  # tune if you like
S3_MAX_CONN = 32                          # S3 socket pool per worker

if __name__ == "__main__":
    print("Will write leads (hours):", LEADS_H)

# ------------------------
# S3 public client (NOAA CFS)
# ------------------------
BUCKET = "noaa-cfs-pds"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def _worker_init():
    # Each process gets its own unsigned S3 client with a larger connection pool
    global _S3
    _S3 = boto3.client(
        "s3",
        config=Config(signature_version=UNSIGNED, max_pool_connections=S3_MAX_CONN)
    )

def _process_one_key(args):
    """
    args = (key, basin_geoms_serialized)
    Returns: list of rows (dicts) for both basins, or [] on failure.
    """
    key, basins_payload = args
    rows = []
    try:
        # parse times & lead
        valid, init = parse_valid_init_from_key(key)
        lead_h = int((valid - init).total_seconds() // 3600)
        if lead_h not in LEADS_SET:
            return rows  # empty

        # download to temp file
        with tempfile.NamedTemporaryFile(suffix=".grb2", delete=False) as tf:
            tmp_path = tf.name
        try:
            _S3.download_file(BUCKET, key, tmp_path)
            with open(tmp_path, "rb") as f:
                grib = f.read()
        finally:
            try: os.remove(tmp_path)
            except Exception: pass

        # decode precip to mm/6h
        sixh = open_cfs_precip_6h(grib)

        # rebuild basin GeoSeries from lightweight payload
        for bname, geojson, crs_epsg in basins_payload:
            g = gpd.GeoSeries([shape(geojson)], crs=f"EPSG:{crs_epsg}")
            masked = mask_to_basin(sixh, g)
            mm = basin_mean_mm(masked)
            rows.append({
                "lead_h":  lead_h,
                "valid":   valid.isoformat(),
                "init":    init.isoformat(),
                "cycle":   key.split("/")[1],   # '00','06','12','18'
                "member":  int(key.split("/")[2][-2:]),
                "basin":   bname,
                "mm_6h":   float(mm),
            })
        return rows
    except Exception:
        return rows
    
def _serialize_basins(basins_dict):
    payload = []
    for bname, gs in basins_dict.items():
        g = gs.to_crs("EPSG:4326").geometry.iloc[0]
        payload.append((bname, mapping(g), 4326))
    return payload


def load_basin_from_gpkg(path: str, huc2_code: str) -> gpd.GeoSeries:
    """
    Load a single HUC2 basin polygon from a local WBD GPKG.

    - Uses the 'WBDHU2' layer (HUC2 polygons)
    - Filters by the 'huc2' attribute (lowercase)
    - Dissolves to a single geometry and returns as GeoSeries (EPSG:4326)
    """
    gdf = gpd.read_file(path, layer="WBDHU2")

    if gdf.empty:
        raise RuntimeError(f"No features found in layer 'WBDHU2' of {path}")

    if "huc2" not in gdf.columns:
        raise RuntimeError(f"Column 'huc2' not found in WBDHU2 layer of {path}")

    gdf_sel = gdf[gdf["huc2"] == huc2_code]

    if gdf_sel.empty:
        raise RuntimeError(f"No rows with huc2='{huc2_code}' in {path}")

    gdf_sel = gdf_sel.to_crs("EPSG:4326").dissolve()
    geom = gdf_sel.geometry.iloc[0]

    return gpd.GeoSeries([geom], crs="EPSG:4326")


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
        try: os.remove(tmp_path)
        except Exception: pass

def open_cfs_precip_6h(grib_bytes: bytes) -> xr.DataArray:
    if not (len(grib_bytes) >= 4 and grib_bytes[:4] == b"GRIB"):
        raise ValueError("Not a GRIB file")

    # PRATE, stepType=avg (common in flxf)
    try:
        ds = _open_cfgrib_from_bytes(
            grib_bytes, {"shortName": "prate", "typeOfLevel": "surface", "stepType": "avg"}
        )
        da = ds["prate"] * (6 * 3600.0)  # kg/m2/s -> mm/6h
        if "latitude" in da.coords:
            da = da.rename({"latitude": "lat", "longitude": "lon"})
        da = da.rename("cfs_prcp_6h")
        da.attrs["units"] = "mm/6h"
        return _wrap_lon_to_180(da)
    except Exception:
        pass

    # PRATE (looser)
    try:
        ds = _open_cfgrib_from_bytes(grib_bytes, {"shortName": "prate"})
        da = ds["prate"]
        factor = 6 * 3600.0
        if "step" in da.coords:
            try:
                step_sec = pd.to_timedelta(da["step"].values).astype("timedelta64[s]").astype(int)
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
                grib_bytes, {"shortName": sn, "typeOfLevel": "surface", "stepType": "accum"}
            )
        except Exception:
            try:
                ds = _open_cfgrib_from_bytes(grib_bytes, {"shortName": sn})
            except Exception:
                ds = None
        if ds is not None and ds.data_vars:
            v = sn if sn in ds.data_vars else list(ds.data_vars)[0]
            da = ds[v]
            if da.attrs.get("units", "").lower() in ("m", "meter", "metre", "meters", "metres"):
                da = da * 1000.0  # m -> mm
            if "latitude" in da.coords:
                da = da.rename({"latitude": "lat", "longitude": "lon"})
            da = da.rename("cfs_prcp_6h")
            da.attrs["units"] = "mm/6h"
            return _wrap_lon_to_180(da)

    # Last resort: show what's inside
    anyds = _open_cfgrib_from_bytes(grib_bytes, None)
    raise RuntimeError(f"Precip not found. Vars present: {list(anyds.data_vars)}")

def _wrap_lon_to_180(da: xr.DataArray) -> xr.DataArray:
    if "lon" in da.coords:
        lon = da["lon"].values
        if np.nanmax(lon) > 180.0:                 # 0..360 -> -180..180
            lon2 = ((lon + 180.0) % 360.0) - 180.0
            da = da.assign_coords(lon=xr.DataArray(lon2, dims=da["lon"].dims))
            da = da.sortby("lon")
    return da

def mask_to_basin(da: xr.DataArray, basin: gpd.GeoSeries) -> xr.DataArray:
    """
    Ensure CRS + spatial dims are set for rioxarray, then clip to the basin.
    Works whether the data uses (lat,lon) or (y,x).
    """
    # 1) normalize coord names to lat/lon if they came in as latitude/longitude
    ren = {}
    if "latitude" in da.coords:  ren["latitude"]  = "lat"
    if "longitude" in da.coords: ren["longitude"] = "lon"
    if ren:
        da = da.rename(ren)

    # 2) write CRS and declare which dims are spatial
    da = da.rio.write_crs("EPSG:4326", inplace=False)

    # common cases: (lat,lon) or (y,x)
    if ("lat" in da.dims and "lon" in da.dims):
        da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        # rioxarray expects y,x order, so transpose if needed
        if list(da.dims[-2:]) != ["lat", "lon"]:
            da = da.transpose(..., "lat", "lon")
    elif ("y" in da.dims and "x" in da.dims):
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
        if list(da.dims[-2:]) != ["y", "x"]:
            da = da.transpose(..., "y", "x")
    else:
        # last resort: try to infer dims from coords named lat/lon
        if "lat" in da.coords and "lon" in da.coords:
            # make sure lat/lon are dimensions (sometimes they’re 1D coords)
            if "lat" not in da.dims:
                da = da.assign_coords(lat=da["lat"]).expand_dims("lat" if da["lat"].ndim==0 else ())
            if "lon" not in da.dims:
                da = da.assign_coords(lon=da["lon"]).expand_dims("lon" if da["lon"].ndim==0 else ())
            da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
        else:
            raise RuntimeError(f"Cannot determine spatial dims from {da.dims} / {list(da.coords)}")

    # 3) clip (basin geometries must be EPSG:4326)
    basin = basin.to_crs("EPSG:4326")
    return da.rio.clip(basin.geometry.apply(mapping), basin.crs, drop=False)


def basin_mean_mm(masked: xr.DataArray) -> float:
    arr = masked.where(np.isfinite(masked))
    if "lat" not in arr.coords or "lon" not in arr.coords:
        return float("nan")
    w = np.cos(np.deg2rad(arr["lat"])).broadcast_like(arr)
    num = (arr * w).sum(dim=("lat","lon"), skipna=True)
    den = w.sum(dim=("lat","lon"), skipna=True)
    val = (num / den).values
    return float(val) if np.isfinite(val) else float("nan")


# ------------------------
# Path helpers
# ------------------------
FNAME_RE = re.compile(r"flxf(?P<valid>\d{10})\.(?P<mem>\d{2})\.(?P<init>\d{10})\.grb2$")
def parse_valid_init_from_key(key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    m = FNAME_RE.search(key)
    if not m:
        raise ValueError(f"cannot parse times from {key}")
    valid = pd.to_datetime(m.group("valid"), format="%Y%m%d%H", utc=True)
    init  = pd.to_datetime(m.group("init"),  format="%Y%m%d%H", utc=True)
    return valid, init

def s3_prefix(yyyymmdd: str, cycle: str, member: int) -> str:
    return f"cfs.{yyyymmdd}/{cycle}/6hrly_grib_{member:02d}/"

def out_csv_path(member: int, lead_h: int) -> Path:
    """One CSV per member+lead, no date folder."""
    p = OUT_ROOT / f"mem{member:02d}" / f"timeseries_{lead_h:03d}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def buffer_rows_init():
    """Return a dict: (member, lead_h) -> list[dict rows]."""
    return defaultdict(list)

def buffer_rows_add(buf, member: int, lead_h: int, valid_ts, init_ts, cycle, basin_name, mm_val):
    buf[(member, lead_h)].append({
        "time": valid_ts,       # valid (verification) time
        "init_utc": init_ts,    # cycle init time
        "cycle": cycle,         # '00','06','12','18'
        "basin": basin_name,    # if you’re writing both basins together; or drop this column
        "mm_6h": float(mm_val),
    })

def flush_buffer(buf):
    """Append rows to each target CSV exactly once per buffer (no overwrite)."""
    for (member, lead_h), rows in buf.items():
        if not rows:
            continue
        path = out_csv_path(member, lead_h)
        df = pd.DataFrame(rows)
        df.sort_values("time", inplace=True)
        write_header = not path.exists()
        df.to_csv(path, mode="a", header=write_header, index=False)
        print(f"  appended {len(df):4d} rows -> {path}")

def append_row_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=header, index=False)

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


def run(basins_payload):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for ymd in each_date(START_DATE, END_DATE):
        print(f"\nDate {ymd}")
        buf = buffer_rows_init()

        for cyc in CYCLES:
            for mem in MEMBERS:
                prefix = s3_prefix(ymd, cyc, mem)
                keys = sorted(list(list_keys_for(prefix)))
                if not keys:
                    continue

                ctx = get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=MAX_WORKERS,
                    mp_context=ctx,
                    initializer=_worker_init,
                ) as ex:
                    futures = [
                        ex.submit(_process_one_key, (k, basins_payload))
                        for k in keys
                    ]
                    for fut in tqdm(as_completed(futures),
                                    total=len(futures),
                                    leave=False,
                                    desc=f"{ymd} {cyc} mem{mem:02d}"):
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

        flush_buffer(buf)

    print("Done.")

  

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        filename="run.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Build HUC basins ONCE from local GPKG files
    BASINS = {
        "HUC2_07_UpperMiss": load_basin_from_gpkg(
            "/mnt/12TB/Sujan/LRF_RC/NWM_Q_to_GH/Hypothesis/Checker/WBD_07_HU2_GPKG.gpkg",
            huc2_code="07",
        ),
        "HUC2_10_Missouri": load_basin_from_gpkg(
            "/mnt/12TB/Sujan/LRF_RC/NWM_Q_to_GH/Hypothesis/Checker/WBD_10_HU2_GPKG.gpkg",
            huc2_code="10",
        ),
    }

    basins_payload = _serialize_basins(BASINS)

    print("Will write leads (hours):", LEADS_H)
    run(basins_payload)

