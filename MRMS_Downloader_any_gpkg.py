'''python mrms_forcing_from_gpkg.py \
  --start 2018,10,30,00 \
  --end   2018,12,31,23 \
  --gpkg  /path/to/your_basins.gpkg \
  --layer divides \
  --id-field id \
  --rawdir /media/12TB/Sujan/MRMS_raw \
  --outdir /media/12TB/Sujan/MRMS_out \
  --freq 6h \
  --tmpdir /media/12TB/Sujan/tmp
'''

#!/usr/bin/env python
import os
import gzip
import shutil
import logging
from pathlib import Path
from datetime import datetime

import requests
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from shapely.ops import transform
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import zonal_stats
import pyproj
import time
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

# ----------------------------------------------------------------------
# Download helpers
# ----------------------------------------------------------------------
def download_file(task):
    """
    Download a single MRMS GRIB2 file (.gz), decompress to .grib2,
    and return the path to the .grib2 file, or None on failure.
    """
    url, gz_file = task
    grib_file = gz_file.replace(".gz", "")

    if os.path.exists(grib_file):
        logging.info(f"Skipping download; file already exists: {grib_file}")
        return grib_file

    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 200:
            with open(gz_file, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            with gzip.open(gz_file, "rb") as f_in, open(grib_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(gz_file)
            logging.info(f"Downloaded and extracted: {grib_file}")
            return grib_file
        else:
            logging.warning(f"Failed to download {url} - status {r.status_code}")
            return None
    except Exception as e:
        logging.warning(f"Error downloading {url}: {e}")
        return None


# ----------------------------------------------------------------------
# Crop + zonal stats
# ----------------------------------------------------------------------
def crop_raster(task):
    """
    Crop a GRIB file to a given bounding box (in raster CRS).
    Returns path to cropped GRIB, or None on failure.
    """
    grib_file, bbox_raster_crs, path = task
    if grib_file is None or not os.path.exists(grib_file):
        return None

    cropped_file = os.path.join(path, "cropped_" + os.path.basename(grib_file))
    if os.path.exists(cropped_file):
        return cropped_file

    try:
        with rasterio.open(grib_file) as src:
            out_image, out_transform = mask(src, [bbox_raster_crs], crop=True)
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GRIB",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )

        with rasterio.open(cropped_file, "w", **out_meta) as dest:
            dest.write(out_image)

        logging.info(f"Cropped: {cropped_file}")
        return cropped_file
    except Exception as e:
        logging.warning(f"Error cropping {grib_file}: {e}")
        return None


def compute_zonal_stats(task):
    """
    Compute basin-mean precip for a single hourly file.

    task = (file_path, basin_shapes, valid_time)

    basin_shapes: list of (geometry, feature_id)
    Returns: list of dicts: {"time": ..., "feature_id": ..., "precip_mm_1h": ...}
    """
    file_path, basin_shapes, valid_time = task
    rows = []
    if file_path is None or not os.path.exists(file_path):
        return rows

    try:
        geoms = [g for g, _ in basin_shapes]
        ids   = [i for _, i in basin_shapes]

        stats = zonal_stats(
            geoms,
            file_path,
            stats="mean",
            all_touched=True,
            nodata=np.nan,
        )

        for fid, st in zip(ids, stats):
            val = st["mean"]
            if val is None or np.isnan(val):
                continue
            rows.append(
                {
                    "time":       valid_time,
                    "feature_id": str(fid),
                    "precip_mm_1h": float(val),
                }
            )

        return rows
    except Exception as e:
        logging.warning(f"Error computing zonal stats for {file_path}: {e}")
        return rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download MRMS 1H QPE, clip to polygons from a GPKG, "
                    "and compute basin-mean 6H precipitation."
    )

    parser.add_argument(
        "--start", "-s", required=True,
        help="Start date as 'YYYY,MM,DD,HH' (e.g. 2018,10,30,00)"
    )
    parser.add_argument(
        "--end", "-e", required=True,
        help="End date as 'YYYY,MM,DD,HH'"
    )
    parser.add_argument(
        "--gpkg", required=True,
        help="Path to input GeoPackage (.gpkg)"
    )
    parser.add_argument(
        "--layer", default=None,
        help="Layer name in the GeoPackage (default: first layer)"
    )
    parser.add_argument(
        "--id-field", default="id",
        help="Attribute field to use as feature ID (default: 'id')"
    )
    parser.add_argument(
        "--rawdir", default=None,
        help="Directory to store raw/cropped MRMS files "
             "(default: ./mrms_raw next to this script)"
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Directory to store output CSV "
             "(default: ./out_mrms next to this script)"
    )
    parser.add_argument(
        "--freq", default="6h",
        help="Aggregation frequency for resampling (default: '6h')"
    )
    parser.add_argument(
        "--tmpdir", default=None,
        help="Optional temp directory (e.g. on a large disk) for GDAL/tmp files."
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Temp dir override (important on clusters where / is nearly full)
    # ------------------------------------------------------------------
    if args.tmpdir is not None:
        os.makedirs(args.tmpdir, exist_ok=True)
        os.environ["TMPDIR"] = args.tmpdir
        os.environ["TMP"] = args.tmpdir
        os.environ["TEMP"] = args.tmpdir
        logging.info(f"Using {args.tmpdir} as temporary directory")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    here = Path(__file__).resolve().parent
    raw_dir = Path(args.rawdir) if args.rawdir else (here / "mrms_raw")
    out_dir = Path(args.outdir) if args.outdir else (here / "out_mrms")

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Parse dates
    # ------------------------------------------------------------------
    start_components = [int(x) for x in args.start.split(",")]
    end_components   = [int(x) for x in args.end.split(",")]
    start_date = datetime(*start_components)
    end_date   = datetime(*end_components)

    if end_date < start_date:
        raise ValueError("end date must be >= start date")

    logging.info(f"Time window: {start_date} to {end_date}")

    # ------------------------------------------------------------------
    # Read polygons from GPKG
    # ------------------------------------------------------------------
    if args.layer:
        gdf = gpd.read_file(args.gpkg, layer=args.layer)
    else:
        # Default: first layer
        gdf = gpd.read_file(args.gpkg)

    if gdf.empty:
        raise RuntimeError(f"No features found in {args.gpkg} (layer={args.layer})")

    if args.id_field not in gdf.columns:
        raise RuntimeError(f"ID field '{args.id_field}' not found in {args.gpkg} "
                           f"(available: {list(gdf.columns)})")

    gdf = gdf.to_crs("EPSG:4326")
    gdf[args.id_field] = gdf[args.id_field].astype(str)

    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    bbox_wgs84 = box(min_lon, min_lat, max_lon, max_lat)

    # ------------------------------------------------------------------
    # Build time sequences
    # ------------------------------------------------------------------
    tseq_hourly = pd.date_range(start=start_date, end=end_date, freq="h")

    # Daily loop for URL construction
    tseq_daily = pd.date_range(start=start_date, end=end_date, freq="D")

    # ------------------------------------------------------------------
    # Build download tasks
    # ------------------------------------------------------------------
    download_tasks = []
    for date in tseq_daily:
        year = date.year
        month = f"{date.month:02d}"
        day = f"{date.day:02d}"
        hours = [f"{h:02d}" for h in range(24)]

        # MRMS archive availability
        if date < datetime(2016, 1, 1):
            logging.info(f"Data not available for {date.date()}. Skipping day.")
            continue
        elif date <= datetime(2020, 10, 14):
            base_url = (
                "https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/mrms/ncep/"
                "GaugeCorr_QPE_01H/GaugeCorr_QPE_01H_00.00_"
                "{year}{month}{day}-{hour}0000.grib2.gz"
            )
        else:
            base_url = (
                "https://mtarchive.geol.iastate.edu/{year}/{month}/{day}/mrms/ncep/"
                "MultiSensor_QPE_01H_Pass2/MultiSensor_QPE_01H_Pass2_00.00_"
                "{year}{month}{day}-{hour}0000.grib2.gz"
            )

        for hour in hours:
            url = base_url.format(year=year, month=month, day=day, hour=hour)
            gz_file = raw_dir / f"{year}{month}{day}-{hour}.grib2.gz"
            download_tasks.append((url, str(gz_file)))

    # ------------------------------------------------------------------
    # Download all MRMS hourly files (in parallel for speed)
    # ------------------------------------------------------------------
    logging.info("Starting downloads...")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as executor:
        downloaded_files = list(executor.map(download_file, download_tasks))

    downloaded_files = [f for f in downloaded_files if f is not None]
    if not downloaded_files:
        logging.warning("No files downloaded successfully. Exiting.")
        return

    # ------------------------------------------------------------------
    # Figure out raster CRS and transform bbox + polygons
    # ------------------------------------------------------------------
    with rasterio.open(downloaded_files[0]) as src:
        raster_crs = src.crs

    # Transform bbox to raster CRS
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", raster_crs, always_xy=True
    )
    bbox_raster_crs = transform(
        transformer.transform,
        bbox_wgs84
    )

    # Transform polygons to raster CRS
    gdf_raster = gdf.to_crs(raster_crs)
    basin_shapes = [
        (geom, fid) for geom, fid in zip(gdf_raster.geometry, gdf_raster[args.id_field])
    ]

    # ------------------------------------------------------------------
    # Build expected hourly filenames (in raw_dir)
    # ------------------------------------------------------------------
    year  = tseq_hourly.year
    month = tseq_hourly.month.map("{:02d}".format)
    day   = tseq_hourly.day.map("{:02d}".format)
    hour  = tseq_hourly.hour.map("{:02d}".format)

    hourly_fnames = [
        raw_dir / f"{y}{m}{d}-{h}.grib2"
        for y, m, d, h in zip(year, month, day, hour)
    ]

    # ------------------------------------------------------------------
    # Process hour by hour: crop → zonal stats → delete
    # ------------------------------------------------------------------
    logging.info("Starting cropping + zonal statistics (hourly)...")
    all_rows = []

    for grib_file, valid_time in zip(hourly_fnames, tseq_hourly):
        grib_file = Path(grib_file)

        # skip hours that never downloaded
        if not grib_file.exists():
            continue

        # crop
        cropped = crop_raster((str(grib_file), bbox_raster_crs, str(raw_dir)))

        # zonal stats
        rows = compute_zonal_stats(
            (cropped, basin_shapes, valid_time.to_pydatetime())
        )
        all_rows.extend(rows)

        # delete raw + cropped
        try:
            if grib_file.exists():
                grib_file.unlink()
            if cropped and os.path.exists(cropped):
                os.remove(cropped)
        except Exception:
            pass

    if not all_rows:
        logging.warning("No zonal stats produced. Exiting.")
        return

    # ------------------------------------------------------------------
    # Build DataFrame + aggregate to 6h (or custom freq)
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    freq = args.freq.lower()  # e.g. "6h"
    logging.info(f"Aggregating to {freq} totals...")

    df_agg = (
        df.groupby("feature_id")
          .resample(freq)["precip_mm_1h"]
          .sum()
          .reset_index()
          .rename(columns={"precip_mm_1h": "precip_mm"})
    )

    # time label like 2018103000
    df_agg = df_agg.sort_values(["feature_id", "time"])
    df_agg["time_str"] = df_agg["time"].dt.strftime("%Y%m%d%H")

    # final columns
    out_cols = ["time_str", "feature_id", "precip_mm"]
    out_df = df_agg[out_cols]

    out_csv = out_dir / "mrms_forcing_from_gpkg.csv"
    out_df.to_csv(out_csv, index=False)

    elapsed = time.time() - start_date.timestamp()
    logging.info(f"Wrote {out_csv}")
    logging.info(f"Done in {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
