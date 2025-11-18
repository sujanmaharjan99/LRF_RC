import os
import requests
import gzip
import shutil
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime
import optparse
from rasterstats import zonal_stats
import pyproj
import time
import concurrent.futures
import logging
import multiprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ----------------------------------------------------------------------
# Basin helpers (Upper Mississippi = HUC2 07, Missouri = HUC2 10)
# ----------------------------------------------------------------------
def load_basin_from_gpkg(path: str, huc2_code: str) -> gpd.GeoSeries:
    """
    Load a single HUC2 polygon from a WBD GPKG (layer WBDHU2, column 'huc2').
    Returns a GeoSeries with one dissolved polygon in EPSG:4326.
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


def build_basins() -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame with two rows:
      - HUC2_07_UpperMiss
      - HUC2_10_Missouri
    using your WBD GPKGs.
    """
    upper_miss = load_basin_from_gpkg(
        "./Checker/WBD_07_HU2_GPKG.gpkg",
        huc2_code="07",
    )

    missouri = load_basin_from_gpkg(
        "./Checker/WBD_10_HU2_GPKG.gpkg",
        huc2_code="10",
    )

    gdf = gpd.GeoDataFrame(
        {
            "basin": ["HUC2_07_UpperMiss", "HUC2_10_Missouri"],
            "geometry": [upper_miss.geometry.iloc[0], missouri.geometry.iloc[0]],
        },
        crs="EPSG:4326",
    )
    return gdf


# ----------------------------------------------------------------------
# Download + crop
# ----------------------------------------------------------------------
def download_file(task):
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


def crop_raster(task):
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


# ----------------------------------------------------------------------
# Zonal stats → per-basin hourly means
# ----------------------------------------------------------------------
def compute_zonal_stats(task):
    """
    Return a list of rows:
        {"time": datetime, "basin": name, "precip_mm": value}
    for this one hourly file.
    """
    file_path, basin_shapes, valid_time = task
    rows = []
    if file_path is None or not os.path.exists(file_path):
        return rows

    try:
        geoms = [g for g, _ in basin_shapes]
        names = [n for _, n in basin_shapes]

        stats = zonal_stats(
            geoms,
            file_path,
            stats="mean",
            all_touched=True,
            nodata=np.nan,
        )

        for name, st in zip(names, stats):
            val = st["mean"]
            if val is None or np.isnan(val):
                continue
            # MRMS 1H fields are mm over the previous hour; mean over basin is fine
            rows.append(
                {
                    "time": valid_time,
                    "basin": name,
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
    start_time = time.time()

    usage = "python %prog -p <path> -o <outdir> -s <start> -e <end>"
    p = optparse.OptionParser(usage=usage)
    p.add_option("--path", "-p", help="Path to input GRIB files directory.")
    p.add_option("--outdir", "-o", help="Path to output directory for CSV.")
    p.add_option("--start", "-s", help="Start date in format YYYY,MM,DD,HH")
    p.add_option("--end", "-e", help="End date in format YYYY,MM,DD,HH")

    options, arguments = p.parse_args()

    if None in [options.path, options.outdir, options.start, options.end]:
        p.print_help()
        return

    path = options.path
    outdir = options.outdir
    start_components = [int(x) for x in options.start.split(",")]
    end_components = [int(x) for x in options.end.split(",")]
    start_date = datetime(*start_components)
    end_date = datetime(*end_components)

    os.makedirs(path, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    # 1) build basin polygons (Upper Mississippi + Missouri)
    basins = build_basins()          # EPSG:4326
    min_lon, min_lat, max_lon, max_lat = basins.total_bounds
    bbox_wgs84 = box(min_lon, min_lat, max_lon, max_lat)

    # 2) build time sequences (hourly)
    tseq_hourly = pd.date_range(start=start_date, end=end_date, freq="H")

    # 3) build download tasks (same logic as your original: GaugeCorr / MultiSensor)
    download_tasks = []
    tseq_daily = pd.date_range(start=start_date, end=end_date, freq="D")

    for date in tseq_daily:
        year = date.year
        month = f"{date.month:02d}"
        day = f"{date.day:02d}"
        hours = [f"{h:02d}" for h in range(24)]

        if date < datetime(2016, 1, 1):
            logging.info(f"Data not available for {date}. Skipping day.")
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
            gz_file = os.path.join(path, f"{year}{month}{day}-{hour}.grib2.gz")
            download_tasks.append((url, gz_file))

    logging.info("Starting downloads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        downloaded_files = list(
            executor.map(download_file, download_tasks)
        )

    downloaded_files = [f for f in downloaded_files if f is not None]
    if not downloaded_files:
        logging.warning("No files downloaded successfully. Exiting.")
        return

    # 4) figure out raster CRS and transform bbox/basins
    with rasterio.open(downloaded_files[0]) as src:
        raster_crs = src.crs

    project = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True).transform
    bbox_raster_crs = project(*bbox_wgs84.exterior.coords.xy)
    bbox_raster_crs = box(min(bbox_raster_crs[0]), min(bbox_raster_crs[1]),
                          max(bbox_raster_crs[0]), max(bbox_raster_crs[1]))

    basins_raster = basins.to_crs(raster_crs)
    basin_shapes = [
        (geom, name) for geom, name in zip(basins_raster.geometry, basins_raster["basin"])
    ]

    # 5) crop all hourly rasters to bbox
    year = tseq_hourly.year
    month = tseq_hourly.month.map("{:02d}".format)
    day = tseq_hourly.day.map("{:02d}".format)
    hour = tseq_hourly.hour.map("{:02d}".format)
    hourly_fnames = [
        os.path.join(path, f"{y}{m}{d}-{h}.grib2") for y, m, d, h in zip(year, month, day, hour)
    ]

    logging.info("Starting cropping...")
    crop_tasks = [(f, bbox_raster_crs, path) for f in hourly_fnames]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        cropped_files = list(executor.map(crop_raster, crop_tasks))

    # 6) zonal stats: basin-mean hourly precip
    logging.info("Starting zonal statistics...")
    zonal_tasks = [
        (cf, basin_shapes, ts.to_pydatetime())
        for cf, ts in zip(cropped_files, tseq_hourly)
    ]

    all_rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        for rows in executor.map(compute_zonal_stats, zonal_tasks):
            all_rows.extend(rows)

    if not all_rows:
        logging.warning("No zonal stats produced. Exiting.")
        return

    df = pd.DataFrame(all_rows)
    df["time"] = pd.to_datetime(df["time"])

    # 7) aggregate to 6-hour totals (this is where your "leads 6,12,...,720" comes in)
    #    we sum 1-hour basin means over 6 hours.
    df = df.set_index("time")

    df6 = (
        df.groupby("basin")
          .resample("6H")["precip_mm_1h"]
          .sum()
          .reset_index()
          .rename(columns={"precip_mm_1h": "precip_mm_6h"})
    )

    # pivot so each basin is a column, one row per time
    df_wide = df6.pivot(index="time", columns="basin", values="precip_mm_6h").reset_index()

    # nice time string
    df_wide["time_str"] = df_wide["time"].dt.strftime("%Y%m%d%H")

    # lead time in hours since first 6-hour period
    df_wide = df_wide.sort_values("time")
    df_wide["lead_h"] = (
        (df_wide["time"] - df_wide["time"].min())
        .dt.total_seconds() // 3600
    )

    # reorder and rename columns a bit
    col_map = {
        "HUC2_07_UpperMiss": "precip_mm_6h_HUC2_07_UpperMiss",
        "HUC2_10_Missouri": "precip_mm_6h_HUC2_10_Missouri",
    }
    df_wide = df_wide.rename(columns=col_map)

    out_cols = ["time_str", "lead_h"] + list(col_map.values())
    out_csv = os.path.join(outdir, "mrms_6h_basin_precip_wide.csv")
    df_wide[out_cols].to_csv(out_csv, index=False)

    elapsed = time.time() - start_time
    logging.info(f"Wrote {out_csv}")
    logging.info(f"Total execution time: {elapsed:.2f} seconds")

    # tidy time label like 2018103000
    df6["time_str"] = df6["time"].dt.strftime("%Y%m%d%H")

    # optional: if you want an explicit "lead_h" relative to the first 6h step
    df6 = df6.sort_values(["basin", "time"])
    df6["lead_h"] = (
        df6.groupby("basin")["time"]
           .transform(lambda s: (s - s.iloc[0]).dt.total_seconds() // 3600)
    )

    # keep what you care about
    out_csv = os.path.join(outdir, "mrms_6h_basin_precip.csv")
    df6[["time_str", "basin", "lead_h", "precip_mm_6h"]].to_csv(out_csv, index=False)

    elapsed = time.time() - start_time
    logging.info(f"Wrote {out_csv}")
    logging.info(f"Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
