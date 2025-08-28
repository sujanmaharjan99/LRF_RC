#!/usr/bin/env python3
import sys
import os
import textwrap
from datetime import datetime
import numpy as np

def try_open_xr(path):
    import xarray as xr
    engines_to_try = ["netcdf4", "h5netcdf", "scipy"]
    errors = {}
    for eng in engines_to_try:
        try:
            ds = xr.open_dataset(path, engine=eng)  # lazy, no load
            return ds, eng, None
        except Exception as e:
            errors[eng] = str(e)
    return None, None, errors

def format_attr(value, maxlen=120):
    try:
        s = str(value)
    except Exception:
        s = repr(value)
    s = s.replace("\n", " ")
    if len(s) > maxlen:
        s = s[:maxlen-3] + "..."
    return s

def human_size(num):
    for unit in ["B","KB","MB","GB","TB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"

def summarize_time(coord):
    try:
        vals = coord.values
        if vals.size == 0:
            return "empty"
        t0 = np.datetime_as_string(vals.min(), unit="s")
        t1 = np.datetime_as_string(vals.max(), unit="s")
        return f"{t0} to {t1} ({vals.size} steps)"
    except Exception:
        return "not datetime or could not parse"

def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_netcdf.py <path_to_nc> [var_to_preview]")
        sys.exit(1)

    path = sys.argv[1]
    var_preview = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    print("="*80)
    print("NetCDF quick summary")
    print("="*80)
    print(f"File: {path}")
    try:
        size = os.path.getsize(path)
        print(f"Size: {human_size(size)}")
    except Exception:
        pass

    # Import here so the script prints a clear message if xarray is missing
    try:
        import xarray as xr
    except Exception as e:
        print("xarray is not installed in this environment.")
        print("Install: pip install xarray netcdf4 h5netcdf")
        sys.exit(1)

    ds, engine, open_errors = try_open_xr(path)
    if ds is None:
        print("Failed to open with available engines.")
        for eng, msg in open_errors.items():
            print(f"  {eng}: {msg}")
        print("Try installing: pip install netcdf4 h5netcdf")
        sys.exit(1)

    print(f"Opened with engine: {engine}")
    print()

    # Global attributes
    print("Global attributes:")
    if ds.attrs:
        for k, v in ds.attrs.items():
            print(f"  - {k}: {format_attr(v)}")
    else:
        print("  (none)")
    print()

    # Dimensions
    print("Dimensions:")
    for dim, size in ds.dims.items():
        print(f"  - {dim}: {size}")
    print()

    # Coordinates
    print("Coordinates:")
    if ds.coords:
        for name, coord in ds.coords.items():
            info = f"shape={tuple(coord.shape)}, dtype={coord.dtype}"
            if str(name).lower() == "time":
                info += f", range={summarize_time(coord)}"
            print(f"  - {name}: {info}")
    else:
        print("  (none)")
    print()

    # Data variables
    print("Data variables:")
    if ds.data_vars:
        for name, var in ds.data_vars.items():
            chunks = getattr(getattr(var.data, "chunks", None), "shape", None)
            chunk_str = f", chunks={chunks}" if chunks else ""
            print(f"  - {name}: dims={var.dims}, shape={tuple(var.shape)}, dtype={var.dtype}{chunk_str}")
            # A few key attrs
            for key in ["long_name", "standard_name", "units", "grid_mapping"]:
                if key in var.attrs:
                    print(f"      {key}: {format_attr(var.attrs[key])}")
    else:
        print("  (none)")
    print()

    # Likely NWM channel_rt helpers
    likely_vars = ["streamflow", "velocity", "nudge", "depth", "q_lateral"]
    channel_dim = None
    if "feature_id" in ds.dims:
        channel_dim = "feature_id"
    elif "station_id" in ds.dims:
        channel_dim = "station_id"

    if channel_dim:
        print(f"Detected a channel-like dimension: {channel_dim} with size {ds.dims[channel_dim]}")
        for v in likely_vars:
            if v in ds:
                da = ds[v]
                print(f"  Sample {v}:")
                try:
                    # print a small slice without loading the whole array
                    idx = min(5, da.shape[-1]) if da.ndim > 0 else 0
                    arr = da.isel({channel_dim: slice(0, idx)}) if channel_dim in da.dims else da.isel({da.dims[-1]: slice(0, idx)})
                    print(f"    preview dims={arr.dims}, shape={tuple(arr.shape)}, units={da.attrs.get('units','')}")
                    # Use .values sparingly to avoid big loads
                    print(f"    first values: {np.array(arr.values).ravel()[:5]}")
                except Exception as e:
                    print(f"    could not preview: {e}")
        print()

    # Time coverage quick check
    if "time" in ds.coords:
        print("Time coverage:")
        print(f"  {summarize_time(ds['time'])}")
        print()

    # Optional variable preview
    if var_preview:
        if var_preview in ds:
            var = ds[var_preview]
            print(f"Preview of variable '{var_preview}':")
            print(f"  dims={var.dims}, shape={tuple(var.shape)}, dtype={var.dtype}")
            for key in ["long_name", "standard_name", "units"]:
                if key in var.attrs:
                    print(f"  {key}: {format_attr(var.attrs[key])}")
            # Print a tiny sample
            try:
                take = tuple(slice(0, min(3, s)) for s in var.shape)
                sample = var.isel(**{d: slice(0, min(3, var.sizes[d])) for d in var.dims}).values
                print(f"  sample values (truncated): {np.array(sample).ravel()[:10]}")
            except Exception as e:
                print(f"  could not sample: {e}")
        else:
            print(f"Variable '{var_preview}' not found in dataset.")
        print()

    # CF conventions hint
    conv = ds.attrs.get("Conventions") or ds.attrs.get("conventions")
    if conv:
        print(f"Conventions: {conv}")
    print("Done.")

if __name__ == "__main__":
    main()
