#!/usr/bin/env python3
"""
NWM NetCDF: list available forecast info for a given station (feature_id).

Usage:
  python nwm_forecast_probe_v2.py /path/to/file.nc 5092598
"""

import sys
import numpy as np
from datetime import datetime, timezone
import xarray as xr

def classify_var(vname, var):
    name = vname.lower()
    attrs = " ".join(str(var.attrs.get(k, "")).lower()
                     for k in ["standard_name", "long_name", "units", "description"])
    text = f"{name} {attrs}"

    # Discharge / streamflow
    if ("streamflow" in text or "discharge" in text or "qout" in text) and ("m3" in text or "m^3" in text):
        return "discharge"
    # Lateral inflow used by routing
    if "q_lateral" in name or "qlat" in name or "lateral" in text:
        return "lateral_inflow"
    # Velocity
    if "velocity" in text or ("m s-1" in text or "m/s" in text) and "discharge" not in text:
        return "velocity"
    # Depth (often a proxy for stage)
    if "depth" in text or "water_depth" in text:
        return "depth"
    # Stage / water surface height
    if ("stage" in text or "gage_height" in text or
        "water_surface_height" in text or "water_surface_elevation" in text or
        "water_sfc_elev" in text or "wse" in text):
        return "stage"
    return "other"
    
def _fmt_dt(val):
    if isinstance(val, np.datetime64):
        return np.datetime_as_string(val, unit="s")
    if isinstance(val, (np.integer, np.floating)):
        try:
            return datetime.utcfromtimestamp(float(val)).replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            return str(val)
    if hasattr(val, "values"):
        return _fmt_dt(val.values)
    return str(val)


def _find_feature_dim(ds):
    # Try common patterns
    candidates = [
        ("feature_id", "feature_id"),
        ("stations", "feature_id"),
        ("station", "station_id"),
        ("id", "feature_id"),
    ]
    for dim_name, id_guess in candidates:
        if dim_name in ds.sizes:
            # Try to find an id variable aligned with the dim
            for v in [id_guess, dim_name, "feature_id", "station_id", "stations"]:
                if v in ds.variables:
                    # Must align on the same length
                    if ds.variables[v].sizes.get(dim_name, None) == ds.sizes[dim_name]:
                        return dim_name, v
            # Fallback: the dim itself might be the coordinate
            return dim_name, None

    # Sometimes feature_id is a coordinate, not a true dim
    for coord in ds.coords:
        if "feature_id" in coord.lower():
            return coord, coord

    return None, None


def _time_coords(ds):
    # Collect plausible time-like coordinates
    names = []
    for c in list(ds.coords) + list(ds.variables):
        lc = c.lower()
        if lc in ("time", "valid_time", "model_output_valid_time", "forecast_time"):
            names.append(c)
    # Keep unique order
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def list_forecasts(nc_path, station_id):
    ds = xr.open_dataset(nc_path)

    feat_dim, id_var = _find_feature_dim(ds)
    if not feat_dim:
        print("Could not find a feature dimension such as 'feature_id' or 'station'.")
        return

    # Build ID array
    if id_var and id_var in ds:
        id_values = ds[id_var].values
    else:
        # Use the feature coordinate itself if it exists
        id_values = ds.coords[feat_dim].values if feat_dim in ds.coords else None
    if id_values is None:
        print(f"Found feature dim '{feat_dim}', but could not find an ID array.")
        return

    # Locate the station index
    try:
        station_id_cast = int(station_id)
    except Exception:
        station_id_cast = str(station_id)

    idx = np.where(id_values == station_id_cast)[0]
    if idx.size == 0:
        # Try string match as fallback
        idx = np.where(id_values.astype(str) == str(station_id_cast))[0]
    if idx.size == 0:
        print(f"Station {station_id} not found. Sample IDs: {id_values[:10]}")
        return
    i = int(idx[0])

    time_like = _time_coords(ds)
    frt_names = [c for c in ds.coords if "forecast_reference_time" in c.lower() or c.lower() == "reference_time"]

    print(f"\nFile: {nc_path}")
    print(f"Feature dimension: {feat_dim}")
    print(f"ID variable: {id_var or '(none, used feature coord)'}")
    print(f"Target station: {station_id} at index {i}")
    print(f"Time coordinate(s): {time_like or 'None'}")
    print(f"Forecast reference time coordinate(s): {frt_names or 'None'}")

    # Heuristic: forecast-like vars have the feature dimension and numeric dtype.
    # Do NOT require time as a dimension, since many NWM files have 1 global time.
    forecast_vars = []
    for vname, v in ds.variables.items():
        if vname in {id_var, feat_dim}:
            continue
        if feat_dim in v.dims:
            # Robust numeric check
            is_numeric = False
            try:
                dt = v.dtype  # xarray gives a numpy dtype here
                # floats or ints count as numeric
                is_numeric = np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.integer)
            except TypeError:
                is_numeric = False

            if is_numeric:
                forecast_vars.append(vname)


    print("\nForecast-like variables:")
    if forecast_vars:
        for vname in forecast_vars:
            v = ds[vname]
            units = v.attrs.get("units", "")
            kind = classify_var(vname, v)
            longn = v.attrs.get("long_name", "")
            print(f"  - {vname} [{kind}]  dims={tuple(v.dims)}  units={units}  long_name={longn}")
    else:
        print("  None detected with feature dimension and numeric data.")
        ds.close()
        return

    # Use the first available time coordinate if present
    tcoord = time_like[0] if time_like else None
    times = ds[tcoord].values if tcoord else np.array([])

    # Reference time, if present
    rcoord = frt_names[0] if frt_names else None
    rtimes = ds[rcoord].values if rcoord else np.array([])

    # Print timing summary per variable for the target station
    for vname in forecast_vars:
        v = ds[vname]
        print(f"\n=== {vname} ===")
        # Slice to station
        try:
            data_slice = v.isel({feat_dim: i})
        except Exception:
            print("Could not slice variable by station index.")
            continue

        # Case A: variable has an explicit time dimension
        explicit_t = next((t for t in (time_like or []) if t in v.dims), None)

        if explicit_t:
            vt = ds[explicit_t].values
            print(f"Valid times: {vt.shape[0]}")
            if vt.shape[0] > 0:
                print(f"  First: {_fmt_dt(vt[0])}")
                print(f"  Last:  {_fmt_dt(vt[-1])}")
            # Compute leads if reference_time aligns
            if rcoord and rcoord in data_slice.coords:
                frt = np.array(data_slice[rcoord].values)
                if frt.size == 1:
                    frt = np.full(vt.shape, frt.item())
                if frt.shape == vt.shape:
                    lead_h = ((vt.astype("datetime64[s]") - frt.astype("datetime64[s]")).astype("timedelta64[h]")).astype(int)
                    leads = ", ".join(map(str, sorted(set(lead_h.tolist()))))
                    print(f"Lead hours: [{leads}]")
                else:
                    print("Reference time present but not aligned with time for lead calculation.")
            continue

        # Case B: no explicit time dimension on the var; use dataset-level time (per-lead file)
        if tcoord and np.size(times) >= 1:
            vt = np.atleast_1d(times)
            print(f"Per-file valid time count: {vt.shape[0]}")
            for k in range(min(3, vt.shape[0])):
                print(f"  Valid time: {_fmt_dt(vt[k])}")

            if rcoord and rtimes.size >= 1:
                frt = np.atleast_1d(rtimes)
                # If both are length-1, compute a single lead
                if frt.size == 1 and vt.size == 1:
                    lead_h = int(((vt[0].astype("datetime64[s]") - frt[0].astype("datetime64[s]"))
                                  .astype("timedelta64[h]")).astype(int))
                    print(f"Reference time: {_fmt_dt(frt[0])}")
                    print(f"Lead hour: {lead_h}")
                else:
                    # Try to align by broadcasting
                    try:
                        lead = ((vt.astype("datetime64[s]") - frt.astype("datetime64[s]"))
                                .astype("timedelta64[h]")).astype(int)
                        leads = ", ".join(map(str, np.unique(lead).tolist()))
                        print(f"Reference time(s): {frt.size}, Lead hours: [{leads}]")
                    except Exception:
                        print("Could not compute lead hours from available time and reference_time.")
        else:
            print("No time coordinate found in dataset to describe forecast validity.")

        # Sample the first few values
        try:
            nshow = 3
            if tcoord and tcoord in v.dims:
                # Time-varying, handled above
                pass
            else:
                # Single time per file, just show a few values exist
                vals = data_slice.values
                if vals.size > 0:
                    if np.ndim(vals) == 0:
                        print(f"Sample value: {float(vals)} {v.attrs.get('units','')}")
                    else:
                        print("Sample values:", ", ".join(str(float(x)) for x in np.ravel(vals)[:nshow]),
                              v.attrs.get('units',''))
        except Exception:
            pass

    ds.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python nwm_forecast_probe_v2.py /path/to/file.nc 5092598")
        sys.exit(2)
    list_forecasts(sys.argv[1], sys.argv[2])
