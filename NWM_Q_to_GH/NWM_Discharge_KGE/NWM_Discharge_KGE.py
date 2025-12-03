import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
#  Basic KGE + season utils
# =========================

def compute_kge(sim, obs):
    """
    Compute Kling-Gupta Efficiency (KGE).
    """
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)

    mask = np.isfinite(sim) & np.isfinite(obs)
    if mask.sum() < 2:
        return np.nan

    sim = sim[mask]
    obs = obs[mask]

    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1)
    beta = np.mean(sim) / np.mean(obs)

    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return kge


def season_from_month(m):
    """
    Dry: Sep(9), Oct(10), Nov(11), Dec(12), Jan(1), Feb(2)
    Wet: Mar(3), Apr(4), May(5), Jun(6), Jul(7), Aug(8)
    """
    if m in [9, 10, 11, 12, 1, 2]:
        return "dry"
    elif m in [3, 4, 5, 6, 7, 8]:
        return "wet"
    else:
        return None


# =========================
#  Data loading
# =========================

def load_usgs_timeseries(usgs_csv, time_col="timestamp", discharge_col="discharge_cumecs"):
    """
    Load USGS discharge time series and standardize column names.
    """
    df = pd.read_csv(usgs_csv)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    return df[[time_col, discharge_col]].rename(columns={time_col: "time", discharge_col: "obs"})


def load_nwm_timeseries_for_lead(station_dir, lead_hours, time_col="time", flow_col="streamflow"):
    """
    Load NWM time series for a given station and lead time.

    Expects files like:
        station_dir / "All" / f"timeseries_{lead_hours:03d}.csv"
    """
    station_dir = Path(station_dir)
    all_dir = station_dir / "All"
    fpath = all_dir / f"timeseries_{lead_hours:03d}.csv"

    if not fpath.exists():
        print(f"  No NWM file for lead {lead_hours}h at {fpath}")
        return None

    df = pd.read_csv(fpath)
    df[time_col] = pd.to_datetime(df[time_col])

    if flow_col in df.columns:
        # Standard case: single column called 'streamflow'
        df = df[[time_col, flow_col]].rename(columns={flow_col: "sim"})
    else:
        # Fallback: average all numeric columns except time
        value_cols = [c for c in df.columns if c != time_col and np.issubdtype(df[c].dtype, np.number)]
        if not value_cols:
            print(f"  No numeric flow columns in {fpath}")
            return None
        df["sim"] = df[value_cols].mean(axis=1, skipna=True)
        df = df[[time_col, "sim"]]

    return df.sort_values(time_col)


# =========================
#  Per-station KGE
# =========================

def compute_kge_for_station(
    station_name,
    nwm_station_dir,
    usgs_csv,
    lead_times,
    time_col_nwm="time",
    flow_col_nwm="streamflow",
    time_col_usgs="timestamp",
    flow_col_usgs="discharge_cumecs",
):
    """
    Compute KGE (all, dry, wet) vs lead time for one station.
    """
    print(f"\nProcessing station: {station_name}")
    usgs = load_usgs_timeseries(usgs_csv, time_col=time_col_usgs, discharge_col=flow_col_usgs)

    results = []

    for lead in lead_times:
        nwm = load_nwm_timeseries_for_lead(
            nwm_station_dir,
            lead,
            time_col=time_col_nwm,
            flow_col=flow_col_nwm,
        )

        if nwm is None:
            results.append(
                {
                    "station": station_name,
                    "lead_hours": lead,
                    "KGE_all": np.nan,
                    "KGE_dry": np.nan,
                    "KGE_wet": np.nan,
                    "n_pairs_all": 0,
                    "n_pairs_dry": 0,
                    "n_pairs_wet": 0,
                }
            )
            continue

        merged = pd.merge(usgs, nwm, on="time", how="inner")

        if merged.empty:
            print(f"  No overlapping timestamps for lead {lead}h")
            results.append(
                {
                    "station": station_name,
                    "lead_hours": lead,
                    "KGE_all": np.nan,
                    "KGE_dry": np.nan,
                    "KGE_wet": np.nan,
                    "n_pairs_all": 0,
                    "n_pairs_dry": 0,
                    "n_pairs_wet": 0,
                }
            )
            continue

        merged["season"] = merged["time"].dt.month.map(season_from_month)

        # all
        kge_all = compute_kge(merged["sim"].values, merged["obs"].values)
        n_all = len(merged)

        # dry
        dry = merged[merged["season"] == "dry"]
        if len(dry) > 1:
            kge_dry = compute_kge(dry["sim"].values, dry["obs"].values)
        else:
            kge_dry = np.nan
        n_dry = len(dry)

        # wet
        wet = merged[merged["season"] == "wet"]
        if len(wet) > 1:
            kge_wet = compute_kge(wet["sim"].values, wet["obs"].values)
        else:
            kge_wet = np.nan
        n_wet = len(wet)

        results.append(
            {
                "station": station_name,
                "lead_hours": lead,
                "KGE_all": kge_all,
                "KGE_dry": kge_dry,
                "KGE_wet": kge_wet,
                "n_pairs_all": n_all,
                "n_pairs_dry": n_dry,
                "n_pairs_wet": n_wet,
            }
        )

    df = pd.DataFrame(results).sort_values("lead_hours")
    return df


# =========================
#  Plotting
# =========================

def plot_station_seasonal_kge(kge_df, station_name, out_dir):
    """
    Create separate plots for dry and wet season KGE vs lead time for one station.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dry
    plt.figure(figsize=(7, 4))
    plt.plot(kge_df["lead_hours"], kge_df["KGE_dry"], marker="o")
    plt.xlabel("Lead time (hours)")
    plt.ylabel("KGE (dry season)")
    plt.title(f"{station_name} - Dry season")
    plt.grid(True)
    plt.tight_layout()
    dry_path = out_dir / f"{station_name}_KGE_dry.png"
    plt.savefig(dry_path, dpi=200)
    plt.close()

    # Wet
    plt.figure(figsize=(7, 4))
    plt.plot(kge_df["lead_hours"], kge_df["KGE_wet"], marker="o")
    plt.xlabel("Lead time (hours)")
    plt.ylabel("KGE (wet season)")
    plt.title(f"{station_name} - Wet season")
    plt.grid(True)
    plt.tight_layout()
    wet_path = out_dir / f"{station_name}_KGE_wet.png"
    plt.savefig(wet_path, dpi=200)
    plt.close()

    print(f"  Saved plots to:\n    {dry_path}\n    {wet_path}")

def plot_all_stations_seasonal_kge(combined_df, out_dir):
    """
    Plot KGE vs lead time for all stations together:
    - dry season in one plot
    - wet season in another plot
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = combined_df["station"].unique()

    # DRY SEASON PLOT
    plt.figure(figsize=(8, 5))
    for st in stations:
        sub = combined_df[combined_df["station"] == st]
        # line plot, no markers
        plt.plot(sub["lead_hours"], sub["KGE_dry"], label=st)
    plt.xlabel("Lead time (hours)")
    plt.ylabel("KGE (dry season)")
    plt.title("Dry season KGE for all stations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    dry_out = out_dir / "AllStations_KGE_dry.png"
    plt.savefig(dry_out, dpi=200)
    plt.close()

    # WET SEASON PLOT
    plt.figure(figsize=(8, 5))
    for st in stations:
        sub = combined_df[combined_df["station"] == st]
        plt.plot(sub["lead_hours"], sub["KGE_wet"], label=st)
    plt.xlabel("Lead time (hours)")
    plt.ylabel("KGE (wet season)")
    plt.title("Wet season KGE for all stations")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    wet_out = out_dir / "AllStations_KGE_wet.png"
    plt.savefig(wet_out, dpi=200)
    plt.close()

    print(f"\nSaved multi-station plots to:\n  {dry_out}\n  {wet_out}")

# =========================
#  Main driver
# =========================

if __name__ == "__main__":
    # Base paths
    NWM_BASE_DIR = Path("/mnt/12TB/Sujan/NWM/Csv")
    USGS_DIR = Path("/mnt/12TB/Sujan/NWM/USGS_data")

    # Lead times in hours (6, 12, ..., 720)
    LEAD_TIMES = range(6, 721, 6)

    # Output locations
    OUTPUT_TABLE_DIR = Path("./kge_outputs")
    OUTPUT_TABLE_DIR.mkdir(exist_ok=True, parents=True)
    PLOT_DIR = Path("./kge_plots")

    all_stations_results = []

    # Loop over station folders in NWM
    for station_dir in sorted(NWM_BASE_DIR.iterdir()):
        if not station_dir.is_dir():
            continue

        station_name = station_dir.name  # e.g. "01_Rulo"

        # Match USGS file: <station_name>_*.csv
        pattern = str(USGS_DIR / f"{station_name}_*.csv")
        matches = glob.glob(pattern)

        if not matches:
            print(f"\nNo USGS file found for station {station_name} (pattern {pattern}), skipping.")
            continue

        usgs_csv = matches[0]  # if multiple, take first
        print(f"\nMatched station {station_name}: USGS file {usgs_csv}")

        station_kge = compute_kge_for_station(
            station_name=station_name,
            nwm_station_dir=station_dir,
            usgs_csv=usgs_csv,
            lead_times=LEAD_TIMES,
            time_col_nwm="time",
            flow_col_nwm="streamflow",
            time_col_usgs="timestamp",
            flow_col_usgs="discharge_cumecs",
        )

        # Save per-station CSV
        out_csv = OUTPUT_TABLE_DIR / f"{station_name}_KGE_seasonal.csv"
        station_kge.to_csv(out_csv, index=False)
        print(f"  Saved station KGE table to {out_csv}")

        # Plots (dry and wet separately)
        plot_station_seasonal_kge(station_kge, station_name, PLOT_DIR)

        all_stations_results.append(station_kge)

    # Save combined table
    if all_stations_results:
        combined = pd.concat(all_stations_results, ignore_index=True)
        combined_csv = OUTPUT_TABLE_DIR / "KGE_seasonal_all_stations.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"\nSaved combined KGE table for all stations to {combined_csv}")

        # New: plots for all stations together (dry and wet)
        plot_all_stations_seasonal_kge(combined, PLOT_DIR)
    else:
        print("\nNo stations processed.")

