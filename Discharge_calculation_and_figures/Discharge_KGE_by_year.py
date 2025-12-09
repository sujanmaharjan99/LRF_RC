import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_aggregate_overall_kge_by_year(combined_df, out_dir, years=None):
    """
    Plot aggregated overall KGE vs lead time for multiple years (no seasons).

    For each year, the function:
      - aggregates KGE_all across stations for each lead_days (mean)
      - plots one line per year in a single figure

    Parameters
    ----------
    combined_df : pandas.DataFrame
        Must contain columns:
            - 'year'      (int, e.g. 2019)
            - 'lead_days'
            - 'KGE_all'
        Each row should represent one station, one lead, one year.
    out_dir : str or Path
        Directory where the figure will be saved.
    years : list of int, optional
        List of years to plot. If None, defaults to [2019, 2020, 2021, 2022, 2023, 2024]
        and uses only those that are present in the DataFrame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "year" not in combined_df.columns:
        raise ValueError("combined_df must contain a 'year' column to plot KGE by year.")

    if years is None:
        years = [2019, 2020, 2021, 2022, 2023, 2024]

    plt.figure(figsize=(9, 5))

    for yr in years:
        sub = combined_df[combined_df["year"] == yr]
        if sub.empty:
            # Skip years that are not present in the data
            continue

        # Aggregate across stations for this year and lead time
        grouped = (
            sub.groupby("lead_days")["KGE_all"]
            .mean()
            .sort_index()
        )

        lead_days = grouped.index.values
        kge_mean = grouped.values

        plt.plot(lead_days, kge_mean, label=str(yr))

    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    plt.title("Aggregate overall KGE by year (all stations)")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend(title="Year")
    plt.tight_layout()

    out_path = out_dir / "AllStations_KGE_overall_by_year.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\nSaved overall aggregate KGE-by-year plot to:\n  {out_path}")

def compute_kge_for_station_by_year(
    station_name,
    nwm_station_dir,
    usgs_csv,
    lead_times,
    years=None,
    time_col_nwm="time",
    flow_col_nwm="streamflow",
    time_col_usgs="timestamp",
    flow_col_usgs="discharge_cumecs",
):
    """
    Compute overall KGE vs lead time by calendar year for one station.

    Returns a DataFrame with columns:
        station, year, lead_hours, lead_days, KGE_all, n_pairs
    """
    if years is None:
        years = [2019, 2020, 2021, 2022, 2023, 2024]

    print(f"\nProcessing station (by year): {station_name}")
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
            # No data for this lead at this station, skip
            continue

        merged = pd.merge(usgs, nwm, on="time", how="inner")

        if merged.empty:
            # No overlapping timestamps
            continue

        merged["year"] = merged["time"].dt.year

        for yr in years:
            sub = merged[merged["year"] == yr]
            if len(sub) < 2:
                # Not enough data to compute KGE for this year/lead
                continue

            kge_y = compute_kge(sub["sim"].values, sub["obs"].values)
            n_pairs = len(sub)

            results.append(
                {
                    "station": station_name,
                    "year": yr,
                    "lead_hours": lead,
                    "lead_days": lead / 24.0,
                    "KGE_all": kge_y,
                    "n_pairs": n_pairs,
                }
            )

    if not results:
        return pd.DataFrame(columns=["station", "year", "lead_hours", "lead_days", "KGE_all", "n_pairs"])

    df_year = pd.DataFrame(results).sort_values(["year", "lead_hours"])
    return df_year

if __name__ == "__main__":
    # Base paths
    NWM_BASE_DIR = Path("/media/12TB/Sujan/NWM/Csv")
    USGS_DIR = Path("/media/12TB/Sujan/NWM/USGS_data")

    # Lead times in hours (6, 12, ..., 720)
    LEAD_TIMES = range(6, 721, 6)

    # Output locations
    OUTPUT_TABLE_DIR = Path("./kge_outputs")
    OUTPUT_TABLE_DIR.mkdir(exist_ok=True, parents=True)
    PLOT_DIR = Path("./kge_plots")

    all_stations_results_year = []

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

        station_kge_year = compute_kge_for_station_by_year(
            station_name=station_name,
            nwm_station_dir=station_dir,
            usgs_csv=usgs_csv,
            lead_times=LEAD_TIMES,
            years=[2019, 2020, 2021, 2022, 2023, 2024],
            time_col_nwm="time",
            flow_col_nwm="streamflow",
            time_col_usgs="timestamp",
            flow_col_usgs="discharge_cumecs",
        )

        if not station_kge_year.empty:
            out_csv_year = OUTPUT_TABLE_DIR / f"{station_name}_KGE_by_year.csv"
            station_kge_year.to_csv(out_csv_year, index=False)
            print(f"  Saved station yearly KGE table to {out_csv_year}")
            all_stations_results_year.append(station_kge_year)

        
    # Save combined yearly table and plot by-year aggregates
    if all_stations_results_year:
        combined_year = pd.concat(all_stations_results_year, ignore_index=True)
        combined_year_csv = OUTPUT_TABLE_DIR / "KGE_by_year_all_stations.csv"
        combined_year.to_csv(combined_year_csv, index=False)
        print(f"\nSaved combined yearly KGE table for all stations to {combined_year_csv}")

        # This uses the same plot_aggregate_overall_kge_by_year function you already defined
        plot_aggregate_overall_kge_by_year(combined_year, PLOT_DIR)
    else:
        print("\nNo yearly KGE results computed.")
 
