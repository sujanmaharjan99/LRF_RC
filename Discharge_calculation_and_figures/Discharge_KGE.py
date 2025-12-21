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
    if m in [ 7, 8,9, 10, 11, 12, 1, 2,3]:
        return "dry"
    elif m in [ 4,5,6]:
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
    df["lead_days"] = df["lead_hours"] / 24.0

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
    plt.plot(kge_df["lead_days"], kge_df["KGE_dry"])
    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    plt.title(f"{station_name} - Dry season")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.tight_layout()
    dry_path = out_dir / f"{station_name}_KGE_dry.png"
    plt.savefig(dry_path, dpi=200)
    plt.close()

    # Wet
    plt.figure(figsize=(7, 4))
    plt.plot(kge_df["lead_days"], kge_df["KGE_wet"])
    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    plt.title(f"{station_name} - Wet season")
    plt.ylim(0.0, 1.0)
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
        plt.plot(sub["lead_days"], sub["KGE_dry"], label=st)
    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    plt.title("Dry season KGE for all stations")
    plt.ylim(0.0, 1.0)
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
        plt.plot(sub["lead_days"], sub["KGE_wet"], label=st)
    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    plt.title("Wet season KGE for all stations")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    wet_out = out_dir / "AllStations_KGE_wet.png"
    plt.savefig(wet_out, dpi=200)
    plt.close()

    print(f"\nSaved multi-station plots to:\n  {dry_out}\n  {wet_out}")
def plot_aggregate_overall_kge(combined_df, out_dir):
    """
    Plot aggregated overall KGE vs lead time over all stations (no wet/dry split).
    Shows mean ± std across stations.

    Parameters
    ----------
    combined_df : pandas.DataFrame
        Must contain columns:
            - 'lead_days'
            - 'KGE_all'
    out_dir : str or Path
        Directory where the figure will be saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by lead time
    grouped = combined_df.groupby("lead_days")

    kge_mean = grouped["KGE_all"].mean()

    lead_days = kge_mean.index.values

    plt.figure(figsize=(8, 5))

    # Mean line
    plt.plot(lead_days, kge_mean.values, label="Mean KGE (all stations)")

    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    plt.title("Aggregate overall KGE (all stations)")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "AllStations_KGE_overall_aggregate.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\nSaved overall aggregate KGE plot to:\n  {out_path}")
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
        grouped = sub.groupby("lead_days")["KGE_all"].mean().sort_index()

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

def compute_kge_months_years_for_station(
    station_name,
    nwm_station_dir,
    usgs_csv,
    lead_times,
    months=None,
    years=None,
    time_col_nwm="time",
    flow_col_nwm="streamflow",
    time_col_usgs="timestamp",
    flow_col_usgs="discharge_cumecs",
):
    """
    Compute KGE vs lead time for a station, filtered by selected months and/or years.

    Parameters
    ----------
    months : int or list[int] or None
        Months to include (1-12). None means all months.
    years : int or list[int] or None
        Years to include (e.g. 2021). None means all years.
    """
    # Normalize months/years inputs
    if months is None:
        months_list = None
    else:
        months_list = [months] if isinstance(months, int) else list(months)
        months_list = sorted(set(months_list))

    if years is None:
        years_list = None
    else:
        years_list = [years] if isinstance(years, int) else list(years)
        years_list = sorted(set(years_list))

    usgs = load_usgs_timeseries(usgs_csv, time_col=time_col_usgs, discharge_col=flow_col_usgs)

    results = []
    for lead in lead_times:
        nwm = load_nwm_timeseries_for_lead(
            nwm_station_dir,
            lead,
            time_col=time_col_nwm,
            flow_col=flow_col_nwm,
        )

        base_row = {
            "station": station_name,
            "lead_hours": lead,
            "lead_days": lead / 24.0,
            "months": "all" if months_list is None else ",".join(map(str, months_list)),
            "years": "all" if years_list is None else ",".join(map(str, years_list)),
        }

        if nwm is None:
            results.append({**base_row, "KGE": np.nan, "n_pairs": 0})
            continue

        merged = pd.merge(usgs, nwm, on="time", how="inner")
        if merged.empty:
            results.append({**base_row, "KGE": np.nan, "n_pairs": 0})
            continue

        # Apply filters
        if months_list is not None:
            merged = merged[merged["time"].dt.month.isin(months_list)]
        if years_list is not None:
            merged = merged[merged["time"].dt.year.isin(years_list)]

        n_pairs = len(merged)
        kge = compute_kge(merged["sim"].values, merged["obs"].values) if n_pairs >= 2 else np.nan

        results.append({**base_row, "KGE": kge, "n_pairs": n_pairs})

    return pd.DataFrame(results).sort_values("lead_hours")

def plot_station_kge_for_months_years(
    kge_df,
    station_name,
    months=None,
    years=None,
    out_dir=None,
    ylim=(-1.0, 1.0),
):
    """
    Plot KGE vs lead time for a station given month/year filters.
    """
    # Make nice labels
    if months is None:
        months_label = "all_months"
        title_months = "All months"
    else:
        mlist = [months] if isinstance(months, int) else sorted(set(months))
        months_label = "m" + "-".join(map(str, mlist))
        title_months = f"Months {mlist}"

    if years is None:
        years_label = "all_years"
        title_years = "All years"
    else:
        ylist = [years] if isinstance(years, int) else sorted(set(years))
        years_label = "y" + "-".join(map(str, ylist))
        title_years = f"Years {ylist}"

    plt.figure(figsize=(7, 4))
    plt.plot(kge_df["lead_days"], kge_df["KGE"])
    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    plt.title(f"{station_name} - {title_months}, {title_years}")
    plt.ylim(*ylim)
    plt.grid(True)
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{station_name}_KGE_{months_label}_{years_label}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved plot to: {out_path}")
        return out_path

    plt.show()
    return None

def run_single_station_month_year(
    station_name,
    nwm_base_dir,
    usgs_dir,
    lead_times,
    months,
    years,
    plot_dir,
    table_dir=None,
):
    station_dir = Path(nwm_base_dir) / station_name

    pattern = str(Path(usgs_dir) / f"{station_name}_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No USGS file found for {station_name} (pattern: {pattern})")

    usgs_csv = matches[0]

    print(f"\n=== Single station run ===")
    print(f"Station: {station_name}")
    print(f"NWM dir: {station_dir}")
    print(f"USGS file: {usgs_csv}")
    print(f"Months: {months}, Years: {years}")

    kge_m_y = compute_kge_months_years_for_station(
        station_name=station_name,
        nwm_station_dir=station_dir,
        usgs_csv=usgs_csv,
        lead_times=lead_times,
        months=months,
        years=years,
    )

    # Optional save table
    if table_dir is not None:
        table_dir = Path(table_dir)
        table_dir.mkdir(parents=True, exist_ok=True)
        out_csv = table_dir / f"{station_name}_KGE_m{'-'.join(map(str, months))}_y{years}.csv"
        kge_m_y.to_csv(out_csv, index=False)
        print(f"Saved table: {out_csv}")

    # Plot
    plot_station_kge_for_months_years(
        kge_m_y,
        station_name,
        months=months,
        years=years,
        out_dir=plot_dir,
    )


def plot_aggregate_seasonal_kge(combined_df, out_dir):
    """
    Plot aggregated KGE vs lead time over all stations:
    - One figure for dry season (mean ± std across stations)
    - One figure for wet season (mean ± std across stations)

    Parameters
    ----------
    combined_df : pandas.DataFrame
        Output from concatenating all per-station KGE tables.
        Must contain columns:
            - 'lead_days'
            - 'KGE_dry'
            - 'KGE_wet'
    out_dir : str or Path
        Directory where the figures will be saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by lead time
    grouped = combined_df.groupby("lead_days")

    # Dry season stats
    dry_mean = grouped["KGE_dry"].mean()
    dry_std = grouped["KGE_dry"].std()

    # Wet season stats
    wet_mean = grouped["KGE_wet"].mean()
    wet_std = grouped["KGE_wet"].std()

    # Convert index to ndarray for plotting
    lead_days = dry_mean.index.values

    # DRY SEASON AGGREGATE PLOT
    plt.figure(figsize=(8, 5))
    plt.plot(lead_days, dry_mean.values, label="Mean KGE (dry)")
    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    plt.title("Aggregate dry season KGE (all stations)")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    dry_out = out_dir / "AllStations_KGE_dry_aggregate.png"
    plt.savefig(dry_out, dpi=200)
    plt.close()

    # WET SEASON AGGREGATE PLOT
    plt.figure(figsize=(8, 5))
    plt.plot(lead_days, wet_mean.values, label="Mean KGE (wet)")
    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    plt.title("Aggregate wet season KGE (all stations)")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    wet_out = out_dir / "AllStations_KGE_wet_aggregate.png"
    plt.savefig(wet_out, dpi=200)
    plt.close()

    print(f"\nSaved aggregate KGE plots to:\n  {dry_out}\n  {wet_out}")
# =========================
#  Main driver
# =========================

if __name__ == "__main__":
    NWM_BASE_DIR = Path("/mnt/12TB/Sujan/Csv/")
    USGS_DIR = Path("/mnt/12TB/Sujan/USGS_data/")
    LEAD_TIMES = range(6, 721, 6)

    OUTPUT_TABLE_DIR = Path("./kge_outputs")
    OUTPUT_TABLE_DIR.mkdir(exist_ok=True, parents=True)

    PLOT_DIR = Path("./kge_plots")
    PLOT_DIR.mkdir(exist_ok=True, parents=True)

    # ----------------------------
    # 1) Quick single-station plot
    # ----------------------------
    run_single_station_month_year(
        station_name="01_Rulo",
        nwm_base_dir=NWM_BASE_DIR,
        usgs_dir=USGS_DIR,
        lead_times=LEAD_TIMES,
        months=[4],     # April
        years=2024,     # year filter
        plot_dir=PLOT_DIR,
        table_dir=OUTPUT_TABLE_DIR,  # optional, set to None if you don't want csv
    )

    # -----------------------------------
    # 2) Continue with your existing loop
    # -----------------------------------
    all_stations_results = []

    for station_dir in sorted(NWM_BASE_DIR.iterdir()):
        if not station_dir.is_dir():
            continue

        station_name = station_dir.name

        pattern = str(USGS_DIR / f"{station_name}_*.csv")
        matches = glob.glob(pattern)
        if not matches:
            print(f"\nNo USGS file found for station {station_name} (pattern {pattern}), skipping.")
            continue

        usgs_csv = matches[0]
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

        out_csv = OUTPUT_TABLE_DIR / f"{station_name}_KGE_seasonal.csv"
        station_kge.to_csv(out_csv, index=False)
        print(f"  Saved station KGE table to {out_csv}")

        plot_station_seasonal_kge(station_kge, station_name, PLOT_DIR)

        all_stations_results.append(station_kge)

    if all_stations_results:
        combined = pd.concat(all_stations_results, ignore_index=True)
        combined_csv = OUTPUT_TABLE_DIR / "KGE_seasonal_all_stations.csv"
        combined.to_csv(combined_csv, index=False)
        print(f"\nSaved combined KGE table for all stations to {combined_csv}")

        plot_all_stations_seasonal_kge(combined, PLOT_DIR)
        plot_aggregate_seasonal_kge(combined, PLOT_DIR)
        plot_aggregate_overall_kge(combined, PLOT_DIR)
        plot_aggregate_overall_kge_by_year(combined, PLOT_DIR)
    else:
        print("\nNo stations processed.")
