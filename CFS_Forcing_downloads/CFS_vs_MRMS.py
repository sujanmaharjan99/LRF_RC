#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
# USER SETTINGS
# -------------------------------------------------------------------
MRMS_DIR = Path("/mnt/12TB/Sujan/LRF_RC/out_mrms/")
CFS_DIR  = Path("/mnt/12TB/Sujan/LRF_RC/out_cfs_gpkg_ts/")
out_dir = Path("/mnt/12TB/Sujan/LRF_RC/CFS_Forcing_downloads/")
MRMS_FILE = MRMS_DIR / "mrms_forcing_from_gpkg.csv"


EVAL_START = "2024-04-01"
EVAL_END   = "2024-06-30"
# Which CFS member to compare: e.g. "mem_ensmean", "mem01", "mem02", ...
CFS_MEMBER_LABEL = "mem_ensmean"

# Basin identifiers in each dataset
#   MRMS: matches 'feature_id' column (string)
#   CFS : matches 'basin' column
MRMS_FEATURE_ID = "0"   # change to whatever is in 'feature_id'
CFS_BASIN_NAME  = "BASIN"   # change to whatever is in 'basin'

# Time alignment: MRMS file has 6-hourly totals; set shift to align
# with CFS valid time (often +6 hours if MRMS bins labeled at window start)
MRMS_TIME_SHIFT_HOURS = 6

# Optional seasonal filter (None for all months)
WET_MONTHS = [4,5,6]   # example; change or set to None

# Minimum number of paired values to compute stats for a lead
MIN_SAMPLES = 30

# -------------------------------------------------------------------
# METRIC FUNCTIONS
# -------------------------------------------------------------------
def kge(sim, obs):
    """Kling-Gupta efficiency."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(sim) == 0:
        return np.nan

    r = np.corrcoef(sim, obs)[0, 1] if len(sim) > 1 else np.nan
    alpha = np.std(sim, ddof=1) / np.std(obs, ddof=1) if np.std(obs, ddof=1) != 0 else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan

    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def nse(sim, obs):
    """Nash-Sutcliffe efficiency."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(sim) == 0:
        return np.nan

    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((sim - obs) ** 2) / denom


def rmse(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(sim) == 0:
        return np.nan
    return np.sqrt(np.mean((sim - obs) ** 2))


def bias(sim, obs):
    """Relative bias (sim/obs - 1)."""
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]
    if len(sim) == 0:
        return np.nan
    denom = np.mean(obs)
    if denom == 0:
        return np.nan
    return np.mean(sim) / denom - 1.0


# -------------------------------------------------------------------
# LOAD MRMS (OBS)
# -------------------------------------------------------------------
def load_mrms():
    df = pd.read_csv(MRMS_FILE)
    df["time"] = pd.to_datetime(df["time_str"], format="%Y%m%d%H", utc=True)

    # Shift to align with CFS valid times if needed
    if MRMS_TIME_SHIFT_HOURS != 0:
        df["time"] = df["time"] + pd.Timedelta(hours=MRMS_TIME_SHIFT_HOURS)

    # Select basin
    df = df[df["feature_id"].astype(str) == str(MRMS_FEATURE_ID)].copy()

    # Clip to evaluation window
    df = df[(df["time"] >= EVAL_START) & (df["time"] <= EVAL_END)]

    df = df[["time", "precip_mm"]].dropna().sort_values("time")
    return df



# -------------------------------------------------------------------
# LOAD ONE LEAD FROM CFS (FORECAST)
# -------------------------------------------------------------------
def load_cfs_lead(lead_h):
    """
    lead_h: forecast lead in hours, e.g. 6, 12, ...
    Reads the corresponding timeseries_XXX.csv for chosen member.
    """
    fname = CFS_DIR / CFS_MEMBER_LABEL / f"timeseries_{lead_h:03d}.csv"
    if not fname.exists():
        return None

    df = pd.read_csv(fname)
    # CFS CSV has: time, init_utc, cycle, basin, mm_6h
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[df["basin"] == CFS_BASIN_NAME].copy()
    df = df[["time", "mm_6h"]].dropna()
    df = df.sort_values("time")
    return df


# -------------------------------------------------------------------
# MAIN COMPARISON
# -------------------------------------------------------------------
def compute_stats_by_lead(max_lead_hours=720, step_hours=6):
    mrms = load_mrms()

    if WET_MONTHS is not None:
        mrms = mrms[mrms["time"].dt.month.isin(WET_MONTHS)].copy()

    leads = range(step_hours, max_lead_hours + 1, step_hours)
    rows = []

    for lead_h in leads:
        cfs = load_cfs_lead(lead_h)
        if cfs is None:
            continue

        merged = pd.merge(
            cfs, mrms,
            on="time",
            how="inner",
            suffixes=("_cfs", "_obs")
        )
        
        # Clip merged series to evaluation window
        merged = merged[(merged["time"] >= EVAL_START) & (merged["time"] <= EVAL_END)]


        if WET_MONTHS is not None:
            merged = merged[merged["time"].dt.month.isin(WET_MONTHS)]

        if len(merged) < MIN_SAMPLES:
            rows.append({
                "lead_h": lead_h,
                "lead_days": lead_h / 24.0,
                "KGE": np.nan,
                "NSE": np.nan,
                "RMSE": np.nan,
                "Bias": np.nan,
                "r": np.nan,
                "N": len(merged)
            })
            continue

        sim = merged["mm_6h"].values
        obs = merged["precip_mm"].values

        # metrics
        kge_val = kge(sim, obs)
        nse_val = nse(sim, obs)
        rmse_val = rmse(sim, obs)
        bias_val = bias(sim, obs)
        r_val = np.corrcoef(sim, obs)[0, 1]

        rows.append({
            "lead_h": lead_h,
            "lead_days": lead_h / 24.0,
            "KGE": kge_val,
            "NSE": nse_val,
            "RMSE": rmse_val,
            "Bias": bias_val,
            "r": r_val,
            "N": len(merged)
        })

    return pd.DataFrame(rows,columns=["lead_h", "lead_days", "KGE", "NSE", "RMSE", "Bias", "r", "N"],)



# -------------------------------------------------------------------
# PLOT EXAMPLE: KGE vs LEAD
# -------------------------------------------------------------------
def plot_kge(stats_df, basin_name="", season_label=""):
    plt.figure(figsize=(8, 4))
    plt.plot(stats_df["lead_days"], stats_df["KGE"])
    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    title = basin_name
    if season_label:
        title += f" - {season_label}"
    plt.title(title)
    #plt.ylim(0.0, 1.0)
    plt.grid(True, axis="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    stats = compute_stats_by_lead(max_lead_hours=720, step_hours=6)

    # Save stats table if you like
    out_stats = out_dir / f"stats_{CFS_MEMBER_LABEL}_{CFS_BASIN_NAME}.csv"
    stats.to_csv(out_stats, index=False)
    print(f"Saved stats to {out_stats}")

    season_label = "Wet season" if WET_MONTHS is not None else ""
    plot_kge(stats, basin_name=CFS_BASIN_NAME, season_label=season_label)
