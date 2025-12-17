#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NWM_BASE = Path("/mnt/12TB/Sujan/Csv")
USGS_BASE = Path("/mnt/12TB/Sujan/USGS_data")

MEMBERS = ["All", "mem1", "mem2", "mem3", "mem4"]
LEAD_HOURS = list(range(6, 721, 6))
LEAD_DAYS = np.array(LEAD_HOURS) / 24.0

OUTPUT_ROOT = Path(__file__).resolve().parent / "Persistance"

NWM_STATIONS = [
    "01_Rulo",
    "02_St_Joseph",
    "03_Kansas_City",
    "04_Waverly",
    "05_Boonville",
    "06_Hermann",
    "07_St_Charles",
    #"08_Grafton",
    "09_ST_Louis",
    "10_Chester",
    "11_Thebes",
]

USGS_FILES = {
    "01_Rulo": "01_Rulo_06813500.csv",
    "02_St_Joseph": "02_St_Joseph_06818000.csv",
    "03_Kansas_City": "03_Kansas_City_06893000.csv",
    "04_Waverly": "04_Waverly_06895500.csv",
    "05_Boonville": "05_Boonville_06909000.csv",
    "06_Hermann": "06_Hermann_06934500.csv",
    "07_St_Charles": "07_St_Charles_06935965.csv",
    "08_Grafton": "08_Grafton_05587450.csv",
    "09_ST_Louis": "09_ST_Louis_07010000.csv",
    "10_Chester": "10_Chester_07020500.csv",
    "11_Thebes": "11_Thebes_07022000.csv",
}

UPSTREAM_OVERRIDE = {
    "01_Rulo": "06601200.csv",
    "09_ST_Louis": "07_St_Charles",# special upstream file for Rulo
}

USGS_TIME_COL = "timestamp"
USGS_Q_COL = "discharge_cumecs" 

def parse_datetime_flex(s: pd.Series) -> pd.Series:
    # Try month-first (US style) first
    dt1 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    # Try day-first second
    dt2 = pd.to_datetime(s, errors="coerce", dayfirst=True)

    # Choose whichever produces more valid timestamps
    if dt2.notna().sum() > dt1.notna().sum():
        return dt2
    return dt1

def get_upstream_file(station_key: str):
    if station_key in UPSTREAM_OVERRIDE:
        val = UPSTREAM_OVERRIDE[station_key]
        # if they gave a csv filename
        if val.lower().endswith(".csv"):
            return USGS_BASE / val
        # otherwise treat as station key
        return USGS_BASE / USGS_FILES[val]

    # default: previous station in ordered list
    try:
        i = NWM_STATIONS.index(station_key)
    except ValueError:
        return None
    if i == 0:
        return None
    upstream_key = NWM_STATIONS[i - 1]
    return USGS_BASE / USGS_FILES[upstream_key]


def kge(sim: np.ndarray, obs: np.ndarray) -> float:
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)

    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[mask]
    obs = obs[mask]
    if sim.size < 3:
        return np.nan

    obs_std = np.std(obs, ddof=1)
    sim_std = np.std(sim, ddof=1)
    if obs_std == 0 or sim_std == 0:
        return np.nan

    r = np.corrcoef(sim, obs)[0, 1]
    alpha = sim_std / obs_std

    obs_mean = np.mean(obs)
    sim_mean = np.mean(sim)
    if obs_mean == 0:
        return np.nan
    beta = sim_mean / obs_mean

    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def read_timeseries_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" not in df.columns or "streamflow" not in df.columns:
        raise ValueError(f"Expected columns ['time','streamflow'] in {path}")

    df["time"] = parse_datetime_flex(df["time"])
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    df["streamflow"] = pd.to_numeric(df["streamflow"], errors="coerce")
    return df[["streamflow"]].dropna()



def find_nwm_file(station_dir: Path, member: str, lead_hr: int) -> Path:
    return station_dir / member / f"timeseries_{lead_hr:03d}.csv"

def read_usgs_file_15min(usgs_path: Path) -> pd.DataFrame:
    df = pd.read_csv(usgs_path)

    if "timestamp" not in df.columns:
        raise ValueError(f"{usgs_path} missing 'timestamp'. Columns: {df.columns.tolist()}")

    df["timestamp"] = parse_datetime_flex(df["timestamp"])
    df = df.dropna(subset=["timestamp"])

    colmap = {c.strip(): c for c in df.columns}
    if "discharge_cumecs" not in colmap:
        raise ValueError(f"{usgs_path} missing 'discharge_cumecs'. Columns: {df.columns.tolist()}")
    q_col = colmap["discharge_cumecs"]

    df["obs"] = pd.to_numeric(df[q_col], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    return df[["obs"]].dropna()

def align_nearest(usgs_df, target_times, tol_minutes=20):
    target_times = pd.DatetimeIndex(target_times).dropna().sort_values()
    base = pd.DataFrame({"time": target_times})

    obs = usgs_df.reset_index()
    obs = obs.rename(columns={obs.columns[0]: "time"}).sort_values("time")
    obs = obs.dropna(subset=["time"])

    out = pd.merge_asof(
        base, obs,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tol_minutes),
    )

    return out.set_index("time")["obs"]


def compute_kge_vs_lead(station_key: str) -> pd.DataFrame:
    station_dir = NWM_BASE / station_key
    if not station_dir.exists():
        raise FileNotFoundError(f"NWM station folder not found: {station_dir}")

    # Downstream (station itself)
    down_path = USGS_BASE / USGS_FILES[station_key]
    usgs_down_15min = read_usgs_file_15min(down_path)

    # Upstream (spatial persistence source)
    up_path = get_upstream_file(station_key)
    usgs_up_15min = read_usgs_file_15min(up_path) if (up_path and up_path.exists()) else None

    out_rows = []

    for lead_hr, lead_day in zip(LEAD_HOURS, LEAD_DAYS):
        row = {"lead_hours": lead_hr, "lead_days": lead_day}

        # Reference NWM timeline (prefer All)
        ref_path = find_nwm_file(station_dir, "All", lead_hr)
        if not ref_path.exists():
            ref_path = None
            for m in MEMBERS:
                p = find_nwm_file(station_dir, m, lead_hr)
                if p.exists():
                    ref_path = p
                    break

        if ref_path is None:
            for member in MEMBERS:
                row[f"N_{member}"] = 0
                row[f"KGE_{member}"] = np.nan
            row["N_persistence_temporal"] = 0
            row["KGE_persistence_temporal"] = np.nan
            row["N_persistence_spatial"] = 0
            row["KGE_persistence_spatial"] = np.nan
            out_rows.append(row)
            continue

        ref_nwm = read_timeseries_csv(ref_path)
        t = ref_nwm.index  # NWM valid times

        # Downstream obs at t
        obs_t = align_nearest(usgs_down_15min, t)

        # Lead-dependent temporal persistence: down(t-lead) vs down(t)
        lag = pd.Timedelta(hours=lead_hr)
        down_tlag = align_nearest(usgs_down_15min, t - lag)
        down_tlag.index = t

        mask_temp = obs_t.notna() & down_tlag.notna()
        row["N_persistence_temporal"] = int(mask_temp.sum())
        row["KGE_persistence_temporal"] = kge(down_tlag[mask_temp].values, obs_t[mask_temp].values)

        # Lead-dependent spatial persistence: up(t-lead) vs down(t)
        if usgs_up_15min is None:
            row["N_persistence_spatial"] = 0
            row["KGE_persistence_spatial"] = np.nan
        else:
            up_tlag = align_nearest(usgs_up_15min, t - lag)
            up_tlag.index = t
            mask_spat = obs_t.notna() & up_tlag.notna()
            row["N_persistence_spatial"] = int(mask_spat.sum())
            row["KGE_persistence_spatial"] = kge(up_tlag[mask_spat].values, obs_t[mask_spat].values)

        # NWM member KGEs
        for member in MEMBERS:
            nwm_path = find_nwm_file(station_dir, member, lead_hr)
            if not nwm_path.exists():
                row[f"N_{member}"] = 0
                row[f"KGE_{member}"] = np.nan
                continue

            nwm_df = read_timeseries_csv(nwm_path).rename(columns={"streamflow": "sim"})
            obs_tm = align_nearest(usgs_down_15min, nwm_df.index)
            mask = obs_tm.notna() & nwm_df["sim"].notna()

            row[f"N_{member}"] = int(mask.sum())
            row[f"KGE_{member}"] = kge(nwm_df["sim"][mask].values, obs_tm[mask].values)

        out_rows.append(row)

    return pd.DataFrame(out_rows)



def plot_kge(station_key: str, df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure()
    x = df["lead_days"].values

    for member in MEMBERS:
        col = f"KGE_{member}"
        if col in df.columns:
            plt.plot(x, df[col].values, linewidth=1, label=member)

    # lead-dependent persistence
    plt.plot(x, df["KGE_persistence_temporal"].values, linewidth=1, label="Temporal persistence (down lag=lead)")
    plt.plot(x, df["KGE_persistence_spatial"].values,  linewidth=1, label="Spatial persistence (up lag=lead)")

    plt.xlabel("Lead time (days)")
    plt.ylabel("KGE")
    plt.title(f"{station_key}: KGE vs Lead Time")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.ylim(-1.0, 1.0)
    plt.legend()

    out_path = out_dir / "kge_vs_leadtime.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for station_key in NWM_STATIONS:
        print(f"\nProcessing {station_key} ...")
        out_dir = OUTPUT_ROOT / station_key
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            df = compute_kge_vs_lead(station_key)
            df.to_csv(out_dir / "kge_vs_leadtime.csv", index=False)
            plot_kge(station_key, df, out_dir)
            (out_dir / "done.txt").write_text("OK\n", encoding="utf-8")
            print(f"  Saved results to: {out_dir}")
        except Exception as e:
            print(f"  ERROR: {e}")
            (out_dir / "error.txt").write_text(str(e) + "\n", encoding="utf-8")
            
if __name__ == "__main__":
    main()