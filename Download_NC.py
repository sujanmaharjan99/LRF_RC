import argparse
import asyncio
import aiohttp
import os
import xarray as xr
import pandas as pd
import h5py
import netCDF4
from google.cloud import storage
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/media/12TB/Sujan/NWM/Codes/Key/steady-library-470316-r2-5ea9851180c5.json"

# Define directories
input_directory = "/media/12TB/Sujan/NWM/Data"
output_directory = "/media/12TB/Sujan/NWM/output1"
 
# Set of feature_ids to extract
feature_ids_set = {880478,2252949,3624735,3702540,4388401,4391417,5089904,5092616,5166621,6010106,6013072}

# List to track incomplete dates
incomplete_dates = []

def parse_args():
    """Parse command-line arguments for start date, end date and ensemble members."""
    parser = argparse.ArgumentParser(
        description="Download and process National Water Model long-range ensembles."
    )
    parser.add_argument(
        "--start", required=True, help="Start date in YYYYMMDD format"
    )
    parser.add_argument(
        "--end", required=True, help="End date in YYYYMMDD format"
    )
    parser.add_argument(
        "--ensemble",
        default="all",
        help="Comma-separated ensemble members to download (mem1, mem2, mem3, mem4) or 'all'"
    )
    return parser.parse_args()

async def async_download_file(session, blob, destination_folder, retries=3, delay=5):
    """Downloads a file asynchronously with retry logic."""
    destination_file = os.path.join(destination_folder, os.path.basename(blob.name))
    if os.path.exists(destination_file):
        return
    
    for attempt in range(retries):
        try:
            async with session.get(blob.public_url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status == 200:
                    with open(destination_file, 'wb') as f:
                        f.write(await response.read())
                    return
                else:
                    print(f"Attempt {attempt+1}: Failed to download {blob.name} (HTTP {response.status})")
        except asyncio.TimeoutError:
            print(f"Attempt {attempt+1}: Timeout error while downloading {blob.name}")
        except Exception as e:
            print(f"Attempt {attempt+1}: Error downloading {blob.name}: {e}")
        
        await asyncio.sleep(delay)

async def download_gcs_folder(bucket_name, folder_path, destination_folder, max_concurrent_downloads=5):
    """Downloads NetCDF files from GCS with concurrency control."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = [blob for blob in bucket.list_blobs(prefix=folder_path) if "channel" in blob.name.lower() and not blob.name.endswith("/")]
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    semaphore = asyncio.Semaphore(max_concurrent_downloads)
    async with aiohttp.ClientSession() as session:
        tasks = [async_download_file(session, blob, destination_folder) for blob in blobs]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    return len(blobs)

def main():
    args = parse_args()
    try:
        start_date = datetime.strptime(args.start, "%Y%m%d")
        end_date = datetime.strptime(args.end, "%Y%m%d")
    except ValueError:
        print("Invalid date format. Use YYYYMMDD.")
        return

    if start_date > end_date:
        print("Start date must not exceed end date.")
        return

    # Determine ensembles to download
    valid = {'mem1', 'mem2', 'mem3', 'mem4'}
    if args.ensemble.lower() == "all":
        ensembles = sorted(valid)
    else:
        ensembles = [e.strip().lower() for e in args.ensemble.split(',') if e.strip()]
        if not ensembles or any(e not in valid for e in ensembles):
            print(f"Invalid ensemble selection. Choose from {', '.join(sorted(valid))} or 'all'.")
            return

    bucket_name = "national-water-model"
    incomplete_by_ensemble = {e: [] for e in ensembles}

    # Loop over dates and ensembles
    date = start_date
    while date <= end_date:
        date_str = date.strftime("%Y%m%d")
        print(f"Processing {date.strftime('%Y-%m-%d')}...")
        for ensemble in ensembles:
            folder_path = f"nwm.{date_str}/long_range_{ensemble}/"
            dest_folder = f"{input_directory}/{date_str}/long_range_{ensemble}"
            print(f"  Downloading {ensemble}…")
            file_count = asyncio.run(
                download_gcs_folder(bucket_name, folder_path, dest_folder)
            )
            if file_count < 480:
                incomplete_by_ensemble[ensemble].append(date.strftime("%Y-%m-%d"))
            else:
                process_netcdf_files(dest_folder)
                # Optionally remove raw .nc files
                # os.system(f"rm -rf {dest_folder}")
        date += timedelta(days=1)

    # Summary
    for ensemble, missing_dates in incomplete_by_ensemble.items():
        if missing_dates:
            print(f"Incomplete dates for {ensemble}: {missing_dates}")
        else:
            print(f"All dates for {ensemble} processed successfully.")

if __name__ == "__main__":
    main()
