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
  # Keep the set as it is
# List to track incomplete dates
incomplete_dates = []

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

 # Get user input for date range
start_date_str = input("Enter start date (YYYYMMDD): ")
end_date_str = input("Enter end date (YYYYMMDD): ")

try:
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    if start_date > end_date:
        print("Error: Start date must be before end date.")
    else:
        bucket_name = "national-water-model"

        while start_date <= end_date:
            folder_path = f"nwm.{start_date.strftime('%Y%m%d')}/long_range_mem3/"
            destination_folder = f"{input_directory}/{start_date.strftime('%Y%m%d')}/long_range_mem3"

            print(f"Downloading {start_date.strftime('%Y-%m-%d')}...")
            loop = asyncio.get_event_loop()
            file_count = loop.run_until_complete(download_gcs_folder(bucket_name, folder_path, destination_folder))

            start_date += timedelta(days=1)

        if incomplete_dates:
            print("Incomplete dates (missing files):", incomplete_dates)
        else:
            print("All dates processed successfully.")

except ValueError:
    print("Error: Invalid date format. Use YYYYMMDD.")