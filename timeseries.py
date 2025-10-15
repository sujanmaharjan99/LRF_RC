# -*- coding: utf-8 -*-
import os
import pandas as pd
import re
import multiprocessing

# Directories
leadlist_dir = "/media/12TB/Fari-troute/NWM/leadlist/mem3"
output_dir = "/media/12TB/Sujan/NWM/Csv/03_Kansas_City/mem3/"

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)

# Target feature_id for model data filtering
target_feature_id = 4391417

# Date range for filtering
start_date = pd.to_datetime('2018-09-17')
end_date = pd.to_datetime('2024-12-31')

# Function to extract timestamp from filename
def extract_timestamp_from_filename(file_path):
    try:
        parts = file_path.split('/')
        date_str = parts[6]  # Extract YYYYMMDD from path

        cycle_match = re.search(r'nwm\.t(\d{2})z', file_path)
        lead_match = re.search(r'f(\d+)', file_path)

        if cycle_match and lead_match:
            cycle_hour = int(cycle_match.group(1))
            lead_hour = int(lead_match.group(1))
            timestamp = pd.to_datetime(date_str, format='%Y%m%d') + pd.Timedelta(hours=cycle_hour + lead_hour)
            return timestamp
    except:
        return None

# Process model file list and save CSV
def process_model_file_list(lead_time):
    lead_time_str = f"{lead_time:03d}"
    model_file_list_path = os.path.join(leadlist_dir, f"{lead_time_str}.txt")
    output_csv_path = os.path.join(output_dir, f"timeseries_{lead_time_str}.csv")

    if not os.path.exists(model_file_list_path):
        print(f"File list does not exist: {model_file_list_path}")
        return
    
    with open(model_file_list_path, 'r') as file:
        file_paths = [line.strip() for line in file.readlines() if os.path.exists(line.strip())]

    print(f"Processing {model_file_list_path} with {len(file_paths)} files...")

    data_frames = []
    for file_path in file_paths:
        try:
            timestamp = extract_timestamp_from_filename(file_path)
            if timestamp is None or not (start_date <= timestamp <= end_date):
                continue

            df = pd.read_csv(file_path, usecols=['feature_id', 'streamflow'])
            df = df[df['feature_id'] == target_feature_id]
            
            if not df.empty:
                df['time'] = timestamp
                data_frames.append(df[['time', 'streamflow']])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True).sort_values(by='time')

        # Generate complete time range (every 6 hours)
        complete_df = pd.DataFrame({'time': pd.date_range(start=start_date, end=end_date, freq='6H')})

        # Merge with model data
        merged_df = complete_df.merge(combined_df, on='time', how='left').fillna({'streamflow': -99})

        # Save to CSV
        merged_df.to_csv(output_csv_path, index=False)
        print(f"Saved: {output_csv_path}")
    else:
        print(f"No valid data found for {model_file_list_path}")

# Use multiprocessing for speedup
if __name__ == '__main__':
    with multiprocessing.Pool(processes=4) as pool:  # Adjust based on CPU cores
        pool.map(process_model_file_list, range(6, 721, 6))
