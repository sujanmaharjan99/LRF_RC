import os
import pandas as pd
from functools import reduce
import numpy as np

# Define your four folders
folders = [
    '/media/12TB/Sujan/NWM/Csv/10_Chester/mem1/',
    '/media/12TB/Sujan/NWM/Csv/10_Chester/mem2/',
    '/media/12TB/Sujan/NWM/Csv/10_Chester/mem3/',
    '/media/12TB/Sujan/NWM/Csv/10_Chester/mem4/'
]
output_folder = '/media/12TB/Sujan/NWM/Csv/10_Chester/All/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

all_files = set(os.listdir(folders[0]))
csv_files = [f for f in all_files if f.endswith('.csv')]

for filename in csv_files:
    dfs = []
    print(f"\n Processing {filename}")
    for i, folder in enumerate(folders):
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['time'])
            df['time'] = df['time'].dt.round('min')  # Normalize timestamp
            df = df[['time', 'streamflow']]
            col_name = f'streamflow_{i+1}'
            df = df.rename(columns={'streamflow': col_name})
            df[col_name] = df[col_name].replace(-99.0, np.nan)  # Replace -99.0 with NaN
            print(f"  - {col_name}: {len(df)} timestamps | min={df[col_name].min()}, max={df[col_name].max()}")
            dfs.append(df)
        else:
            print(f"  ?? Missing: {file_path}")

    if not dfs:
        continue

    merged_df = reduce(lambda left, right: pd.merge(left, right, on='time', how='outer'), dfs)
    merged_df = merged_df.sort_values('time').drop_duplicates('time')
    print(f"  ?? Merged timestamps: {len(merged_df)}")

    streamflow_cols = [col for col in merged_df.columns if col.startswith('streamflow_')]
    merged_df['streamflow'] = merged_df[streamflow_cols].mean(axis=1, skipna=True).round(2)

    result_df = merged_df[['time', 'streamflow']]
    print(f"  ? Final mean timestamps: {len(result_df)} | min={result_df['streamflow'].min()}, max={result_df['streamflow'].max()}")

    result_df.to_csv(os.path.join(output_folder, filename), index=False)
    print(f"  ?? Saved average for {filename}")

print("\n? All done!")