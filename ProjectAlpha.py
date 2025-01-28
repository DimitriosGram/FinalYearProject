import kagglehub
import pandas as pd
import streamlit as st
import os

# Download the latest version of the dataset
path = kagglehub.dataset_download("whisperingkahuna/premier-league-2324-team-and-player-insights")

# Function to load CSV files from a specified subfolder into DataFrames
def load_csv_files_from_subfolder(folder_path):
    if os.path.isdir(folder_path):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        dataframes = {}
        if csv_files:
            for csv_file in csv_files:
                csv_path = os.path.join(folder_path, csv_file)
                dataframes[csv_file] = pd.read_csv(csv_path)
            return dataframes
        else:
            st.write("No CSV files found in the subfolder.")
            return None
    else:
        st.write(f"Subfolder '{folder_path}' not found.")
        return None

# Specify the subfolder containing the CSV files
subfolder_name = 'Premleg_23_24'
subfolder_path = os.path.join(path, subfolder_name)

# Load the data into DataFrames
dataframes = load_csv_files_from_subfolder(subfolder_path)

# st.write the names and previews of the DataFrames
if dataframes:
    for name, df in dataframes.items():
        st.write(f"Preview of {name}:")
        st.write(df.head())
