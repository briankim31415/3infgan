from datetime import datetime, time
import csv
import pandas as pd
import json
from haversine import haversine
import matplotlib.pyplot as plt
import os

TEMP_PATH = "temp"

def read_plt_file(file_path):
    """
    Read the plt file and return the content as a string.
    """
    if file_path.endswith(".plt"):
        with open(file_path, "r") as f:
            return f.read()
    else:
        raise ValueError("File must be a .plt file")

def read_csv_file_as_df(file_path) -> pd.DataFrame:
    """
    Read the csv file and return the content as a list of tuples.
    """
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError("File must be a .csv file")
    

def read_geolife_trajectory(file_path, user_id, file_name):
    """
    Read the geolife trajectory file and return a list of tuples, each containing:
    - latitude
    - longitude
    - altitude
    - days passed
    - date
    - time
    """
    DATE_FORMAT = "%Y-%m-%d"
    file = read_plt_file(file_path)
    lines = file.split("\n")
    trajectory = []
    for i, line in enumerate(lines):
        if i < 6 or line == "":
            continue
        else:
            row = line.split(",")
            time_split = row[6].split(":")
            traj = []
            traj.append(f'{user_id}_{file_name.split(".")[0]}') # trajectory id
            traj.append(float(row[0]))  # latitude
            traj.append(float(row[1]))  # longitude
            traj.append(float(row[3]))  # altitude
            traj.append(float(row[4]))  # days passed
            traj.append(row[5]) # date
            traj.append(row[6]) # time
            # traj.append(datetime.strptime(row[5], DATE_FORMAT)) # date
            # traj.append(time(int(time_split[0]), int(time_split[1]), int(time_split[2]))) # time
            trajectory.append(traj)
    return trajectory

def write_json_file(data, file_path):
    """
    Write the data to a json file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def draw_trajectory_plots(trajectory_df, debug=False):
    """
    Draw plots of the trajectory.
    """
    # Reset index
    trajectory_df = trajectory_df.copy()
    trajectory_df.reset_index(drop=True, inplace=True)

    # Plot the time difference between each point for each trajectory
    if "datetime" not in trajectory_df.columns:
        DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        trajectory_df["datetime"] = pd.to_datetime(trajectory_df["date"] + " " + trajectory_df["time"], format=DATE_FORMAT)
        trajectory_df.sort_values(by=["datetime"], inplace=True)

    if debug:
        # Plot the time difference between each point for each trajectory
        time_diffs = trajectory_df["datetime"].diff().dropna()
        plt.plot(time_diffs.dt.total_seconds())
        plt.title("Time Difference Between Points")
        plt.show()
        plt.close()

        # Plot the latitude for each trajectory
        plt.plot(trajectory_df["latitude"])
        plt.xlabel("Time")
        plt.ylabel("Latitude")
        plt.xticks(rotation=45)
        plt.title("Latitude over Time")
        plt.show()
        plt.close()

        # Plot the longitude for each trajectory
        plt.plot(trajectory_df["longitude"])
        plt.xlabel("Time")
        plt.ylabel("Longitude")
        plt.xticks(rotation=45)
        plt.title("Longitude over Time")
        plt.show()
        plt.close()

    # Plot the trajectory on a map with time gradient
    fig, ax = plt.subplots()
    sc = ax.scatter(
        trajectory_df["longitude"],
        trajectory_df["latitude"],
        c=range(len(trajectory_df)),
        cmap="viridis",
        s=10
    )
    plt.colorbar(sc, label="Time Progression")
    plt.xlabel("Longitude")
    plt.xticks(rotation=45)
    plt.ylabel("Latitude")
    plt.title("Trajectory with Time Gradient")
    plt.show()
    plt.savefig(os.path.join(TEMP_PATH, f"trajectory_{trajectory_df['trajectory_id'].iloc[0]}.png"))
    plt.close()

def distance_between_coordinates(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two coordinates in meters.
    """
    # Check if lat1 and lon1 are not out of range
    if lat1 < 39.4 or lat1 > 41.1 or lon1 < 115.7 or lon1 > 117.6:
        return 0
    if lat2 < 39.4 or lat2 > 41.1 or lon2 < 115.7 or lon2 > 117.6:
        return 0
    return haversine((lat1, lon1), (lat2, lon2)) * 1000

def interpolate_selected_columns(df, columns_to_interpolate):
    """
    Interpolates only selected numeric columns to 1-second intervals using linear interpolation.

    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' column.
        columns_to_interpolate (list): List of numeric columns to interpolate.

    Returns:
        pd.DataFrame: Interpolated DataFrame with 1-second intervals.
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime')

    # Part 1: Interpolate only numeric columns
    df_interp = (
        df[columns_to_interpolate]
        .resample('1s')
        .mean()
        .interpolate(method='linear')
    )

    # Part 2: Carry forward non-interpolated columns
    other_cols = df.columns.difference(columns_to_interpolate)
    df_other = (
        df[other_cols]
        .resample('1s')
        .first()
        .ffill()
    )

    # Combine and return
    df_combined = pd.concat([df_interp, df_other], axis=1).reset_index()
    return df_combined

def check_valid_beijing_trajectory(trajectory_df):
    """
    Check if the trajectory is valid in Beijing.
    """
    # Check longitude and latitude are within Beijing
    if trajectory_df["longitude"].min() < 115.7 or trajectory_df["longitude"].max() > 117.6:
        return False
    if trajectory_df["latitude"].min() < 39.4 or trajectory_df["latitude"].max() > 41.1:
        return False
    return True

def next_timestep(d, t):
    """Next timestep for mobility data"""
    if t == 47:
        if d == 90:
            raise ValueError("ERROR: End of trajectory")
        return d + 1, 0
    else:
        return d, t + 1
