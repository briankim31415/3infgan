import pandas as pd
import os
from .data_utils import *
import csv
import requests
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import gzip
import shutil
from tqdm import tqdm

DATA_PATH = "data"
OUTPUT_PATH = "output"
TEMP_PATH = "temp"
MOBILITY_INPUT_PATH = "data/mobility"
OPENROUTE_API_KEY = os.getenv("OPENROUTE_API_KEY")

###########################
#      RUN PARAMETERS     #
###########################
DEBUG = True
NUM_TRAJECTORIES = 15000
RUN_METHOD = ["expand_mobility"]
###########################

# def debug():
#     faulty_df = read_csv_file_as_df(os.path.join(TEMP_PATH, "faulty_trajectories.csv"))
#     grouped = faulty_df.groupby("trajectory_id")
#     first_5_grouped = list(grouped)[:5]
#     for trajectory_id, trajectory in grouped:
#         print(trajectory_id)
#         print(trajectory.head(20))
#         break

def debug():
    df = read_csv_file_as_df(os.path.join(TEMP_PATH, "faulty_trajectories.csv"))
    grouped = df.groupby("trajectory_id")
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    df["datetime"] = pd.to_datetime(df["datetime"], format=DATE_FORMAT)
    for trajectory_id, group in grouped:
        # count number of datetime differences greater than 1 second
        time_diffs = group["datetime"].diff().dropna()
        gt_1_seconds = time_diffs[time_diffs > pd.Timedelta(seconds=1)]
        if len(gt_1_seconds) > 0:
            # Print time step before and after the time difference
            for i in range(len(gt_1_seconds)):
                print(f"Time step before: {group.iloc[gt_1_seconds.index[i] - 1]['datetime']}")
                print(f"Time at time difference: {group.iloc[gt_1_seconds.index[i]]['datetime']}")
                print(f"Time step after: {group.iloc[gt_1_seconds.index[i] + 1]['datetime']}")
                print(f"Time difference: {gt_1_seconds.iloc[i]}")
                print("--------------------------------")
    print(f"Total number of trajectories with time differences greater than 1 second: {len(grouped)}")

def debug1():
    BEIJING_PATH = os.path.join(TEMP_PATH, "beijing_15000_trajectories_resampled.csv")
    df = read_csv_file_as_df(BEIJING_PATH)

    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    df["datetime"] = pd.to_datetime(df["datetime"], format=DATE_FORMAT)

    # Only keep datetime, latitude, longitude, trajectory_id
    df = df[["datetime", "latitude", "longitude", "trajectory_id"]]

    # Get datetime differences
    count = 0
    faulty_trajectories = []
    grouped = df.groupby("trajectory_id")
    for trajectory_id, trajectory in tqdm(grouped):
        trajectory["time_diff"] = trajectory["datetime"].diff().dropna()
        
        # Count number of time differences greater than 1 second
        if len(trajectory[trajectory["time_diff"] > pd.Timedelta(seconds=1)]) > 0:
            faulty_trajectories.append(trajectory)
            count += 1
    
    print(f"Number of time differences greater than 1 second: {count}")

    # Save faulty trajectories to a csv file
    faulty_df = pd.concat(faulty_trajectories)
    faulty_df.to_csv(os.path.join(TEMP_PATH, "faulty_trajectories.csv"), index=False)

###########################
#     GEOLIFE DATASET     #
###########################
def make_geolife_dataset():
    """
    Read the Geolife dataset and return a list of trajectories, each containing:
    - latitude
    - longitude
    - altitude
    - days passed
    - date
    - time
    """
    GEOLIFE_PATH = os.path.join(DATA_PATH, "Geolife")
    GEOLIFE_OUTPUT_PATH = os.path.join(OUTPUT_PATH, "Geolife")
    trajectory_list = []

    user_ids = [id for id in os.listdir(GEOLIFE_PATH) if id.isdigit()]
    user_ids.sort()
    for user_id in user_ids:
        # Read the trajectory file
        trajectory_path = os.path.join(GEOLIFE_PATH, user_id, "Trajectory")
        for file in os.listdir(trajectory_path):
            if file.endswith(".plt"):
                # Add trajectories to the list
                trajectories = read_geolife_trajectory(os.path.join(trajectory_path, file), user_id, file)
                trajectory_list.extend(trajectories)
        
        # For debugging
        if DEBUG and user_id == "001":
            break
    
    # Save trajectory list to a csv file and add header
    header = ["trajectory_id", "latitude", "longitude", "altitude", "days_passed", "date", "time"]
    with open(os.path.join(GEOLIFE_OUTPUT_PATH, "all_trajectories.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(trajectory_list)
    
    return trajectory_list

def find_geolife_ranges():
    """
    Find the maximum altitude in the Geolife dataset
    """
    GEOLIFE_PATH = os.path.join(OUTPUT_PATH, "Geolife")
    trajectory_list = read_csv_file_as_df(os.path.join(GEOLIFE_PATH, "all_trajectories.csv"))
    # print(f"Max latitude: {trajectory_list['latitude'].max()}")
    # print(f"Min latitude: {trajectory_list['latitude'].min()}")
    # print(f"Max longitude: {trajectory_list['longitude'].max()}")
    # print(f"Min longitude: {trajectory_list['longitude'].min()}")
    # print(f"Max altitude: {trajectory_list['altitude'].max()}")
    # print(f"Min altitude: {trajectory_list['altitude'].min()}")
    # print(f"Max days passed: {trajectory_list['days_passed'].max()}")
    # print(f"Min days passed: {trajectory_list['days_passed'].min()}")
    # print(f"Lowest 30 altitudes: {trajectory_list['altitude'].sort_values().head(30)}")
    # print(f"Highest 30 altitudes: {trajectory_list['altitude'].sort_values().tail(30)}")
    altitude_list = trajectory_list['altitude'].sort_values().tolist()

    # Remove outliers
    altitude_list = [altitude for altitude in altitude_list if altitude > -2000 and altitude < 30000]

    # Histogram of altitude
    plt.hist(altitude_list, bins=100)
    plt.show()

def get_geolife_cities():
    """
    Add city information to Geolife trajectories using city shapefile.
    """
    GEOLIFE_PATH = os.path.join(OUTPUT_PATH, "Geolife")
    trajectory_df = read_csv_file_as_df(os.path.join(GEOLIFE_PATH, "all_trajectories.csv"))

    # Load city shapefile
    city_shp_file = os.path.join(DATA_PATH, "gadm41_CHN_shp", "gadm41_CHN_2.shp")
    city_gdf = gpd.read_file(city_shp_file)
    city_gdf = city_gdf.to_crs("EPSG:4326")

    # Prepare trajectory dataframe as GeoDataFrame
    trajectory_df["geometry"] = trajectory_df.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
    traj_gdf = gpd.GeoDataFrame(trajectory_df, geometry="geometry", crs="EPSG:4326")

    # Spatial join every point with city boundaries
    joined = gpd.sjoin(traj_gdf, city_gdf[["NAME_2", "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={"NAME_2": "city"})

    # Merge back with original to ensure all points are preserved
    traj_gdf["city"] = joined["city"].fillna("unknown")

    # Save the result
    output_path = os.path.join(GEOLIFE_PATH, "all_trajectories_with_city.csv")
    traj_gdf.drop(columns="geometry").to_csv(output_path, index=False)
    print(f"Saved with city info to {output_path}")

def split_geolife_by_city():
    """
    Split the Geolife trajectory dataset into separate CSV files by city.
    """
    GEOLIFE_PATH = os.path.join(OUTPUT_PATH, "Geolife")
    input_path = os.path.join(GEOLIFE_PATH, "all_trajectories_with_city.csv")
    city_output_path = os.path.join(GEOLIFE_PATH, "city_split")
    os.makedirs(city_output_path, exist_ok=True)

    df = read_csv_file_as_df(input_path)
    if "city" not in df.columns:
        raise ValueError("Expected 'city' column in data. Please run get_geolife_cities() first.")
    
    print(df.head(5))
    print(df.columns)

    # Group trajectories by their starting city's name
    print(len(df))
    grouped = df.groupby("trajectory_id")
    print(len(grouped))
    city_groups = {}
    for traj_id, group in grouped:
        start_city = group.iloc[0]["city"]
        if start_city not in city_groups:
            city_groups[start_city] = []
        city_groups[start_city].append(group)

    for city, traj_list in city_groups.items():
        city_df = pd.concat(traj_list)
        safe_city = str(city).replace(" ", "_").replace("/", "-")
        city_df.to_csv(os.path.join(city_output_path, f"{safe_city}.csv"), index=False)

    print(f"Saved {len(grouped)} city files to {city_output_path}")

def count_geolife_city_trajectories():
    """
    Count the number of trajectories in each city
    """
    CITY_PATH = os.path.join(OUTPUT_PATH, "Geolife", "city_split")
    cities = os.listdir(CITY_PATH)
    cities.sort()
    
    # Count trajectories in each city
    city_counts = {}
    for city in cities:
        if city.endswith(".csv"):
            df = read_csv_file_as_df(os.path.join(CITY_PATH, city))
            city_counts[city] = df['trajectory_id'].nunique()
    
    # Sort cities by number of trajectories
    city_counts = dict(sorted(city_counts.items(), key=lambda item: item[1]))
    for city, count in city_counts.items():
        print(f"{city}: {count} trajectories")

def get_average_trajectory_length():
    """
    Get the average trajectory length for each city
    """
    # CITY_PATH = os.path.join(OUTPUT_PATH, "Geolife", "city_split")
    # cities = os.listdir(CITY_PATH)
    # cities.sort()

    # for city in cities:
    #     df = read_csv_file_as_df(os.path.join(CITY_PATH, city))
    #     traj_groups = df.groupby("trajectory_id").groups
    #     lengths = []
    #     for traj_id, indices in traj_groups.items():
    #         lengths.append(len(indices))
    #     print(f"{city}: Average trajectory length = {sum(lengths) / len(lengths):.2f} points")
    
    # Get average trajectory length of entire dataset
    df = read_csv_file_as_df(os.path.join(OUTPUT_PATH, "Geolife", "all_trajectories.csv"))

    # traj_groups = df.groupby("trajectory_id").groups
    # lengths = []
    # for traj_id, indices in traj_groups.items():
    #     lengths.append(len(indices))
    # print(f"Average trajectory length of entire dataset = {sum(lengths) / len(lengths):.2f} points")

    # Get total number of data points
    print(f"Total number of data points: {df.shape[0]}")
        
def clean_beijing_for_inf_gan():
    """
    Clean the Beijing dataset for Infinite GANs.
    """
    BEIJING_PATH = os.path.join(OUTPUT_PATH, "Geolife", "city_split", "Beijing.csv")
    df = read_csv_file_as_df(BEIJING_PATH)

    # Remove trajectories with less than 100 points
    print(f'Number of trajectories before cleaning: {df["trajectory_id"].nunique()}')
    traj_counts = df["trajectory_id"].value_counts()
    valid_ids = traj_counts[traj_counts >= 100].index
    df = df[df["trajectory_id"].isin(valid_ids)]
    print(f'Number of trajectories after cleaning: {df["trajectory_id"].nunique()}')

    # Only keep the trajectory_id, latitude, longitude, and altitude columns
    df = df[["trajectory_id", "latitude", "longitude", "altitude"]]

    # Save the cleaned dataset
    df.to_csv(os.path.join(OUTPUT_PATH, "Geolife", "Beijing_cleaned.csv"), index=False)

def create_temp_beijing_dataset():
    """
    Create a temp Beijing dataset for Infinite GANs.
    """
    BEIJING_PATH = os.path.join(OUTPUT_PATH, "Geolife", "city_split", "Beijing.csv")
    df = read_csv_file_as_df(BEIJING_PATH)
    print(df.head(5))

    # Save first 10000 trajectories to a temp parquet file
    grouped = df.groupby("trajectory_id")
    count = 0
    save_df = pd.DataFrame()
    for traj_id, group in grouped:
        if count == NUM_TRAJECTORIES:
            break
        count += 1
        save_df = pd.concat([save_df, group])
    save_df.to_parquet(os.path.join(TEMP_PATH, f"beijing_{NUM_TRAJECTORIES}_trajectories.parquet"))

def analyze_temp_beijing_dataset():
    """
    Analyze the temp Beijing dataset for Infinite GANs.
    """
    df = pd.read_parquet(os.path.join(TEMP_PATH, "beijing_10k_trajectories.parquet"))

    # plot the time difference between each point for each trajectory
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    grouped = df.groupby("trajectory_id")
    for traj_id, group in grouped:
        print(len(group))
        # Convert time to datetime
        group["datetime"] = pd.to_datetime(group["date"] + " " + group["time"], format=DATE_FORMAT)

        # Get time difference in seconds
        time_diffs = group["datetime"].diff().dropna()
        # print(time_diffs.describe())

        # Show line plot of time difference for each trajectory
        # plt.plot(time_diffs.dt.total_seconds())
        # plt.show()

        # plt.hist(time_diffs.dt.total_seconds(), bins=100)
        # plt.show()

        # average time difference
        print(time_diffs.dt.total_seconds().mean())
        break

def split_large_gaps_in_trajectory(trajectory_df, data_point_length=100, time_gap_threshold=50, distance_gap_threshold=100):
    """
    Split large gaps in the trajectory.
    """
    # Reset index
    trajectory_df = trajectory_df.copy()
    trajectory_df.reset_index(drop=True, inplace=True)

    if "datetime" not in trajectory_df.columns:
        DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        trajectory_df["datetime"] = pd.to_datetime(trajectory_df["date"] + " " + trajectory_df["time"], format=DATE_FORMAT)
        trajectory_df.sort_values(by=["datetime"], inplace=True)

    # Find large time gaps
    large_gaps = [0, len(trajectory_df)]
    time_diffs = trajectory_df["datetime"].diff().dropna()
    large_gaps.extend(time_diffs[time_diffs > pd.Timedelta(seconds=time_gap_threshold)].index.tolist())

    # Find large distance gaps
    for i in range(len(trajectory_df) - 1):
        lat_1 = trajectory_df.iloc[i]["latitude"]
        long_1 = trajectory_df.iloc[i]["longitude"]
        lat_2 = trajectory_df.iloc[i+1]["latitude"]
        long_2 = trajectory_df.iloc[i+1]["longitude"]
        dist = distance_between_coordinates(lat_1, long_1, lat_2, long_2)
        if dist > distance_gap_threshold:
            if i + 1 not in large_gaps:
                large_gaps.append(i + 1)

    # Split the trajectory at the large gaps
    large_gaps.sort()
    split_trajectories = []
    for i in range(len(large_gaps) - 1):
        if large_gaps[i+1] - large_gaps[i] >= data_point_length:
            split_trajectories.append(trajectory_df.iloc[large_gaps[i]:large_gaps[i+1]])

    # Return the split trajectories
    return split_trajectories

def resample_beijing_dataset():
    """
    Resample the Beijing dataset for Infinite GANs.
    """
    # Read the parquet file
    df = pd.read_parquet(os.path.join(TEMP_PATH, f"beijing_{NUM_TRAJECTORIES}_trajectories.parquet"))

    # Convert time to datetime
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format=DATE_FORMAT)
    df.sort_values(by=["trajectory_id", "datetime"], inplace=True)

    # Resample the data
    resampled_data = []
    for (trajectory_id), group in tqdm(df.groupby(["trajectory_id"])):
        # Split large gaps in trajectory
        split_trajectories = split_large_gaps_in_trajectory(group)

        # Interpolate the trajectory
        for trajectory in split_trajectories:
            if check_valid_beijing_trajectory(trajectory):
                interpolated = interpolate_selected_columns(trajectory, ["latitude", "longitude", "altitude"])
                resampled_data.append(interpolated)
            else:
                print(f"Trajectory {trajectory_id} is not valid in Beijing")
        
    # Save the resampled data to a csv file
    resampled_df = pd.concat(resampled_data)
    resampled_df.to_csv(os.path.join(TEMP_PATH, f"beijing_{NUM_TRAJECTORIES}_trajectories_resampled.csv"), index=False)


###########################
#    MOBILITY DATASET     #
###########################

def unpack_mobility_dataset():
    """
    Unpack the mobility dataset.
    """ 
    MOBILITY_PATH = os.path.join(DATA_PATH, "mobility")
    for file in os.listdir(MOBILITY_PATH):
        if file.endswith(".gz"):
            print(f"Unpacking {file}...")
            with gzip.open(os.path.join(MOBILITY_PATH, file), 'rb') as f_in:
                with open(os.path.join(MOBILITY_PATH, file.replace(".gz", "")), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Unpacked {file} to {file.replace('.gz', '')}")
            os.remove(os.path.join(MOBILITY_PATH, file))

def get_mobility_data():
    """
    Get the mobility data.
    """
    MOBILITY_PATH = os.path.join(DATA_PATH, "mobility")
    # df = read_csv_file_as_df(os.path.join(MOBILITY_PATH, "yjmob100k-dataset1.csv"))
    # print(df.head(5))

    # # Save first 10000 users to a temp parquet file
    # df = df[df["uid"].isin(range(10000))]
    # df.to_parquet(os.path.join(TEMP_PATH, "mobility_10k_users.parquet"))

    # Read the parquet file
    df = pd.read_parquet(os.path.join(TEMP_PATH, "mobility_10k_users.parquet"))
    print(df.head(5))

def expand_mobility(count=10000):
    """
    Expand the mobility dataset.
    """
    # Read the csv file
    df = read_csv_file_as_df(os.path.join(DATA_PATH, "mobility", "yjmob100k-dataset2.csv"))
    expanded = []

    # Group by uid
    grouped = df.groupby("uid")
    grouped = list(grouped)[:count]
    for uid, group in tqdm(grouped):
        # Get the first date and time
        d = group.iloc[0]["d"]
        t = group.iloc[0]["t"]

        # Get last date and time
        end_d = group.iloc[-1]["d"]
        end_t = group.iloc[-1]["t"]

        # Get first x, y
        x = group.iloc[0]["x"]
        y = group.iloc[0]["y"]

        # Get the first row index
        row_index = 0

        # Iterate through the dates and times
        while (d < end_d) or (d == end_d and t <= end_t):
            # Check if current date and time matches the current row
            if d == group.iloc[row_index]["d"] and t == group.iloc[row_index]["t"]:
                x = group.iloc[row_index]["x"]
                y = group.iloc[row_index]["y"]
                row_index += 1
            
            # Add x, y for current date and time to list
            expanded.append({"uid": uid, "d": d, "t": t, "x": x, "y": y})

            # Get next date and time
            d, t = next_timestep(d, t)
        
    # Save the expanded data to a csv file
    expanded_df = pd.DataFrame(expanded)
    expanded_df.to_csv(os.path.join(TEMP_PATH, f"mobility2_expanded_{count}.csv"), index=False)

###########################
#    OPENSTREETMAP API    #
###########################
def fetch_overpass():
    """
    Fetch data from the Overpass API using a POST request.
    Returns the JSON response.
    """
    url = "https://overpass-api.de/api/interpreter"
    query = """
        [bbox:30.618338,-96.323712,30.591028,-96.330826]
        [out:json]
        [timeout:90]
        ;
        (
            way
                (
                    30.626917110746,
                    -96.348809105664,
                    30.634468750236,
                    -96.339893442898
                );
        );
        out geom;
    """
    data = {"data": query}
    response = requests.post(url, data=data)
    response.raise_for_status()
    resp = response.json()
    output_file = os.path.join(OUTPUT_PATH, "overpass_response.json")
    write_json_file(resp, output_file)

def fetch_openroute():
    """
    Fetch data from the OpenRoute API using a POST request.
    Returns the JSON response.
    """
    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
    }
    call = requests.get(f'https://api.openrouteservice.org/v2/directions/driving-car?api_key={OPENROUTE_API_KEY}&start=8.681495,49.41461&end=8.687872,49.420318', headers=headers)
    output_file = os.path.join(OUTPUT_PATH, "openroute_response.json")
    write_json_file(call.json(), output_file)

###########################
#      MAIN FUNCTION      #
###########################
def main():
    if DEBUG:
        print("DEBUGGING...")
        debug()
    else:
        for method in RUN_METHOD:
            if method in globals() and callable(globals()[method]):
                print(f"Running {method}...")
                globals()[method]()
            else:
                raise ValueError(f"Method '{method}' is not defined.")

if __name__ == "__main__":
    main()