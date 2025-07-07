# Dataset Overview for Infinite GANs

This document describes the datasets used in our study. Two main categories are presented:

-   [**Testing Datasets**](#1-testing-datasets) – Time-series / stochastic data used for model benchmarking.
-   [**Urban Datasets**](#2-urban-datasets) – Urban trajectory, mobility, and road-network datasets.

## 1. Testing Datasets

The [Ornstein-Uhlenbeck process](#11-ornsteinuhlenbeck-process), [Google Stock](#12-google-stock), [Weather](#13-weather-max-planck-institute-weather-station), and [Air Quality](#14-air-quality-uci-ml-repository) datasets were used in the original [Infinite GAN](https://arxiv.org/abs/2209.12894) paper.

### 1.1 Ornstein–Uhlenbeck Process

-   **Dimensions**: 1
-   **Data points**: Synthetic (no fixed size)
-   **Description**: A classic stochastic process exhibiting mean-reverting behavior, commonly used in physics and finance. This synthetic dataset serves as a controlled testbed to benchmark the model's ability to learn and generate time-series with noise and temporal correlation.

### 1.2 Google Stock

-   **Dimensions**: 6
-   **Data points**: 4,431
-   **Time Granularity**: Daily
-   **Time span**: Aug 19, 2004 – Mar 24, 2022
-   **Description**: A multivariate financial time-series dataset capturing daily trading statistics for Google stock, including open, close, high, low prices, adjusted close, and volume. Used to evaluate model performance on real-world market data.

### 1.3 Weather ([Max Planck Institute Weather Station](https://www.bgc-jena.mpg.de/wetter/))

-   **Dimensions**: 21
-   **Data points**: 52,696
-   **Time Granularity**: 10 minutes
-   **Time span**: Jan 1, 2020 – Jan 1, 2021
-   **Location**: Jena, Germany
-   **Description**: High-resolution meteorological dataset collected from a Max Planck Institute weather station in Jena, Germany. It includes temperature, wind, humidity, solar radiation, and other variables across 21 channels. Designed to test multi-channel prediction and pattern generation in weather forecasting.

### 1.4 Air Quality ([UCI ML Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data))

-   **Dimensions**: 11
-   **Data points**: 35,064
-   **Time Granularity**: Hourly
-   **Time span**: Mar 1, 2013 – Feb 28, 2017
-   **Location**: Aoti Zhongxin, Beijing, China
-   **Description**: Air quality data collected from multiple locations in Beijing, with this study focusing on the Aoti Zhongxin station for consistency. Contains hourly measurements of PM2.5, PM10, NO2, CO, O3, SO2, and meteorological data, allowing evaluation of model performance on environmental monitoring tasks.

### 1.5 Electricity Transformer Temperature ([ETTm1 Dataset](https://github.com/zhouhaoyi/ETDataset))

-   **Dimensions**: 7
-   **Data points**: 69,680
-   **Time Granularity**: 15 minutes
-   **Time span**: Jul 1, 2016 – Jun 26, 2018
-   **Location**: China
-   **Description**: Time-series data collected from electric power transformers, recording internal temperatures and operational variables. Widely used in long-term time-series forecasting benchmarks due to its periodicity and variability across time.

## 2. Urban Datasets

### 2.1 [Geolife](https://www.microsoft.com/en-us/download/details.aspx?id=52367)

> Not included in GitHub repository due to large file size.

-   **Dimensions**: 3
-   **Data points**: 24,876,978
-   **Location precision**: Latitude, longitude in degrees; altitude in feet
-   **Time granularity**: 91% of trajectories logged every 1~5 seconds or every 5~10 meters
-   **Time span**: Apr 5, 2007 – Jul 12, 2012
-   **Location**: Primarily Beijing, China
-   **Description**: Dataset of 18,670 GPS trajectories of 182 users over five years primarily in Beijing, China. Captures human mobility patterns with fine-grained spatial and temporal resolution. Each trajectory includes latitude, longitude, and altitude, offering a rich source of movement behaviors for urban modeling tasks. The average trajectory length is 1,332 data points.

### 2.2 Human Mobility Prediction Challenge 2023 ([_Metropolitan Scale and Longitudinal Dataset of Anonymized Human Mobility Trajectories_](https://arxiv.org/pdf/2307.03401))

-   **Dimensions**: 2
-   **Data points**: 140,924,924
-   **Location precision**: 500x500 meter tiles on 200x200 tile grid
-   **Time granularity**: 30 minutes
-   **Time span**: 90 days
-   **Location**: Anonymized city in Japan
-   **Description**: An extensive mobility dataset from an anonymized Japanese city, covering 100,000 users over 90 days. Each user's movement is mapped to 500x500m spatial tiles in a 200x200 grid, updated every 30 minutes. Includes dense metadata such as POI category counts per tile. Designed for urban-scale trajectory and flow modeling.

### 2.3 [Overpass](https://wiki.openstreetmap.org/wiki/Overpass_API) & [OpenRouteService](https://openrouteservice.org/) API (via [OpenStreetMap](https://www.openstreetmap.org/))

-   **Dimensions**: 2
-   **Data points**: Limited by API
    -   _Overpass_: 10,000 queries/day & 1 GB data/day
    -   _OpenRouteService_: ~40 queries/min & ~2,000 queries/day
-   **Location precision**: Latitude, longitude in degrees
-   **Time granularity**: N/A
-   **Time span**: N/A
-   **Location**: Global
-   **Description**: Urban spatial data retrieved from OpenStreetMap through the Overpass and OpenRouteService APIs. Overpass provides raw vector geometries (e.g. roads, buildings, POIs) and metadata within a bounding box. OpenRouteService offers route planning services, returning detailed navigation steps, distance, and estimated travel time between two coordinates under different transport modes (e.g., walking, cycling, driving). Valuable for incorporating road network structures and accessibility constraints into trajectory generation tasks.
