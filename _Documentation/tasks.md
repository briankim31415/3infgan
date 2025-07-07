# Project Task Timeline

## May 2025

-   Forked the `torchsde` GitHub repository and executed experiments on the one-dimensional Ornstein-Uhlenbeck process.
-   Deployed the codebase to the TACC server and verified successful execution.
-   Conducted an analysis of the codebase to assess parameter influence.

## June 2025

-   Collected multi-dimensional datasets referenced in the publication experiments:
    -   [Google Stock](./datasets.md#12-google-stock)
    -   [Air Quality](./datasets.md#14-air-quality-uci-ml-repository)
-   Acquired additional datasets for evaluating various noise types:
    -   [Weather](./datasets.md#13-weather-max-planck-institute-weather-station)
    -   [ETTm1](./datasets.md#15-electricity-transformer-temperature-ettm1-dataset)
-   Compiled and analyzed performance metrics across all acquired datasets.
-   Identified key arguments causing massive computational overhead.
-   Integrated Ryan Farrell's `infinite-gans` repository with the `torchsde` example codebase.
    -   Enabled logging to Weights & Biases to facilitate tracking of training progress.
    -   Extended the training logic to support multiple discriminator updates per generator update.
-   Evaluated and compared the performance of baseline versus extended training protocols.
-   Gathered urban datasets for context-specific sample generation:
    -   [Geolife](./datasets.md#21-geolife)
    -   [Human Mobility Challenge](./datasets.md#22-human-mobility-prediction-challenge-2023-metropolitan-scale-and-longitudinal-dataset-of-anonymized-human-mobility-trajectories)
    -   [Overpass/OpenRouteService API](./datasets.md#23-overpass--openrouteservice-api-via-openstreetmap)

## July 2025

-   [_In Progress_] Execute training and evaluation on urban datasets.
