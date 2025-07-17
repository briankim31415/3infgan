# Infinite GAN: 2D TSM Simulator Model Guide

This guide explains how to train an Infinite GAN model for use in the 2D TSM simulator. While we demonstrate the process using the cleaned **Geolife** dataset, the same steps can be adapted for any custom dataset with minor code additions.

## Overview

The workflow for using Infinite GAN in the 2D simulator involves:

1. Setting up a configuration file.
2. Preprocessing and loading data.
3. Training the model locally or on TACC.
4. Extracting the final trained generator for simulation.

## 📦 Using a Pre-Existing Dataset

If you're working with an existing dataset (e.g., Geolife), you can skip data preprocessing. Simply follow the **Local Training** instructions in the main [README](../README.md#️-local-training) to begin training.

## 🆕 Using a New Dataset

To train Infinite GAN on a new dataset, follow the steps below.

### Step 1: Create a Configuration File

Create a `.yaml` configuration file in the `confs/` directory. At a minimum, define:

-   `data_source`: Name of the CSV file containing the data.
-   `wandb_proj`: Project name for logging in Weights & Biases.
-   `wandb_name`: Unique run name for tracking the experiment.

You may optionally override additional training or architecture parameters by copying them from `default.yaml`. Any parameters not included in your custom config will fall back to the default values.

### Step 2: Add Data Preprocessing Logic

If you're using a custom dataset, you'll need to add a new loading method in `data.py`.

1. **Create a new method** that:

    - Loads and parses the dataset (from the CSV).
    - Groups rows into time-series trajectories.
    - Converts the result into a tensor with shape `(n_trajectories, t_size, feature_dim)`.
    - Sets `self.cols` to a list of feature column names for logging.

2. **Update the dataset loading logic:**
    - Navigate to the `if-else` block starting at line 72 in `data.py`.
    - Add an `elif` case that checks for your new `data_source` string and calls the newly defined data loader.

> **Note:** Each trajectory must be at least `t_size` time steps long.

### Step 3: Train the Infinite GAN

You can run the model either locally or on a compute cluster.

#### ✅ Local Training

Use the following command:

```bash
python -m src.run --cfg_name={config_name}.yaml
```

To enable online logging with Weights & Biases:

1. Set your API key as an environment variable (see [README](../README.md#-setup-weights--biases-api-on-linuxmacos)).
2. Add the following flags:

```bash
--use_wandb --online
```

#### 🚀 Running on TACC

1. In the `sbatch/` directory, create a new `.bat` file with appropriate SBATCH directives (e.g., job name, time, GPU settings).

2. Add the following commands to set up the environment and run training:

```bash
cd ..
source ./venv/bin/activate
module load gcc cuda python3
python3 -m src.run --cfg_name={config_name}.yaml {--additional_flags}
```

3. Submit the job with:

```bash
sbatch {your_script_name}.bat
```

## 📤 Exporting the Final Generator Model

After training, the final generator model will be saved in the `models/` directory as:

```text
{wandb_name}_generator_{timestamp}.pth
```

To use this model in the 2D TSM simulator:

1. Copy the model file to the following directory in the simulator repo:

```
tsm-barrage-sim/infgan/
```

2. Make sure you're on the `feature/inf-gan-movement` branch of the repository:

> [tsm_barrage_sim/feature/inf-gan-movement](https://github.com/CVC-DBG/tsm-barrage-sim/tree/feature/inf-gan-movement)
