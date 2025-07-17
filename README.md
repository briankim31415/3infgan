# Infinite-GANs for Synthetic Urban Scene Generation

This repository explores the use of **Neural SDEs as Infinite-Dimensional GANs** for generating synthetic data, with the goal of producing **urban scene data**. The codebase supports training, sample generation, and evaluation using **Wasserstein distance**, with tracking via [Weights & Biases](https://wandb.ai/).

> Originally developed and tested on [UT Austin's TACC cluster](https://www.tacc.utexas.edu/), but can be run locally.

## 🔬 Background

This project builds on the paper:

> **Neural SDEs as Infinite-Dimensional GANs**  
> _Patrick Kidger et al._  
> [arXiv:2209.12894](https://arxiv.org/abs/2209.12894)

The architecture leverages a **Neural SDE-based generator** and **Neural CDE-based discriminator**, enabling data generation in function space rather than fixed-dimensional representations.

## 🚀 Features

-   ✔️ Infinite-dimensional GAN framework (SDE/CDE architecture)
-   🧠 Neural SDE Generator + Neural CDE Discriminator
-   📉 Wasserstein loss implementation
-   📊 Real-time experiment tracking with Weights & Biases
-   🖼️ Synthetic sample generation + image logging
-   ⚙️ Modular config files via YAML config files
-   🧮 Batch job support for UT TACC SLURM cluster

## 📁 Project Structure

```
infinite-gans/
├── _Documentation/       # Markdown documents
│   ├── problem_statement.md # Formulation of project goal
│   ├── breakdown_ryan.md    # Infinite GAN breakdown by Ryan Roby
│   ├── datasets.md          # Overview of included datasets
│   ├── demo.md              # Showcase of current results
│   ├── tasks.md             # Project tasks timeline
│   └── 2d_sim_guide.md      # Guided steps on how to generate 2D TSM sim model
├── src/                  # Core implementation
│   ├── run.py               # Command-line interface
│   ├── train.py             # Training loop
│   ├── data.py              # Real data loader
│   ├── generator.py         # Generator (Neural SDE)
│   ├── discriminator.py     # Discriminator (Neural CDE)
│   ├── losses.py            # Wasserstein loss
│   ├── logging.py           # Sample logging
│   └── utils.py             # Helper methods
├── confs/                # Configuration files
│   ├── default.yaml         # Default parameters
│   ├── {dataset}.yaml       # Dataset-specific configs
│   └── {dataset}_basic.yaml # Simplified logic configs
├── data/                 # Datasets
├── sbatch/               # SLURM batch scripts for TACC
└── figures/              # Figures for documentation
```

## 🛠️ Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/briankim31415/3infgan.git
    cd 3infgan
    ```

2. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

### 📊 Setup Weights & Biases API (on Linux/macOS)

1. Add your Weights & Biases API key to your shell config file as an environment variable:

    ```bash
    export WANDB_API_KEY=your_api_key_here
    ```

2. Apply the changes:

    ```bash
    source ~/.bashrc  # or source ~/.zshrc
    ```

3. Verify environment variable is set correctly:

    ```bash
    # In the Terminal
    echo $WANDB_API_KEY
    ```

    ```python
    # In Python
    import os
    print(os.getenv("WANDB_API_KEY"))
    ```

## 🧪 Running the Code

### ▶️ Local Training

```bash
python -m src.run   # Generates Ornstein-Uhlenbeck process by default
```

Optional flags:

-   ` --cfg_name=<config_name>`: Specify a config file to run.
-   `--use_wandb`: Enable Weights & Biases logging.
-   `--online`: Set Weights & Biases mode to online.
-   `--cfg_name`: Specify a config from `confs/`.

Example:

```bash
python -m src.run --cfg_name=weather.yaml --use_wandb --online
```

### 🧬 TACC Cluster (SLURM)

To run on the UT TACC cluster:

```bash
cd sbatch/
sbatch <job_file>.sbatch
```

Edit the `.sbatch` file to point to the config and parameters you want to use.

### ⚙️ Config Directory Structure

Each config file begins with the data source name. In the case of the Ornstein-Uhlenbeck process, it is set as the `default` data source. There are 2 types of configuration files per data source:

-   `{data_source}.yaml` runs the expanded training logic with 5 discriminator updates per generator update.
-   `{data_source}_basic.yaml` runs the basic training logic with 1 discriminator update per generator update.

The `simple` designation for the `geolife` dataset tests performance on only 1 column of the dataset for validation purposes.

## 📚 Citation

If you use this codebase, please cite the original work:

```
@article{kidger2022neural,
  title={Neural SDEs as Infinite-Dimensional GANs},
  author={Kidger, Patrick and Foster, James and Lyons, Terry and Salvi, Corrado},
  journal={arXiv preprint arXiv:2209.12894},
  year={2022}
}
```
