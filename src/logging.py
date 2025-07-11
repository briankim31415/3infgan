"""
LOGGING.PY

This file contains the logging functions for the Infinite GAN.

It contains the following functions:
- start_wandb: Start wandb for logging.
- close_wandb: Close wandb instance.
- plot_wandb_samples: Generate and plot samples locally and upload to wandb.
"""


import wandb
import os
import matplotlib.pyplot as plt
from. data import Data

def start_wandb(cfg):
    """Start wandb for logging."""
    if cfg.use_wandb:
        if cfg.config_num == 0:
            name = f'{cfg.wandb_name}_[{cfg.timestamp}]'
        else:
            name = f'{cfg.wandb_name}_cfg{cfg.config_num}_[{cfg.timestamp}]'

        wandb.login()
        wandb.init(
                project=cfg.wandb_proj,
                config=vars(cfg),
                name=name,
                tags=[cfg.wandb_name, cfg.data_source, cfg.timestamp],
                mode="online" if cfg.wandb_online else "offline"
            )

def close_wandb(cfg):
    """Close wandb instance."""
    if cfg.use_wandb:
        wandb.finish()

def plot_wandb_samples(cfg, ts, step, generator, data_loader: Data, avg=False):
    """Generate and plot samples locally and upload to wandb."""
    # Get real and fake data.
    real_samples = data_loader.get_real_samples()
    fake_samples = data_loader.get_fake_samples(generator)

    # Get only the first num_plot_samples samples
    num_plot_samples = min(cfg.num_plot_samples, real_samples.size(0))
    real_samples = real_samples[:num_plot_samples]
    fake_samples = fake_samples[:num_plot_samples]

    # Determine how many value channels we have (excluding the time channel)
    num_channels = real_samples.size(-1) - 1
    channel_indices = range(1, num_channels + 1)

    safe_timestamp = cfg.timestamp.replace("/", "-")
    output_dir = f"output/{safe_timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Plot each channel
    for v in channel_indices:
        for i, real_sample_ in enumerate(real_samples):
            kwargs = {'label': 'Real'} if i == 0 else {}
            plt.plot(ts.cpu(), real_sample_[:, v].cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
        for i, generated_sample_ in enumerate(fake_samples):
            kwargs = {'label': 'Fake'} if i == 0 else {}
            plt.plot(ts.cpu(), generated_sample_[:, v].cpu(), color='crimson', linewidth=0.5, alpha=0.7, **kwargs)
        plt.legend()
        channel_title = f" - {data_loader.cols[v - 1]}" if num_channels > 1 else ""
        title = f"{'[AVG] ' if avg else ''}Step {step}: {num_plot_samples} real vs fake samples{channel_title}"
        fig, ax = plt.subplots()
        ax.annotate(f"STEP {step}",
                xy=(0, 1), xycoords='axes fraction',
                xytext=(27, 21), textcoords='offset points',
                ha='right', va='bottom',
                fontsize=15,
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", ec="black", lw=1))
        plt.title(title)
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        fig = wandb.Image(plt)
        if avg:
            key = f"Averaged model samples{channel_title}"
            image_path = f"{output_dir}/{cfg.wandb_name}_avg_c{v}_{step}.png"
        else:
            key = f"Samples{channel_title}"
            image_path = f"{output_dir}/{cfg.wandb_name}_c{v}_{step}.png"
        wandb.log({key: fig})
        plt.savefig(image_path)
        plt.close()