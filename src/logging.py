import wandb
import matplotlib.pyplot as plt

from. data import Data

def start_wandb(cfg):
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
    if cfg.use_wandb:
        wandb.finish()

def plot_wandb_samples(cfg, ts, step, generator, data_loader: Data, avg=False):
    # Get real and fake data
    real_samples = data_loader.get_real_samples()
    fake_samples = data_loader.get_fake_samples(generator)

    # Get only the first num_plot_samples samples
    num_plot_samples = min(cfg.num_plot_samples, real_samples.size(0))
    real_samples = real_samples[:num_plot_samples]
    fake_samples = fake_samples[:num_plot_samples]

    # Determine how many value channels we have (excluding the time channel)
    num_channels = real_samples.size(-1) - 1
    channel_indices = range(1, num_channels + 1)

    # Plot each channel
    for v in channel_indices:
        for i, real_sample_ in enumerate(real_samples):
            kwargs = {'label': 'Real'} if i == 0 else {}
            plt.plot(ts.cpu(), real_sample_[:, v].cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
        for i, generated_sample_ in enumerate(fake_samples):
            kwargs = {'label': 'Generated'} if i == 0 else {}
            plt.plot(ts.cpu(), generated_sample_[:, v].cpu(), color='crimson', linewidth=0.5, alpha=0.7, **kwargs)
        plt.legend()
        channel_title = f" - {data_loader.cols[v - 1]}" if num_channels > 1 else ""
        title = f"{'[AVG] ' if avg else ''}Step {step}: {num_plot_samples} real vs fake samples{channel_title}"
        plt.title(title)
        fig = wandb.Image(plt)
        key = f"Averaged model samples{channel_title}" if avg else f"Samples{channel_title}"
        wandb.log({key: fig})
        plt.close()