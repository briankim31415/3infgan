import torch
import torchcde
import wandb
import matplotlib.pyplot as plt


def start_wandb(cfg):
    if cfg.use_wandb:
        wandb.login()
        wandb.init(
                project=cfg.wandb_proj,
                config=vars(cfg),
                name=f'{cfg.data_source}_[{cfg.timestamp}]',
                tags=[cfg.data_source, cfg.timestamp],
                mode="online" if cfg.wandb_online else "offline"
            )

def close_wandb(cfg):
    if cfg.use_wandb:
        wandb.finish()

def plot_wandb_samples(cfg, ts, generator, data_loader, step, avg=False):
    num_plot_samples = cfg.num_plot_samples

    # Get real and fake data
    real_data = data_loader.next()
    assert num_plot_samples <= real_data.size(0)
    real_data = torchcde.LinearInterpolation(real_data).evaluate(ts)

    generator.eval()
    with torch.no_grad():
        generated_data = generator(ts, real_data.size(0)).cpu()
    generated_data = torchcde.LinearInterpolation(generated_data).evaluate(ts)
    generator.train()


    real_data = real_data[:num_plot_samples]
    generated_data = generated_data[:num_plot_samples]

    # Plot samples for each feature
    # real_data = real_data.permute(2,0,1)  # Reshape to (features, num_plot_samples, sample_size)
    # generated_data = generated_data.permute(2,0,1)    # Reshape to (features, num_plot_samples, sample_size)
    # num_features = 1 # TODO: change for multi features
    # for i in range(num_features):

    # Get data for each feature and denormalize
    real_data = real_data * data_loader.std[0] + data_loader.mean[0]
    generated_data = generated_data * data_loader.std[0] + data_loader.mean[0]

    for j, real_sample_ in enumerate(real_data):
        kwargs = {'label': 'Real'} if j == 0 else {}
        plt.plot(ts.cpu(), real_sample_[:, 1].cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
    for j, generated_sample_ in enumerate(generated_data):
        kwargs = {'label': 'Generated'} if j == 0 else {}
        plt.plot(ts.cpu(), generated_sample_[:, 1].cpu(), color='crimson', linewidth=0.5, alpha=0.7, **kwargs)
    plt.legend()
    plt.title(f"{'[AVG] ' if avg else ''}Step {step}: {num_plot_samples} samples from both real and generated distributions.")
    # plt.savefig(f'{cfg.output_dir}/samples_{i+1}.png')
    fig = wandb.Image(plt)
    wandb.log({'Averaged model samples': fig} if avg else {'Samples': fig})
    plt.close()