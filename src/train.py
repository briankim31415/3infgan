import torch
from torch.optim.swa_utils import AveragedModel, SWALR
import tqdm
import wandb

from .data import Data
from .generator import Generator
from .discriminator import Discriminator
from .losses import generator_loss, evaluate_loss, wasserstein_loss
from .logging import start_wandb, close_wandb, plot_wandb_samples
from .utils import set_seed



def train(cfg):
    set_seed(cfg.seed)
    start_wandb(cfg)
    
    data_loader = Data(cfg)
    ts = data_loader.ts
    data_size = data_loader.data_size

    generator = Generator(data_size, cfg.initial_noise_size, cfg.noise_size, cfg.hidden_size, cfg.mlp_size, cfg.num_layers, cfg.g_dt).to(cfg.device)
    discriminator = Discriminator(data_size, cfg.hidden_size, cfg.mlp_size, cfg.num_layers, cfg.d_dt).to(cfg.device)

    gen_optm = torch.optim.Adadelta(generator.parameters(), lr=cfg.generator_lr, weight_decay=cfg.weight_decay)
    dis_optm = torch.optim.Adadelta(discriminator.parameters(), lr=cfg.discriminator_lr, weight_decay=cfg.weight_decay)

    averaged_generator = AveragedModel(generator)
    averaged_discriminator = AveragedModel(discriminator)
    swa_scheduler_g = SWALR(gen_optm, swa_lr=cfg.generator_lr * 0.1)
    swa_scheduler_d = SWALR(dis_optm, swa_lr=cfg.discriminator_lr * 0.1)

    # Picking a good initialisation is important!
    # In this case these were picked by making the parameters for the t=0 part of the generator be roughly the right
    # size that the untrained t=0 distribution has a similar variance to the t=0 data distribution.
    # Then the func parameters were adjusted so that the t>0 distribution looked like it had about the right variance.
    # What we're doing here is very crude -- one can definitely imagine smarter ways of doing things.
    # (e.g. pretraining the t=0 distribution)
    with torch.no_grad():
        for param in generator._initial.parameters():
            param *= cfg.init_mult1
        for param in generator._func.parameters():
            param *= cfg.init_mult2

    trange = tqdm.tqdm(range(cfg.num_steps))
    for step in trange:
        if cfg.basic:
            real_samples = data_loader.next()

            generated_samples = generator(ts, cfg.batch_size)
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            # loss = generated_score - real_score
            loss = wasserstein_loss(cfg, real_score, generated_score)
            loss.backward()

            for param in generator.parameters():
                param.grad *= -1
            gen_optm.step()
            dis_optm.step()
            gen_optm.zero_grad()
            dis_optm.zero_grad()

            # Constrain the Lipschitz constant of the discriminator
            with torch.no_grad():
                for module in discriminator.modules():
                    if isinstance(module, torch.nn.Linear):
                        lim = 1 / module.out_features
                        module.weight.clamp_(-lim, lim)
            
            # Stochastic weight averaging typically improves performance.
            if step > cfg.swa_step_start: # TODO: make this a multiple of a config param (i.e. cfg.0.5 * steps)
                averaged_generator.update_parameters(generator)
                averaged_discriminator.update_parameters(discriminator)
                swa_scheduler_g.step()
                swa_scheduler_d.step()

            # Logging
            if step % cfg.log_interval == 0:
                total_unavg_loss = evaluate_loss(ts, cfg.batch_size, data_loader, generator, discriminator)

                metrics = {
                    'metrics/total_unavg_loss': total_unavg_loss,
                    'metrics/real_score_mean': real_score.mean().item(),
                    'metrics/fake_score_mean': generated_score.mean().item(),
                    'metrics/step': step
                }

                if step > cfg.swa_step_start:
                    total_averaged_loss = evaluate_loss(ts, cfg.batch_size, data_loader, averaged_generator.module, averaged_discriminator.module)
                    metrics['metrics/total_avg_loss'] = total_averaged_loss


                trange.set_postfix({k: f"{v:.3f}" for k, v in metrics.items()})

                if cfg.use_wandb:
                    wandb.log(metrics)

                    ################
                    # PROBING
                    ################                        
                    # MODULE B: X_0
                    wandb.log({
                        'probe/moduleB_x0_mean': generator.x0.mean().item(),
                        'probe/moduleB_x0_std': generator.x0.std().item()
                    })

                    # MODULE E: Y
                    wandb.log({
                        'probe/moduleE_y_fake_mean': generated_samples[..., 1].mean().item(),
                        'probe/moduleE_y_real_mean': real_samples[..., 1].mean().item(),
                        'probe/moduleE_y_fake_std': generated_samples[..., 1].std().item(),
                        'probe/moduleE_y_real_std': real_samples[..., 1].std().item()
                    })

                    # MODULE F: H_0
                    wandb.log({
                        'probe/moduleF_h0_mean': discriminator.h0.mean().item(),
                        'probe/moduleF_h0_std': discriminator.h0.std().item()
                    })

                    # MODULE H: D
                    wandb.log({
                        'probe/moduleH_d_mean': generated_score.mean().item(),
                        'probe/moduleH_d_std': generated_score.std().item()
                    })
                    ################
                
                    # Per channel logging (skip time channel at ch=0)
                    for ch in range(1, generated_samples.shape[-1]):
                        ch_loss = generated_samples[..., ch].mean() - real_samples[..., ch].mean()
                        wandb.log({
                            f'channel_stats/channel_{ch}_gen_mean': generated_samples[..., ch].mean().item(),
                            f'channel_stats/channel_{ch}_real_mean': real_samples[..., ch].mean().item(),
                            f'channel_stats/channel_{ch}_gen_std': generated_samples[..., ch].std().item(),
                            f'channel_stats/channel_{ch}_real_std': real_samples[..., ch].std().item(),
                            f'channel_stats/channel_{ch}_loss': ch_loss
                        })
                
                
            if step % cfg.samples_interval == 0 and cfg.use_wandb:
                plot_wandb_samples(cfg, ts, step, generator, data_loader)
                
                if step > cfg.swa_step_start:
                    plot_wandb_samples(cfg, ts, step, averaged_generator, data_loader, avg=True)

        else:
            # Update discriminator 5 times per generator update
            dis_loss_sum = 0.0
            for _ in range(cfg.d_updates_per_g):
                # Get real and fake data
                real_data = data_loader.next()
                with torch.no_grad():
                    fake_data = generator(ts, cfg.batch_size)

                # Get real and fake scores
                real_scores = discriminator(real_data)
                fake_scores = discriminator(fake_data)

                # Get loss (fake_scores - real_scores)
                dis_loss = wasserstein_loss(cfg, real_scores, fake_scores)
                dis_loss_sum += dis_loss.item()
                dis_loss.backward()

                dis_optm.step()
                dis_optm.zero_grad()

                # Constrain the Lipschitz constant of the discriminator
                with torch.no_grad():
                    for module in discriminator.modules():
                        if isinstance(module, torch.nn.Linear):
                            lim = 1 / module.out_features
                            module.weight.clamp_(-lim, lim)
            
            dis_loss = dis_loss_sum / cfg.d_updates_per_g

            # Get real and fake data
            real_data = data_loader.next()
            fake_data = generator(ts, cfg.batch_size)

            # Get real and fake scores
            with torch.no_grad():
                real_scores = discriminator(real_data)
            fake_scores = discriminator(fake_data)

            # Get loss
            gen_loss = wasserstein_loss(cfg, real_scores, fake_scores)
            gen_loss.backward()

            for param in generator.parameters():
                param.grad *= -1

            gen_optm.step()
            gen_optm.zero_grad()


            # Stochastic weight averaging typically improves performance.
            if step > cfg.swa_step_start:
                averaged_generator.update_parameters(generator)
                averaged_discriminator.update_parameters(discriminator)
                swa_scheduler_g.step()
                swa_scheduler_d.step()
            
            # Logging
            if step % cfg.log_interval == 0:
                total_unavg_loss = evaluate_loss(ts, cfg.batch_size, data_loader, generator, discriminator)

                metrics = {
                    'discriminator_loss': dis_loss,
                    'generator_loss': gen_loss.item(),
                    'total_unavg_loss': total_unavg_loss,
                    'real_score_mean': real_scores.mean().item(),
                    'fake_score_mean': fake_scores.mean().item(),
                    'step': step
                }
                trange.set_postfix({k: f"{v:.3f}" for k, v in metrics.items()})

                if cfg.use_wandb:
                    wandb.log(metrics)

                    ################
                    # PROBING
                    ################
                    # MODULE A: V ~ N(0,I_v)
                    # wandb.log({
                    #     'probe/moduleA_v_mean': generator.noise.mean().item(),
                    #     'probe/moduleA_v_std': generator.noise.std().item()
                    # })
                        
                    # MODULE B: X_0
                    wandb.log({
                        'probe/moduleB_x0_mean': generator.x0.mean().item(),
                        'probe/moduleB_x0_std': generator.x0.std().item()
                    })

                    # MODULE E: Y
                    wandb.log({
                        'probe/moduleE_y_fake_mean': fake_data[..., 1].mean().item(),
                        'probe/moduleE_y_real_mean': real_data[..., 1].mean().item(),
                        'probe/moduleE_y_fake_std': fake_data[..., 1].std().item(),
                        'probe/moduleE_y_real_std': real_data[..., 1].std().item()
                    })

                    # MODULE F: H_0
                    wandb.log({
                        'probe/moduleF_h0_mean': discriminator.h0.mean().item(),
                        'probe/moduleF_h0_std': discriminator.h0.std().item()
                    })

                    # MODULE H: D
                    wandb.log({
                        'probe/moduleH_d_mean': fake_scores.mean().item(),
                        'probe/moduleH_d_std': fake_scores.std().item()
                    })
                    ################
                
                    # Per channel logging (skip time channel at ch=0)
                    for ch in range(1, fake_data.shape[-1]):
                        ch_loss = fake_data[..., ch].mean() - real_data[..., ch].mean()
                        wandb.log({
                            f'channel_stats/channel_{ch}_gen_mean': fake_data[..., ch].mean().item(),
                            f'channel_stats/channel_{ch}_real_mean': real_data[..., ch].mean().item(),
                            f'channel_stats/channel_{ch}_gen_std': fake_data[..., ch].std().item(),
                            f'channel_stats/channel_{ch}_real_std': real_data[..., ch].std().item(),
                            f'channel_stats/channel_{ch}_loss': ch_loss
                        })
                
            if step % cfg.samples_interval == 0 and cfg.use_wandb:
                plot_wandb_samples(cfg, ts, step, generator, data_loader)
                
                if step > cfg.swa_step_start:
                    plot_wandb_samples(cfg, ts, step, averaged_generator, data_loader, avg=True)

    close_wandb(cfg)

    return generator, discriminator