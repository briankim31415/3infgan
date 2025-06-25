import torch
import torch.optim.swa_utils as swa_utils
import tqdm
import wandb

from .data import Data
from .generator import Generator
from .discriminator import Discriminator
from .losses import discriminator_loss, generator_loss, evaluate_loss
from .utils import start_wandb, close_wandb, set_seed, plot_wandb_samples

'''
Init g/d
Loop sample/train
Wandb logging loss/scores
Call plot wandb samples
Call losses
'''


def train(cfg):
    set_seed(cfg.seed)
    start_wandb(cfg)
    
    data_loader = Data(cfg)
    ts = data_loader.ts
    data_size = data_loader.data_size

    generator = Generator(data_size, cfg.initial_noise_size, cfg.noise_size, cfg.hidden_size, cfg.mlp_size, cfg.num_layers, cfg.g_dt).to(cfg.device)
    discriminator = Discriminator(data_size, cfg.hidden_size, cfg.mlp_size, cfg.num_layers, cfg.d_dt).to(cfg.device)

    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)
    # swa_scheduler_g = SWALR(g_optimizer, swa_lr=cfg.eta_g * 0.1)
    # swa_scheduler_d = SWALR(d_optimizer, swa_lr=cfg.eta_d * 0.1)

    # TODO UPDATE THESE PARAMETERS
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

    gen_optm = torch.optim.Adadelta(generator.parameters(), lr=cfg.generator_lr, weight_decay=cfg.weight_decay)
    dis_optm = torch.optim.Adadelta(discriminator.parameters(), lr=cfg.discriminator_lr, weight_decay=cfg.weight_decay)

    trange = tqdm.tqdm(range(cfg.num_steps))
    for step in trange:
        if cfg.basic:
            real_samples = data_loader.next()

            generated_samples = generator(ts, cfg.batch_size)
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = generated_score - real_score
            loss.backward()

            for param in generator.parameters():
                param.grad *= -1
            gen_optm.step()
            dis_optm.step()
            gen_optm.zero_grad()
            dis_optm.zero_grad()

            ###################
            # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
            # LipSwish activation functions).
            ###################
            with torch.no_grad():
                for module in discriminator.modules():
                    if isinstance(module, torch.nn.Linear):
                        lim = 1 / module.out_features
                        module.weight.clamp_(-lim, lim)
            
            # Stochastic weight averaging typically improves performance.
            if step > cfg.swa_step_start: # TODO: make this a multiple of a config param (i.e. cfg.0.5 * steps)
                averaged_generator.update_parameters(generator)
                averaged_discriminator.update_parameters(discriminator)

            if step % cfg.log_interval == 0:
                total_unavg_loss = evaluate_loss(ts, cfg.batch_size, data_loader.dataloader, generator, discriminator)

                metrics = {
                    'total_unavg_loss': total_unavg_loss,
                    'real_score_mean': real_score.item(),
                    'fake_score_mean': generated_score.item(),
                    'step': step
                }

                if step > cfg.swa_step_start:
                    total_averaged_loss = evaluate_loss(ts, cfg.batch_size, data_loader.dataloader, averaged_generator.module, averaged_discriminator.module)
                    metrics['total_avg_loss'] = total_averaged_loss


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
                        'probe/moduleH_d_mean': generated_score.item(),
                        'probe/moduleH_d_std': generated_score.item()
                    })
                    ################
                
            if step % cfg.samples_interval == 0 and cfg.use_wandb:
                plot_wandb_samples(cfg, ts, generator, data_loader, step)
                
                if step > cfg.swa_step_start:
                    plot_wandb_samples(cfg, ts, averaged_generator, data_loader, step, avg=True)

        else:
            # Update discriminator 5 times per generator update
            for _ in range(cfg.d_updates_per_g):
                # Get real and fake data
                real_data = data_loader.next()
                with torch.no_grad():
                    fake_data = generator(ts, cfg.batch_size)

                # Get real and fake scores
                real_score = discriminator(real_data)
                fake_score = discriminator(fake_data)

                # Get discriminator loss (fake_score - real_score)
                dis_loss = discriminator_loss(real_score, fake_score)
                dis_loss.backward()

                dis_optm.step()
                dis_optm.zero_grad()

                ###################
                # TODO: MOVE TO ANOTHER FILE?
                ###################
                # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
                # LipSwish activation functions).
                ###################
                with torch.no_grad():
                    for module in discriminator.modules():
                        if isinstance(module, torch.nn.Linear):
                            lim = 1 / module.out_features
                            module.weight.clamp_(-lim, lim)

            # Get fake data and score
            fake_data = generator(ts, cfg.batch_size)
            fake_score = discriminator(fake_data)

            # Get generator loss
            gen_loss = generator_loss(fake_score)
            gen_loss.backward()

            # TODO WHAT DOES THIS DO??
            for param in generator.parameters():
                param.grad *= -1

            gen_optm.step()
            gen_optm.zero_grad()

            ###################
            # TODO: MOVE TO ANOTHER FILE?
            ###################
            # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
            # LipSwish activation functions).
            ###################
            with torch.no_grad():
                for module in discriminator.modules():
                    if isinstance(module, torch.nn.Linear):
                        lim = 1 / module.out_features
                        module.weight.clamp_(-lim, lim)


            # Stochastic weight averaging typically improves performance.
            if step > cfg.swa_step_start: # TODO: make this a multiple of a config param (i.e. cfg.0.5 * steps)
                averaged_generator.update_parameters(generator)
                averaged_discriminator.update_parameters(discriminator)
                # swa_scheduler_g.step()
                # swa_scheduler_d.step()
            
            # Logging
            if step % cfg.log_interval == 0:
                total_unavg_loss = evaluate_loss(ts, cfg.batch_size, data_loader.dataloader, generator, discriminator)

                metrics = {
                    'discriminator_loss': dis_loss.item(),
                    'generator_loss': gen_loss.item(),
                    'total_unavg_loss': total_unavg_loss,
                    'real_score_mean': real_score.item(),
                    'fake_score_mean': fake_score.item(),
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
                        'probe/moduleH_d_mean': fake_score.item(),
                        'probe/moduleH_d_std': fake_score.item()
                    })
                    ################
                
            if step % cfg.samples_interval == 0 and cfg.use_wandb:
                plot_wandb_samples(cfg, ts, generator, data_loader, step)
                
                if step > cfg.swa_step_start:
                    plot_wandb_samples(cfg, ts, averaged_generator, data_loader, step, avg=True)

    close_wandb(cfg)

    return generator, discriminator