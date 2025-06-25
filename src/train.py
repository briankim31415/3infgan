import torch
import torch.optim.swa_utils as swa_utils
import tqdm
import wandb

from .data import Data
from .generator import Generator
from .discriminator import Discriminator
from .losses import discriminator_loss, generator_loss
from .utils import start_wandb, close_wandb

'''
Init g/d
Loop sample/train
Wandb logging loss/scores
Call plot wandb samples
Call losses
'''


def train(cfg):
    start_wandb(cfg)
    
    data_loader = Data(cfg)
    ts = data_loader.ts

    generator = Generator()
    discriminator = Discriminator()

    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

    # TODO WHAT DOES THIS DO??
    with torch.no_grad():
        for param in generator._initial.parameters():
            param *= cfg.init_mult1
        for param in generator._func.parameters():
            param *= cfg.init_mult2

    gen_optm = torch.optim.Adadelta(generator.parameters(), lr=cfg.generator_lr, weight_decay=cfg.weight_decay)
    dis_optm = torch.optim.Adadelta(discriminator.parameters(), lr=cfg.discriminator_lr, weight_decay=cfg.weight_decay)

    trange = tqdm.tqdm(range(cfg.num_steps))
    for step in trange:
        for _ in range(cfg.d_updates_per_g):
            dis_optm.zero_grad()

            fake_data = generator(ts, cfg.batch_size)

            real_data = data_loader.get_real_samples()

            real_score = discriminator(real_data)
            fake_score = discriminator(fake_data)

            dis_loss = discriminator_loss(real_score, fake_score)

            dis_loss.backward()
            dis_optm.step()
            dis_optm.zero_grad()

        fake_data = generator(ts, cfg.batch_size)
        fake_score = discriminator(fake_data)

        gen_loss = generator_loss(fake_score)

        gen_loss.backward()
        gen_optm.step()

        # TODO WHAT DOES THIS DO??
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
        if step > cfg.swa_step_start:
            averaged_generator.update_parameters(generator)
            averaged_discriminator.update_parameters(discriminator)

    close_wandb(cfg)