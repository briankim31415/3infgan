import torch
import wandb

def wasserstein_distance(x, y, p=1):
    """
    Approximate Wasserstein-p distance between two 1D tensors using sorting.
    Retains gradients.

    Args:
        x (Tensor): shape [batch_size, 1] or [batch_size]
        y (Tensor): shape [batch_size, 1] or [batch_size]
        p (int): power (1 for W1, 2 for W2, etc.)

    Returns:
        Tensor: scalar Wasserstein distance
    """
    x = x.view(-1)
    y = y.view(-1)

    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)

    # Element-wise distance, then take p-norm
    return torch.mean(torch.abs(x_sorted - y_sorted) ** p) ** (1. / p)

def wasserstein_loss(cfg, real_scores, fake_scores, discriminator=None, real_data=None, fake_data=None):
    w_loss = None
    if real_scores.size() == fake_scores.size():
        # Calculate Wasserstein distance if arrays are same size
        direction = torch.sign((fake_scores.mean() - real_scores.mean()).detach())
        w_loss = direction * wasserstein_distance(real_scores, fake_scores)  # Signed Wasserstein distance
    
    # Calculate simple difference of mean if arrays are not same size
    m_loss = fake_scores.mean() - real_scores.mean()

    # Log both mean and wasserstein loss for analysis
    if cfg.use_wandb:
        wandb.log({
            'losses/was_dist': w_loss.item() if w_loss is not None else 0,
            'losses/mean': m_loss.item()
        })
    
    # Add gradient penalty (todo later)
    # if self.lipschitz_method == 'gp' and discriminator is not None:
    #     # Add gradient penalty
    #     gp = gradient_penalty(discriminator, real_data, fake_data, self.gp_lambda)
    #     return loss + gp

    loss = w_loss if w_loss is not None else m_loss
    return loss

def generator_loss(d_fake):
        """Compute generator loss."""
        return -d_fake.mean()

def apply_weight_clipping(model, clip_value=0.01):
    """Apply weight clipping to enforce Lipschitz constraint."""
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)

def evaluate_loss(ts, batch_size, data_loader, generator, discriminator):
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for real_samples, in data_loader.dataloader:
            generated_samples = generator(ts, batch_size)
            generated_score = discriminator(generated_samples).mean()
            real_score = discriminator(real_samples).mean()
            loss = generated_score - real_score # TODO maybe update with wasserstein?
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    generator.train()
    discriminator.train()
    return total_loss / total_samples




# TODO Look into later, since WGAN + GP paper shows good results
# def gradient_penalty(discriminator, real_data, fake_data, lambda_gp=10.0):
#     """Compute gradient penalty for WGAN-GP with time-augmented paths.
    
#     Args:
#         discriminator: Discriminator model
#         real_data: Real trajectories with time channel (batch, time, channels)
#         fake_data: Fake trajectories with time channel (batch, time, channels)
#         lambda_gp: Gradient penalty coefficient
    
#     Returns:
#         Gradient penalty scalar
#     """
#     batch_size, seq_len, channels = real_data.shape
#     device = real_data.device
    
#     # Random interpolation weights for each time step
#     alpha = torch.rand(batch_size, 1, 1, device=device)
    
#     # Interpolate between real and fake data
#     interpolated = alpha * real_data + (1 - alpha) * fake_data
#     interpolated.requires_grad_(True)
    
#     # Get discriminator output
#     d_interpolated = discriminator(interpolated).mean()
    
#     # Create ones tensor for gradient computation
#     ones = torch.ones_like(d_interpolated, requires_grad=False)
    
#     # Compute gradients with respect to interpolated samples
#     gradients = autograd.grad(
#         outputs=d_interpolated,
#         inputs=interpolated,
#         grad_outputs=ones,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True
#     )[0]
    
#     # Reshape gradients to (batch, -1)
#     gradients = gradients.reshape(batch_size, -1)
    
#     # Compute L2 norm of gradients
#     gradient_norm = gradients.norm(2, dim=1)
    
#     # Compute penalty as E[(||grad|| - 1)^2]
#     penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
#     return penalty