import torch

def discriminator_loss(real_score, generated_score, discriminator=None, real_data=None, fake_data=None):
        """Compute discriminator loss with optional gradient penalty.
        
        Args:
            real_score: Discriminator scores for real data
            generated_score: Discriminator scores for fake data
            discriminator: Discriminator model (required for gradient penalty)
            real_data: Real trajectories (required for gradient penalty)
            fake_data: Fake trajectories (required for gradient penalty)
        
        Returns:
            Total discriminator loss
        """
        # Wasserstein distance estimate (to maximize)
        wasserstein_distance = real_score - generated_score
        
        # Base loss (to minimize)
        base_loss = -wasserstein_distance
        
        # if self.lipschitz_method == 'gp' and discriminator is not None:
        #     # Add gradient penalty
        #     gp = gradient_penalty(discriminator, real_data, fake_data, self.gp_lambda)
        #     return base_loss + gp
        
        return base_loss

def generator_loss(d_fake):
        """Compute generator loss."""
        return -d_fake.mean()

def apply_weight_clipping(model, clip_value=0.01):
    """Apply weight clipping to enforce Lipschitz constraint."""
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)

def evaluate_loss(ts, batch_size, dataloader, generator, discriminator):
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for real_samples, in dataloader:
            generated_samples = generator(ts, batch_size)
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = generated_score - real_score
            total_samples += batch_size
            total_loss += loss.item() * batch_size
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
#     d_interpolated = discriminator(interpolated)
    
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