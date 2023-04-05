import torch
import torch.nn.functional as functional
from . import configs

def D_loss_r1(G, D, z, real_samples, regularization = False):
    fake_samples = G(z).type(real_samples.type())
    real_scores = D(real_samples)
    fake_scores = D(fake_samples)
    main_loss = torch.mean(functional.softplus(fake_scores)) + torch.mean(functional.softplus(-real_scores))
    
    if not(regularization):
        return main_loss

    temp_samples = real_samples.detach().requires_grad_(True)
    real_scores = D(temp_samples)
    grads = torch.autograd.grad(
        outputs=torch.sum(real_scores),
        inputs=temp_samples)
    
    regularization = 0.5 * torch.mean(torch.sum(torch.square(grads[0]), dim = [1,2,3]))
    return main_loss + configs.R1_GAMMA * regularization

def D_WGAN_loss_gp(G, D, z, real_samples, regularization = False):
    batch_size, C, H, W = real_samples.shape

    fake_samples = G(z).type(real_samples.type())
    real_scores = D(real_samples)
    fake_scores = D(fake_samples)
    main_loss = functional.softplus(fake_scores).mean() + functional.softplus(-real_scores).mean()

    if (regularization == False):
        return main_loss
    
    epsilon = torch.rand(size = (batch_size,1,1,1)).to('cuda')
    epsilon = torch.tile(epsilon, [1,3,H,W]).requires_grad_(True)
    inter_images = real_samples.detach() * epsilon + fake_samples.detach() * (1-epsilon)
    mixed_score = D(inter_images)
    grad = torch.autograd.grad(
        outputs=mixed_score.sum(),
        inputs=inter_images)
    gp = (grad[0].norm() - 1).square().mean()
    return main_loss + 10 * gp

def G_loss(G, D, z):
    fake_samples = G(z)
    fake_scores = D(fake_samples)
    main_loss = torch.mean(functional.softplus(-fake_scores))
    return main_loss

def G_loss_pl(G, D, z, regularization = False):
    fake_samples = G(z)
    fake_scores = D(fake_samples)
    main_loss = torch.mean(functional.softplus(-fake_scores))

    if not(regularization):
        return main_loss

    batch_size, _ = z.shape
    pl_minibatch_size = int(batch_size * configs.PL_BATCH_SIZE_RATIO)
    w_samples = G.mapping_network(z[:pl_minibatch_size])
    fake_samples = G.synthesis(w_samples)
    _, _, H, W = fake_samples.shape
    pl_noise = torch.randn_like(fake_samples) / H
    pl_grads = torch.autograd.grad(
        outputs=(fake_samples * pl_noise).sum(),
        inputs = w_samples
    )[0]
    pl_lengths = pl_grads.square().sum(1).sqrt()
    pl_mean = None
    if (G.pl_mean is not None):
        pl_mean = G.pl_mean.lerp(pl_lengths.mean(), configs.PL_DECAY)
    else:
        pl_mean = pl_lengths.mean()
    G.pl_mean = pl_mean.detach()
    pl_penalty = (pl_lengths - pl_mean).square().mean()

    return main_loss + configs.PL_WEIGHT * pl_penalty

def G_loss_test(G, D, z, real_samples):
    fake_samples = G(z)
    return (real_samples - fake_samples).square().mean()