import torch
import torch.nn.functional as functional
from . import configs

def D_loss_r1(G, D, z, noise, real_samples):
    fake_samples = G(z,noise).type(real_samples.type())
    real_scores = D(real_samples)
    fake_scores = D(fake_samples)
    main_loss = torch.mean(functional.softplus(fake_scores)) + torch.mean(functional.softplus(-real_scores))
        
    temp_samples = real_samples.detach().requires_grad_(True)
    real_scores = D(temp_samples)
    grads = torch.autograd.grad(
        outputs=torch.sum(real_scores),
        inputs=temp_samples)
    
    regularization = 0.5 * configs.R1_GAMMA * torch.mean(torch.sum(torch.square(grads[0]), dim = [1,2,3]))
    return main_loss + regularization

def D_WGAN_loss_gp(G, D, z, noise, real_samples):
    batch_size, C, H, W = real_samples.shape

    fake_samples = G(z,noise).type(real_samples.type())
    real_scores = D(real_samples)
    fake_scores = D(fake_samples)
    main_loss = functional.softplus(fake_scores).mean() + functional.softplus(-real_scores).mean()

    epsilon = torch.rand(size = (batch_size,1,1,1)).to('cuda')
    epsilon = torch.tile(epsilon, [1,3,H,W]).requires_grad_(True)
    inter_images = real_samples.detach() * epsilon + fake_samples.detach() * (1-epsilon)
    mixed_score = D(inter_images)
    grad = torch.autograd.grad(
        outputs=mixed_score.sum(),
        inputs=inter_images)
    gp = (grad[0].norm() - 1).square().mean()
    return main_loss + 10 * gp

def G_loss(G, D, z, noise):
    fake_samples = G(z, noise)
    fake_scores = D(fake_samples)
    return torch.mean(functional.softplus(-fake_scores))

def G_loss_pl(G, D, fake_samples, real_samples):
    pass

def G_loss_test(G, D, z, noise, real_samples):
    fake_samples = G(z, noise)
    return (real_samples - fake_samples).square().mean()