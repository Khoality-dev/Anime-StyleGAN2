# adapted from https://github.com/NVlabs/stylegan2-ada-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms as T
import math
import numpy as np
from .configs import *

class ModulatedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ModulatedConv2D, self).__init__()

        self.w = nn.Parameter(torch.randn(size = (out_channels, in_channels, kernel_size, kernel_size)))
        self.w_gain = 1. / np.sqrt(in_channels * kernel_size * kernel_size)
        self.b = nn.Parameter(torch.zeros(size = (out_channels,)))
        self.activation = nn.LeakyReLU(0.2)
        self.stride = stride
        self.padding = padding

    def forward(self, x, s, noise = None, demodulate = True):
        N, in_channels, H, W = x.shape
        out_channels, _, kH, kW = self.w.shape
        w = self.w * self.w_gain
        w = self.w.unsqueeze(0)
        w = w * s.reshape(N, 1, -1, 1, 1)
        if (demodulate):
            w = w * (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt().reshape(N, -1, 1, 1, 1)
        w = w.reshape(N*out_channels, in_channels, kH,kW).to(x.dtype)
        x = x.reshape(1, N*in_channels, H, W)
        x = functional.conv2d(x, w, stride = self.stride, padding = self.padding, groups=N)
        x = x.reshape(N, out_channels, H, W)
        if noise is not None:
            x = x.add(noise)

        b = self.b.repeat(N, 1, 1, 1).reshape(N, -1, 1, 1)
        x = x.add(b)

        x = self.activation(x) * np.sqrt(2)
        return x

class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, bias_init = 0, activation = False, lr_multiplyer = 1.0):
        super(FullyConnectedLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(size = (out_features, in_features)) / lr_multiplyer)
        self.b = None
        if bias_init is not None:
            self.b = nn.Parameter(torch.full(size = [out_features,], fill_value = 1.0 * bias_init))
        self.w_gain = lr_multiplyer / np.sqrt(in_features)
        self.b_gain = lr_multiplyer
        
        self.activation = None
        if (activation):
            self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        w = self.w * self.w_gain
        x = x.matmul(w.t())

        if self.b is not None:
            b = self.b * self.b_gain
            x = x.add(b)

        if self.activation is not None:
            x = self.activation(x) * np.sqrt(2)

        return x

class Conv2DLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size = 1, stride=1, padding='same', bias_init = 0, activation = False):
        super(Conv2DLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(out_features, in_features, kernel_size, kernel_size))

        self.b = None
        if bias_init is not None:
            self.b = nn.Parameter(torch.full(size = (out_features,), fill_value=1. * bias_init))

        self.w_gain = 1. / np.sqrt(in_features * kernel_size * kernel_size)

        self.stride = stride
        self.padding = padding

        self.activation = None
        if activation:
            self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x, gain = 1.0):
        w = self.w * self.w_gain
        b = self.b
        x = functional.conv2d(x, w, b, self.stride, self.padding)
        if self.activation is not None:
            act_gain = np.sqrt(2) * gain #LeakyRelu gain * layer gain
            x = self.activation(x) * act_gain
        else:
            x = x * gain

        return x

class toRGBLayer(nn.Module):
    def __init__(self, in_channels, final_resolution):
        super(toRGBLayer, self).__init__()
        
        self.conv = Conv2DLayer(in_channels, 3, 1,padding='same')
        self.upsampling = nn.UpsamplingBilinear2d(size=final_resolution)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.upsampling(x)
        return x

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, num_layers):
        super(MappingNetwork, self).__init__()
        self.fcs = []
        for i in range(num_layers):
            fc = FullyConnectedLayer(latent_dim, latent_dim, activation=True, lr_multiplyer=0.01)
            self.fcs.append(fc)
            self.add_module(f'fc{i}', fc)

    def forward(self, z):
        for i in range(len(self.fcs)):
            z = self.fcs[i](z)
        return z
        
class StyleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super(StyleLayer, self).__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, 1.)
        self.noise_strength = nn.Parameter(torch.full([], 0.))
        self.conv = ModulatedConv2D(in_channels, out_channels, 3)

    def forward(self, x, w):
        s = self.affine(w)
        N, C, H, W = x.shape
        noise = torch.randn(size = (N, 1, H, W)).to(DEVICE)
        noise = noise * self.noise_strength
        x = self.conv(x, s, noise)
        return x

class StyleBlock(nn.Module):
    def __init__(self,in_channels, out_channels, w_dim, final_resolution):
        super(StyleBlock, self).__init__()
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.style_layer0 = StyleLayer(in_channels, out_channels, w_dim)
        self.style_layer1 = StyleLayer(out_channels, out_channels, w_dim)
        self.toRGB = toRGBLayer(out_channels, final_resolution)

    def forward(self, x, w):
        if (self.upsampling is not None):
            x = self.upsampling(x)
        if self.style_layer0 is not None:
            x = self.style_layer0(x, w)
        x = self.style_layer1(x, w)
        x_rgb = self.toRGB(x)
        return x, x_rgb

class MinibatchStdDev(nn.Module):
    def __init__(self):
        super(MinibatchStdDev, self).__init__()

    def forward(self, x): 
        N, C, H, W = x.shape #NxCxHxW
        y = x.std(dim = 0) #CxHxW
        y = y.mean()    #1
        y = y.repeat(N,1,H,W) #Nx1xHxW
        x = torch.cat([x,y], dim = 1) #NxC+1xHxW
        return x

class DiscriminiatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminiatorBlock, self).__init__()
        self.downsample0 = nn.AvgPool2d(2)
        self.skip = Conv2DLayer(in_channels, out_channels, 1, 1, 'same', bias_init=None)

        self.conv0 = Conv2DLayer(in_channels, in_channels, 3, 1, 'same', activation=True)

        self.downsample1 = nn.AvgPool2d(2)
        self.conv1 = Conv2DLayer(in_channels, out_channels, 3, 1, 'same', activation=True)
        
    def forward(self, x):
        y = self.downsample0(x)
        y = self.skip(y, gain = np.sqrt(0.5))
        x = self.conv0(x)
        x = self.downsample1(x)
        x = self.conv1(x, gain = np.sqrt(0.5))
        x = y.add(x)
            
        return x

class Synthesis(nn.Module):
    def __init__(self, latent_dims, final_resolution):
        super(Synthesis, self).__init__()
        
        self.constant_layer = nn.Parameter(torch.randn(NUM_FEATURE_MAP[4], 4, 4))

        self.constant_layer_cache = self.constant_layer.unsqueeze(0).repeat(8, 1, 1, 1)
        self.first_block = StyleLayer(NUM_FEATURE_MAP[4], NUM_FEATURE_MAP[4], latent_dims)
        self.first_block_toRGB = toRGBLayer(NUM_FEATURE_MAP[4], final_resolution)

        self.blocks = []
        current_resolution = 4
        while (current_resolution < final_resolution):
            current_resolution = current_resolution << 1
            block = StyleBlock(NUM_FEATURE_MAP[current_resolution/2], NUM_FEATURE_MAP[current_resolution], latent_dims, final_resolution)
            self.blocks.append(block)
            self.add_module("style_block_"+str(NUM_FEATURE_MAP[current_resolution])+"x"+str(current_resolution)+"x"+str(current_resolution), block)

    def forward(self, w):
        batch_size, _ = w.shape
        
        if (batch_size != self.constant_layer.shape[0]):
            self.constant_layer_cache = self.constant_layer.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        x = self.constant_layer_cache
        x = self.first_block(x, w)
        rgb = self.first_block_toRGB(x)

        for i in range(len(self.blocks)):
            x, x_rgb = self.blocks[i](x, w)
            rgb = rgb.add(x_rgb)
        
        return rgb

class Generator(nn.Module):
    def __init__(self, latent_dims, num_mapping_network_layers, final_resolution):
        super(Generator, self).__init__()
        self.mapping_network = MappingNetwork(latent_dims, num_mapping_network_layers)
        self.synthesis = Synthesis(latent_dims, final_resolution)
        self.pl_mean = None
        self.w_mean = torch.zeros(size=(latent_dims,)).to(DEVICE)
        self.iteration = 0

    def forward(self, z, trunc_factor = 1.0):
        N, _ = z.shape
        w = self.mapping_network(z)
        
        # interpolate w mean for truncation trick
        self.w_mean = self.w_mean.lerp(w.mean(dim=0), 1e-4).detach()
        w = self.w_mean.unsqueeze(0).repeat(N, 1).lerp(w, trunc_factor)
        y = self.synthesis(w)
        return y

class Discriminator(nn.Module):
    def __init__(self, orginal_resolution):
        super(Discriminator, self).__init__()
        self.blocks = []
        self.fromRGB = Conv2DLayer(3, NUM_FEATURE_MAP[orginal_resolution], 1, 1, 'same', activation=True)
        current_resolution = orginal_resolution
        while (current_resolution > 4):
            block = DiscriminiatorBlock(NUM_FEATURE_MAP[current_resolution], NUM_FEATURE_MAP[current_resolution//2])
            self.blocks.append(block)
            self.add_module("disciminator_block_"+str(NUM_FEATURE_MAP[current_resolution])+"x"+str(current_resolution)+"x"+str(current_resolution), block)
            current_resolution = (current_resolution >> 1)

        self.minibatch_stddev = MinibatchStdDev()
        self.conv = Conv2DLayer(NUM_FEATURE_MAP[4] + 1, NUM_FEATURE_MAP[4], 3, 1, 'same', activation=True)
        self.fc = FullyConnectedLayer(NUM_FEATURE_MAP[4] * 4 * 4, NUM_FEATURE_MAP[4], activation = True)
        self.flatten = nn.Flatten()
        self.out = FullyConnectedLayer(NUM_FEATURE_MAP[4], 1)

    def forward(self, x):
        x = self.fromRGB(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.minibatch_stddev(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.out(x)
        return x

 