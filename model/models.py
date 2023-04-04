import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms as T
import math
from .configs import *

class ModulatedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(ModulatedConv2D, self).__init__()

        self.w = nn.Parameter(torch.randn(size = (out_channels, in_channels, kernel_size, kernel_size)))
        self.b = nn.Parameter(torch.zeros(size = (out_channels,)))
        self.activation = nn.LeakyReLU(0.2)
        self.stride = stride
        self.padding = padding

    def forward(self, x, s, noise = None, demodulate = True):
        N, in_channels, H, W = x.shape
        out_channels, _, kH, kW = self.w.shape
        w = self.w.unsqueeze(0)
        w = w * s.reshape(N, 1, -1, 1, 1)
        if (demodulate):
            w = w * (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt().reshape(N, -1, 1, 1, 1)
        w = w.reshape(N*out_channels, in_channels, kH,kW).to(x.dtype)
        x = x.reshape(1, N*in_channels, H, W)
        x = functional.conv2d(x, w, stride = self.stride, padding = self.padding, groups=N)
        x = x.reshape(N, out_channels, H, W)
        b = self.b.repeat(N, 1, 1, 1).reshape(N, -1, 1, 1)
        x = x.add(b)
        if noise is not None:
            x = x.add(noise)
        x = self.activation(x)
        return x

class FullyConnectedLayer(nn.Module):
    def __init__(self, in_features, out_features, bias_init, activation = False):
        super(FullyConnectedLayer, self).__init__()
        self.fc = nn.Linear(in_features, out_features, False) 
        self.b = nn.Parameter(torch.full(size = [out_features,], fill_value= bias_init))

        self.activation = None
        if (activation):
            self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.fc(x)
        x = x.add(self.b)
        if self.activation is not None:
            x = self.activation(x)
        return x

class toRGBLayer(nn.Module):
    def __init__(self, in_channels, final_resolution):
        super(toRGBLayer, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, 3, 1,padding='same')
        self.upsampling = nn.UpsamplingBilinear2d(size=final_resolution)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.upsampling(x)
        return x

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, num_layers):
        super(MappingNetwork, self).__init__()
        self.layers = []
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2))
            self.layers.append(layer)
            self.add_module(f'fc{i}', layer)

    def forward(self, z):
        for i in range(len(self.layers)):
            z = self.layers[i](z)
        return z
        
class StyleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super(StyleLayer, self).__init__()
        self.affine = FullyConnectedLayer(w_dim, in_channels, 1.)
        self.noise_strength = nn.Parameter(torch.ones([]))
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
    def __init__(self, in_channels, out_channels, down = True, ministddev = False):
        super(DiscriminiatorBlock, self).__init__()

        self.downsample = nn.AvgPool2d(2) if (down) else None

        self.skip = None
        self.ministddev = None

        if (ministddev):
            self.ministddev = MinibatchStdDev()
        elif not(down):
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 'same'),
                nn.LeakyReLU(0.2))

        self.conv0 = None
        if (ministddev):
            self.conv0 = nn.Sequential(
                nn.Conv2d(in_channels+1, out_channels, 3, 1, 'same'),
                nn.LeakyReLU(0.2))
        else:
            self.conv0 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 'same'),
                nn.LeakyReLU(0.2))

        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 4, 1, 'valid'),
            nn.LeakyReLU(0.2)) if ministddev else nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 'same'),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        y = None
        if self.skip is not None:
            y = self.skip(x)

        if self.ministddev is not None:
            x = self.ministddev(x)

        x = self.conv0(x)

        if self.conv1 is not None:
            x = self.conv1(x)

        if y is not None:
            x = y.add(x)
            
        return x

class Synthesis(nn.Module):
    def __init__(self, latent_dims, final_resolution):
        super(Synthesis, self).__init__()
        
        self.constant_layer = nn.Parameter(torch.ones(NUM_FEATURE_MAP[4], 4, 4))

        self.constant_layer_cache = self.constant_layer.repeat(8, 1, 1, 1)
        self.first_block = StyleLayer(NUM_FEATURE_MAP[4], NUM_FEATURE_MAP[4], latent_dims)
        self.first_block_toRGB = toRGBLayer(NUM_FEATURE_MAP[4], final_resolution)

        self.blocks = []
        current_resolution = 4
        while (current_resolution < final_resolution):
            current_resolution = current_resolution << 1
            block = StyleBlock(NUM_FEATURE_MAP[current_resolution/2], NUM_FEATURE_MAP[current_resolution], latent_dims, final_resolution)
            self.blocks.append(block)
            self.add_module("style_block_"+str(NUM_FEATURE_MAP[current_resolution])+"x"+str(current_resolution)+"x"+str(current_resolution), block)

        self.tanh = nn.Tanh()

    def forward(self, w):
        batch_size, _ = w.shape
        
        if (batch_size != self.constant_layer.shape[0]):
            self.constant_layer_cache = self.constant_layer.repeat(batch_size, 1, 1, 1)

        x = self.constant_layer_cache
        x = self.first_block(x, w)
        rgb = self.first_block_toRGB(x)

        for i in range(len(self.blocks)):
            x, x_rgb = self.blocks[i](x, w)
            rgb = rgb.add(x_rgb)
        
        rgb = self.tanh(rgb)
        return rgb

class Generator(nn.Module):
    def __init__(self, latent_dims, num_mapping_network_layers, final_resolution):
        super(Generator, self).__init__()
        self.mapping_network = MappingNetwork(latent_dims, num_mapping_network_layers)
        self.synthesis = Synthesis(latent_dims, final_resolution)

    def forward(self, z):
        w = self.mapping_network(z)
        y = self.synthesis(w)
        return y

class Discriminator(nn.Module):
    def __init__(self, orginal_resolution):
        super(Discriminator, self).__init__()
        self.blocks = []

        current_resolution = orginal_resolution
        while (current_resolution > 4):
            block = None
            if (len(self.blocks) == 0):
                block = DiscriminiatorBlock(3, NUM_FEATURE_MAP[orginal_resolution], down=False)
            else:
                block = DiscriminiatorBlock(NUM_FEATURE_MAP[current_resolution*2], NUM_FEATURE_MAP[current_resolution])
            self.blocks.append(block)
            self.add_module("disciminator_block_"+str(NUM_FEATURE_MAP[current_resolution])+"x"+str(current_resolution)+"x"+str(current_resolution), block)
            current_resolution = (current_resolution >> 1)

        block = DiscriminiatorBlock(NUM_FEATURE_MAP[current_resolution*2], NUM_FEATURE_MAP[current_resolution], down=True, ministddev=True)
        self.blocks.append(block)
        self.add_module("disciminator_block_"+str(NUM_FEATURE_MAP[current_resolution])+"x"+str(current_resolution)+"x"+str(current_resolution), block)
        current_resolution = 4
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(NUM_FEATURE_MAP[current_resolution], 1)

    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

