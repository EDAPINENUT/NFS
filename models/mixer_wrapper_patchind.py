from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class EquiMixerBlock(nn.Module):
    def __init__(self, 
                 embed_size: int,
                 resolution: tuple,
                 token_mixer, 
                 channel_mixer,
                 token_demixer=nn.Identity,
                 temporal_mixer=nn.Identity,
                 temporal_size=None,
                 norm_layer=nn.Identity,
                 residual_layer=nn.Identity,
                 affine_transform=nn.Identity,
                 activation=F.gelu):
        super().__init__()
        
        self.embed_size = embed_size
        self.resolution = resolution
        self.temporal_size = temporal_size
        self.affine_transform = affine_transform(embed_size)
        self.channel_mixer = channel_mixer(embed_size, resolution, temporal_size)
        self.token_mixer = token_mixer(embed_size, resolution, temporal_size)
        self.token_demixer = token_demixer(embed_size, resolution, temporal_size)
        self.temporal_mixer = temporal_mixer(temporal_size)
        self.activation = activation
        
        if temporal_size is not None:
            assert(isinstance(temporal_size, int)), 'the size of temporal domian must be integer!'
            self.temporal_mixer = temporal_mixer(temporal_size)
            
        norm_dim = resolution + (embed_size,) if temporal_size is None else resolution + (temporal_size, embed_size)
        self.norm_layer = norm_layer(normalized_shape = norm_dim)        
        
        self.residual_layer = residual_layer(embed_size)
        
    def forward(self, x):
        # x: shape [batch_size, embed_size, resolution1, resolution2, ...]
        x1 = self.token_mixer(x)
        x1 = self.temporal_mixer(x1)
        x1 = self.channel_mixer(x1)
        x1 = self.token_demixer(x1)
        x1 = self.affine_transform(x1)
        x1 = self.norm_layer(x1)
        
        x2 = self.residual_layer(x)
        
        return self.activation(x1 + x2)
        
        
class MixerWrapper(nn.Module):
    def __init__(self,
                 input_size,
                 output_size, 
                 embed_size,
                 resolution,
                 lifter,
                 projector,
                 token_mixer, 
                 channel_mixer,
                 patch_size=None,
                 token_demixer=nn.Identity,
                 temporal_mixer=nn.Identity,
                 temporal_size=None,
                 norm_layer=nn.Identity,
                 residual_layer=nn.Identity,
                 affine_transform=nn.Identity,
                 activation=F.gelu,
                 grid_in=True,
                 depth=3):
        
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.patch_size = patch_size
        self.resolution = resolution
        self.spatial_dim = len(resolution)
        self.rearrange1, self.rearrange2, self.patch_to_pixel = nn.Identity(), nn.Identity(), nn.Identity()
        
        if patch_size is not None:
            assert len(self.resolution) == len(self.patch_size), ('Length of resolution does not match it of patch size.')
            assert np.prod([(self.resolution[i]%self.patch_size[i]==0) for i in range(len(self.resolution))]), ('the patch size is not an factor of resolution.')
            self.resolution = tuple([resolution[i]//self.patch_size[i] for i in range(len(self.resolution))])
            # input_size = np.prod(self.patch_size) * input_size
            
            if len(self.resolution) == 3:
                self.rearrange1 = Rearrange('b (x p1) (y p2) (t p3) c -> b x y t (p1 p2 p3) c', p1=patch_size[0], p2=patch_size[1], p3=patch_size[2]) 
                self.rearrange2 = Rearrange('b x y t (p1 p2 p3 c) -> b (x p1) (y p2) (t p3) c', p1=patch_size[0], p2=patch_size[1], p3=patch_size[2]) 
            elif len(self.resolution) == 2:
                self.rearrange1 = Rearrange('b (x p1) (y p2) t c -> b x y t (p1 p2 c)' , p1=patch_size[0], p2=patch_size[1])
                self.rearrange2 = Rearrange('b x y t (p1 p2 c) -> b (x p1) (y p2) t c', p1=patch_size[0], p2=patch_size[1], p3=patch_size[2]) 
            else:
                raise NotImplementedError('The resolution must be of dimension 2 or 3.')     
            
            # self.patch_to_pixel = nn.Sequential(Rearrange('b x y t e -> b e x y t'),
            #                                     nn.Conv3d(embed_size, np.prod(self.patch_size)*embed_size, 1),
            #                                     Rearrange('b e x y t -> b x y t e'), nn.GELU())
            self.patch_to_pixel = nn.Sequential(nn.Linear(embed_size, np.prod(self.patch_size)*embed_size), 
                                                nn.GELU())
            
            
        self.grid_in = grid_in
        self.lifter = lifter(input_size=input_size, embed_size=embed_size, dim=self.spatial_dim)
        self.projector = projector(output_size=output_size, embed_size=embed_size, dim=self.spatial_dim)
        self.mixer_layers = nn.ModuleList([EquiMixerBlock(embed_size, self.resolution, token_mixer, channel_mixer, token_demixer, 
                                                           temporal_mixer, temporal_size, norm_layer, residual_layer, 
                                                           affine_transform, activation) for d in range(depth)])
        
    def forward(self, x):
        grids = self.get_grid(x)
        # x = torch.cat((x, grids), dim=-1)
            
        input_shape = x.shape
        x = self.rearrange1(x)
        grids = self.rearrange1(grids)
        x_l = self.lifter(x, grids)
        for layer in self.mixer_layers:
            x_l = layer(x_l)
        x_l = self.patch_to_pixel(x_l)
        x_l = self.rearrange2(x_l)
        x_p = self.projector(x_l, grids)   
        return x_p.reshape(*input_shape[:-1], self.output_size)
    
    def get_grid(self, x):
        batchsize, size_x, size_y, size_z = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(x)
        
        