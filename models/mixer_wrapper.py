import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_layers import *

class EquiMixerBlock(nn.Module):
    def __init__(self, 
                 embed_size: int,
                 resolution: tuple,
                 token_mixer, 
                 channel_mixer,
                 token_demixer=nn.Identity,
                 norm_layer=nn.Identity,
                 residual_layer=nn.Identity,
                 affine_transform=nn.Identity,
                 activation=F.gelu):
        super().__init__()
        
        self.embed_size = embed_size
        self.resolution = resolution
        self.affine_transform = affine_transform(embed_size)
        self.channel_mixer = channel_mixer(embed_size, resolution)
        self.token_mixer = token_mixer(embed_size, resolution)
        self.token_demixer = token_demixer(embed_size, resolution)
        self.activation = activation
        
            
        norm_dim = resolution + (embed_size,) 
        self.norm_layer = norm_layer(normalized_shape = norm_dim)        
        
        self.residual_layer = residual_layer(embed_size)
        
    def forward(self, x):
        # x: shape [batch_size, embed_size, resolution1, resolution2, ...]
        x1 = self.token_mixer(x)
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
                 equispace=True,
                 node_attr_size=None,
                 patch_size=None,
                 token_demixer=nn.Identity,
                 norm_layer=nn.Identity,
                 residual_layer=nn.Identity,
                 affine_transform=nn.Identity,
                 activation=F.gelu,
                 ker_width: int=64,
                 interpolation: str='gaussian',
                 depth=3):
        
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.patch_size = patch_size
        self.resolution = resolution
        self.original_resolution = resolution
        self.equispace = equispace
        self.dim = len(self.resolution)
        
        self.up_sampler, self.down_sampler = Identity(), Identity()
        
        if node_attr_size is not None:
            assert self.equispace == False, ('The param node_attr_size should only used in nonequispaced sampling.')
            if interpolation == 'gaussian':
                self.up_to_equi = GaussianMean(node_attr_size, embed_size, ker_width, self.dim)
                self.down_to_pixel = GaussianMean(node_attr_size, embed_size, ker_width, self.dim)
            else:
                self.up_to_equi = AttrMean(node_attr_size, embed_size, ker_width, self.dim)
                self.down_to_pixel = AttrMean(node_attr_size, embed_size, ker_width, self.dim)
            
            self.up_sampler = self.up_to_equi
            self.down_sampler = self.down_to_pixel
            
        if patch_size is not None:
            assert self.equispace == True, ('The param patch_size should only used in equispaced sampling.')
            assert len(self.resolution) == len(self.patch_size), ('Length of resolution does not match it of patch size.')
            assert np.prod([(self.resolution[i]%self.patch_size[i]==0) for i in range(len(self.resolution))]), ('the patch size is not an factor of resolution.')
            
            self.resolution = tuple([resolution[i]//self.patch_size[i] for i in range(len(self.resolution))])
            
            self.up_to_patch = PatchMean(patch_size, embed_size, to_patch=True)
            self.down_to_pixel = PatchMean(patch_size, embed_size, to_patch=False)
            self.up_sampler = self.up_to_patch
            self.down_sampler = self.down_to_pixel
            
            self.patch_to_pixel = nn.Sequential(nn.Linear(embed_size, np.prod(self.patch_size)*embed_size), 
                                                nn.GELU())        
            
        self.lifter = lifter(input_size=input_size, embed_size=embed_size)
        self.projector = projector(output_size=output_size, embed_size=embed_size)
        self.mixer_layers = nn.ModuleList([EquiMixerBlock(embed_size, self.resolution, token_mixer, channel_mixer, token_demixer, norm_layer, residual_layer, 
                                                          affine_transform, activation) for d in range(depth)])
    
    def get_edge_attr(self, u_in, x_in, a_in, resample_grids, edge_from, edge_to):
        # Assume the weight is not dynamic, thus decrease the computational cost
        u_attr = u_in[:,edge_from,...,0,:]
        x_in = x_in[...,0,:-1][:,edge_from]
        resample_grids_attr = resample_grids[...,0,:-1][:,edge_to]
        x_attr = x_in - resample_grids_attr
        edge_attr = torch.cat([u_attr, x_attr, resample_grids_attr], dim=-1)
        if a_in is not None: edge_attr = torch.cat([edge_attr, a_in[...,0,:][:,edge_from]], dim=-1) 
        return edge_attr, resample_grids.shape[:-1]
    
    def forward(self, batch):
        u_in = batch.u_in
        x_in = batch.x_in
        a_in = batch.a_in
        resample_grids = batch.resample_grids
        edge_original = batch.edge_original
        edge_resample = batch.edge_resample
        
        batch_size = u_in.shape[0]
        input_shape = u_in.shape 
        
        if self.equispace:
            u_in = u_in.reshape(batch_size, *self.original_resolution, -1)
            
        u = torch.cat([u_in, x_in], dim=-1)
        u_l = self.lifter(u)        
        
        if not self.equispace:
            edge_attr, grid_size = self.get_edge_attr(u_in, x_in, a_in, resample_grids, edge_original, edge_resample)
        else:
            edge_attr, grid_size = None, None
            
        u_equi = self.up_sampler(u_l, edge_attr, grid_size, edge_original, edge_resample)
        u_equi = u_equi.reshape(batch_size, *self.resolution, -1)
        
        for layer in self.mixer_layers:
            u_equi = layer(u_equi)
            
        if not self.equispace:
            u_equi = u_equi.reshape(batch_size, np.prod(self.resolution[:-1]), self.resolution[-1], -1)  
              
        u_unequi = self.down_sampler(u_equi, edge_attr, input_shape[:-1], edge_resample, edge_original)
        
        u_p = self.projector(u_unequi)   
        
        return u_p.reshape(*input_shape[:-1], self.output_size)
    
        
        