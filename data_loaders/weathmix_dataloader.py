from ast import Not
from textwrap import indent
import torch 
import pickle
import numpy as np
from timeit import default_timer
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from models.libs.utils import UnitGaussianNormalizer, get_grids
from .phymix_dataloader import *

def load_dataset(dataset_path, batch_size, resample_resolution=[64, 64], dis_type='euclid', eps=0.05, spatial_sample_num=None,
                 val_batch_size=None, equispace=True, sub=1, T_in=10, T_pred=40, device='cuda', data_level=-1, **kwargs):
    
    if val_batch_size == None:
        val_batch_size = batch_size
        
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
        
    u = data['u']
    a = data['a']
    
    sample_num = u.shape[0]
    s = u.shape[1] // sub
    
    ntrain, nval, ntest = int(sample_num * 0.7), int(sample_num * 0.1), int(sample_num * 0.2)
    
    u = torch.from_numpy(u)
    a = torch.from_numpy(a)

    spatial_dim = len(u.shape) - 3
    for dim in range(spatial_dim): 
        idx = torch.arange(0, u.shape[dim+1], sub)
        u = torch.index_select(u, dim=dim+1, index=idx)
        a = torch.index_select(a, dim=dim+1, index=idx)
    
    train_in_org = u[:ntrain,...,:T_in,data_level]
    train_param_org = a[:ntrain,...,data_level]
    train_out_org = u[:ntrain,...,T_in:T_pred+T_in,data_level]
   

    val_in_org = u[ntrain:nval+ntrain,...,:T_in,data_level]
    val_param_org = a[ntrain:nval+ntrain,...,data_level]
    val_out_org = u[ntrain:nval+ntrain,...,T_in:T_pred+T_in,data_level]

    test_in_org = u[-ntest:,...,:T_in,data_level]
    test_param_org = a[-ntest:,...,data_level]
    test_out_org = u[-ntest:,...,T_in:T_pred+T_in,data_level]

    original_resolution = train_out_org.shape[1:]
    in_size = train_in_org.shape[-1]

    train_in = train_in_org.reshape(train_in_org.shape[0], -1, T_in)
    train_param = train_param_org.reshape(train_param_org.shape[0], -1)
    train_out = train_out_org.reshape(train_out_org.shape[0], -1, T_pred)
    val_in = val_in_org.reshape(val_in_org.shape[0], -1, T_in)
    val_param = val_param_org.reshape(val_param_org.shape[0], -1)
    val_out = val_out_org.reshape(val_out_org.shape[0], -1, T_pred)
    
    test_in = test_in_org.reshape(test_in_org.shape[0], -1, T_in)
    test_param = test_param_org.reshape(test_param_org.shape[0], -1)
    test_out = test_out_org.reshape(test_out_org.shape[0], -1, T_pred)
    
    dimension = len(original_resolution)
    
    print('This is a {}-D problem.'.format(dimension), \
        'Original spatial point number is {}.'.format(train_in.shape[1]),\
        'Temporal input number is {}.'.format(train_in.shape[-1]),\
        'Temporal output number is {}.'.format(train_out.shape[-1]))
    
    if not equispace and spatial_sample_num is not None:
        # randomly sample spatial points on the regular meshgrids
        
        idx = np.sort(np.random.permutation(np.arange(0, train_in.shape[1]))[:spatial_sample_num])
        train_in = train_in[:,idx,:]
        train_param = train_param[:,idx]
        train_out = train_out[:,idx,:]
        
        val_in = val_in[:,idx,:]
        val_param = val_param[:,idx]
        val_out = val_out[:,idx,:]
        
        test_in = test_in[:,idx,:]
        test_param = test_param[:,idx]
        test_out = test_out[:,idx,:]

    if equispace:
        x = get_grids(original_resolution) # the default is [0,1] equispaced samples
    elif spatial_sample_num is not None:
        x = get_grids(original_resolution).reshape(-1, T_pred, dimension)[idx, :, :]
    else:
        x = get_grids(original_resolution).reshape(-1, T_pred, dimension)
           
    in_normalizer = UnitGaussianNormalizer(train_in)
    train_in = in_normalizer.encode(train_in).unsqueeze(dim=-2).repeat([1,1,T_pred,1])
    val_in = in_normalizer.encode(val_in).unsqueeze(dim=-2).repeat([1,1,T_pred,1])
    test_in = in_normalizer.encode(test_in).unsqueeze(dim=-2).repeat([1,1,T_pred,1])
    
    param_normalizer = UnitGaussianNormalizer(train_param)
    train_param = param_normalizer.encode(train_param)[...,None,None].repeat([1,1,T_pred,1])
    val_param = param_normalizer.encode(val_param)[...,None,None].repeat([1,1,T_pred,1])
    test_param = param_normalizer.encode(test_param)[...,None,None].repeat([1,1,T_pred,1])

    out_normalizer = UnitGaussianNormalizer(train_out)
    train_out = out_normalizer.encode(train_out)
    val_out = out_normalizer.encode(val_out)
    test_out = out_normalizer.encode(test_out)

    if not equispace:
        mesh_generator = MeshGenerator(x, resample_resolution, T_pred, dis_type, eps, device)
    else:
        mesh_generator = None
    
    trn_set = PhyMixLoader(
        in_data=train_in, mesh_generator=mesh_generator, out_data=train_out, in_x=x, param=train_param, 
        in_data_ori=train_in_org, out_data_ori=train_out_org, resolution=resample_resolution, device=device
    )

    val_set = PhyMixLoader(
        in_data=val_in, mesh_generator=mesh_generator, out_data=val_out, in_x=x, param=val_param, 
        in_data_ori=val_in_org, out_data_ori=val_out_org, resolution=resample_resolution, device=device
    )

    test_set = PhyMixLoader(
        in_data=test_in, mesh_generator=mesh_generator, out_data=test_out, in_x=x, param=test_param, 
        in_data_ori=test_in_org, out_data_ori=test_out_org, resolution=resample_resolution, device=device
    )
    
    data = {}
    data['train_loader'], data['val_loader'], data['test_loader'] = \
        DataLoader(trn_set, batch_size=batch_size, shuffle=True, collate_fn=collate),\
        DataLoader(val_set, batch_size=val_batch_size, shuffle=False, collate_fn=collate),\
        DataLoader(test_set, batch_size=val_batch_size, shuffle=False, collate_fn=collate)
    data['scaler'] = out_normalizer.to(device)
    data['mesh_generator'] = mesh_generator
    data['original_resolution'] = tuple(original_resolution)
    data['resample_resolution'] = tuple(resample_resolution) + (original_resolution[-1],)
    data['param_dim'] = train_param.shape[-1]
    data['dimension'] = dimension
    return data
    
       