from ast import Not
from textwrap import indent
import torch 
import pickle
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from models.libs.utils import UnitGaussianNormalizer, get_grids


def load_dataset(dataset_path, batch_size, resample_resolution=[64, 64], dis_type='euclid', eps=0.05, spatial_sample_num=None,
                 val_batch_size=None, equispace=True, sub=1, T_in=10, T_pred=40, device='cuda', data_level=-1, **kwargs):
    
    if val_batch_size == None:
        val_batch_size = batch_size
        
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
        
    u = data['u']
    a = data['a']
    t = data['time_feature']

    sample_num = u.shape[0]
    s = u.shape[1] // sub
    
    ntrain, nval, ntest = int(sample_num * 0.7), int(sample_num * 0.1), int(sample_num * 0.2)
    
    u = torch.from_numpy(u)
    a = torch.from_numpy(a)
    t = torch.from_numpy(t)

    spatial_dim = len(u.shape) - 3
    for dim in range(spatial_dim): 
        idx = torch.arange(0, u.shape[dim+1], sub)
        u = torch.index_select(u, dim=dim+1, index=idx)
        a = torch.index_select(a, dim=dim+1, index=idx)
    
    train_in_org = u[:ntrain,...,:T_in,data_level]
    train_param_org = a[:ntrain,...,data_level]
    train_out_org = u[:ntrain,...,T_in:T_pred+T_in,data_level]
    train_time = t[:ntrain,T_in:T_pred+T_in]

    val_in_org = u[ntrain:nval+ntrain,...,:T_in,data_level]
    val_param_org = a[ntrain:nval+ntrain,...,data_level]
    val_out_org = u[ntrain:nval+ntrain,...,T_in:T_pred+T_in,data_level]
    val_time = t[ntrain:nval+ntrain,T_in:T_pred+T_in,]

    test_in_org = u[-ntest:,...,:T_in,data_level]
    test_param_org = a[-ntest:,...,data_level]
    test_out_org = u[-ntest:,...,T_in:T_pred+T_in,data_level]
    test_time = t[-ntest:,T_in:T_pred+T_in]

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
        train_in = train_in[:,idx,]
        train_param = train_param[:,idx]
        train_out = train_out[:,idx,]
        
        val_in = val_in[:,idx,]
        val_param = val_param[:,idx]
        val_out = val_out[:,idx,]
        
        test_in = test_in[:,idx,]
        test_param = test_param[:,idx]
        test_out = test_out[:,idx,]
    
    if 'x' in data.keys():
        x = torch.from_numpy(data['x'])[::sub, ::sub, :].reshape(-1, spatial_dim)
        for i in range(x.shape[-1]):
            x[...,i] = (x[...,i] - x[...,i].min()) / (x[...,i].max() - x[...,i].min())
        x = x.unsqueeze(dim=-2).repeat(1, T_pred, 1)
    elif equispace:
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
    
    trn_set = WeathMixLoader(
        in_data=train_in, mesh_generator=mesh_generator, out_data=train_out, in_x=x, in_t=train_time, param=train_param, 
        in_data_ori=train_in_org, out_data_ori=train_out_org, resolution=resample_resolution, device=device
    )

    val_set = WeathMixLoader(
        in_data=val_in, mesh_generator=mesh_generator, out_data=val_out, in_x=x, in_t=val_time, param=val_param, 
        in_data_ori=val_in_org, out_data_ori=val_out_org, resolution=resample_resolution, device=device
    )

    test_set = WeathMixLoader(
        in_data=test_in, mesh_generator=mesh_generator, out_data=test_out, in_x=x, in_t=test_time, param=test_param, 
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
    
        
class MeshGenerator(object):
    def __init__(self, x, sample_resolution=[64, 64], T_pred=10, dis_type='euclid', eps=0.1, device='cuda'):
        self.x = x
        self.x_spatial = self.x[:,0,:]

        for i in range(self.x_spatial.shape[-1]):
            assert self.x_spatial[:,i].max() <= 1.0 and self.x_spatial[:,i].min() >= 0, \
                ('The scale of x should be [0,1]')
        
        assert (len(sample_resolution) == self.x_spatial.shape[-1])
        
        self.grids = get_grids(sample_resolution).reshape(-1, self.x_spatial.shape[-1])

        if dis_type == 'euclid':
            self.dis_func = self.euclid_distance
        elif dis_type == 'sphere':
            self.dis_func = self.sphere_distance
        
        self.edge_original, self.edge_resample = self.get_connectivity(self.x_spatial, self.grids, eps=eps)
        
        self._resample_point_num = self.grids.shape[0]
        self.resample_grids = self.grids[:,None,:].repeat(1,T_pred,1)
    
    def get_attributes(self):
        return self.resample_grids, self.edge_original, self.edge_resample
        
    def get_connectivity(self, X, Y, eps):
        distance = self.dis_func(X, Y)
        edge_index = torch.vstack(torch.where(distance <= eps))
        edge_original = edge_index[0]
        edge_resample = edge_index[1]
        return edge_original, edge_resample
        
    def sphere_distance(self, X, Y=None):
        if Y is None:
            Y = X
        assert(Y.shape[1] == X.shape[1])
        if (torch.abs(torch.square(X).sum(dim=-1).sqrt() - 1.0) < 1e-7).prod() == False:
            X = torch.divide(X,torch.square(X).sum(dim=-1).sqrt()+1e-7)
            Y = torch.divide(Y,torch.square(Y).sum(dim=-1).sqrt()+1e-7)
            
        cos_theta = torch.matmul(X, Y.transpose(0,1))
        cos_theta = torch.clip(cos_theta, min=-1.0, max=1.0)
        theta = torch.arccos(cos_theta)
        return theta 
    
    def euclid_distance(self, X, Y=None):
        if Y is None:
            Y = X
        assert(Y.shape[1] == X.shape[1])
        dist = torch.cdist(X, Y)
        return dist     

class WeathMixLoader(Dataset):
    def __init__(
        self, 
        in_data,
        out_data,
        in_x,
        in_t=None,
        param=None,
        in_data_ori=None,
        out_data_ori=None,
        mesh_generator=None,
        normalizer=None,
        resolution=None,
        device='cuda',
        ) -> None:
        
        self.device = device
        self.u_in = in_data
        self.u_out = out_data
        self.a = param
        self.u_ori_in = in_data_ori
        self.u_ori_out = out_data_ori
        self.mesh_generator = mesh_generator
        self.normalizer = normalizer
        self.resolution=resolution
        self.x_in = in_x
        self.t_in = in_t
        self.sample_size = in_data.shape[0]
        
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        u_in = self.u_in[idx]
        u_out = self.u_out[idx]
        t_in = self.t_in[idx] 
        x_in = self.x_in
        
        if self.mesh_generator is not None and self.a is not None and self.t_in is None:
            a_in = self.a[idx]
            resample_grids, edge_original, edge_resample = self.mesh_generator.get_attributes()
            return u_in.to(self.device).float(), u_out.to(self.device).float(), x_in.to(self.device).float(), a_in.to(self.device).float(),\
                resample_grids.to(self.device).float(), edge_original.to(self.device).long(), edge_resample.to(self.device).long()
        
        elif self.mesh_generator is None and self.a is not None and self.t_in is None:
            a_in = self.a[idx]
            return u_in.to(self.device).float().float(), u_out.to(self.device).float(), x_in.to(self.device).float(), a_in.to(self.device).float()
        
        elif self.mesh_generator is None and self.t_in is not None and self.a is not None:
            t_in = self.t_in[idx]
            a_in = self.a[idx]
            return u_in.to(self.device).float().float(), u_out.to(self.device).float(), x_in.to(self.device).float(), a_in.to(self.device).float(), t_in.to(self.device).float()

        elif self.mesh_generator is not None and self.a is not None and self.t_in is not None:
            a_in = self.a[idx]
            t_in = self.t_in[idx]
            resample_grids, edge_original, edge_resample = self.mesh_generator.get_attributes()
            return u_in.to(self.device).float(), u_out.to(self.device).float(), x_in.to(self.device).float(), a_in.to(self.device).float(),\
                resample_grids.to(self.device).float(), edge_original.to(self.device).long(), edge_resample.to(self.device).long(),\
                t_in.to(self.device).float()
        
        
def collate(batch):
    if len(batch[0]) == 7:
        u_in = torch.stack([item[0] for item in batch], dim=0)
        u_out = torch.stack([item[1] for item in batch], dim=0)   
        x_in = torch.stack([item[2] for item in batch], dim=0)    
        a_in = torch.stack([item[3] for item in batch], dim=0)
        t_in = None   
        resample_grids = torch.stack([item[4] for item in batch], dim=0)  
        edge_original = batch[0][5]
        edge_resample = batch[0][6] 
    
    elif len(batch[0]) == 4:
        u_in = torch.stack([item[0] for item in batch], dim=0)
        u_out = torch.stack([item[1] for item in batch], dim=0)   
        x_in = torch.stack([item[2] for item in batch], dim=0)    
        a_in = torch.stack([item[3] for item in batch], dim=0)
        t_in = None   
        resample_grids = None
        edge_original = None
        edge_resample = None
    
    elif len(batch[0]) == 5:
        u_in = torch.stack([item[0] for item in batch], dim=0)
        u_out = torch.stack([item[1] for item in batch], dim=0)   
        x_in = torch.stack([item[2] for item in batch], dim=0)
        a_in = torch.stack([item[3] for item in batch], dim=0)  
        t_in =  torch.stack([item[4] for item in batch], dim=0) 
        resample_grids = None
        edge_original = None
        edge_resample = None
    
    elif len(batch[0]) == 8:
        u_in = torch.stack([item[0] for item in batch], dim=0)
        u_out = torch.stack([item[1] for item in batch], dim=0)   
        x_in = torch.stack([item[2] for item in batch], dim=0)    
        a_in = torch.stack([item[3] for item in batch], dim=0)   
        resample_grids = torch.stack([item[4] for item in batch], dim=0)  
        edge_original = batch[0][5]
        edge_resample = batch[0][6] 
        t_in =  torch.stack([item[7] for item in batch], dim=0) 

    return Batch(u_in, u_out, x_in, a_in, t_in, resample_grids, edge_original, edge_resample)

class Batch():
    def __init__(self, u_in, u_out, x_in, a_in, t_in=None, resample_grids=None, edge_original=None, edge_resample=None):
        self.u_in = u_in
        self.u_out = u_out
        self.x_in = x_in
        self.a_in = a_in
        self.t_in = t_in
        self.resample_grids = resample_grids
        self.edge_original = edge_original
        self.edge_resample = edge_resample
        