import torch 
import pickle
import numpy as np
from torch._C import device
from models.libs.utils import *
from timeit import default_timer
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def load_dataset(
    data_path,
    batch_size,
    val_batch_size=None,
    manifold='planar',
    eps=0.05,
    sub=1, 
    T_in=10, 
    T_pred=40,
    spatial_sample_num=4096,
    device='cuda', 
    data_level=None,
    **kwargs
    ):

    if val_batch_size == None:
        val_batch_size = batch_size
        
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        
    u = data['u']
    a = data['a']
    sample_num = u.shape[0]
    s = u.shape[1] // sub
    
    ntrain, nval, ntest = int(sample_num * 0.7), int(sample_num * 0.1), int(sample_num * 0.2)
    
    u = torch.from_numpy(u)
    a = torch.from_numpy(a)

    spatial_dim = len(u.shape) - 2 if data_level is None else len(u.shape) - 3
    for dim in range(spatial_dim): 
        idx = torch.arange(0, u.shape[dim+1], sub)
        u = torch.index_select(u, dim=dim+1, index=idx)
        a = torch.index_select(a, dim=dim+1, index=idx)
    if data_level is not None:
        train_in_org = u[:ntrain,...,:T_in,data_level]
        train_param_org = a[:ntrain,...,data_level]
        train_out_org = u[:ntrain,...,T_in:T_pred+T_in,data_level]

        val_in_org = u[ntrain:nval+ntrain,...,:T_in,data_level]
        val_param_org = a[ntrain:nval+ntrain,...,data_level]
        val_out_org = u[ntrain:nval+ntrain,...,T_in:T_pred+T_in,data_level]

        test_in_org = u[-ntest:,...,:T_in,data_level]
        test_param_org = a[-ntest:,...,data_level]
        test_out_org = u[-ntest:,...,T_in:T_pred+T_in,data_level]
    else:
        train_in_org = u[:ntrain,...,:T_in]
        train_param_org = a[:ntrain,...]
        train_out_org = u[:ntrain,...,T_in:T_pred+T_in]

        val_in_org = u[ntrain:nval+ntrain,...,:T_in]
        val_param_org = a[ntrain:nval+ntrain,...]
        val_out_org = u[ntrain:nval+ntrain,...,T_in:T_pred+T_in]

        test_in_org = u[-ntest:,...,:T_in]
        test_param_org = a[-ntest:,...]
        test_out_org = u[-ntest:,...,T_in:T_pred+T_in]

    original_resolution = train_out_org.shape[1:]
    
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

    if spatial_sample_num is not None:
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
    
    grid_info = {}
    if 'x' in data.keys():
        grid_info['all'] = data['x'] 
    else:
        grid_info['all'] = get_grids(original_resolution).reshape(-1, T_pred, dimension)

    if spatial_sample_num is not None:
        grid_info['used'] = grid_info['all'][idx, :, :]
    else:
        grid_info['used'] = grid_info['all']
    
    grid = grid_info['used'][:,0,:-1]
    dim = grid.shape[-1]
    grid = grid.reshape(-1, dim)
    
    if len(train_in.shape) == 3: 
        train_in = train_in.unsqueeze(dim=-1)
        val_in = val_in.unsqueeze(dim=-1)
        test_in = test_in.unsqueeze(dim=-1)

    if len(train_out.shape) == 3: 
        train_out = train_out.unsqueeze(dim=-1) 
        val_out = val_out.unsqueeze(dim=-1)
        test_out = test_out.unsqueeze(dim=-1)

    in_normalizer = UnitGaussianNormalizer(train_in)
    train_in = in_normalizer.encode(train_in)
    val_in = in_normalizer.encode(val_in)
    test_in = in_normalizer.encode(test_in)
    
    param_normalizer = UnitGaussianNormalizer(train_param)
    train_param = param_normalizer.encode(train_param)
    val_param = param_normalizer.encode(val_param)
    test_param = param_normalizer.encode(test_param)

    out_normalizer = UnitGaussianNormalizer(train_out)
    train_out = out_normalizer.encode(train_out)
    val_out = out_normalizer.encode(val_out)
    test_out = out_normalizer.encode(test_out)
    
    trn_set = Seq2SeqLoader(
        in_data=train_in, out_data=train_out, device=device
    )

    val_set = Seq2SeqLoader(
        in_data=val_in,  out_data=val_out, device=device
    )

    test_set = Seq2SeqLoader(
        in_data=test_in,  out_data=test_out, device=device
    )

    data = {}
    data['train_loader'], data['val_loader'], data['test_loader'] = \
        DataLoader(trn_set, batch_size=batch_size, shuffle=True, collate_fn=collate),\
        DataLoader(val_set, batch_size=val_batch_size, shuffle=False, collate_fn=collate),\
        DataLoader(test_set, batch_size=val_batch_size, shuffle=False, collate_fn=collate)
    data['scaler'] = out_normalizer.to(device)
    data['sparse_idx'] = get_connectivity(grid, grid, eps=eps).to(device)
    data['original_resolution'] = tuple(original_resolution)
    data['node_num'] = train_in.shape[1]
    data['dimension'] = dimension
    return data

def euclid_distance(X, Y=None):
    if Y is None:
        Y = X
    assert(Y.shape[1] == X.shape[1])
    dist = torch.cdist(X, Y)
    return dist  

def get_connectivity(X, Y, eps):
    distance = euclid_distance(X, Y)
    edge_index = torch.vstack(torch.where(distance <= eps))
    return edge_index

class Seq2SeqLoader(Dataset):
    def __init__(
        self, 
        in_data, out_data, device
        ) -> None:
        
        self.device = device

        self.data_signal_in = in_data
        self.data_signal_out = out_data
        
        self.sample_size, self.in_len, self.feature_num = self.data_signal_in.shape[0], self.data_signal_in.shape[1], self.data_signal_in.shape[-1]
        self.out_horizon = self.data_signal_out.shape[1]

        self.data_signal_in = self.data_signal_in.reshape(self.sample_size, self.in_len, -1, self.feature_num)
        self.data_signal_out = self.data_signal_out.reshape(self.sample_size, self.out_horizon, -1, self.feature_num)

    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        data = self.data_signal_in[idx].to(self.device)
        labels = self.data_signal_out[idx].to(self.device)
        return data, labels

def collate(batch):
    data = torch.stack([item[0] for item in batch], dim=0)   
    labels = torch.stack([item[1] for item in batch], dim=0)    
    return Batch(data, labels)
    
class Batch():
    def __init__(self, data, labels):
        self.u_in = data.float()
        self.u_out = labels.float()



