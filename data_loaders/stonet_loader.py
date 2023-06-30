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
    level_sizes=[2048, 1024, 1024],
    radius_inner=np.array([0.5/10* 1.41, 0.5/8* 1.41, 0.5/8* 1.41]) * np.pi,
    radius_inter=np.array([0.5/8  , 0.5/4]) * np.pi * 1.41,
    out_time_num=None,
    eval_induct=False,
    manifold='planar',
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
        grid_info['edge_feature'] = grid_info['all'][idx, :, :]
        idx_induct = [i for i in range(grid_info['edge_feature'].shape[0]) if i not in idx]
        grid_info['inductive_edge_feature'] = grid_info['all'][idx_induct, :, :]
    else:
        grid_info['edge_feature'] = grid_info['all']

    grid = grid_info['edge_feature'][:,0,:-1]
    inductive_grid = grid_info['inductive_edge_feature'][:,0,:-1] if 'inductive_edge_feature' in grid_info else None
    
    dim = grid.shape[-1]
    grid = grid.reshape(-1, dim)
    radius_inner = np.array(radius_inner)
    radius_inter = np.array(radius_inter)
    
    mesh_generator = MeshGenerator(
        grid=grid, inductive_grid=inductive_grid, level=len(level_sizes), m=level_sizes, 
        radius_inner=radius_inner, radius_inter=radius_inter, manifold=manifold, device=device
        )

    
    if len(train_in.shape) == 3: 
        train_in = train_in.unsqueeze(dim=-1)
        val_in = val_in.unsqueeze(dim=-1)
        test_in = test_in.unsqueeze(dim=-1)

    if len(train_out.shape) == 3: 
        train_out = train_out.unsqueeze(dim=-1) 
        val_out = val_out.unsqueeze(dim=-1)
        test_out = test_out.unsqueeze(dim=-1)
    
    train_in = train_in.permute(0,2,1,3)
    train_out = train_out.permute(0,2,1,3)
    val_in = val_in.permute(0,2,1,3)
    val_out = val_out.permute(0,2,1,3)
    test_in = test_in.permute(0,2,1,3)
    test_out = test_out.permute(0,2,1,3)

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
        in_data=train_in, mesh_generator=mesh_generator, out_data=train_out,
        out_time_num=out_time_num, device=device, eval_induct=False,
    )

    val_set = Seq2SeqLoader(
        in_data=val_in, mesh_generator=mesh_generator, out_data=val_out,
        out_time_num=out_time_num, device=device, eval_induct=eval_induct,
    )

    test_set = Seq2SeqLoader(
        in_data=test_in, mesh_generator=mesh_generator, out_data=test_out,
        out_time_num=out_time_num, device=device, eval_induct=eval_induct,
    )

    data = {}
    data['train_loader'], data['val_loader'], data['test_loader'] = \
        DataLoader(trn_set, batch_size=batch_size, shuffle=True, collate_fn=collate),\
        DataLoader(val_set, batch_size=val_batch_size, shuffle=False, collate_fn=collate),\
        DataLoader(test_set, batch_size=val_batch_size, shuffle=False, collate_fn=collate)
    data['scaler'] = out_normalizer.to(device)
    data['mesh_generator'] = mesh_generator
    data['node_level'] = mesh_generator.node_level
    data['original_resolution'] = tuple(original_resolution)
    data['dimension'] = dimension
    return data


class MeshGenerator(object):
    def __init__(
        self,
        grid, 
        level, 
        m, 
        radius_inner,
        radius_inter,
        manifold,
        inductive_grid=None,
        device='cuda',
        seed=2022) -> None:
        super().__init__()
        # set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        self.device = device
        self.m = m
        self.level = level
        self.grid = grid.float()
        self.n = self.grid.shape[0]
        self.d = self.grid.shape[-1]
        self.radius_inner = radius_inner
        self.radius_inter = radius_inter
        self.manifold = manifold
        if self.manifold == 'sphere':
            self.distance_func = self.sphere_distance
        elif self.manifold == 'planar':
            self.distance_func = self.euclid_distance
        else:
            raise NotImplementedError('the manifold is not supprted.')
        
        self.idx, self.idx_used, self.grid_sample = self.sample(n=self.n)
        self.grid_sample_all = torch.cat(self.grid_sample, dim=0)
        
        self.edge_index_mid, self.edge_index_down, self.edge_index_up, self.n_edges_inner, self.n_edges_inter = \
            self.ball_connectivity(self.radius_inner, self.radius_inter, self.grid_sample, stop_passing_index=None)
            
        self.edge_index_mid_range, self.edge_index_down_range, self.edge_index_up_range = \
            self.get_edge_index_range(self.n_edges_inner, self.n_edges_inter)
            
        self.edge_attr, self.edge_attr_down, self.edge_attr_up = \
            self.edge_attributes(self.grid_sample, self.edge_index_mid, self.edge_index_down, 
                                 self.edge_index_up, self.n_edges_inner, self.n_edges_inter,
                                 self.edge_index_mid_range, self.edge_index_down_range,
                                 self.edge_index_up_range)
        self.edge_index_all = self.construct_overall_graph(self.radius_inner[0]/2, self.grid_sample)
        
        self.node_level = [grid.shape[0] for grid in self.grid_sample]
        
        if inductive_grid is not None:
            self.induct_grid = inductive_grid.float()
            self.induct_n = self.induct_grid.shape[0]
            self.idx_all_induct = torch.arange(0, self.n + self.induct_n, 1)
            
            
            self.induct_idx = [ii for ii in self.idx]
            self.induct_idx[0] = torch.cat([self.idx[0], self.idx_all_induct[-self.induct_n:]])
            self.induct_idx_used = torch.cat(self.induct_idx)
            
            self.induct_grid_sample = [ii for ii in self.grid_sample]
            self.induct_grid_sample[0] = torch.cat([self.grid_sample[0], self.induct_grid], dim=0)
            self.induct_grid_sample_all = torch.cat(self.induct_grid_sample, dim=0)
            
            self.induct_point_idx = torch.arange(len(self.idx[0]), len(self.idx[0]) + self.induct_n)
            self.induct_edge_index_mid, self.induct_edge_index_down, self.induct_edge_index_up, self.induct_n_edges_inner, self.induct_n_edges_inter = \
                self.ball_connectivity(self.radius_inner, self.radius_inter, self.induct_grid_sample, stop_passing_index=None)
            
            self.induct_edge_index_mid_range, self.induct_edge_index_down_range, self.induct_edge_index_up_range = \
                self.get_edge_index_range(self.induct_n_edges_inner, self.induct_n_edges_inter)
                
            self.induct_edge_attr, self.induct_edge_attr_down, self.induct_edge_attr_up = \
                self.edge_attributes(self.induct_grid_sample, self.induct_edge_index_mid, self.induct_edge_index_down, 
                                    self.induct_edge_index_up, self.induct_n_edges_inner, self.induct_n_edges_inter,
                                    self.induct_edge_index_mid_range, self.induct_edge_index_down_range,
                                    self.induct_edge_index_up_range)
            self.induct_edge_index_all = self.construct_overall_graph(self.radius_inner[0]/2, self.induct_grid_sample, None)
            self.induct_node_level = [grid.shape[0] for grid in self.induct_grid_sample]
            
            
            
    def sample(self, n):
        idx_all = torch.arange(0, n, 1)

        idx = []
        grid_sample = []
        
        perm = torch.randperm(n)
        
        index = 0
        for l in range(self.level):
            idx_temp = idx_all[perm[index: index+self.m[l]]]
            idx.append(idx_temp)
            grid_sample.append(self.grid[idx[l]])
            index = index+self.m[l]

        idx_used = idx_all[perm[:index]]

        return idx, idx_used, grid_sample

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
    
    def construct_overall_graph(self, r, grid_sample, stop_passing_index=None):
        grid_sample = torch.cat(grid_sample, dim=0)
        pwd = self.distance_func(grid_sample)
        edge_index = torch.vstack(torch.where(pwd <= r))
        edge_index = edge_index[[1,0]].long()
        if stop_passing_index is not None:
            not_from_stop = ~(edge_index[0][...,None] == stop_passing_index).any(-1)
            edge_index = edge_index[:,torch.where(not_from_stop)[0]]
        edge_index_all = edge_index
        return edge_index_all
        
    def ball_connectivity(self, radius_inner, radius_inter, grid_sample, stop_passing_index=None):
        assert len(radius_inner) == self.level
        assert len(radius_inter) == self.level - 1
        edge_index_mid = []
        edge_index_down = []
        edge_index_up = []
        n_edges_inner = []
        n_edges_inter = []
        
        index = 0
        for l in range(self.level):
            pwd = self.distance_func(grid_sample[l])
            # pwd += (torch.eye(pwd.shape[0])*1e2).to(self.device) # remove the self-loop
            edge_index = torch.vstack(torch.where(pwd <= radius_inner[l])) + index
            # edge_index[0] --> from
            # edge_index[1] --> to
            edge_index = edge_index[[1,0]]
            
            if stop_passing_index is not None:
                not_from_stop = ~(edge_index[0][...,None] == stop_passing_index).any(-1)
                edge_index = edge_index[:,torch.where(not_from_stop)[0]]
            
            edge_index_mid.append(edge_index.long())
            n_edges_inner.append(edge_index.shape[1])
            index = index + grid_sample[l].shape[0]

        index = 0
        for l in range(self.level-1):
            pwd = self.distance_func(grid_sample[l], grid_sample[l+1]) # n * m
            edge_index = torch.vstack(torch.where(pwd <= radius_inter[l])) + index
            edge_index[1, :] = edge_index[1, :] + grid_sample[l].shape[0]
            edge_index = edge_index[[1,0]]
            
            if stop_passing_index is not None:
                not_from_stop = ~(edge_index[0][...,None] == stop_passing_index).any(-1)
                edge_index = edge_index[:,torch.where(not_from_stop)[0]]
                
            edge_index_down.append(edge_index.long())
            
            edge_index = edge_index[[1,0]]
            
            if stop_passing_index is not None:
                not_from_stop = ~(edge_index[0][...,None] == stop_passing_index).any(-1)
                edge_index = edge_index[:,torch.where(not_from_stop)[0]]
                
            edge_index_up.append(edge_index.long())
                
            n_edges_inter.append(edge_index.shape[1])
            index = index + grid_sample[l].shape[0]

        edge_index_mid = torch.cat(edge_index_mid, dim=1)
        edge_index_down = torch.cat(edge_index_down, dim=1) if len(edge_index_down)!=0 else torch.zeros_like(edge_index_mid)
        edge_index_up = torch.cat(edge_index_up, dim=1) if len(edge_index_up)!=0 else torch.zeros_like(edge_index_mid)

        return edge_index_mid, edge_index_down, edge_index_up, n_edges_inner, n_edges_inter

    def get_edge_index_range(self, n_edges_inner, n_edges_inter):
        # in order to use graph network's data structure,
        # the edge index shall be stored as tensor instead of list
        # we concatenate the edge index list and label the range of each level

        edge_index_mid_range = torch.zeros((self.level,2), dtype=torch.long)
        edge_index_down_range = torch.zeros((self.level-1,2), dtype=torch.long)
        edge_index_up_range = torch.zeros((self.level-1,2), dtype=torch.long)

        n_edge_index = 0
        for l in range(self.level):
            edge_index_mid_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + n_edges_inner[l]
            edge_index_mid_range[l, 1] = n_edge_index

        n_edge_index = 0
        for l in range(self.level-1):
            edge_index_down_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + n_edges_inter[l]
            edge_index_down_range[l, 1] = n_edge_index
        
        n_edge_index = 0
        for l in range(self.level-1):
            edge_index_up_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + n_edges_inter[l]
            edge_index_up_range[l, 1] = n_edge_index


        return edge_index_mid_range, edge_index_down_range, edge_index_up_range

    def edge_attributes(self, grid_sample, edge_index_mid, edge_index_down, edge_index_up, n_edges_inner, n_edges_inter,
                        edge_index_mid_range, edge_index_down_range, edge_index_up_range):
        grid_sample = torch.cat(grid_sample, dim=0)
        
        edge_attr_mid = []
        edge_attr_down = []
        edge_attr_up = []

        for l in range(self.level):
            idx_from = edge_index_mid_range[l][0]
            idx_to = edge_index_mid_range[l][1]
            edge_attr = grid_sample[edge_index_mid[:,idx_from:idx_to].T].reshape((n_edges_inner[l], 2*self.d))
            edge_attr_mid.append(edge_attr)

        for l in range(self.level - 1):
            idx_from = edge_index_down_range[l][0]
            idx_to = edge_index_down_range[l][1]
            edge_attr = grid_sample[edge_index_down[:,idx_from:idx_to].T].reshape((n_edges_inter[l], 2*self.d))
            edge_attr_down.append(edge_attr)

        for l in range(self.level - 1):
            idx_from = edge_index_up_range[l][0]
            idx_to = edge_index_up_range[l][1]
            edge_attr = grid_sample[edge_index_up[:,idx_from:idx_to].T].reshape((n_edges_inter[l], 2*self.d))
            edge_attr_up.append(edge_attr)

        edge_attr_mid = torch.cat(edge_attr_mid, dim=0)
        edge_attr_down = torch.cat(edge_attr_down, dim=0) if len(edge_attr_down)!=0 else torch.zeros_like(edge_attr_mid)
        edge_attr_up = torch.cat(edge_attr_up, dim=0) if len(edge_attr_up)!=0 else torch.zeros_like(edge_attr_mid)
        return edge_attr_mid, edge_attr_down, edge_attr_up

    def get_attributes(self):
        return self.node_level, self.idx_used, self.grid_sample_all, self.edge_index_all, \
               self.edge_index_mid, self.edge_index_down, self.edge_index_up, \
               self.edge_index_mid_range, self.edge_index_down_range, self.edge_index_up_range,\
               self.edge_attr, self.edge_attr_down, self.edge_attr_up     

    def get_induct_attributes(self):
        return self.induct_node_level, self.induct_idx_used, self.induct_grid_sample_all, self.induct_edge_index_all, \
               self.induct_edge_index_mid, self.induct_edge_index_down, self.induct_edge_index_up, \
               self.induct_edge_index_mid_range, self.induct_edge_index_down_range, self.induct_edge_index_up_range,\
               self.induct_edge_attr, self.induct_edge_attr_down, self.induct_edge_attr_up, self.induct_point_idx
    
class Seq2SeqLoader(Dataset):
    def __init__(
        self, 
        in_data, mesh_generator, out_data,
        out_time_num, device, eval_induct,
        data_time_in=None, data_time_out=None
        ) -> None:
        
        self.device = device

        self.data_time_in = data_time_in
        self.data_time_out = data_time_out
        self.data_signal_in = in_data
        self.data_signal_out = out_data
        
        self.sample_size, self.in_len, self.feature_num = self.data_signal_in.shape[0], self.data_signal_in.shape[1], self.data_signal_in.shape[-1]
        self.out_horizon = self.data_signal_out.shape[1]

        self.data_signal_in = self.data_signal_in.reshape(self.sample_size, self.in_len, -1, self.feature_num)
        self.data_signal_out = self.data_signal_out.reshape(self.sample_size, self.out_horizon, -1, self.feature_num)
        self.eval_induct = eval_induct
        
        if eval_induct:
            assert 'in_inductive_node_feature' in self.data
            assert 'out_inductive_node_feature' in self.data
            
            self.data_inductive_in = torch.from_numpy(self.data['in_inductive_node_feature']).to(self.device)
            self.data_inductive_out = torch.from_numpy(self.data['out_inductive_node_feature']).to(self.device)
            
            self.data_signal_in = torch.cat([self.data_signal_in, self.data_inductive_in], dim=-2)
            self.data_signal_out = torch.cat([self.data_signal_out, self.data_inductive_out], dim=-2)

        self.mesh_generator = mesh_generator

        self.random_mesh_sample()
        self.out_time_num = out_time_num if out_time_num is not None else self.out_horizon

    
    def random_mesh_sample(self):
        
        if self.eval_induct == False:
            self.m, self.idx, self.grid_sample_all, self.edge_index_all, \
            self.edge_index_mid, self.edge_index_down, self.edge_index_up, \
            self.edge_index_mid_range, self.edge_index_down_range, self.edge_index_up_range,\
            self.edge_attr_mid, self.edge_attr_down, self.edge_attr_up = self.mesh_generator.get_attributes()
            self.induct_idx = None
            
        else:
            self.m, self.idx, self.grid_sample_all, self.edge_index_all, \
            self.edge_index_mid, self.edge_index_down, self.edge_index_up, \
            self.edge_index_mid_range, self.edge_index_down_range, self.edge_index_up_range,\
            self.edge_attr_mid, self.edge_attr_down, self.edge_attr_up, self.induct_idx = self.mesh_generator.get_induct_attributes()
            
        self.data_signal_in = self.data_signal_in[:,:,self.idx]
        self.data_signal_out = self.data_signal_out[:,:,self.idx]
    
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        data = self.data_signal_in[idx].to(self.device)
        labels = self.data_signal_out[idx].to(self.device)
        
        edge_attr = [self.edge_attr_mid.to(self.device),
                     self.edge_attr_down.to(self.device),
                     self.edge_attr_up.to(self.device)]
        edge_index = [self.edge_index_mid.to(self.device),
                      self.edge_index_down.to(self.device),
                      self.edge_index_up.to(self.device)]
        edge_range = [self.edge_index_mid_range.to(self.device),
                      self.edge_index_down_range.to(self.device),
                      self.edge_index_up_range.to(self.device)]
        
        edge_index_all = self.edge_index_all.to(self.device)
        time_in = self.data_time_in[idx].to(self.device) if self.data_time_in is not None \
            else torch.arange(0, self.in_len).unsqueeze(dim=-1).to(self.device)
        time_out = self.data_time_out[idx].to(self.device)if self.data_time_out is not None \
            else torch.arange(self.in_len, self.in_len+self.out_horizon).unsqueeze(dim=-1).to(self.device)
        grid = self.grid_sample_all.to(self.device)
        out_time_num = self.out_time_num
        induct_idx = self.induct_idx.to(self.device) if self.induct_idx is not None else None
        
        return grid, data, labels, edge_attr, \
            edge_index, edge_range, time_in, time_out,\
            edge_index_all, out_time_num, induct_idx

def collate(batch):

    grid = torch.stack([item[0] for item in batch], dim=0)
    data = torch.stack([item[1] for item in batch], dim=0)   
    labels = torch.stack([item[2] for item in batch], dim=0)    
    edge_attr = batch[0][3] 
    edge_index = batch[0][4]
    edge_range = batch[0][5]
    time_in = torch.stack([item[6] for item in batch], dim=0)    
    time_out = torch.stack([item[7] for item in batch], dim=0) 
    edge_index_all = batch[0][8]
    out_time_num = batch[0][9]
    induct_idx = batch[0][10]
    return Batch(grid, data, labels, edge_attr, edge_index, edge_range, time_in, time_out, edge_index_all, out_time_num, induct_idx)

class Batch():
    def __init__(self, grid, data, labels, edge_attr, edge_index, edge_range, time_in, time_out, edge_index_all, out_time_num, induct_idx):
        self.horizon = labels.shape[1]
        self.out_idx = torch.randperm(self.horizon)[:out_time_num].sort().values
        
        self.x = grid.float()
        self.data = data.float()
        self.u_out = labels.float()[:,self.out_idx]
        self.edge_attr = edge_attr
        self.edge_index = edge_index
        self.edge_range = edge_range
        self.time_in = time_in.float()
        self.time_out = time_out.float()[:,self.out_idx]
        self.edge_index_all = edge_index_all
        self.induct_idx = induct_idx
        self.t_eval = self.out_idx.float().to(self.x)


