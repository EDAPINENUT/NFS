from random import sample
import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
from pathlib import Path
from functools import reduce
from functools import partial
# from models.libs.utils import UnitGaussianNormalizer

def load_dataset(dataset_dir, batch_size, val_batch_size=None, sub=1, T_in=10, T_pred=40, device=None, **kwargs):
    dataset_path = Path(dataset_dir) / 'data.mat'
    
    if val_batch_size == None:
        val_batch_size = batch_size
        
    reader = MatReader(dataset_path)
    u = reader.read_field('u')
    a = reader.read_field('a')
    sample_num = u.shape[0]
    s = u.shape[1] // sub
    
    ntrain, nval, ntest = int(sample_num * 0.7), int(sample_num * 0.1), int(sample_num * 0.2)
    train_in = u[:ntrain,::sub,::sub,:T_in]
    train_out = u[:ntrain,::sub,::sub,T_in:T_pred+T_in]
    
    val_in = u[ntrain:nval+ntrain,::sub,::sub,:T_in]
    val_out = u[ntrain:nval+ntrain,::sub,::sub,T_in:T_pred+T_in]
    
    test_in = u[-ntest:,::sub,::sub,:T_in]
    test_out = u[-ntest:,::sub,::sub,T_in:T_pred+T_in]
    
    in_normalizer = UnitGaussianNormalizer(train_in)
    train_in = in_normalizer.encode(train_in).unsqueeze(dim=-2).repeat([1,1,1,T_pred,1])
    val_in = in_normalizer.encode(val_in).unsqueeze(dim=-2).repeat([1,1,1,T_pred,1])
    test_in = in_normalizer.encode(test_in).unsqueeze(dim=-2).repeat([1,1,1,T_pred,1])
    
    resolution = train_in.shape[1:-1]
    out_normalizer = UnitGaussianNormalizer(train_out)
    train_out = out_normalizer.encode(train_out)
    val_out = out_normalizer.encode(val_out)
    test_out = out_normalizer.encode(test_out)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_in, train_out), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(val_in, val_out), batch_size=val_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_in, test_out), batch_size=val_batch_size, shuffle=False)

    return {'train_loader':train_loader, 'val_loader':val_loader, 'test_loader':test_loader}, resolution, out_normalizer

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float
    
dataset_path = '/usr/commondata/public/Neural_Dynamics/CTmixer/dataset/burgers_equation/burgers_v1000_t200_r1024_N2048.mat'
reader = MatReader(dataset_path)
reader