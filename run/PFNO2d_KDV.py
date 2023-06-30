from data_loaders.phymix_dataloader import load_dataset
from models.mixer_wrapper import MixerWrapper
from models.basic_layers import *
from models.token_mixers import *
from models.channel_mixers import *
import torch.nn as nn
from functools import partial
from trainers.trainer import Trainer 

data_path = './dataset/kdv_equation/dataset_sr1024.pkl'
device = torch.device('cuda')
batch_size = 16
embed_size = 32
lr = 5e-4
max_epochs = 500
scheduler_step = 100
scheduler_gamma = 0.5
equispace = True
T_in = 10
T_pred = 40
spatial_sample_num = None
sub = 2
data = load_dataset(dataset_path=data_path, batch_size=batch_size, spatial_sample_num=spatial_sample_num, 
                    T_in=T_in, T_pred=T_pred, equispace=equispace, sub=sub, device=device)
resolution = data['original_resolution'] if equispace else data['resample_resolution']
input_size = T_in + data['dimension']
node_attr_size = 2 * (data['dimension'] - 1) + T_in + 1
patch_size = [4,1]

model = MixerWrapper(input_size=input_size,
                    output_size=1,
                    embed_size=embed_size,
                    resolution=resolution,
                    lifter=MLPLifter,
                    projector=MLPProjector,
                    token_mixer=FourierMixer,
                    channel_mixer=LowFreqMixer,
                    token_demixer=FourierDemixer,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    residual_layer=ConvResLayer2D,
                    patch_size=patch_size,
                    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

trainer = Trainer(data=data,
                  model=model,
                  log_dir='experiments/',
                  experiment_name='PFNO2d_KDV_s{}_ti{}_to{}_equi'.format(sub, T_in, T_pred),
                  optimizer=optimizer,
                  lr_scheduler=scheduler,
                  device=device,
                  normalizer=data['scaler'])

trainer.train()