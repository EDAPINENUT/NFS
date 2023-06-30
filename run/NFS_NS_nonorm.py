from data_loaders.phymix_dataloader import load_dataset
from models.mixer_wrapper import MixerWrapper
from models.basic_layers import *
from models.token_mixers import *
from models.channel_mixers import *
import torch.nn as nn
from functools import partial
from trainers.trainer import Trainer 
from models.libs.utils import SetSeed
SetSeed(2024)

data_path = './dataset/ns_equation/dataset_sr128.pkl'
device = torch.device('cuda')
batch_size = 4
embed_size = 32
lr = 1e-3
max_epochs = 500
scheduler_step = 100
scheduler_gamma = 0.5
equispace = False
T_in = 10
T_pred = 40
spatial_sample_num = 4096
sub = 1
interpolation = 'learned' # gaussian #learned
time_k = 8 if T_pred//2 >=8 else T_pred//2 + 1
eps=0.05

data = load_dataset(dataset_path=data_path, batch_size=batch_size, spatial_sample_num=spatial_sample_num, eps=eps,
                    T_in=T_in, T_pred=T_pred, resample_resolution=[64,64], equispace=equispace, sub=sub, device=device)
resolution = data['original_resolution'] if equispace else data['resample_resolution']
input_size = T_in + data['dimension']
node_attr_size = 2 * (data['dimension'] - 1) + T_in + 1

model = MixerWrapper(input_size=input_size,
                    output_size=1,
                    embed_size=embed_size,
                    resolution=resolution,
                    lifter=MLPLifter,
                    projector=MLPProjector,
                    token_mixer=FourierMixer,
                    channel_mixer=partial(LowFreqMixer, k=8, time_k=time_k),
                    token_demixer=FourierDemixer,
                    residual_layer=ConvResLayer3D,
                    node_attr_size=node_attr_size,
                    equispace=equispace,
                    interpolation=interpolation
                    ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

trainer = Trainer(data=data,
                  model=model,
                  log_dir='experiments/',
                  experiment_name='NFS3d_s{}_ti{}_to{}_nonequi_{}_nonorm'.format(sub, T_in, T_pred, interpolation),
                  optimizer=optimizer,
                  lr_scheduler=scheduler,
                  device=device,
                  normalizer=data['scaler'])

trainer._test_final_n_epoch()
trainer.train()
