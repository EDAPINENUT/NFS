from data_loaders.stonet_loader import load_dataset
import torch
from stmodels.stonet import *
from trainers.trainer import Trainer 

device = torch.device('cuda')
data_path = './dataset/ns_equation/dataset_sr128.pkl'

batch_size = 4
lr = 1e-3
spatial_sample_num = 1024
T_in = 10
T_pred = 20
sub = 1
embed_size = 24
scheduler_step = 10
max_epochs = 500
scheduler_gamma = 0.5

level_sizes = [512, 256, 256]
radius_inner = [0.04, 0.05, 0.05]
radius_inter = [0.05, 0.05]
data = load_dataset(data_path=data_path, batch_size=batch_size, device=device, sub=sub, radius_inner=radius_inner,
        radius_inter=radius_inter, spatial_sample_num=spatial_sample_num, T_in=T_in, T_pred=T_pred, level_sizes=level_sizes)

model_kwargs = {'seq_len':T_in, 'horizon': T_pred, 'input_dim':1, 
                'output_dim':1, 'enc_layer_num':1, 'dec_layer_num':1, 
                'embed_size':embed_size, 'time_dim':1, 'cont_dec':False,
                'location_dim':data['dimension'] - 1 }

model = STONet(level_sizes=level_sizes, **model_kwargs)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

trainer = Trainer(data=data,
                  model=model,
                  log_dir='experiments/',
                  experiment_name='STONet3d_sn{}_ti{}_to{}'.format(spatial_sample_num, T_in, T_pred),
                  optimizer=optimizer,
                  lr_scheduler=scheduler,
                  device=device,
                  normalizer=data['scaler'])

trainer.train()