from data_loaders.gstmodel_dataloader import load_dataset
import torch
from stmodels.baselines.recurrent import *
from trainers.trainer import Trainer 

device = torch.device('cuda')
data_path = './dataset/burgers_equation/dataset_v100_sr1024.pkl'

batch_size = 8
lr = 1e-3
spatial_sample_num = 512
T_in = 10
T_pred = 10
sub = 1
embed_size = 64
scheduler_step = 10
max_epochs = 200
scheduler_gamma = 0.5
conv_method = 'AGCRN' #GConvGRU, DCRNN, AGCRN
data = load_dataset(data_path=data_path, batch_size=batch_size, device=device, sub=sub, 
                  spatial_sample_num=spatial_sample_num, T_in=T_in, T_pred=T_pred, eps=0.05)

model_kwargs = {'seq_len':T_in, 'horizon': T_pred, 'input_size':1, 'max_view':2,
                'output_size':1, 'depth':2, 'embed_size':embed_size,'node_num':data['node_num'],
                'embed_dim':2 }

model = RNNModel(sparse_idx=data['sparse_idx'], 
                conv_method=conv_method,
                **model_kwargs)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

trainer = Trainer(data=data,
                  model=model,
                  log_dir='experiments/',
                  experiment_name='{}BGS_s{}_ti{}_to{}'.format(conv_method, sub, T_in, T_pred),
                  optimizer=optimizer,
                  lr_scheduler=scheduler,
                  device=device,
                  normalizer=data['scaler'])

# trainer.train()
trainer._test_final_n_epoch(n=3)