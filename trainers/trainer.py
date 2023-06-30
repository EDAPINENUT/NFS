from functools import partial
import torch 
from models.libs.logger import get_logger
from models.libs import utils
from pathlib import Path
import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from models.libs.loss import LpLoss
import numpy as np

class Trainer:
    def __init__(self,
                 data,
                 model,
                 log_dir,
                 normalizer=None,
                 lr=0.001,
                 max_epoch=500,
                 eval_func=None,
                 loss_func=None,
                 lr_scheduler=None,
                 optimizer=None,
                 max_grad_norm=5.0,
                 load_epoch=0,
                 experiment_name='physics_expriment',
                 device='cuda',
                 **kwargs):
        
        self._device = device
        self.data = data
        self.data_num = \
            {'train_num': len(data['train_loader'].dataset), 'val_num': len(data['val_loader'].dataset), 'test_num': len(data['test_loader'].dataset)}
        
        self._model = model.to(device)
        self._max_epoch = max_epoch
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)\
            if optimizer is None else optimizer
        
        self.eval_func = eval_func if eval_func is not None else LpLoss(size_average=False, p=2, relative_error=False)
        self.loss_func = loss_func if loss_func is not None else partial(F.mse_loss, reduction='mean')
        
        scheduler_step = 100
        scheduler_gamma = 0.5
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)\
            if lr_scheduler is None else lr_scheduler
            
        self.normalizer = normalizer.to(device) if normalizer is not None else nn.Identity()
        
        self.max_grad_norm = max_grad_norm
        
        self._experiment_name = experiment_name
        self._log_dir = self._get_log_dir(self, log_dir)
        self._logger = get_logger(self._log_dir, __name__, 'info.log', level='INFO')
        
        self._epoch_num = load_epoch
        if self._epoch_num > 0:
            self.load_model(self._epoch_num)
            
    @staticmethod
    def _get_log_dir(self, log_dir):
        log_dir = Path(log_dir)/self._experiment_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        model_path = Path(self._log_dir)/'saved_model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        config = {}
        config['model_state_dict'] = self._model.state_dict()
        config['epoch'] = epoch
        model_name = model_path/('epo%d.tar' % epoch)
        torch.save(config, model_name)
        self._logger.info("Saved model at {}".format(epoch))
        return model_name
    
    def load_model(self, epoch_num):
        model_path = Path(self._log_dir)/'saved_model'
        model_name = model_path/('epo%d.tar' % epoch_num)

        assert os.path.exists(model_name), 'Weights at epoch %d not found' % epoch_num

        checkpoint = torch.load(model_name, map_location='cpu')
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch_num))
    
    def _evaluate(self, dataset, epoch_num=0, load_model=False, save_results=False):
        if load_model == True:
            self.load_model(epoch_num)
        
        self._model.eval()
        val_iterator = self.data['{}_loader'.format(dataset)]
        val_loss = 0.0
        y_truths = []
        y_preds = []
        losses = []
        with torch.no_grad():
            for batch in val_iterator:
                y = batch.u_out.squeeze(dim=-1)

                out = self._model(batch).squeeze(dim=-1)

                out = self.normalizer.decode(out)
                y = self.normalizer.decode(y)

                loss = self.eval_func(out.reshape(out.shape[0], -1), y.reshape(y.shape[0], -1))
                val_loss += loss.item()

                losses.append(loss.item())
                y_truths.append(y.detach().cpu().numpy())
                y_preds.append(out.detach().cpu().numpy())
        if save_results:
            y_truths = np.concatenate(y_truths, axis=0)
            y_preds = np.concatenate(y_preds, axis=0)
            np.save('ns_truth_infer', np.array(y_truths))
            np.save('ns_pred_infer', np.array(y_preds))

        val_loss /= self.data_num['{}_num'.format(dataset)]
        return val_loss
    
    def train(self, log_every=1, test_every_n_epochs=10, save_model=True, patience=100):
        
        message = "the number of trainable parameters: " + str(utils.count_parameters(self._model))
        self._logger.info(message)
        self._logger.info('Start training the model ...')
        
        train_iterator = self.data['train_loader']
        
        min_val_loss = float('inf')
        
        for epoch_num in range(self._epoch_num, self._max_epoch):

            self._model = self._model.train()
            train_loss = 0
            progress_bar = tqdm(train_iterator, unit="batch")

            for batch in progress_bar:
                
                y = batch.u_out.squeeze(dim=-1)
                self.optimizer.zero_grad()
                out = self._model(batch).squeeze(dim=-1)
                mse = self.loss_func(out, y, reduction='mean')
                
                out = self.normalizer.decode(out)
                y = self.normalizer.decode(y)
                loss = self.eval_func(out.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1))
                
                mse.backward()
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                train_loss += loss.item()
                progress_bar.set_postfix(training_loss=loss.item())

                # clear the cache
                # with torch.cuda.device('cuda:{}'.format(self._device.index)):
                #     torch.cuda.empty_cache()
                # del loss
                
            self.lr_scheduler.step()
            train_loss /= self.data_num['train_num']
            
            val_loss = self._evaluate(dataset='val')
            
            if (epoch_num % log_every) == log_every - 1:
                message = '---Epoch.{} Training Loss per sample: {:6f}. ' \
                    .format(epoch_num, train_loss)
                self._logger.info(message)
                message = '---Epoch.{} Validation Loss per sample: {:6f}. ' \
                    .format(epoch_num, val_loss)
                self._logger.info(message)
            
            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss = self._evaluate(dataset='test')
                message = '---Epoch.{} Test Loss per sample: {:6f}. ' \
                    .format(epoch_num, test_loss)
                self._logger.info(message)
            
            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    best_epoch = epoch_num
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Validation Loss decrease from {:.6f} to {:.6f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
               
            # early stopping
            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num, 'the best epoch is: %d' % best_epoch)
                    break    

        self._test_final_n_epoch(n=3)

    def _test_final_n_epoch(self, n=1, eval_funcs=[LpLoss(size_average=False, relative_error=False, p=1), LpLoss(size_average=False, relative_error=False, p=2)]):
        model_path = Path(self._log_dir)/'saved_model'
        model_list = os.listdir(model_path)
        import re

        epoch_list = []
        for filename in model_list:
            epoch_list.append(int(re.search(r'\d+', filename).group()))

        epoch_list = np.sort(epoch_list)[-n:]
        for i in range(n):
            epoch_num = epoch_list[i]

            for j in range(len(eval_funcs)):
                self.eval_func = eval_funcs[j]
                test_loss = self._evaluate('test', epoch_num, load_model=True)
                message = "Loaded the {}-th epoch.".format(epoch_num) + \
                          " Loss_{} : {}".format(j, test_loss)
                self._logger.info(message)
        
        self._logger.handlers.clear()

    def test_final_n_epoch(self, n=1, eval_funcs=[LpLoss(size_average=False, relative_error=False, p=1), LpLoss(size_average=False, relative_error=False, p=2)]):
        model_path = Path(self._log_dir)/'saved_model'
        model_list = os.listdir(model_path)
        import re

        epoch_list = []
        for filename in model_list:
            epoch_list.append(int(re.search(r'\d+', filename).group()))

        epoch_list = np.sort(epoch_list)[-n:]
        for i in range(n):
            epoch_num = epoch_list[i]
            test_losses = []
            for j in range(len(eval_funcs)):
                self.eval_func = eval_funcs[j]
                test_loss = self._evaluate('test', epoch_num, load_model=True)
                message = "Loaded the {}-th epoch.".format(epoch_num) + \
                          " Loss_{} : {}".format(j, test_loss)
                test_losses.append(test_loss)
                self._logger.info(message)

        return test_losses
