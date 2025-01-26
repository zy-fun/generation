import os
import torch
from data_provider.data_factory import data_provider
from torch import optim
import torch
import torch.nn as nn
import time
import numpy as np
from models import Transformer, VAETransformer
from tqdm import tqdm
from datetime import datetime, timedelta
from utils.load_params import load_embeddings
from utils.tools import SmartSave, get_timeF
from utils.metrics import metric
from layers.Loss import MaskedMSELoss

class Exp(object):
    def __init__(self, args):
        self.args = args

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        # print(self.model.enc_embedding.edge_embedding.weight)
        self._get_data()

    def _build_model(self):
        self.model_dict = {
            'Transformer': Transformer,
            'VAETransformer': VAETransformer,
        }
        model = self.model_dict[self.args.model].Model
        if self.args.use_pretrained:
            emb_path = os.path.join('dataset/processed', self.args.data)
            emb = torch.tensor(load_embeddings(emb_path, emb_type='cat'), dtype=torch.float32)
            self.args.n_vocab, self.args.enc_emb = emb.shape
            model = model(self.args, emb).to(self.device)
        else:
            model = model(self.args).to(self.device)
        return model

    def _acquire_device(self):
        # if self.args.use_gpu:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(
        #         self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
        #     device = torch.device('cuda:{}'.format(self.args.gpu))
        #     print('Use GPU: cuda:{}'.format(self.args.gpu))
        # else:
        #     device = torch.device('cpu')
        #     print('Use CPU')
        device = torch.device('cuda:0')
        return device
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = MaskedMSELoss()
        return criterion

    def _get_data(self):
        self.train_data, self.train_loader, self.val_data, self.val_loader, self.test_data, self.test_loader = data_provider(self.args)
        pass

    def vali(self, data='val'):
        if data == 'val':
            vali_data, vali_loader = self.val_data, self.val_loader
        elif data == 'test':
            vali_data, vali_loader = self.test_data, self.test_loader
        else:
            vali_data, vali_loader = self.train_data, self.train_loader

        total_loss = []
        criterion = self._select_criterion()
        self.model.eval()
        with torch.no_grad():
            for i, (edge_seq, edge_feature, timeF, y, y_mask) in tqdm(enumerate(vali_loader), desc='Batches'):
                edge_seq = edge_seq.to(self.device)
                edge_feature = edge_feature[:, :, :self.args.enc_dim].to(self.device)
                timeF = timeF.to(self.device)
                y = y.to(self.device)
                y_mask = y_mask.to(self.device)

                out = self.model(edge_seq, edge_feature, timeF)
                out = out.detach().cpu()
                y = y.detach().cpu()
                y_mask = y_mask.detach().cpu()
                loss = criterion(out, y, y_mask)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self.train_data, self.train_loader

        path = os.path.join(self.args.checkpoints, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        smart_save = SmartSave(verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (edge_seq, edge_feature, timeF, y, y_mask) in enumerate(train_loader):
                # iter
                iter_count += 1
                model_optim.zero_grad()

                edge_seq = edge_seq.to(self.device)
                edge_feature = edge_feature[:, :, :self.args.enc_dim].to(self.device)
                timeF = timeF.to(self.device)
                y = y.to(self.device)
                y_mask = y_mask.to(self.device)
                # print(edge_seq.shape, edge_feature.shape, timeF.shape, y.shape, y_mask.shape)

                out = self.model(edge_seq, edge_feature, timeF)
                loss = criterion(out, y, y_mask)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
            
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i+1, epoch+1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            # epoch
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            y = (y.squeeze() + 0.5) * 86399
            out = (out.squeeze() + 0.5) * 86399
            y = [str(timedelta(seconds=int(i))) for i in y[-1].tolist()]
            out = [str(timedelta(seconds=int(i))) for i in out[-1].tolist()]
            print(y)
            print(out)
            train_loss = np.average(train_loss)
            vali_loss = self.vali(data='val')
            test_loss = self.vali(data='test')

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            smart_save(vali_loss, self.model, path)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self):
        test_data, test_loader = self.test_data, self.test_loader

        preds = []
        trues = []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
             for i, (edge_seq, edge_feature, timeF, y, y_mask) in tqdm(enumerate(test_loader)):
                edge_seq = edge_seq.to(self.device)
                edge_feature = edge_feature[:, :, :self.args.enc_dim].to(self.device)
                dec_in = timeF.to(self.device)[:, 0, :].unsqueeze(1)
                y = y.to(self.device)
                y_mask = y_mask.to(self.device)

                # autoregressive
                autoregress_steps = y.shape[1]
                for step in range(autoregress_steps):
                    # if self.args.model == 'VAETransformer':
                    #     out = self.model.predict(edge_seq, edge_feature, dec_in)[:,-1]
                    # else:
                    out = self.model(edge_seq, edge_feature, dec_in)[:,-1]
                    out = get_timeF(out).to(self.device)
                    # print(dec_in[0, :, 0])
                    # print(out[0, 0])
                    dec_in = torch.cat((dec_in, out.unsqueeze(1)), dim=1) 

                    # out = dec_in[0, :, 0]
                    # yy = y[0, :step+2]
                    # out = (out + 0.5) * 86399
                    # yy = (yy + 0.5) * 86399
                out = dec_in[:, 1:, 0]
                    
                y = ((y + 0.5) * 86399).detach().cpu().numpy()
                out = ((out + 0.5) * 86399).detach().cpu().numpy()
                y_mask = y_mask.detach().cpu().numpy()

                for i in range(y.shape[0]):
                    indices = np.nonzero(y_mask[i])
                    pred = out[i][indices]
                    true = y[i][indices]
                    preds.append(pred)
                    trues.append(true) 

        print(pred)
        print(true)
        result = metric(preds, trues)
        for k, value in result.items():
            print(f'{k}: {value}')

        preds = [x[-1:] for x in preds]
        trues = [x[-1:] for x in trues]
        result = metric(preds, trues)
        for k, value in result.items():
            print(f'{k}: {value}')
        return