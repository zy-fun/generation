import os
import torch
from data_provider.data_factory import data_provider
from torch import optim
import torch
import torch.nn as nn
import time

class Exp(object):
    def __init__(self, args):
        self.args = args
        self.model = None
        pass

        self.device = self._acquire_device()
        # self.model = self._build_model().to(self.device)

    def _build_model(self):
        # raise NotImplementedError
        return None

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
        criterion = nn.MSELoss()
        return criterion

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args.data, self.args.batch_size, split=flag)
        return data_set, data_loader

    def vali(self):
        pass

    def train(self):
        train_data, train_loader = self._get_data(flag='train')

        # path = 
        # if not os.path.exists(path):
        #     os.makedirs(path)

        time_now = time.time()

        # train_steps = len(train_loader)
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # self.model.train()
            epoch_time = time.time()
            for i, (edge_seq, edge_feature, timeF) in enumerate(train_loader):
                iter_count += 1
                # model_optim.zero_grad()
                edge_seq = edge_seq.to(self.device)
                edge_feature = edge_feature.to(self.device)
                timeF = timeF.to(self.device)
                exit()
            exit()

        return self.model

        pass

    def test(self):
        pass