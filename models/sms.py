# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from datasets import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer, FeaBuffer
import torch
from backbone.VAEmodels.vanilla_vae import *
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#####################
class sms(ContinualModel):
    NAME = 'sms'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='SMS')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        print('build ours!')

        self.index = 0
        self.bufferList = []

        #self.mynetWeight = nn.Parameter(torch.randn((2), requires_grad=True))

        for param in self.net.vitmodel_1.parameters():
            param.requires_grad = False

        for param in self.net.vitmodel_2.parameters():
            param.requires_grad = False

        for param in self.net.vitmodel_3.parameters():
            param.requires_grad = False

    def get_myoptimizer(self):

        # check if optimizer is in torch.optim
        supported_optims = {optim_name.lower(): optim_name for optim_name in dir(optim) if optim_name.lower() in self.AVAIL_OPTIMS}
        opt = None
        if self.args.optimizer.lower() in supported_optims:
            if self.args.optimizer.lower() == 'sgd':
                opt = getattr(optim, supported_optims[self.args.optimizer.lower()])(self.get_parameters(), lr=self.args.lr,
                                                                                    weight_decay=self.args.optim_wd,
                                                                                    momentum=self.args.optim_mom,
                                                                                    nesterov=self.args.optim_nesterov == 1)
            elif self.args.optimizer.lower() == 'adam' or self.args.optimizer.lower() == 'adamw':
                opt = getattr(optim, supported_optims[self.args.optimizer.lower()])(self.get_parameters(), lr=self.args.lr,
                                                                                    weight_decay=self.args.optim_wd)

        if opt is None:
            raise ValueError('Unknown optimizer: {}'.format(self.args.optimizer))
        return opt

    def begin_task(self, dataset: ContinualDataset) -> None:
        #begin task
        # print('buffer_size:',self.args.buffer_size // dataset.N_TASKS)
        self.index += 1
        print("index:", self.index, self.args.buffer_size // self.index)
        self.buffer = FeaBuffer(self.args.buffer_size - self.args.buffer_size // self.index * (self.index - 1))
        self.tempbuffer = FeaBuffer(self.maxNum)
        self.opt = self.get_optimizer()
        # print('opt:', self.args.optimizer.lower())

    def end_task(self, dataset: ContinualDataset) -> None:
        #end task
        # self.net.CreateNewExper()
        self.opt = self.get_optimizer()

        # buffer operation
        
        buffer_outputs_1 = []
        buffer_outputs_2 = []
        buffer_logits_1 = []
        buffer_logits_2 = []
        for buffer in self.bufferList:
            if not buffer.is_empty():
                buf_inputs, _, buf_logits = buffer.get_all_data(transform=self.transform, device=self.device)

                buf_outputs, buf_features = self.net(buf_inputs)
                buffer_outputs_1.append(buf_outputs)
                buffer_logits_1.append(buf_logits)

                buf_inputs, buf_labels, _ = buffer.get_all_data(transform=self.transform, device=self.device)

                buf_outputs, buf_features = self.net(buf_inputs)
                buffer_outputs_2.append(buf_outputs)
                buffer_logits_2.append(buf_logits)
                
        buf_outputs_1 = torch.cat(buffer_outputs_1, dim = 0)
        buf_outputs_2 = torch.cat(buffer_outputs_2, dim = 0)
        buf_logits_1 = torch.cat(buffer_logits_1, dim = 0)
        buf_logits_2 = torch.cat(buffer_logits_2, dim = 0)
        loss_mse = self.args.alpha * F.mse_loss(buf_outputs_1, buf_logits_1)
        loss_mse.backward()
        loss_ce = self.args.beta * self.loss(buf_outputs_2, buf_logits_2)
        loss_ce.backward()


    def myPrediction(self,x,k):
        with torch.no_grad():
            #Perform the prediction according to the seloeced expert
            out = self.net.myprediction(x,k)
        return out


    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()

        outputs, ori_features = self.net(inputs)

        loss = self.loss(outputs, labels)
        # print('loss:', loss, loss.item())
        loss.backward()
        tot_loss = loss.item()

        self.opt.step()

        if epoch == 0:
            self.tempbuffer.add_data(examples=ori_features,
                                labels=labels,
                                logits=outputs.data)

        return tot_loss
    