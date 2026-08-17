# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.nn import functional as F

import numpy as np
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


###################################
class MixtureAdvancedVit(ContinualModel):
    NAME = 'mixtureadvancedvit'
    #COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    COMPATIBILITY = ['task-il']
    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay++.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(MixtureAdvancedVit, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size)
        self.currentTaskIndex = 0

        #self.mynetWeight = nn.Parameter(torch.randn((2), requires_grad=True))

        for param in self.net.vitmodel.parameters():
            param.requires_grad = False

        self.opt = self.get_myoptimizer()

    def end_task(self, dataset):
        n = np.shape(self.net.classifierArr)[0]
        self.net.createNewExpert()
        self.currentTaskIndex = n
        self.net.currentTaskIndex = self.currentTaskIndex
        self.opt = self.get_myoptimizer()

    def get_myoptimizer(self):
        # check if optimizer is in torch.optim
        parameterSet = self.net.ReturnParameterSetAccordingTaskLabel(self.net.currentTaskIndex)
        opt = torch.optim.Adam(parameterSet, lr=self.args.lr, weight_decay=self.args.optim_wd)
        return opt

    def forward(self, x, task_label):

        start_idx, end_idx = self.dataset.get_offsets(task_label)
        self.net.currentTaskInde =task_label

        out = self.net(x)
        # mask out previous tasks - Task-IL forward
        if start_idx > 0:
            out[:, :start_idx] = -torch.inf
        out[:, end_idx:] = -torch.inf
        return out

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.net.currentTaskIndex = self.currentTaskIndex
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
