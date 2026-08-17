# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_dataset
from torch.optim import SGD
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args

from utils.conf import get_device
from models.utils.continual_model import ContinualModel

import numpy as np

def get_backbone(bone,model, old_cols=None, x_shape=None,args = None):
    from backbone.MNISTMLP import MNISTMLP
    from backbone.MNISTMLP_PNN import MNISTMLP_PNN,MNISTMLP_MyDynamic
    from backbone.ResNet18 import ResNet
    from backbone.ResNet18_PNN import resnet18_pnn
    from backbone.ResNet18_Dynamic import resnet18_pnn_2,resnet18_pnn_3, ResNetDynamic

    #classCount = model.dataset.N_CLASSES#model.n_remaining_classes + model.n_seen_classes
    #print("class count")
    #print(classCount)

    print(args.dataset)
    if isinstance(bone, MNISTMLP):
        return MNISTMLP_MyDynamic(bone.input_size, bone.output_size, old_cols)
    elif isinstance(bone, ResNet):
        if args.dataset == "seq-tinyimg":
            return resnet18_pnn_3(bone.num_classes, bone.nf, old_cols, x_shape)
        else:
            return resnet18_pnn_2(bone.num_classes, bone.nf, old_cols, x_shape)

        #return resnet18_pnn_2(bone.num_classes, model.args.nf, old_cols, x_shape)
    else:
        raise NotImplementedError('Progressive Neural Networks is not implemented for this backbone')

class DerppDynamic(ContinualModel):

    NAME = 'derppdynamic'
    COMPATIBILITY = ['task-il']


    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Progressive Neural Networks')
        add_rehearsal_args(parser)
        parser.add_argument('--nf', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform):

        self.nets = [get_backbone(backbone,self,args=args).to(get_device())]
        backbone = self.nets[-1]
        self.args = args
        super(DerppDynamic, self).__init__(backbone, loss, args, transform)
        self.x_shape = None
        self.soft = torch.nn.Softmax(dim=0)
        self.logsoft = torch.nn.LogSoftmax(dim=0)
        self.task_idx = 0
        self.args = args

    def forward(self, x, task_label):
        if self.x_shape is None:
            self.x_shape = x.shape

        start_idx, end_idx = self.dataset.get_offsets(task_label)
        if self.task_idx == 0:
            out = self.net(x)

        else:
            self.nets[task_label].to(self.device)
            out = self.nets[task_label](x)
            if self.task_idx != task_label:
                self.nets[task_label].cpu()

        # mask out previous tasks - Task-IL forward
        if start_idx > 0:
            out[:, :start_idx] = -torch.inf
        out[:, end_idx:] = -torch.inf
        return out

    def end_task(self, dataset):

        #if self.task_idx > 0:
        #    self.net.returnComponentWeight()

        # instantiate new column
        self.task_idx += 1

        for name,param in self.nets[-1].named_parameters():
            param.requires_grad = False

        self.nets[-1].cpu()
        self.nets.append(get_backbone(dataset.get_backbone(),self, self.nets, self.x_shape,args=self.args).to(self.device))
        self.net = self.nets[-1]
        self.opt = self.get_optimizer()
        '''
        self.opt.add_param_group({'params': self.net.componentWeight, 'lr': 0.01})
        for c1 in range(np.shape(self.net.IndividualWeightArr)[0]):
            self.opt.add_param_group({'params': self.net.IndividualWeightArr[c1], 'lr': 0.01})
        '''

    ###############
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if self.x_shape is None:
            self.x_shape = inputs.shape

        self.net.to(self.device)

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()
