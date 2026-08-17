# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import numpy as np
import torch as th
from backbone.HSIC_ import *
import copy

############
class Derpp(ContinualModel):
    NAME = 'myderppresnet50'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay++.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')

        parser.add_argument('--t1', type=float, required=False,
                            help='Penalty1.')

        parser.add_argument('--t2', type=float, required=False,
                            help='Penalty2.')

        parser.add_argument('--isMNIST', type=float, required=False,
                            help='Penalty3.')

        return parser

    def begin_task(self, dataset):
        self.old_net = copy.deepcopy(self.net)

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size)


    def Calculate_DistanceMLP(self,inputs,labels):
        old_feature1, old_feature2 = self.old_net.GetAllFeaturs(inputs)
        feature1, feature2 = self.net.GetAllFeaturs(inputs)

        d1 = HSIC(old_feature1, feature1)
        d2 = HSIC(old_feature2, feature2)

        d = d1 + d2

        #Calculate distance between the common z and specific z
        dd1 = HSIC(old_feature1, old_feature2)
        dd2 = HSIC(feature1, feature2)
        sumn1 = dd1 + dd2

        return d,sumn1

    def Calculate_Distance(self,inputs,labels):
        Oldout1, Oldout2, Oldfeature,Oldout1_, Oldout2_, Oldfeature_ = self.old_net.GetAllFeaturs(inputs)
        out1_, out2_, feature_,out1__, out2__, feature__ = self.net.GetAllFeaturs(inputs)

        out1 = th.reshape(Oldout1, (-1, 64 * 32 * 32))
        out1_ = th.reshape(out1_, (-1, 64 * 32 * 32))
        out1__ = th.reshape(out1__, (-1, 64 * 32 * 32))

        out2 = th.reshape(Oldout2, (-1, 128 * 16 * 16))
        out2_ = th.reshape(out2_, (-1, 128 * 16 * 16))
        out2__ = th.reshape(out2__, (-1, 128 * 16 * 16))

        d1 = HSIC(out1, out1_)
        d2 = HSIC(out2, out2_)
        d5 = HSIC(Oldfeature, feature_)

        d = d1 + d2 + d5

        #Calculate distance between the common z and specific z
        dd1 = HSIC(out1_, out1__)
        dd2 = HSIC(out2_, out2__)
        dd5 = HSIC(feature_, feature__)
        sumn1 = dd1 + dd2 + dd5

        return d,sumn1


    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.t1 = self.args.t1#0.1
        self.t2 = self.args.t2#0.5
        self.isMNIST = self.args.isMNIST
        self.opt.zero_grad()

        if self.current_task > 0:
            outputs = self.net(inputs)

            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss = loss.item()

            if not self.buffer.is_empty():
                buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform,
                                                                 device=self.device)

                buf_outputs = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                tot_loss += loss_mse.item()

                buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform,
                                                                 device=self.device)

                buf_outputs = self.net(buf_inputs)
                loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
                loss_ce.backward()
                tot_loss += loss_ce.item()

                t1 = self.t1
                t2 = self.t2

                if self.isMNIST != 1:
                    r_loss, r_disentangled_loss = self.Calculate_Distance(buf_inputs, buf_logits)
                else:
                    r_loss, r_disentangled_loss = self.Calculate_DistanceMLP(buf_inputs, buf_logits)

                # r_loss.backward()
                # r_disentangled_loss.backward()
                rr = r_loss * t1 - r_disentangled_loss * t2
                rr.backward()
                tot_loss += rr.item()
        else:

            outputs = self.net(inputs)

            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss = loss.item()

            if not self.buffer.is_empty():
                buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

                buf_outputs = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                tot_loss += loss_mse.item()

                buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

                buf_outputs = self.net(buf_inputs)
                loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
                loss_ce.backward()
                tot_loss += loss_ce.item()

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return tot_loss
