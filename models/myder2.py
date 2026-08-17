# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import copy
import torch as th
from backbone.HSIC_ import *

class MyDer2(ContinualModel):
    NAME = 'myder2'
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

        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size)

    def begin_task(self, dataset):
        self.old_net = copy.deepcopy(self.net)#self.net.clone()#copy.deepcopy(net)#torch.deepcopy(self.net)

    def Calculate_Distance(self,inputs,labels):
        Oldout1, Oldout2, Oldout3, Oldout4, Oldfeature,Oldout1_, Oldout2_, Oldout3_, Oldout4_, Oldfeature_ = self.old_net.GetAllFeaturs(inputs)
        out1_, out2_, out3_, out4_, feature_,out1__, out2__, out3__, out4__, feature__ = self.net.GetAllFeaturs(inputs)

        out1 = th.reshape(Oldout1, (-1, 64 * 32 * 32))
        out1_ = th.reshape(out1_, (-1, 64 * 32 * 32))
        out1__ = th.reshape(out1__, (-1, 64 * 32 * 32))

        out2 = th.reshape(Oldout2, (-1, 128 * 16 * 16))
        out2_ = th.reshape(out2_, (-1, 128 * 16 * 16))
        out2__ = th.reshape(out2__, (-1, 128 * 16 * 16))

        out3 = th.reshape(Oldout3, (-1, 256 * 8 * 8))
        out3_ = th.reshape(out3_, (-1, 256 * 8 * 8))
        out3__ = th.reshape(out3__, (-1, 256 * 8 * 8))

        out4 = th.reshape(Oldout4, (-1, 512 * 4 * 4))
        out4_ = th.reshape(out4_, (-1, 512 * 4 * 4))
        out4__ = th.reshape(out4__, (-1, 512 * 4 * 4))

        d1 = HSIC(out1, out1_)
        d2 = HSIC(out2, out2_)
        d3 = HSIC(out3, out3_)
        d4 = HSIC(out4, out4_)
        d5 = HSIC(Oldfeature, feature_)

        d = d1 + d2 + d3 + d4 + d5

        #Calculate distance between the common z and specific z
        dd1 = HSIC(out1_, out1__)
        dd2 = HSIC(out2_, out2__)
        dd3 = HSIC(out3_, out3__)
        dd4 = HSIC(out4_, out4__)
        dd5 = HSIC(feature_, feature__)
        sumn1 = dd1 + dd2 + dd3 + dd4 + dd5

        return d,sumn1

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.t1 = 0.1
        self.t2 = 0.5

        if self.current_task > 0:
            self.opt.zero_grad()

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

                r_loss, r_disentangled_loss = self.Calculate_Distance(buf_inputs, buf_logits)
                tot_loss += r_loss * self.t1 + r_disentangled_loss * self.t2

            self.opt.step()

            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels,
                                 logits=outputs.data)
        else:
            self.opt.zero_grad()

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
