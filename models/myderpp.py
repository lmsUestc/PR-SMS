# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from datasets import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from models.SelfGraphMemory2Framework_ import *
import numpy as np

##########################
class MyDerpp(ContinualModel):
    NAME = 'myderpp'
    COMPATIBILITY = [
        'class-il', 'domain-il', 'task-il', 'general-continual']

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

        self.isFirst = True
        count = int(self.args.buffer_size / 2.0)
        self.buffer = Buffer(count)
        self.dynamicBuffer = Buffer(count)
        distanceType = "WDistance"

        self.TSFramework = SelfGraphMemory2Framework("myName", device, 0)
        self.TSFramework.distance_type = distanceType
        self.TSFramework.MaxMemoryCluster = 20

        self.TSFramework.maxSizeForEachMemory = 64

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()

        outputs = self.net(inputs)

        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss = loss.item()

        if self.isFirst == True:
            #print("dddd")
            #print(np.shape(not_aug_inputs))
            #print(np.shape(labels))
            #print(np.shape(outputs.data))
            self.TSFramework.MemoryBegin(not_aug_inputs, labels,outputs.data)
            self.isFirst = False

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_inputs2,_,buf_logits2 = self.TSFramework.GetRandomSamples(32)
            buf_inputs = torch.cat((buf_inputs,buf_inputs2),0)
            buf_logits = torch.cat((buf_logits,buf_logits2),0)

            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            buf_inputs2,buf_labels2,_ = self.TSFramework.GetRandomSamples(32)
            buf_inputs = torch.cat((buf_inputs,buf_inputs2),0)
            buf_labels = torch.cat((buf_labels,buf_labels2),0)

            buf_outputs = self.net(buf_inputs)
            loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss_ce.backward()
            tot_loss += loss_ce.item()

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return tot_loss
