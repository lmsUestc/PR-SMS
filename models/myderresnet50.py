# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser, add_experiment_args, add_management_args, add_rehearsal_args
from utils.buffer import Buffer
import numpy as np
import torch as th
from backbone.HSIC_ import *
import copy

class MyDerResnet50(ContinualModel):
    NAME = 'myderresnet50'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via'
                                ' Dark Experience Replay.')
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')

        parser.add_argument('--t1', type=float, required=False,
                            help='Penalty1.')

        parser.add_argument('--t2', type=float, required=False,
                            help='Penalty2.')

        parser.add_argument('--isMNIST', type=float, required=False,
                            help='Penalty3.')

        return parser

    def __init__(self, backbone, loss, args, transform):
        super(MyDerResnet50, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)

        self.slowArr = []
        self.fastArr = []
        self.lastInput = []


    def begin_task(self, dataset):
        self.old_net = copy.deepcopy(self.net)

    #def end_task(self, dataset):

        '''
        if self.current_task > 0:
            if np.shape(self.slowArr)[0] < self.current_task:
                fastDistance = self.Estimate_FastRepresentation_Distance(self.lastInput)
                slowDistance = self.Estimate_SlowRepresentation_Distance(self.lastInput)
                fastDistance = fastDistance.detach().cpu().numpy()
                slowDistance = slowDistance.detach().cpu().numpy()
                self.slowArr.append(slowDistance)
                self.fastArr.append(fastDistance)

        arr1 = np.array(self.slowArr).astype('str')
        myThirdName = "results/SplitCIFAR10_Slow.txt"
        # myThirdName = "results/Diffusion_Forgetting_RecoLoss_FirstTaskLearning.txt"
        f = open(myThirdName, "w", encoding="utf-8")
        for i in range(np.shape(arr1)[0]):
            f.writelines(arr1[i])
            f.writelines('\n')
        f.flush()
        f.close()
        
        arr1 = np.array(self.fastArr).astype('str')
        myThirdName = "results/SplitCIFAR10_Fast.txt"
        # myThirdName = "results/Diffusion_Forgetting_RecoLoss_FirstTaskLearning.txt"
        f = open(myThirdName, "w", encoding="utf-8")
        for i in range(np.shape(arr1)[0]):
            f.writelines(arr1[i])
            f.writelines('\n')
        f.flush()
        f.close()
        '''

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

    def Estimate_FastRepresentation_Distance(self,inputs):
        with torch.no_grad():
            Oldout1, Oldout2, Oldout3, Oldout4, Oldfeature,Oldout1_, Oldout2_, Oldout3_, Oldout4_, Oldfeature_ = self.old_net.GetAllFeaturs(inputs)
            out1_, out2_, out3_, out4_, feature_,out1__, out2__, out3__, out4__, feature__ = self.net.GetAllFeaturs(inputs)

            out1 = th.reshape(Oldout1_, (-1, 64 * 32 * 32))
            out1_ = th.reshape(out1__, (-1, 64 * 32 * 32))

            out2 = th.reshape(Oldout2_, (-1, 128 * 16 * 16))
            out2_ = th.reshape(out2__, (-1, 128 * 16 * 16))

            out3 = th.reshape(Oldout3_, (-1, 256 * 8 * 8))
            out3_ = th.reshape(out3__, (-1, 256 * 8 * 8))

            out4 = th.reshape(Oldout4_, (-1, 512 * 4 * 4))
            out4_ = th.reshape(out4__, (-1, 512 * 4 * 4))

            d1 = HSIC(out1, out1_)
            d2 = HSIC(out2, out2_)
            d3 = HSIC(out3, out3_)
            d4 = HSIC(out4, out4_)
            d5 = HSIC(Oldfeature_, feature__)

            d = d1 + d2 + d3 + d4 + d5

        return d

    def Estimate_SlowRepresentation_Distance(self,inputs):
        with torch.no_grad():
            Oldout1, Oldout2, Oldout3, Oldout4, Oldfeature,Oldout1_, Oldout2_, Oldout3_, Oldout4_, Oldfeature_ = self.old_net.GetAllFeaturs(inputs)
            out1_, out2_, out3_, out4_, feature_,out1__, out2__, out3__, out4__, feature__ = self.net.GetAllFeaturs(inputs)

            out1 = th.reshape(Oldout1, (-1, 64 * 32 * 32))
            out1_ = th.reshape(out1_, (-1, 64 * 32 * 32))

            out2 = th.reshape(Oldout2, (-1, 128 * 16 * 16))
            out2_ = th.reshape(out2_, (-1, 128 * 16 * 16))

            out3 = th.reshape(Oldout3, (-1, 256 * 8 * 8))
            out3_ = th.reshape(out3_, (-1, 256 * 8 * 8))

            out4 = th.reshape(Oldout4, (-1, 512 * 4 * 4))
            out4_ = th.reshape(out4_, (-1, 512 * 4 * 4))

            d1 = HSIC(out1, out1_)
            d2 = HSIC(out2, out2_)
            d3 = HSIC(out3, out3_)
            d4 = HSIC(out4, out4_)
            d5 = HSIC(Oldfeature, feature_)

            d = d1 + d2 + d3 + d4 + d5

        return d


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


    def Calculate_Distance2(self,inputs,labels):

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

##############
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.t1 = self.args.t1#0.1
        self.t2 = self.args.t2#0.5
        self.isMNIST = self.args.isMNIST

        self.opt.zero_grad()
        tot_loss = 0

        if self.current_task > 0:
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            if not self.buffer.is_empty():
                buf_inputs, buf_logits = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                tot_loss += loss_mse.item()

                t1 = self.t1
                t2 = self.t2

                if self.isMNIST != 1:
                    r_loss, r_disentangled_loss = self.Calculate_Distance(buf_inputs, buf_logits)
                else:
                    r_loss, r_disentangled_loss = self.Calculate_DistanceMLP(buf_inputs, buf_logits)

                #r_loss.backward()
                #r_disentangled_loss.backward()
                rr = r_loss * t1 - r_disentangled_loss * t2
                rr.backward()
                tot_loss += rr.item()#r_loss.item() * t1 + r_disentangled_loss.item() * t2

            self.opt.step()
            self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        else:
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            tot_loss += loss.item()

            if not self.buffer.is_empty():
                buf_inputs, buf_logits = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform, device=self.device)
                buf_outputs = self.net(buf_inputs)
                loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                loss_mse.backward()
                tot_loss += loss_mse.item()

            self.opt.step()
            self.buffer.add_data(examples=not_aug_inputs, logits=outputs.data)

        self.lastInput = inputs
        #Estimate the distance
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        return tot_loss
