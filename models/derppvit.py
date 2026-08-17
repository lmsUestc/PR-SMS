# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import numpy as np
#from backbone.utils.layers import IncrementalClassifier
import torch as th
from torch.nn.functional import avg_pool2d, relu

import torch
import clip
from PIL import Image
import torchvision.models as models
import copy
import torch.nn as nn

#############################
class DerppViT(ContinualModel):
    NAME = 'derppvit'
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

    def GetFeatureVector(self,x,model):

        #f = model(x,returnt='features')

        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        return x

    def GetOutputFromClassifier(self,x):

        features1 = self.GetFeatureVector(x,self.slowNet)
        features2 = self.GetFeatureVector2(x,self.net)

        f1 = avg_pool2d(features1, features1.shape[2])  # -> 512, 1, 1
        f1 = f1.view(f1.size(0), -1)  # 512
        #f1 = features1

        f2 = avg_pool2d(features2, features2.shape[2])  # -> 512, 1, 1
        f2 = f2.view(f2.size(0), -1)  # 512

        feature = torch.cat((f1,f2),1)
        feature = f1

        #feature = f2
        out = self.classifier(feature)
        return out

    def GetFeatureVector2(self,x,model):
        out_0 = model.bn1(model.conv1(x))  # 64, 32, 32
        if model.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(model, 'maxpool'):
            out_0 = model.maxpool(out_0)

        out_1 = model.layer1(out_0)  # -> 64, 32, 32
        out_2 = model.layer2(out_1)  # -> 128, 16, 16
        out_3 = model.layer3(out_2)  # -> 256, 8, 8
        out_4 = model.layer4(out_3)  # -> 512, 4, 4

        return out_4

    def GetNewOptimizer(self):
        newOptimizer = torch.optim.Adam(self.slowNet.parameters(), lr = self.args.lr,
        #newOptimizer = torch.optim.Adam(self.net.parameters(), lr = self.args.lr,
        weight_decay = self.args.optim_wd)
        parameters = self.classifier.parameters()
        newOptimizer.add_param_group({'params': parameters, 'lr': self.args.lr})
        #newOptimizer.add_param_group({'params': self.slowNet.parameters(), 'lr': self.args.lr})

        #newOptimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr,
        #weight_decay = self.args.optim_wd)
        return newOptimizer

    def output(self,x):
        features = self.GetFeatureVector(x,self.slowNet)

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size)
        
        #self.slowNet = copy.deepcopy(self.net)
        num_classes = 10
        #embed_dim = self.net.embed_dim * 2

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        model = models.resnet18(pretrained=False)
        modelfile = '/jmain02/home/J2AD016/jjw02/ffy12-jjw02/resnet18-f37072fd.pth'
        checkpoint = torch.load(modelfile)
        model.load_state_dict(checkpoint)
        self.slowNet = model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.slowNet = self.slowNet.to(device)
        self.classifier = self.classifier.to(device)

        self.newOptimizer = self.GetNewOptimizer()

        #device = "cuda" if torch.cuda.is_available() else "cpu"
        #model, preprocess = clip.load("ViT-B/32", device=device)
        #self.preprocess = preprocess

        '''
        parameterSet1 = self.net.get_params(discard_classifier=True)
        classifierParameter = self.head.parameters()
        parameterSet2 = self.slowNet.get_params(discard_classifier=True)

        self.vitOpt = th.optim.Adam(parameterSet1, lr=self.args.lr,weight_decay=self.args.optim_wd)
        '''

    def MyForward(self,data):
        returnt = 'dd'
        f1 = self.net.forward_features(data, return_all=returnt == 'full')
        f2 = self.slowNet.forward_features(data, return_all=returnt == 'full')

        a1 = self.net.forward_head(f1,True)
        a2 = self.slowNet.forward_head(f2,True)
        a1 = th.cat((a1,a2),0)
        out = self.head(a1)

        return out

    def forward(self, x: torch.Tensor):
        #Define the prediction
        with torch.no_grad():
            outputs = self.GetOutputFromClassifier(x)
            return outputs

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        #self.opt.zero_grad()
        self.newOptimizer.zero_grad()

        #outputs = self.MyForward(inputs) #self.net(inputs)
        outputs = self.GetOutputFromClassifier(inputs)

        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss = loss.item()

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.GetOutputFromClassifier(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.GetOutputFromClassifier(buf_inputs)
            loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss_ce.backward()
            tot_loss += loss_ce.item()

        #self.opt.step()
        self.newOptimizer.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return tot_loss
