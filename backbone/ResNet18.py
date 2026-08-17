# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu
from timm import create_model
import torchvision.transforms as transforms
from backbone.VAEmodels.vanilla_vae import *
from models.utils.expert import Expert, ExpertSelector

from backbone import MammothBackbone
#from backbone.vit import *
# from pytorch_pretrained_vit import ViT
def get_backbone_VAE_CIFAR10(x_shape=None):
    return VanillaVAE(1, 64,input_size=23)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.

    Args:
        in_planes: number of input channels
        out_planes: number of output channels
        stride: stride of the convolution

    Returns:
        convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.

        Args:
            in_planes: the number of input channels
            planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.return_prerelu = False
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, input_size)

        Returns:
            output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        if self.return_prerelu:
            self.prerelu = out.clone()

        out = relu(out)
        return out


class ResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.return_prerelu = False
        self.device = "cpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.classifier = nn.Linear(nf * 8 * block.expansion, num_classes)

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable


    def GetAllFeaturs(self, x: torch.Tensor):
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        return out_1,out_2,out_3,out_4,feature

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'all'] but got {}".format(returnt))


class MyResNet(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
        """
        super(MyResNet, self).__init__()
        self.return_prerelu = False
        self.device = "gpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.in_planes = nf
        self.layer1_ = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2_ = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3_ = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4_ = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.classifier = nn.Linear(nf * 8 * block.expansion*2, num_classes)

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable


    def GetAllFeaturs(self, x: torch.Tensor):
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4

        feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        return out_1,out_2,out_3,out_4,feature,out_1_,out_2_,out_3_,out_4_,feature_

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4

        feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        totalFeature = torch.cat((feature,feature_),1)

        if returnt == 'features':
            return feature

        out = self.classifier(totalFeature)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'all'] but got {}".format(returnt))

class HybridResNet(MammothBackbone):

    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
        """
        super(HybridResNet, self).__init__()
        self.return_prerelu = False
        self.device = "gpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.in_planes = nf

        self.mynetWeight = nn.Parameter(torch.randn((2), requires_grad=True))

        '''
        self.layer1_ = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2_ = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3_ = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4_ = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        '''

        #self.classifier = nn.Linear(nf * 8 * block.expansion*2, num_classes)
        self.classifier = nn.Linear(1280, num_classes)

        filter_fn2 = 'jx_vit_base_p16_224-80ecf9dd.pth'
        model_name = 'vit_base_patch16_224'

        # model = vit_base_patch16_224_prompt_prototype(pretrained=True, pretrain_type='in21k-ft-in1k')
        '''
        self.vitmodel = create_vision_transformer('vit_base_patch16_224_in21k_fn_in1k_old', pretrained=False,
                                              **dict(model_kwargs, **kwargs))
        weights_path = "/jmain02/home/J2AD016/jjw02/ffy12-jjw02/vit_base_patch16_224_in21k_miil-887286df.pth"
        pretrained_weights = torch.load(weights_path)
        self.vitmodel.load_state_dict(pretrained_weights)
        '''
        self.vitProcess = transforms.Compose(
        [transforms.Resize(224)])

        self.vitmodel = create_model(
            model_name,
            pretrained=True,
            num_classes=10,
            pretrained_cfg_overlay=dict(
                file="/jmain02/home/J2AD016/jjw02/ffy12-jjw02/mammoth-master/backbone/jx_vit_base_p16_224-80ecf9dd.pth")
        )

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable


    def GetAllFeaturs(self, x: torch.Tensor):
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4

        feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        return out_1,out_2,out_3,out_4,feature,out_1_,out_2_,out_3_,out_4_,feature_

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        #Extract feature from a pretrained vit

        with torch.no_grad():
            processX = self.vitProcess(x)
            features = self.vitmodel.forward_features(processX)
            features = features[:,0,:]
            #features = self.vitmodel(processX,returnt='features')

        #print(np.shape(features))
        vit_features = features
        #vit_features = F.avg_pool2d(features, features.shape[-2:]).squeeze(-1).squeeze(-1)
        #vit_features = vit_features.view(vit_features.size(0), -1)  # 512
        #print(np.shape(vit_features))

        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        '''
        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4

        feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512
        '''

        softWeights = F.softmax(self.mynetWeight)

        totalFeature = torch.cat((feature * softWeights[0],vit_features * softWeights[0]),1)
        #print(np.shape(totalFeature))

        if returnt == 'features':
            return feature

        out = self.classifier(totalFeature)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'all'] but got {}".format(returnt))


class HybridResNet2(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
        """
        super(HybridResNet2, self).__init__()
        self.return_prerelu = False
        self.device = "gpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf

        '''
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        '''

        self.in_planes = nf

        self.mynetWeight = nn.Parameter(torch.randn((2), requires_grad=True))

        '''
        self.layer1_ = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2_ = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3_ = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4_ = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        '''

        #self.classifier = nn.Linear(nf * 8 * block.expansion*2, num_classes)
        self.classifier = nn.Linear(768, num_classes)

        filter_fn2 = 'jx_vit_base_p16_224-80ecf9dd.pth'
        model_name = 'vit_base_patch16_224'

        # model = vit_base_patch16_224_prompt_prototype(pretrained=True, pretrain_type='in21k-ft-in1k')
        '''
        self.vitmodel = create_vision_transformer('vit_base_patch16_224_in21k_fn_in1k_old', pretrained=False,
                                              **dict(model_kwargs, **kwargs))
        weights_path = "/jmain02/home/J2AD016/jjw02/ffy12-jjw02/vit_base_patch16_224_in21k_miil-887286df.pth"
        pretrained_weights = torch.load(weights_path)
        self.vitmodel.load_state_dict(pretrained_weights)
        '''
        self.vitProcess = transforms.Compose(
        [transforms.Resize(224)])

        self.vitmodel = create_model(
            model_name,
            pretrained=True,
            num_classes=10,
            pretrained_cfg_overlay=dict(
                file="/jmain02/home/J2AD016/jjw02/ffy12-jjw02/mammoth-master/backbone/jx_vit_base_p16_224-80ecf9dd.pth")
        )

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable


    def GetAllFeaturs(self, x: torch.Tensor):
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4

        feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        return out_1,out_2,out_3,out_4,feature,out_1_,out_2_,out_3_,out_4_,feature_

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        #Extract feature from a pretrained vit

        with torch.no_grad():
            processX = self.vitProcess(x)
            features = self.vitmodel.forward_features(processX)
            features = features[:,0,:]
            #features = self.vitmodel(processX,returnt='features')

        #print(np.shape(features))
        vit_features = features
        #vit_features = F.avg_pool2d(features, features.shape[-2:]).squeeze(-1).squeeze(-1)
        #vit_features = vit_features.view(vit_features.size(0), -1)  # 512
        #print(np.shape(vit_features))

        '''
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512
        '''

        '''
        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4

        feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512
        '''

        softWeights = F.softmax(self.mynetWeight)

        #totalFeature = torch.cat((feature * softWeights[0],vit_features * softWeights[0]),1)
        #print(np.shape(totalFeature))
        totalFeature = vit_features

        if returnt == 'features':
            return feature

        out = self.classifier(totalFeature)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'all'] but got {}".format(returnt))

class HybridResNet3(MammothBackbone):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
        """
        super(HybridResNet3, self).__init__()
        self.return_prerelu = False
        self.device = "gpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf


        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.in_planes = nf

        self.mynetWeight = nn.Parameter(torch.randn((2), requires_grad=True))

        '''
        self.layer1_ = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2_ = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3_ = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4_ = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        '''

        #self.classifier = nn.Linear(nf * 8 * block.expansion*2, num_classes)
        #self.classifier = nn.Linear(768, num_classes)
        self.classifier = nn.Linear(1280, num_classes)

        filter_fn2 = 'jx_vit_base_p16_224-80ecf9dd.pth'
        model_name = 'vit_base_patch16_224'

        # model = vit_base_patch16_224_prompt_prototype(pretrained=True, pretrain_type='in21k-ft-in1k')
        '''
        self.vitmodel = create_vision_transformer('vit_base_patch16_224_in21k_fn_in1k_old', pretrained=False,
                                              **dict(model_kwargs, **kwargs))
        weights_path = "/jmain02/home/J2AD016/jjw02/ffy12-jjw02/vit_base_patch16_224_in21k_miil-887286df.pth"
        pretrained_weights = torch.load(weights_path)
        self.vitmodel.load_state_dict(pretrained_weights)
        '''
        self.vitProcess = transforms.Compose(
        [transforms.Resize(224)])

        self.vitmodel = create_model(
            model_name,
            pretrained=True,
            num_classes=10,
            pretrained_cfg_overlay=dict(
                file="/jmain02/home/J2AD016/jjw02/ffy12-jjw02/mammoth-master/backbone/jx_vit_base_p16_224-80ecf9dd.pth")
        )

    def to(self, device, **kwargs):
        self.device = device
        return super().to(device, **kwargs)

    def set_return_prerelu(self, enable=True):
        self.return_prerelu = enable
        for c in self.modules():
            if isinstance(c, self.block):
                c.return_prerelu = enable

    def GetAllFeaturs(self, x: torch.Tensor):
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4

        feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        return out_1,out_2,out_3,out_4,feature,out_1_,out_2_,out_3_,out_4_,feature_

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        #Extract feature from a pretrained vit

        with torch.no_grad():
            processX = self.vitProcess(x)
            features = self.vitmodel.forward_features(processX)
            features = features[:,0,:]
            #features = self.vitmodel(processX,returnt='features')

        #print(np.shape(features))
        vit_features = features
        #vit_features = F.avg_pool2d(features, features.shape[-2:]).squeeze(-1).squeeze(-1)
        #vit_features = vit_features.view(vit_features.size(0), -1)  # 512
        #print(np.shape(vit_features))

        '''
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        if self.return_prerelu:
            out_0_t = out_0.clone()
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512
        '''

        '''
        out_1_ = self.layer1_(out_0)  # -> 64, 32, 32
        out_2_ = self.layer2_(out_1_)  # -> 128, 16, 16
        out_3_ = self.layer3_(out_2_)  # -> 256, 8, 8
        out_4_ = self.layer4_(out_3_)  # -> 512, 4, 4

        feature_ = avg_pool2d(out_4_, out_4_.shape[2])  # -> 512, 1, 1
        feature_ = feature_.view(feature_.size(0), -1)  # 512

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512
        '''

        softWeights = F.softmax(self.mynetWeight)

        #totalFeature = torch.cat((feature * softWeights[0],vit_features * softWeights[0]),1)
        #print(np.shape(totalFeature))
        totalFeature = vit_features

        if returnt == 'features':
            return feature

        out = self.classifier(totalFeature)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'all'] but got {}".format(returnt))



class Ours(MammothBackbone):
    
    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        super(Ours, self).__init__()
        self.return_prerelu = False
        self.device = "cuda"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.weightArr = []

        self.in_planes = nf
        # 初始化网络的可学习参数
        self.mynetWeight = nn.Parameter(torch.randn((2), requires_grad=True))
        #self.classifier = nn.Linear(nf * 8 * block.expansion*2, num_classes)
        #self.classifier = nn.Linear(768, num_classes)
        # 定义全连接层和分类器
        self.fc = nn.Linear(768, 500)
        self.classifier = nn.Linear(500, num_classes)
        self.classifierArr = []
        self.fcArr = []
        self.classifierArr.append(self.classifier)
        self.fcArr.append(self.fc)
        self.weightArr.append([])

        # 加载预训练的 ViT 模型
        model_name_1 = 'name_1'
        model_name_2 = 'name_2'
        model_name_3 = 'name_3'
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.vitProcess = transforms.Compose(
        [transforms.Resize(224)])


        # 通过给定路径加载预训练的模型权重
        self.vitmodel_1 = create_model(
            model_name_1,
            pretrained=True,
            num_classes=10,
            pretrained_cfg_overlay=dict(
                file="path")
        )
        self.vitmodel_2 = create_model(
            model_name_2,
            pretrained=True,
            num_classes=10,
            pretrained_cfg_overlay=dict(
                file="path")
        )
        self.vitmodel_3 = create_model(
            model_name_3,
            pretrained=True,
            num_classes=10,
            pretrained_cfg_overlay=dict(
                file="path")
        )

    def CreateNewExper(self):
        """
        创建一个新的实验，更新权重和网络结构。
        """
        existedN = np.shape(self.fcArr)[0]
        self.mynetWeight = nn.Parameter(torch.randn((existedN), requires_grad=True).to(self.device))
        self.weightArr.append(self.mynetWeight)
        num_classes = self.num_classes
        ##################
        self.fc = nn.Linear(768, 500,device = self.device)
        self.classifier = nn.Linear(500*2, num_classes,device = self.device)
        self.classifierArr.append(self.classifier)
        self.fcArr.append(self.fc)

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def to(self, device, **kwargs):
        """
        将模型迁移到指定的设备（如 GPU）。
        参数:
            device: 目标设备
        """
        self.device = device
        return super().to(device, **kwargs)

    def myprediction(self,x,k):
        """
        基于模型进行预测。

        参数:
            x: 输入数据
            k: 子网络索引

        返回:
            模型的输出
        """
        pass

    def forward(self, x: torch.Tensor, returnt='both') -> torch.Tensor:
        """
        执行前向传播。
        参数:
            x: 输入张量 (batch_size, *input_shape)
            returnt: 返回类型（'out', 'features', 'both', 'full'）

        返回:
            输出张量 (类别数)
        """
        # 提取预训练 ViT 特征
        with torch.no_grad():
            # print('x:', x.shape)
            if x.shape[-1] == 768:
                features = x
            else:
                processX = self.vitProcess(x)
                feature_1 = self.vitmodel_1.forward_features(processX)
                feature_2 = self.vitmodel_2.forward_features(processX)
                feature_3 = self.vitmodel_3.forward_features(processX)
                
                features = feature_1 + feature_2 + feature_3
                features = features[:,0,:] # 提取特征

        vit_features = features

        totalFeature = vit_features.to(self.device)

        if returnt == 'features':
            # 如果需要返回特征
            return features
        # print("s:", np.shape(self.classifierArr)[0])
        if np.shape(self.classifierArr)[0] == 1:
            fcfeatures = self.fc(totalFeature)
            out = self.classifier(fcfeatures)
            
        elif np.shape(self.classifierArr)[0] == 2:
            sumf = self.fcArr[0](totalFeature)
            fcfeatures = self.fc(totalFeature)
            fcfeatures = torch.cat((fcfeatures, sumf), 1)
            out = self.classifier(fcfeatures)
        else:
            softmax = F.softmax(self.mynetWeight)
            existedN = np.shape(self.classifierArr)[0]
            sumf = 0
            for c1 in range(existedN-1):
                with torch.no_grad():
                    f1 = self.fcArr[c1](totalFeature)
                    sumf = sumf + f1 * softmax[c1]

            fcfeatures = self.fc(totalFeature)
            fcfeatures = torch.cat((fcfeatures,sumf),1)
            out = self.classifier(fcfeatures)

        if returnt == 'out':
            # 否则返回模型的输出
            return out
        elif returnt == 'both':
            return (out, features)
        elif returnt == 'full':
            return out, [
                out_0 if not self.return_prerelu else out_0_t,
                out_1 if not self.return_prerelu else self.layer1[-1].prerelu,
                out_2 if not self.return_prerelu else self.layer2[-1].prerelu,
                out_3 if not self.return_prerelu else self.layer3[-1].prerelu,
                out_4 if not self.return_prerelu else self.layer4[-1].prerelu
            ]

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both', 'all'] but got {}".format(returnt))

    def GetFeatureByExpertIndex(self,x,index):
        with torch.no_grad():

            processX = self.vitProcess(x)
            features = self.vitmodel.forward_features(processX)
            features = features[:, 0, :]
            # features = self.vitmodel(processX,returnt='features')

            # print(np.shape(features))
            vit_features = features

            totalFeature = vit_features

            nextFeature = self.fcArr[index](totalFeature)
            nextFeature = F.relu(nextFeature)

        return nextFeature
    
def resnet18(nclasses: int, nf: int = 64) -> ResNet:
    """
    Instantiates a ResNet18 network.

    Args:
        nclasses: number of output classes
        nf: number of filters

    Returns:
        ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)



def myresnet18(nclasses: int, nf: int = 64):

    """
    Instantiates a ResNet18 network.

    Args:
        nclasses: number of output classes
        nf: number of filters

    Returns:
        ResNet network
    """

    return Ours(BasicBlock, [2, 2, 2, 2], nclasses, nf)
