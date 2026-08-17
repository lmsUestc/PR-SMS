# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import numpy as np
import torch
from models.MMDCriterion import *

class DerppDualMemory(ContinualModel):
    NAME = 'derppdualmemory'
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

        myCount = int(self.args.buffer_size/2)
        #self.buffer = Buffer(myCount)
        self.buffer = Buffer(self.args.buffer_size)
        self.SlowBuffer = Buffer(myCount)
        self.mmdCriterion = MMD_loss()

    def end_task(self, dataset):
        print("aaaa")
        print(self.buffer.__len__())
        print("bbb")
        print(self.SlowBuffer.__len__())

    '''
    def end_task(self, dataset):

        t_idx = self.current_task

        n_seen_classes = dataset.N_CLASSES_PER_TASK * (t_idx + 1) if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
            sum(dataset.N_CLASSES_PER_TASK[:t_idx + 1])
        n_past_classes = dataset.N_CLASSES_PER_TASK * t_idx if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
            sum(dataset.N_CLASSES_PER_TASK[:t_idx])

        loader = dataset.train_loader
        norm_trans = dataset.get_normalization_transform()
        a_x, a_y, a_f, a_l = [], [], [], []
        for x, y, not_norm_x in loader:
            mask = (y >= n_past_classes) & (y < n_seen_classes)
            x, y, not_norm_x = x[mask], y[mask], not_norm_x[mask]
            if not x.size(0):
                continue
            a_x.append(not_norm_x.cpu())
            a_y.append(y.cpu())

        return 0
    '''

    def CalculateDistance(self,x1,x2):

        x1 = x1.view( np.shape(x1)[0] ,-1)
        x2 = x2.view( np.shape(x2)[0],-1)

        #torch_loss_fn = torch.nn.MSELoss(reduction='mean')
        #torch_loss = torch_loss_fn(x1,x2)
        torch_loss = self.mmdCriterion.rbf_mmd(x1,x2)

        return torch_loss

    def CalculateDistanceSingle(self,x1,x2):
        x1 = x1.view(-1)
        x2 = x2.view(-1)
        x1 = torch.reshape(x1,(1,-1))
        x2 = torch.reshape(x2,(1,-1))

        #torch_loss_fn = torch.nn.MSELoss(reduction='mean')
        #torch_loss = torch_loss_fn(x1,x2)
        torch_loss = self.mmdCriterion.rbf_mmd(x1,x2)

        return torch_loss

    def CalculateDistance_Centre_OtherCentre(self,rX,centreList,centreIndex):
        #n = centreList.size(0)#np.shape(centreList)[0]
        n = len(centreList)
        sum1 = 0
        for i in range(n):
            a1 = centreList[i]
            if a1 != centreIndex:
                d1 = self.CalculateDistanceSingle(rX[a1],rX[centreIndex])
                sum1 += d1
        sum1 = sum1 / (n - 1)
        return sum1

    def CalculateDistanceBetweenDistributions_Single(self,rX,set1,set2):
        data1 = rX[set1]
        data2 = rX[set2]
        #torch_loss_fn = torch.nn.MSELoss(reduction='mean')
        #torch_loss = torch_loss_fn(data1, data2)
        torch_loss = self.CalculateDistance(data1,data2)
        return torch_loss

    def CalculateDistanceBetweenDistributions(self,rX,set1,runCount):
        clusterSize = np.shape(set1)[0]

        sum1 = 0
        for j in range(runCount):
            choice = np.random.choice(np.shape(rX)[0], size=clusterSize, replace=False)
            d1 = self.CalculateDistanceBetweenDistributions_Single(rX,set1,choice)
            sum1 += d1

        sum1 = sum1 / runCount
        return sum1

    def CalculateDistance_Central(self,rX,centreList):
        centreCount = np.shape(centreList)[0]
        myarr = []
        for i in range(centreCount):
            sum1 = 0
            for j in range(centreCount):
                x1 = rX[centreList[i]]
                x2 = rX[centreList[j]]
                if i != j:
                    d1 = self.CalculateDistanceSingle(x1,x2)
                    sum1 += d1

            sum1 = sum1 / (centreCount-1)
            myarr.append(sum1)
        return myarr

    def CalculateCentre(self,rX,set1):
        #print(set1)
        n = len(set1) #set1.size(0)#np.shape(set1)[0]
        arr1 = []
        for i in range(n):
            a1 = set1[i]
            sum1 = 0
            for j in range(n):
                a2 = set1[j]
                if i != j:
                    #print(np.shape(rX[a1]))
                    #print(np.shape(rX[a2]))
                    t1 = self.CalculateDistanceSingle(rX[a1],rX[a2])
                    sum1 += t1
            arr1.append(sum1)

        arr1 = torch.Tensor(arr1)
        arr1 = arr1.cpu().numpy()
        minIndex = np.argmin(arr1)
        centerIndex = set1[minIndex]
        return centerIndex

    def CalculateDistanceBetween_Centre_Other(self,rX,set1,centreIndex,selectedSize):
        arr1 = []
        arr1DataIndex = []
        #n = np.shape(set1)[0]
        n = len(set1)#set1.size(0)
        myResults = []
        for i in range(n):
            x1Index = set1[i]
            if x1Index != centreIndex:
                d1 = self.CalculateDistanceSingle(rX[x1Index],rX[centreIndex])
                arr1.append(d1)
                arr1DataIndex.append(x1Index)
                myResults.append(x1Index)

        myindex = np.argsort(myResults,kind='mergesort')

        arr1DataIndex2 = []
        for j in range(np.shape(myindex)[0]):
            arr1DataIndex2.append(myResults[myindex[j]])

        arr1DataIndex2 = arr1DataIndex2[0:selectedSize]
        return arr1DataIndex2

    def MerageTwoSets(self,rX,set1,set2):
        setSize = np.shape(set1)[0]
        totalSet = []
        for t1 in range(setSize):
            totalSet.append(set1[t1])

        setSize2 = np.shape(set2)[0]
        for t2 in range(setSize2):
            totalSet.append(set2[t2])

        # 找出不同元素的位置
        unique_indices = np.where(np.diff(totalSet))
        unique_indices = np.array(unique_indices)
        unique_indices = np.reshape(unique_indices,(np.shape(unique_indices)[1]))
        unique_indices = unique_indices.astype(np.int64)

        #print(np.shape(unique_indices))
        #print("myadd")
        #print(unique_indices)
        #print("myoo")
        #print(np.shape(totalSet))

        newSet = []
        for j in range(np.shape(unique_indices)[0]):
            newSet.append(totalSet[unique_indices[j]])

        centreIndex = self.CalculateCentre(rX,newSet)
        selectedSet = self.CalculateDistanceBetween_Centre_Other(rX, newSet, centreIndex, setSize)
        return selectedSet,centreIndex

    def RandomSelectSet(self,rX,maxCount):
        maxSize = len(rX) #rX.size(0)
        maxSize = np.arange(maxSize)
        choice = np.random.choice(maxSize, size=maxCount, replace=False)
        return choice

    def MerageClusters(self,rX,arr1,arr1Center):
        myarr = []
        n = np.shape(arr1)[0]

        print("dada")
        print(np.shape(arr1Center))

        if n == 1:
            return arr1, arr1Center

        if n == 2:
            newSet, newCentre = self.MerageTwoSets(rX, arr1[0], arr1[1])
            resultArr1 = newSet
            resultArr1Center = newCentre
        else:
            for i in range(n):
                set1 = arr1[i]
                centre1 = arr1Center[i]
                d1 = self.CalculateDistanceBetweenDistributions(rX,set1,3)
                d2 = self.CalculateDistance_Centre_OtherCentre(rX,arr1Center,centre1)
                dd = d1 + d2
                myarr.append(dd)

            myarrIndex = np.argsort(myarr,kind='mergesort')
            t1index = myarrIndex[0]
            t2index = myarrIndex[1]

            print(t2index)
            set1 = arr1[t1index]
            set2 = arr1[t2index]

            resultArr1 = []
            resultArr1Center = []

            newSet,newCentre = self.MerageTwoSets(rX,set1,set2)
            resultCount = n - 2
            for j in range(n):
                if j != t1index and j != t2index:
                    resultArr1.append(arr1[j])
                    resultArr1Center.append(arr1Center[j])

            resultArr1.append(newSet)
            resultArr1Center.append(newCentre)
        return resultArr1,resultArr1Center

    def processData(self,dataX,dataY,dataLogit,selectedMaxSize):
        arr1 = []
        arr1Center = []
        clusterCount = 20

        for t1 in range(clusterCount):
            set1 = self.RandomSelectSet(dataX, selectedMaxSize)
            arr1.append(set1)

            centreIndex = self.CalculateCentre(dataX, set1)
            arr1Center.append(centreIndex)

            print("mytttt")
            print(np.shape(set1))

        # Perform the merage class
        n = np.shape(arr1)[0]
        while (n > 2):
            arr1, arr1Center = self.MerageClusters(dataX, arr1, arr1Center)
            n = np.shape(arr1)[0]

        arr1, arr1Center = self.MerageClusters(dataX, arr1, arr1Center)

        sizeOfBlock = np.shape(arr1)[0]
        if sizeOfBlock > selectedMaxSize:
            arr1 = arr1[0:selectedMaxSize]
            arr1Center = arr1Center[0:selectedMaxSize]

        print(np.shape(dataX))
        # Select the data samples into the memory buffer
        selectedX = []
        selectedY = []
        selectedLogit = []
        for c1 in range(np.shape(arr1)[0]):
            a1 = dataX[arr1[c1]]
            a2 = dataY[arr1[c1]]
            a3 = dataLogit[arr1[c1]]
            a1 = torch.reshape(a1, (1, 3, 32, 32))
            a2 = torch.reshape(a2, (1, np.shape(a2)[0]))
            a3 = torch.reshape(a3, (1, np.shape(a3)[0]))

            if np.shape(selectedX)[0] == 0:
                selectedX = a1
                selectedY = a2
                selectedLogit = a3
            else:
                selectedX = torch.cat((selectedX, a1), 0)
                selectedY = torch.cat((selectedY, a2), 0)
                selectedLogit = torch.cat((selectedLogit, a3), 0)

        print("bbbb")
        print(np.shape(selectedX))

        self.SlowBuffer.add_data_new(examples=selectedX,
                             labels=selectedY,
                             logits=selectedLogit)

        #Remove the shot term memory buffer
        n1 = np.shape(selectedX)[0]
        bufferSize = self.buffer.buffer_size - n1
        self.buffer.ResizeBufferSize(bufferSize)

        # self.SlowBuffer.add_data_new(selectedX,selectedY,selectedLogit)
        print("end task")
        # print(np.shape(selectedX))

    def end_task2(self, dataset,train_loader):

        train_iter = iter(train_loader)

        maxSize = self.args.buffer_size
        longTermSize = int(maxSize/2)
        maxNofTasks = dataset.N_TASKS

        selectedMaxSize = int(longTermSize / maxNofTasks)

        dataX, dataY,dataLogit = [],[],[]
        while True:
            try:
                data = next(train_iter)
            except StopIteration:
                isFirst = False
                break

            inputs, labels, not_aug_inputs = data
            inputs = inputs.to(self.device)
            #inputs, labels = inputs.to(self.device), labels.to(self.device, dtype=torch.long)
            #not_aug_inputs = not_aug_inputs.to(self.device)
            #print(inputs.size())

            with torch.no_grad():
                outputs = self.net(inputs)

            inputs = inputs.to(torch.device("cpu"))
            not_aug_inputs = not_aug_inputs.to(torch.device("cpu"))
            outputs = outputs.to(torch.device("cpu"))

            if np.shape(dataX)[0] == 0:
                dataX = not_aug_inputs
                dataY = labels
                dataLogit = outputs
            else:
                #print(dataX.size())
                dataX = torch.cat((dataX,not_aug_inputs),0)
                dataY = torch.cat((dataY,labels),0)
                dataLogit = torch.cat((dataLogit,outputs),0)

        dataX = torch.reshape(dataX,(-1,3,32,32))
        dataY = torch.reshape(dataY,(np.shape(dataY)[0],-1))
        dataLogit = torch.reshape(dataLogit,(np.shape(dataLogit)[0],-1))

        arr1 = []
        arr1Center = []
        clusterCount = 20

        for t1 in range(clusterCount):
            set1 = self.RandomSelectSet(dataX,selectedMaxSize)
            arr1.append(set1)

            centreIndex = self.CalculateCentre(dataX,set1)
            arr1Center.append(centreIndex)

            print("mytttt")
            print(np.shape(set1))

        #Perform the merage class
        n = np.shape(arr1)[0]
        while(n > 2):
            arr1, arr1Center = self.MerageClusters(dataX, arr1, arr1Center)
            n = np.shape(arr1)[0]

        arr1, arr1Center = self.MerageClusters(dataX, arr1, arr1Center)

        sizeOfBlock = np.shape(arr1)[0]
        if sizeOfBlock > selectedMaxSize:
            arr1 = arr1[0:selectedMaxSize]
            arr1Center = arr1Center[0:selectedMaxSize]

        print(np.shape(dataX))
        #Select the data samples into the memory buffer
        selectedX = []
        selectedY = []
        selectedLogit = []
        for c1 in range(np.shape(arr1)[0]):
            a1 = dataX[arr1[c1]]
            a2 = dataY[arr1[c1]]
            a3 = dataLogit[arr1[c1]]
            a1 = torch.reshape(a1,(1,3,32,32))
            a2 = torch.reshape(a2,(1,np.shape(a2)[0]))
            a3 = torch.reshape(a3,(1,np.shape(a3)[0]))

            if np.shape(selectedX)[0] == 0:
                selectedX = a1
                selectedY = a2
                selectedLogit = a3
            else:
                selectedX = torch.cat((selectedX,a1),0)
                selectedY = torch.cat((selectedY,a2),0)
                selectedLogit = torch.cat((selectedLogit,a3),0)

        print("bbbb")
        print(np.shape(selectedX))

        #self.SlowBuffer.add_data_new(examples=selectedX,
        #                     labels=selectedY,
        #                     logits=selectedLogit)

        #self.SlowBuffer.add_data_new(selectedX,selectedY,selectedLogit)
        print("end task")
        #print(np.shape(selectedX))

    def getDataFromBuffers(self,buffer1,buffer2,minibatch_size, transform, device):

        '''
        buf_inputs, _, buf_logits = buffer2.get_data(minibatch_size, transform=transform,
                                                           device=device)

        return buf_inputs,_,buf_logits
        '''

        if buffer1.is_empty() == True:
            buf_inputs, _, buf_logits = buffer2.get_data(minibatch_size, transform=transform,
                                                           device=device)

        else:
            size = int(minibatch_size/2)
            buf_inputs, _, buf_logits = buffer1.get_data(size, transform=transform,
                                                             device=device)
            buf_inputs2, _, buf_logits2 = buffer2.get_data(size, transform=transform,
                                                             device=device)
            buf_inputs = torch.cat((buf_inputs,buf_inputs2),0)
            buf_logits = torch.cat((buf_logits,buf_logits2),0)
        return buf_inputs,_,buf_logits

    def getDataFromBuffers2(self,buffer1,buffer2,minibatch_size, transform, device):

        '''
        buf_inputs, buf_labels, _ = buffer2.get_data(minibatch_size, transform=transform,
                                                     device=device)

        return buf_inputs, buf_labels, _
        '''

        if buffer1.is_empty() == True:
            buf_inputs, buf_labels, _ = buffer2.get_data(minibatch_size, transform=transform,
                                                           device=device)
        else:
            size = int(minibatch_size/2)
            buf_inputs, buf_labels, _ = buffer1.get_data(size, transform=transform,
                                                             device=device)
            buf_inputs2, buf_labels2, _ = buffer2.get_data(size, transform=transform,
                                                             device=device)

            buf_labels2 = torch.reshape(buf_labels2,(-1,1))
            #print(np.shape(buf_labels))
            #print(np.shape(buf_labels2))
            buf_inputs = torch.cat((buf_inputs,buf_inputs2),0)
            buf_labels = torch.cat((buf_labels,buf_labels2),0)
        return buf_inputs,buf_labels,_


    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss = loss.item()

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.getDataFromBuffers(self.SlowBuffer,self.buffer,self.args.minibatch_size, transform=self.transform, device=self.device)#self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()
            buf_inputs, buf_labels, _ = self.getDataFromBuffers2(self.SlowBuffer,self.buffer,self.args.minibatch_size, transform=self.transform, device=self.device)

            '''
            print(np.shape(buf_outputs))
            print(np.shape(buf_labels))
            print("tttt")
            print(buf_labels)
            '''

            buf_labels = buf_labels.reshape(-1)

            buf_outputs = self.net(buf_inputs)
            loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss_ce.backward()
            tot_loss += loss_ce.item()

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return tot_loss
