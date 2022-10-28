import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers
#from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import os

class resnet_lstm4_fle_TADE_light(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm4_fle_TADE_light, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.num_experts = 6
        self.sequence_length = 11
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        # self.conv1 =  resnet.conv1
        # self.bn1 =  resnet.bn1
        # self.relu =  resnet.relu
        # self.maxpool =  resnet.maxpool
        # self.layer1 =  resnet.layer1
        # self.layer2 =  resnet.layer2
        # self.layer3 =  nn.ModuleList([resnet.layer3 for _ in range(self.num_experts)])
        self.layer4 =  nn.ModuleList([resnet.layer4 for _ in range(self.num_experts)])
        self.avgpool =  resnet.avgpool
        # self.share.add_module("layer3", resnet.layer3)
        # self.share.add_module("layer4", resnet.layer4)
        # self.share.add_module("avgpool", resnet.avgpool)
        self.lstm =  nn.ModuleList([nn.LSTM(2048, 512, batch_first=True) for _ in range(self.num_experts)])
        self.fc1 =  nn.ModuleList([nn.Linear(512, 256) for _ in range(self.num_experts)])
        self.fc2 =  nn.ModuleList([nn.Linear(256, 5) for _ in range(self.num_experts)])   


        # self.lstm = nn.LSTM(2048, 512, batch_first=True)
        # self.fc1 = nn.Linear(512, 256)
        # self.fc2 = nn.Linear(256, 5)
        self.dropout = nn.Dropout(p=0.5)

        for _ in range(self.num_experts):
            init.xavier_normal_(self.lstm[_].all_weights[0][0])
            init.xavier_normal_(self.lstm[_].all_weights[0][1])
            init.xavier_uniform_(self.fc1[_].weight)
            init.xavier_uniform_(self.fc2[_].weight)

    def separate_model(self, x, index, length):
        # x = (self.layer3[index])(x)
        x = (self.layer4[index])(x)
        x = self.avgpool(x)
        x = x.view(-1, length, 2048)
        self.lstm[index].flatten_parameters()
        y, _ = self.lstm[index](x)
        y = y.contiguous().view(-1, 512)
        y = self.dropout(y)
        y = self.fc1[index](y)
        y = self.dropout(y)
        y = F.relu(y)
        y = self.fc2[index](y)
        output = y.view(-1,length,5)
        output = torch.mean(output,dim=1)
        output = output.view(output.shape[0],5)
        return output
    def forward(self, x):
        length = x.shape[1]
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        outs = []
        for index in range(self.num_experts):
            outs.append(self.separate_model(x,index,length))
        final_out = torch.stack(outs, dim = 1).mean(dim=1)
        return {
            "output": final_out,
            "logits": torch.stack(outs, dim=1)
        }

class resnet_lstm4_fle_TADE(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm4_fle_TADE, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.num_experts = 3
        self.sequence_length = 11
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        # self.conv1 =  resnet.conv1
        # self.bn1 =  resnet.bn1
        # self.relu =  resnet.relu
        # self.maxpool =  resnet.maxpool
        # self.layer1 =  resnet.layer1
        # self.layer2 =  resnet.layer2
        self.layer3 =  nn.ModuleList([resnet.layer3 for _ in range(self.num_experts)])
        self.layer4 =  nn.ModuleList([resnet.layer4 for _ in range(self.num_experts)])
        self.avgpool =  resnet.avgpool
        # self.share.add_module("layer3", resnet.layer3)
        # self.share.add_module("layer4", resnet.layer4)
        # self.share.add_module("avgpool", resnet.avgpool)
        self.lstm =  nn.ModuleList([nn.LSTM(2048, 512, batch_first=True) for _ in range(self.num_experts)])
        self.fc1 =  nn.ModuleList([nn.Linear(512, 256) for _ in range(self.num_experts)])
        self.fc2 =  nn.ModuleList([nn.Linear(256, 5) for _ in range(self.num_experts)])   

        # self.lstm = nn.LSTM(2048, 512, batch_first=True)
        # self.fc1 = nn.Linear(512, 256)
        # self.fc2 = nn.Linear(256, 5)
        self.dropout = nn.Dropout(p=0.5)

        for _ in range(self.num_experts):
            init.xavier_normal_(self.lstm[_].all_weights[0][0])
            init.xavier_normal_(self.lstm[_].all_weights[0][1])
            init.xavier_uniform_(self.fc1[_].weight)
            init.xavier_uniform_(self.fc2[_].weight)

    def separate_model(self, x, index, length):
        x = (self.layer3[index])(x)
        x = (self.layer4[index])(x)
        x = self.avgpool(x)
        x = x.view(-1, length, 2048)
        self.lstm[index].flatten_parameters()
        y, _ = self.lstm[index](x)
        y = y.contiguous().view(-1, 512)
        y = self.dropout(y)
        y = self.fc1[index](y)
        y = self.dropout(y)
        y = F.relu(y)
        y = self.fc2[index](y)
        output = y.view(-1,length,5)
        output = torch.mean(output,dim=1)
        output = output.view(output.shape[0],5)
        return output
    def forward(self, x):
        length = x.shape[1]
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        outs = []
        for index in range(self.num_experts):
            outs.append(self.separate_model(x,index,length))
        final_out = torch.stack(outs, dim = 1).mean(dim=1)
        return {
            "output": final_out,
            "logits": torch.stack(outs, dim=1)
        }

