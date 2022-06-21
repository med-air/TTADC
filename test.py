# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:00:59 2022

@author: wama
"""

import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from dataset_synapse import Synapse_dataset, RandomGenerator
from test_func_TADE import inference
from  loss import focal_loss,prediction2label,ordinal_regression,ordinal_regression_focal
import torch.nn.functional as F
from tools import get_error_name
import time
from network import *


parser = argparse.ArgumentParser()
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--list_dir', type=str,
                    default='./lists/', help='list dir')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--max_epochs', type=int,
                    default=45, help='maximum epoch number to train')
parser.add_argument('--root_path', type=str,
                    default='../data/final/iCTCF_6_24', help='root dir for data')
parser.add_argument('--model_path', type=str,
                    default='../models_save/iCTCF_6_test_bs4_5e5_de20_40_/')
parser.add_argument('--model_step', type=str,
                    default= 42)
parser.add_argument('--max_step', type=str,
                    default= 30000)
parser.add_argument('--batch_size', type=str,
                    default= 20)
parser.add_argument('--base_lr', type=str,
                    default= 5e-5)
parser.add_argument('--sequence_length', type=str,
                    default= 11)
parser.add_argument('--train_txt', type=str,
                    default= 'iCTCF_train')
parser.add_argument('--val_txt', type=str,
                    default= 'iCTCF_test')
parser.add_argument('--num_classes', type=int,
                    default=6, help='output channel of network')
args = parser.parse_args()


model = resnet_lstm4_fle_TADE_light()

model_path = args.model_path  +'epoch_'+ str(args.model_step) + '.pth'
model.load_state_dict(torch.load(model_path))

error_name  = get_error_name(args.list_dir+args.val_txt+'.txt')

auc0,auc1,auc2,auc3,error_name,result_,Y_val_set = inference(args, model,error_name,args.model_step)