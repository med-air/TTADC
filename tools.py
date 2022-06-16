import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def  get_error_name (list_path):
    sample_list = open(list_path).readlines()
    test_error_count = {}
    for idx in range(len(sample_list)):
        test_error_count[sample_list[idx].strip('\n')] = [0,0,0,0,0,0]
    return test_error_count
