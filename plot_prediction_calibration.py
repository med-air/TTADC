# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 22:09:47 2021

@author: wama
"""

# import numpy as np
# import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression as LR
import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import roc_auc_score as AUC

# from cal.log_loss import log_loss
# from cal.get_diagram_data import get_diagram_data
# from cal.load_data_adult import y, p


def platts_scaling (prediction, output):
    modified_target = np.zeros_like(prediction)
    # modified_target = modified_target - 1
    for i, target in enumerate(output):
        modified_target[i, 0:target+1] = 1
    lr1 = LR()												
    lr1.fit( prediction[:,1].reshape(-1, 1), modified_target[:,1] )		
    lr2 = LR()												
    lr2.fit( prediction[:,2].reshape(-1, 1), modified_target[:,2] )
    lr3 = LR()												
    lr3.fit( prediction[:,3].reshape(-1, 1), modified_target[:,3] )
    lr4 = LR()												
    lr4.fit( prediction[:,4].reshape(-1, 1), modified_target[:,4] )
    return lr1,lr2,lr3,lr4


def train_the_regression():
    path = './results/pro/'
    pro = np.load(path + 'val_608_fold0_pro.npy')
    gt = np.load(path + 'val_608_fold0_gt.npy')
    for i in range(1,5):
        pro_ = np.load(path + 'val_608_fold' + str(i) + '_pro.npy')
        gt_ = np.load(path + 'val_608_fold' + str(i) + '_gt.npy')
        pro = np.concatenate((pro, pro_), axis=0)
        gt = np.concatenate((gt, gt_), axis=0)
    lr1,lr2,lr3,lr4 = platts_scaling (pro, gt)
    return lr1,lr2,lr3,lr4

def prediction_calibration(prediction,lr1,lr2,lr3,lr4):
    
    p_calibrated1 = lr1.predict_proba(prediction[:,1].reshape(-1, 1))
    p_calibrated2 = lr2.predict_proba(prediction[:,2].reshape(-1, 1))
    p_calibrated3 = lr3.predict_proba(prediction[:,3].reshape(-1, 1))
    p_calibrated4 = lr4.predict_proba(prediction[:,4].reshape(-1, 1))
    p_calibrated0 = prediction[:,0]
    p_calibrated0 = p_calibrated0[:,np.newaxis]
    p_calibrated1 = p_calibrated1[:,1][:,np.newaxis]
    p_calibrated2 = p_calibrated2[:,1][:,np.newaxis]
    p_calibrated3 = p_calibrated3[:,1][:,np.newaxis]
    p_calibrated4 = p_calibrated4[:,1][:,np.newaxis]
    prediction_calibrated = np.concatenate((p_calibrated0, p_calibrated1,p_calibrated2,p_calibrated3,p_calibrated4), axis=1)
    return prediction_calibrated

path = './results/pro/'
pro = np.load(path + 'val_608_fold0_pro.npy')
gt = np.load(path + 'val_608_fold0_gt.npy')
for i in range(1,5):
    pro_ = np.load(path + 'val_608_fold' + str(i) + '_pro.npy')
    gt_ = np.load(path + 'val_608_fold' + str(i) + '_gt.npy')
    pro = np.concatenate((pro, pro_), axis=0)
    gt = np.concatenate((gt, gt_), axis=0)
lr1,lr2,lr3,lr4 = platts_scaling (pro, gt)

path = './results/pro/'
pro = np.load(path + 'test_608_fib_pro.npy')
gt = np.load(path + 'test_608_fib_gt.npy')