# -*- coding: utf-8 -*-
# @Author  : wama
from torch import nn
import torch
from torch.nn import functional as F

def ordinal_regression_focal(predictions,targets):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""
    predictions = predictions['logits']
    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions[:,0,:])
    k_rate = 0.5
    x_num = 969/(5*k_rate+1)
    norm_rate = torch.tensor([5/6, 4/6, 3/6, 2/6, 1/6]).cuda()
    rate = torch.tensor([0.748, 0.619, 0.606, 0.169, 0.017]).cuda()
    rate_1 = torch.tensor([(969-x_num)/969, (969-x_num-k_rate*x_num)/969, (969-x_num-2*k_rate*x_num)/969, (969-x_num-3*k_rate*x_num)/969,(969-x_num-4*k_rate*x_num)/969]).cuda()
    rate_2 = torch.tensor([(969-k_rate*x_num)/969, (969-x_num-k_rate*x_num)/969, (969-x_num-2*k_rate*x_num)/969, (969-x_num-3*k_rate*x_num)/969,(969-x_num-4*k_rate*x_num)/969]).cuda()
    rate_3 = torch.tensor([(969-k_rate*x_num)/969, (969-2*k_rate*x_num)/969, (969-x_num-2*k_rate*x_num)/969, (969-x_num-3*k_rate*x_num)/969,(969-x_num-4*k_rate*x_num)/969]).cuda()
    rate_4 = torch.tensor([(969-k_rate*x_num)/969, (969-2*k_rate*x_num)/969, (969-3*k_rate*x_num)/969, (969-x_num-3*k_rate*x_num)/969,(969-x_num-4*k_rate*x_num)/969]).cuda()
    rate_5 = torch.tensor([(969-k_rate*x_num)/969, (969-2*k_rate*x_num)/969, (969-3*k_rate*x_num)/969, (969-4*k_rate*x_num)/969,(969-x_num-4*k_rate*x_num)/969]).cuda()
    rate_6 = torch.tensor([(969-k_rate*x_num)/969, (969-2*k_rate*x_num)/969, (969-3*k_rate*x_num)/969, (969-4*k_rate*x_num)/969,(969-5*k_rate*x_num)/969]).cuda()

    recalibration = ((1-rate+1e-9)*norm_rate)/(rate*(1-norm_rate+1e-9))
    #recalibration_reverse = ((1-rate_+1e-9)*norm_rate)/(rate_*(1-norm_rate+1e-9))
    recalibration_reverse_1 = ((1-rate_1+1e-9)*norm_rate)/(rate_1*(1-norm_rate+1e-9))
    recalibration_reverse_2 = ((1-rate_2+1e-9)*norm_rate)/(rate_2*(1-norm_rate+1e-9))
    recalibration_reverse_3 = ((1-rate_3+1e-9)*norm_rate)/(rate_3*(1-norm_rate+1e-9))
    recalibration_reverse_4 = ((1-rate_4+1e-9)*norm_rate)/(rate_4*(1-norm_rate+1e-9))
    recalibration_reverse_5 = ((1-rate_5+1e-9)*norm_rate)/(rate_5*(1-norm_rate+1e-9))
    recalibration_reverse_6 = ((1-rate_6+1e-9)*norm_rate)/(rate_6*(1-norm_rate+1e-9))
   
    recalibration = recalibration.repeat(predictions.shape[0],1)
    recalibration_reverse_1 = recalibration_reverse_1.repeat(predictions.shape[0],1)
    recalibration_reverse_2 = recalibration_reverse_2.repeat(predictions.shape[0],1)
    recalibration_reverse_3 = recalibration_reverse_3.repeat(predictions.shape[0],1)
    recalibration_reverse_4 = recalibration_reverse_4.repeat(predictions.shape[0],1)
    recalibration_reverse_5 = recalibration_reverse_5.repeat(predictions.shape[0],1)
    recalibration_reverse_6 = recalibration_reverse_6.repeat(predictions.shape[0],1)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        modified_target[i, 0:target] = 1
    loss_sum = 0
    
    for class_index in range(modified_target.shape[1]): 
        #loss_sum  += nn.BCELoss(reduction='none')(torch.sigmoid(predictions[:,0,:][:,class_index]),modified_target[:,class_index])
        #loss_sum  += nn.BCELoss(reduction='none')(torch.sigmoid(predictions[:,1,:][:,class_index]-recalibration[:,class_index].log()),modified_target[:,class_index])
        loss_sum  += nn.BCELoss(reduction='none')(torch.sigmoid(predictions[:,0,:][:,class_index]-recalibration[:,class_index].log()+recalibration_reverse_1[:,class_index].log()),modified_target[:,class_index])
        loss_sum  += nn.BCELoss(reduction='none')(torch.sigmoid(predictions[:,1,:][:,class_index]-recalibration[:,class_index].log()+recalibration_reverse_2[:,class_index].log()),modified_target[:,class_index])
        loss_sum  += nn.BCELoss(reduction='none')(torch.sigmoid(predictions[:,2,:][:,class_index]-recalibration[:,class_index].log()+recalibration_reverse_3[:,class_index].log()),modified_target[:,class_index])
        loss_sum  += nn.BCELoss(reduction='none')(torch.sigmoid(predictions[:,3,:][:,class_index]-recalibration[:,class_index].log()+recalibration_reverse_4[:,class_index].log()),modified_target[:,class_index])
        loss_sum  += nn.BCELoss(reduction='none')(torch.sigmoid(predictions[:,4,:][:,class_index]-recalibration[:,class_index].log()+recalibration_reverse_5[:,class_index].log()),modified_target[:,class_index])
        loss_sum  += nn.BCELoss(reduction='none')(torch.sigmoid(predictions[:,5,:][:,class_index]-recalibration[:,class_index].log()+recalibration_reverse_6[:,class_index].log()),modified_target[:,class_index])
        #loss_sum  += nn.BCELoss(reduction='none')(torch.sigmoid(predictions[:,0,:][:,class_index]),modified_target[:,class_index])
        #weight[:,class_index] = modified_target[:,class_index]*(1-predictions[:,class_index])**2+(1-modified_target[:,class_index])*predictions[:,class_index]**2
    #weights_max,_=torch.max(weight,dim=1)
    return torch.mean(loss_sum)



def ordinal_regression(predictions,targets):
    """Ordinal regression with encoding as in https://arxiv.org/pdf/0704.1028.pdf"""

    # Create out modified target with [batch_size, num_labels] shape
    modified_target = torch.zeros_like(predictions)

    # Fill in ordinal target function, i.e. 0 -> [1,0,0,...]
    for i, target in enumerate(targets):
        modified_target[i, 0:target+1] = 1

    return nn.MSELoss(reduction='mean')(predictions, modified_target)

def prediction2label(pred):
    """Convert ordinal predictions to class labels, e.g.
    
    [0.9, 0.1, 0.1, 0.1] -> 0
    [0.9, 0.9, 0.1, 0.1] -> 1
    [0.9, 0.9, 0.9, 0.1] -> 2
    etc.
    """
    
    pred_tran = (pred > 0.5).cumprod(axis=1)
    max_index = -1
    for j in range(pred_tran.shape[1]):
        if pred_tran[0][j] == 1:
            # max_index = j
            max_index +=  1
    
    return max_index

