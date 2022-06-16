import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from tqdm import tqdm
from sklearn import metrics
from loss import prediction2label
import torch.nn.functional as F
from functools import reduce
from prediction_calibration import *
from tools import *
from eval import *
def tran_prediction(pred,thr):
    pred = pred > thr
    pred_ = pred.astype(int)
    return pred_

def tran_class(pred_sample):
    sample_class = []
    for k in range(len(pred_sample)):
        list_ = pred_sample[:k+1]
        ln = reduce(lambda x,y:x*y,list_)
        sample_class.append(ln*np.sum(list_))
    return np.max(sample_class) 

def Acc_AUC2(predict, gt,class_num):
    predict = np.array(predict)
    predict_ = predict[:,class_num]
    gt = np.array(gt)
    gt_ = np.zeros(len(gt)).astype(np.float32)
    gt_[gt<=class_num] = 0
    gt_[gt>class_num] = 1
    auc, auc_cov = delong_roc_variance(gt_,predict_)
    
    print('{} AUC:'.format(class_num),auc, np.sqrt(auc_cov), auc-1.96*np.sqrt(auc_cov), auc+1.96*np.sqrt(auc_cov))
    pred_half = tran_prediction(predict, np.array(0.5))
    predicted_class = []
    for sample_index in range(len(predict)):
        sample_class = tran_class(pred_half[sample_index]) 
        if sample_class <= class_num:
            predicted_class.append(0)
        else:
            predicted_class.append(1)
    confusion = metrics.confusion_matrix(gt_, predicted_class,labels=[0,1,2,3,4])
    specificity = 0 
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    accuracy = 0
    if np.sum(confusion) != 0:
        accuracy = (float(confusion[1,1]) + float(confusion[0,0])) / np.sum(confusion)
    #print(metrics.classification_report(gt_,pred))
    #print(metrics.confusion_matrix(gt_,pred))
    output_tran = np.float32(output)
    output_tran = tran_prediction(output_tran, np.array(0.5))
    output_tran = tran_class(output_tran) 
    specificity_std = np.sqrt(specificity*(1-specificity)/float(confusion[0,0]+confusion[0,1]))
    sensitivity_std = np.sqrt(sensitivity*(1-sensitivity)/float(confusion[1,1]+confusion[1,0]))
    accuracy_std = np.sqrt(accuracy*(1-accuracy)/np.sum(confusion))
    print('Accuracy:',accuracy, int(float(confusion[0,0])+float(confusion[1,1])), '/', int(np.sum(confusion)), accuracy-1.96*accuracy_std, accuracy+1.96*accuracy_std)
    print('Specificity:',specificity, int(float(confusion[0,0])), '/', int(float(confusion[0,0]+confusion[0,1])), specificity-1.96*specificity_std, specificity+1.96*specificity_std)
    print('Sensitivity:',sensitivity, int(float(confusion[1,1])), '/', int(float(confusion[1,1]+confusion[1,0])),  sensitivity-1.96*sensitivity_std, sensitivity+1.96*sensitivity_std)
    print("============================================")
    return auc


path = '../models_save/train_test_time_630/result_32.npy'
data  = np.load(path)
data = data.item()
result_ = data['result']
Y_val_set = data['label']
#print(metrics.classification_report(Y_val_set,result))
#print(metrics.confusion_matrix(Y_val_set,result,labels=[0,1,2,3,4]))

auc0 = Acc_AUC2(result_, Y_val_set,0)
auc1 = Acc_AUC2(result_, Y_val_set,1)
auc2 = Acc_AUC2(result_, Y_val_set,2)
auc3 = Acc_AUC2(result_, Y_val_set,3)


def inference(args, model, error_name, test_save_path=None):
    db_test = Synapse_dataset(base_dir='../data/final/train_test_608', list_dir=args.list_dir, split=args.val_txt,is_train = False)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    result = []
    result_ = []
    Y_val_set = []
    iteration_error_name = []
    prediction_epoch = []
    y_train =  []
    # lr1,lr2,lr3,lr4 = train_the_regression()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image_batch, label_batch, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name']
        image_batch = image_batch.cuda()
        output = model(image_batch)
        output = torch.mean(output,dim=0)
        output = output.view(1,4)
        output = torch.sigmoid(output)
        output = output.data.cpu().numpy()
        prediction_epoch.append(output)
        # output = prediction_calibration(output,lr1,lr2,lr3,lr4)
        output_tran = np.float32(output)
        output_tran = tran_prediction(output_tran, np.array(0.5))
        output_tran = tran_class(output_tran) 
        # output_tran = prediction2label(output)
        # _, preds_phase = torch.max(output.data, 1)
        
        pred_ = np.float32(output)
        label_batch = np.float32(label_batch.data.cpu().numpy())
        y_train.append(label_batch[0])
        # pred_result = np.float32(output_tran.data.cpu().numpy())
        if output_tran != label_batch[0]:
            error_name[case_name[0]][output_tran] += 1
            error_name[case_name[0]][int(label_batch[0])] = -1
            iteration_error_name.append(case_name)


        result_.append(pred_[0])
        result.append(output_tran)
        Y_val_set.append(label_batch[0])
    prediction_epoch = np.array(prediction_epoch)
    # np.save('./results/pro/'+args.val_txt+'_pro.npy',prediction_epoch)
    # np.save('./results/pro/'+args.val_txt+'_gt.npy',y_train)     
    print(metrics.classification_report(Y_val_set,result))
    print(metrics.confusion_matrix(Y_val_set,result,labels=[0,1,2,3,4]))
    # print(iteration_error_name)
        #print(Y_val_set)
        #print(result_)
    auc0 = Acc_AUC2(result_, Y_val_set,0)
    auc1 = Acc_AUC2(result_, Y_val_set,1)
    auc2 = Acc_AUC2(result_, Y_val_set,2)
    auc3 = Acc_AUC2(result_, Y_val_set,3)
    return auc0,auc1,auc2,auc3,error_name,np.array(result_),np.array(Y_val_set)
        
        
        
       
    # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    # metric_list = metric_list / len(db_test)
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    # return "Testing Finished!"