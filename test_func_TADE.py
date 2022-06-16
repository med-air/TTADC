import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator,RandomGenerator_test
from tqdm import tqdm
from sklearn import metrics
from loss import prediction2label
import torch.nn.functional as F
from functools import reduce
from prediction_calibration import *
from tools import *
from eval import *
import time
def tran_prediction(pred,thr):
    pred = pred > thr
    pred_ = pred.astype(int)
    return pred_

def tran_prediction_(pred,thr):
    pred = pred > 0.5
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
    confusion = metrics.confusion_matrix(gt_, predicted_class,labels=[0,1,2])
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
    specificity_std = np.sqrt(specificity*(1-specificity)/float(confusion[0,0]+confusion[0,1]))
    sensitivity_std = np.sqrt(sensitivity*(1-sensitivity)/float(confusion[1,1]+confusion[1,0]))
    accuracy_std = np.sqrt(accuracy*(1-accuracy)/np.sum(confusion))
    print('Accuracy:',accuracy, int(float(confusion[0,0])+float(confusion[1,1])), '/', int(np.sum(confusion)), accuracy_std, accuracy-1.96*accuracy_std, accuracy+1.96*accuracy_std)
    print('Specificity:',specificity, int(float(confusion[0,0])), '/', int(float(confusion[0,0]+confusion[0,1])), specificity-1.96*specificity_std, specificity+1.96*specificity_std)
    print('Sensitivity:',sensitivity, int(float(confusion[1,1])), '/', int(float(confusion[1,1]+confusion[1,0])),  sensitivity-1.96*sensitivity_std, sensitivity+1.96*sensitivity_std)
    print("============================================")
    return auc


def Acc_AUC(predict, gt,class_num):
    gt = np.array(gt)
    pred = np.zeros(len(predict)).astype(np.float32)
    gt_ = np.zeros(len(gt)).astype(np.float32)
    gt_[gt<=class_num] = 0
    gt_[gt>class_num] = 1
    for j in range(len(predict)):
        pred[j] = predict[j][class_num + 1]
    fpr, tpr, thresholds = metrics.roc_curve(gt_,pred)
    print('{} AUC:'.format(class_num),metrics.auc(fpr,tpr))

    pred = np.array(pred)
    pred = pred>0.5
    pred=[int(i) for i in pred>0.5]
    pred=np.array(pred)


    confusion = metrics.confusion_matrix(gt_,pred)
    specificity = 0 
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    #print(metrics.classification_report(gt_,pred))
    #print(metrics.confusion_matrix(gt_,pred))
    print('Specificity:',specificity)
    print('Sensitivity:',sensitivity)
    print("============================================")
    return metrics.auc(fpr,tpr)

def inference(args, model, error_name,epoch_num, test_save_path=None):
    db_test_training = Synapse_dataset(base_dir='../data/Synapse/iCTCF_test', list_dir=args.list_dir, split=args.val_txt,is_train = False,transform=transforms.Compose(
                                   [RandomGenerator_test(output_size=[args.img_size, args.img_size])]), test_time=True)
    db_test = Synapse_dataset(base_dir='../data/Synapse/iCTCF_test', list_dir=args.list_dir, split=args.val_txt,is_train = False)
    testloader_training = DataLoader(db_test_training, batch_size=1, shuffle=False, num_workers=2)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    
    metric_list = 0.0
    result = []
    result_ = []
    Y_val_set = []
    iteration_error_name = []
    prediction_epoch = []
    y_train =  []
    
    aggregation_weight = torch.nn.Parameter(torch.FloatTensor(6),requires_grad=True)
    aggregation_weight.data.fill_(1/6)
    optimizer_ = torch.optim.SGD([aggregation_weight], lr = 0.2,momentum = 0.9,weight_decay=5e-4,nesterov=True)
    cos = torch.nn.CosineSimilarity(dim=1,eps=1e-6)
    loss = 0
    
    device = torch.device("cuda:1")
    # torch.distributed.init_process_group(backend = "nccl")
    # torch.cuda.set_device('cude:{}'.format(args.device_ids[0]))
    model = torch.nn.DataParallel(model) 
    model = model.to(device)
    model = model.cuda()
    
    optimizer_.zero_grad()
    # aggregation_weight = torch.tensor([0.0,10.0,0.0])
    #aggregation_softmax = torch.nn.functional.softmax(aggregation_weight)
    if epoch_num>=30:
        print("Test Time Training")
        for num_test in range(5):
            for i, sampled_test_batch in tqdm(enumerate(testloader_training)):
                image_batch0,  image_batch1= sampled_test_batch['image0'], sampled_test_batch['image1']
                image_batch0 = image_batch0.cuda()
                image_batch1 = image_batch1.cuda()
                output0 = model(image_batch0)
                output1 = model(image_batch1)
                model1_output0 = torch.sigmoid(output0['logits'][:,0,:])
                model2_output0 =torch.sigmoid(output0['logits'][:,1,:])
                model3_output0 =torch.sigmoid(output0['logits'][:,2,:])
                model4_output0 =torch.sigmoid(output0['logits'][:,3,:])
                model5_output0 =torch.sigmoid(output0['logits'][:,4,:])
                model6_output0 =torch.sigmoid(output0['logits'][:,5,:])
                model1_output1 =torch.sigmoid(output1['logits'][:,0,:])
                model2_output1 =torch.sigmoid(output1['logits'][:,1,:])
                model3_output1 =torch.sigmoid(output1['logits'][:,2,:])
                model4_output1 =torch.sigmoid(output1['logits'][:,3,:])
                model5_output1 =torch.sigmoid(output1['logits'][:,4,:])
                model6_output1 =torch.sigmoid(output1['logits'][:,5,:])
                aggregation_softmax = torch.nn.functional.softmax(aggregation_weight)
                aggregation_output0 = aggregation_softmax[0].cuda()*model1_output0+aggregation_softmax[1].cuda()*model2_output0+aggregation_softmax[2].cuda()*model3_output0+aggregation_softmax[3].cuda()*model4_output0+aggregation_softmax[4].cuda()*model5_output0+aggregation_softmax[5].cuda()*model6_output0
                aggregation_output1 = aggregation_softmax[0].cuda()*model1_output1+aggregation_softmax[1].cuda()*model2_output1+aggregation_softmax[2].cuda()*model3_output1+aggregation_softmax[3].cuda()*model4_output1+aggregation_softmax[4].cuda()*model5_output1+aggregation_softmax[5].cuda()*model6_output1
                cos_similarity = cos(aggregation_output0,aggregation_output1).mean()
                loss = (1-cos_similarity)/4
                loss.backward()
                if i%4== 0:
                    # print(loss)
                    optimizer_.step()
                    optimizer_.zero_grad()
    # lr1,lr2,lr3,lr4 = train_the_regression()
                if aggregation_softmax[0]<0.05 or  aggregation_softmax[1]<0.05 or aggregation_softmax[2]<0.05 or aggregation_softmax[3]<0.05 or aggregation_softmax[4]<0.05 or aggregation_softmax[5]<0.05:
                    break
            print("Model weights:", aggregation_softmax)
    torch.cuda.empty_cache()
    time.sleep(10)
    model.eval()
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        # h, w = sampled_batch["image"].size()[2:]
        image_batch, label_batch, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name']
        image_batch = image_batch.cuda()
        output = model(image_batch)
        # output = torch.mean(output,dim=0)
        # output = output.view(1,5)
        model1_output = torch.sigmoid(output['logits'][:,0,:])
        model2_output =torch.sigmoid(output['logits'][:,1,:])
        model3_output =torch.sigmoid(output['logits'][:,2,:])
        model4_output =torch.sigmoid(output['logits'][:,3,:])
        model5_output =torch.sigmoid(output['logits'][:,4,:])
        model6_output =torch.sigmoid(output['logits'][:,5,:])
        aggregation_softmax = torch.nn.functional.softmax(aggregation_weight)

        aggregation_output = aggregation_softmax[0].cuda()*model1_output+aggregation_softmax[1].cuda()*model2_output+aggregation_softmax[2].cuda()*model3_output+aggregation_softmax[3].cuda()*model4_output+aggregation_softmax[4].cuda()*model5_output+aggregation_softmax[5].cuda()*model6_output
        #aggregation_output = torch.sigmoid(output['logits'][:,0,:])
        output = aggregation_output.data.cpu().numpy()
        prediction_epoch.append(output)
        # output = prediction_calibration(output,lr1,lr2,lr3,lr4)
        output_tran = np.float32(output)
        output_tran = tran_prediction_(output_tran, np.array(0.5))
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
    print(metrics.confusion_matrix(Y_val_set,result))
    # print(iteration_error_name)
        #print(Y_val_set)
        #print(result_)
    auc0 = Acc_AUC2(result_, Y_val_set,0)
    auc1 = Acc_AUC2(result_, Y_val_set,1)
    auc2 = Acc_AUC2(result_, Y_val_set,2)
    auc3 = Acc_AUC2(result_, Y_val_set,3)
    auc4 = Acc_AUC2(result_, Y_val_set,4)
    
    print("mean AUC:", (auc0+auc1+auc2+auc3+auc4)/5)
    return auc0,auc1,auc2,auc3,error_name,np.array(result_),np.array(Y_val_set)
        
        