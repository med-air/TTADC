# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 20:47:29 2022

@author: wama
"""

import numpy as np
import SimpleITK as sitk
import os
import pandas
import cv2
from scipy import ndimage
import copy
import h5py

img_path = './processed_data/resampled_data/'
gt_path = './processed_data/lungAreasFinal_/'
label_path = './Patients.xlsx'

img_path_ = os.listdir(img_path)
gt_path_ = os.listdir(gt_path)

#wb = xlrd.open_workbook(label_path)
df = pandas.read_excel(label_path, engine='openpyxl')
#label = wb.sheet_by_name
label_list = []
def get_gt(patient_ID):
    index = 0
    label = 3
    for l in range(len(df.Patient)):
        if df.Patient[l].split(' ')[-1] == patient_ID:
            index = l
            break
    # if df.Morbidity[index] == 'Control' or df.Morbidity[index] == 'Suspected':
    #     label = 0
    # elif df.Morbidity[index] == 'Mild' or df.Morbidity[index] == 'Regular':
    #     label = 1
    # elif df.Morbidity[index] == 'Severe' or df.Morbidity[index] == 'Critically ill':
    #     label = 2
    if df.Morbidity[index] == 'Control':
        label = 0
    elif df.Morbidity[index] == 'Suspected':
        label = 1
    elif df.Morbidity[index] == 'Mild':
        label = 2
    elif df.Morbidity[index] == 'Regular':
        label = 3
    elif df.Morbidity[index] == 'Severe':
        label = 4
    elif df.Morbidity[index] == 'Critically ill':
        label = 5
    return label

def norm(img):
    

    mean = np.mean(img)
    std = np.std(img)
    img = (img-mean)/std
    
    max_ = np.max(img)
    min_ = np.min(img)
    img = (img-min_)/(max_-min_)
    return img*2 - 1


def norm2(img,gt):
    img = ndimage.interpolation.zoom(img,np.divide([24, 224, 224], img.shape),mode='nearest')
    gt = ndimage.interpolation.zoom(gt,np.divide([24, 224, 224], gt.shape),mode='nearest')
    return(img,gt)
    

def catImg2(img):
    D,H,W = np.where(img==1)
    if len(D)!=0:
        minD = np.min(D)
        maxD = np.max(D)
        num =  0
        index = 0
        for i in range(len(img)):
            single = img[i,:,:]
            a,b = np.where(single==2)
            num_ = len(a)
            if num_ >= num:
                num = num_
                index = i
    else:
        index = 12
        minD = 0
        maxD = 0

    return minD,maxD,index  

for num in range(len(img_path_)):
    patient_ID = img_path_[num].split('.')[0].split('-')[-1]
    label = get_gt(patient_ID)
    label_list.append(label)
    img_file_path = img_path + img_path_[num]
    gt_file_path = gt_path + gt_path_[num]
    
    Liver = sitk.ReadImage(gt_file_path)
    LiverImg = sitk.GetArrayFromImage(Liver)
    LiverImg = LiverImg.astype(np.uint8)
    
    Raw = sitk.ReadImage(img_file_path)
    RawImg = sitk.GetArrayFromImage(Raw)
    
    RawImg, LiverImg = norm2(RawImg, LiverImg)
    RawImg = norm(RawImg)



    RawImg_ = copy.deepcopy(LiverImg)
    # if dir.split('/')[-1]== 'R06921301 2min':
    #     print(1)
    minD, maxD, max_index = catImg2(RawImg_)
    #RawImg_[RawImg_ != 2] = 0
    #RawImg_[RawImg_ == 2] = 1
    #RawImg_ = Resize(RawImg_)
    mat = copy.deepcopy(RawImg_)
    mat[mat == 0] = 1      
    
    RawImg[RawImg_==0] = RawImg[0,0,0]
    # RawImg = Resize(RawImg)
    #RawImg = RawImg * mat

    print(patient_ID)
    
    #length = maxD - minD + 1
    # length = 13
    # length = RawImg.shape[0]
    if RawImg.shape[0] >=24:
        length = 24
        if (max_index+length//2)<= len(RawImg) and (max_index - length//2-1)>=0:
            index = max_index - length//2-1
        elif (max_index+length//2)>len(RawImg):
            index = len(RawImg) - length
        else:
            index = 0
    else:
        length = RawImg.shape[0]
        index = 0

    result = np.zeros((length,RawImg.shape[1],RawImg.shape[2])).astype(np.float32)
    gt = np.zeros((length,RawImg.shape[1],RawImg.shape[2])).astype(np.float32)

    
    ini = 0 
    # RawImg = norm1(RawImg)
    for j in range(index,index+length):
        #im = RawImg[j]
        
        result[ini] = RawImg[j]
        gt[ini] = RawImg_[j]
        ini += 1

  
    resultImage = sitk.GetImageFromArray(result)     

    sitk.WriteImage(resultImage,'./iCTCF_preprocessed_6_24/'+img_path_[num])

    
    f = h5py.File('./iCTCF_preprocessed_6_24/'  + img_path_[num].split('.')[0]+'_fle.npy.h5','w') 
    f['image'] = result                
    f['label'] = label
    # f['pathogeny'] = int(dir.split('/')[-2][-1])      
    f.close()  
    # with open('./lists/' + file_name + '.txt','a') as fp:
    #             fp.write(saved_name+'\n')
    #             fp.close