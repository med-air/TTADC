import SimpleITK as sitk
import os
import numpy as np
import cv2

def normalize(img_):
    img_[img_>=250] = 250
    img_[img_<=-200] = -200
        
    
    
    mean = np.mean(img_)
    std = np.std(img_)
    # mean = -0.29790374886599763
    # std = 0.29469745653088375
    img_ = (img_ - mean) / std
    max_ = np.max(img_)
    min_ = np.min(img_)
    img_ = (img_ - min_) / (max_ - min_ + 1e-9)
    img_ = img_ * 2 - 1
    return img_

train_file_path = '../data/raw/train_test_778/'

for dir, file, images in os.walk(train_file_path):
    if images != []:

        series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dir)
        if series_id == []:
            print(dir)
        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dir, series_id[0])
        print(len(series_file_names))
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        image3d = series_reader.Execute()
   
        Images = sitk.GetArrayFromImage(image3d)
        originImg = np.zeros((Images.shape[0],224,224)).astype(np.float32)
        #originGT = np.zeros((Images.shape[0],256,256)).astype(np.float32)
        Images = normalize(Images)
        for i in range(len(Images)):
            img = Images[i]
            # img = normalize(img)
            img = cv2.resize(img, (224, 224),interpolation=cv2.INTER_AREA)
            originImg[i] = img
        
        resultImage_ = sitk.GetImageFromArray(originImg)  
        sitk.WriteImage(resultImage_, dir+'/Img_raw.nii')
        # with open(train_file_path + '/ben_ct_train_list','a') as fp:
        #         fp.write(dir + '/Img.nii,'+ dir + '/Img.nii\n')
        #         fp.close()

            


