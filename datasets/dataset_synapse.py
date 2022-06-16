import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from transforms3d import *


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         selection = np.random.randint(0,4)
#         rotato3d = RandomRotation3D(15)
#         shift3d = RandomShift3D((5,5,5))
#         flip3d = RandomFlip3D((1,2))
#         Gaussian3d = GaussianDenoising((0.9,1.1))
#         if selection == 0:
#             sample = np.expand_dims(sample, 0)    
#         elif selection == 1:
#             sample = rotato3d(sample)
#         elif selection == 2:
#             sample = Gaussian3d(sample)
#         elif selection == 3:
#             sample = flip3d(sample)
#         else:
#             sample = shift3d(sample)
        

#         return sample

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if label == 0:
            selection = np.random.randint(0,2)
        elif label == 1:
            selection = np.random.randint(0,5)
        elif label == 2:
            selection = np.random.randint(0,5)
        elif label == 3:
            selection = np.random.randint(0,5)
        else: 
            selection = 0
        rotato3d = RandomRotation3D((15,30))
        shift3d = RandomShift3D((5,5,5))
        flip3d = RandomFlip3D(axes = (1,2),p=1.0)
        Gaussian3d = GaussianDenoising()
        if selection == 0:
            image = np.expand_dims(image, 0)           
        elif selection == 1:
            image = rotato3d(image)
        elif selection == 2:
            image = Gaussian3d(image)
        elif selection == 3:
            image = flip3d(image)
        else:
            image = shift3d(image)    
        sample = {'image': image, 'label': label}

        return sample

class RandomGenerator_test(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        selection = np.random.randint(0,5)
        rotato3d = RandomRotation3D((15,30))
        shift3d = RandomShift3D((5,5,5))
        flip3d = RandomFlip3D(axes = (1,2),p=1.0)
        Gaussian3d = GaussianDenoising()
        if selection == 0:
            image = np.expand_dims(image, 0)           
        elif selection == 1:
            image = rotato3d(image)
        elif selection == 2:
            image = Gaussian3d(image)
        elif selection == 3:
            image = flip3d(image)
        else:
            image = shift3d(image)    
        sample = {'image': image, 'label': label}

        return sample

# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
       
#         rotato3d = RandomRotation3D((0,30))
#         shift3d = RandomShift3D((5,5,5))
#         flip3d = RandomFlip3D(axes = (1,2),p=1.0)
#         Gaussian3d = GaussianDenoising((0.9,1.1))

#         if label == 0:
#             image = rotato3d(image)
#         elif label == 1:
#             selection = np.random.randint(0,5)
#         elif label == 2:
#             selection = np.random.randint(0,5)
#         elif label == 3:
#             selection = np.random.randint(0,5)
#         else: 
#             selection = 0

#         if selection == 0:
#             image = np.expand_dims(image, 0)           
#         elif selection == 1:
#             image = rotato3d(image)
#         elif selection == 2:
#             image = Gaussian3d(image)
#         elif selection == 3:
#             image = flip3d(image)
#         else:
#             image = shift3d(image)
#         sample = {'image': image, 'label': label}

#         return sample

class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, is_train = True, transform=None,test_time=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir
        self.is_train = is_train
        self.test_time = test_time

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.is_train:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + "/{}_fle.npy.h5".format(slice_name)
            data = h5py.File(data_path)
            image, label = data['image'][()], data['label'][()]
            # image = np.expand_dims(image,axis=0)
            
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}_fle.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][()], data['label'][()]
            
            
        
        #label = np.random.randint(0,5)
        
        if self.transform and self.test_time==False:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
        elif self.transform and self.test_time==True:
            sample = {'image': image, 'label': label}
            sample0 = self.transform(sample)
            sample1 = self.transform(sample)
            image0 = sample0['image']
            image0 = np.transpose(image0,[1,0,2,3])
            #image = image.repeat(3,1)
            image0_1 = image0[:image0.shape[0]-2]
            image0_2 = image0[1:image0.shape[0]-1]
            image0_3 = image0[2:image0.shape[0]]
            image0_ = np.concatenate((image0_1, image0_2, image0_3), axis = 1)

            image1 = sample1['image']
            image1 = np.transpose(image1,[1,0,2,3])
            #image = image.repeat(3,1)
            image1_1 = image1[:image1.shape[0]-2]
            image1_2 = image1[1:image1.shape[0]-1]
            image1_3 = image1[2:image1.shape[0]]
            image1_ = np.concatenate((image1_1, image1_2, image1_3), axis = 1)
            sample_pair = {'image0': image0_,'image1': image1_}
            return sample_pair
        else:
            image = np.expand_dims(image,axis=0)
            sample = {'image': image, 'label': label}
    
        image, label = sample['image'], sample['label']
        image = np.transpose(image,[1,0,2,3])
        #image = image.repeat(3,1)
        image1 = image[:image.shape[0]-2]
        image2 = image[1:image.shape[0]-1]
        image3 = image[2:image.shape[0]]
        image_ = np.concatenate((image1, image2, image3), axis = 1)
        sample = {'image': image_, 'label': label}
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample



# class Synapse_dataset(Dataset):
#     def __init__(self, base_dir, list_dir, split, transform=None):
#         self.transform = transform  # using transform in torch!
#         self.split = split
#         self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
#         self.data_dir = base_dir

#     def __len__(self):
#         return len(self.sample_list)

#     def __getitem__(self, idx):
#         if self.split == "train":
#             slice_name = self.sample_list[idx].strip('\n')
#             data_path = os.path.join(self.data_dir, slice_name+'.npz')
#             data = np.load(data_path)
#             image, label = data['image'], data['label']
#         else:
#             vol_name = self.sample_list[idx].strip('\n')
#             filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
#             data = h5py.File(filepath)
#             image, label = data['image'][:], data['label'][:]

#         sample = {'image': image, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
#         sample['case_name'] = self.sample_list[idx].strip('\n')
#         return sample
