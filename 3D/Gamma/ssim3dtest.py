#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 22:00:57 2022

@author: arun
"""
import time
import datetime
# import cv2
# import itk
# import h5py
import numpy as np
# from os import listdir
# from os.path import isfile, join
# import matplotlib.pyplot as plt
import random
import os
import sys, getopt
# import multiprocessing

# import cycleganssimetriclib as ssTF

# os.environ['CUDA_VISIBLE_DEVICES'] = "-1" # comment this line when running in eddie
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
# import tensorflow.python.keras.engine
from tensorflow.keras.models import clone_model

import scipy.io
from tensorflow.keras.utils import Sequence

import cycleganssimetriclib as ssTF

tf.keras.backend.clear_session()

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()

#%%
def create_image_array_gen_CT(trainCT_image_names, trainCT_path):
    image_array = []
    for image_name in trainCT_image_names:
        mat_contents=scipy.io.loadmat(os.path.join(trainCT_path,image_name))
        CT_b=mat_contents['CT_b']
        CT_b=np.array(CT_b)
        # CT_b = ((CT_b-np.min(CT_b))/((np.max(CT_b)-np.min(CT_b))*0.5))-1#Normalisation needs proper
        CT_b=np.expand_dims(CT_b, axis=-1)
        image_array.append(CT_b)
    return np.array(image_array)

def create_image_array_gen_CB(trainCT_image_names, trainCT_path):
    image_array = []
    for image_name in trainCT_image_names:
        mat_contents=scipy.io.loadmat(os.path.join(trainCT_path,image_name))
        CT_b=mat_contents['CB_b']
        CT_b=np.array(CT_b)
        # CT_b = 2.*(CT_b-np.min(CT_b))/(np.max(CT_b)-np.min(CT_b))-1
        # CT_b = ((CT_b-np.min(CT_b))/((np.max(CT_b)-np.min(CT_b))*0.5))-1
        CT_b=np.expand_dims(CT_b, axis=-1)
        image_array.append(CT_b)
    return np.array(image_array)

class data_sequence(Sequence):
    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size,batch_set_size):
        # self.newshape=newshape
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        self.batch_set_size=batch_set_size
        image_list_A=random.sample(image_list_A,self.batch_set_size)
        image_list_B=random.sample(image_list_B,self.batch_set_size)
        for image_name in image_list_A:
            # if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_A.append(os.path.join(trainA_path, image_name))
        for image_name in image_list_B:
            # if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                self.train_B.append(os.path.join(trainB_path, image_name))
    
    def __len__(self):
        # no=1
        return int(min(len(self.train_A), len(self.train_B)) / float(self.batch_size))
        # return int(no)
    
    def __getitem__(self, idx):
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
        real_images_A = create_image_array_gen_CT(batch_A, '')
        real_images_B = create_image_array_gen_CB(batch_B, '')
        return real_images_A, real_images_B  # input_data, target_data
                

def loadprintoutgen(trainCT_path,trainCB_path,batch_size,batch_set_size):
    trainCT_image_names = os.listdir(trainCT_path)
    trainCB_image_names = os.listdir(trainCB_path)
    # return trainCT_image_names,trainCB_image_names
    return data_sequence(trainCT_path, trainCB_path, trainCT_image_names, trainCB_image_names,batch_size=batch_size,batch_set_size=batch_set_size)

        
#%%True
          # self.DiscCB.trainable = 

mypath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/'
outputpath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/output'
weightoutputpath='/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/12032022/beta/'

#%%
def randdomblockchoose(trainCT_path,trainCB_path):
    trainCT_image_names = os.listdir(trainCT_path)
    trainCB_image_names = os.listdir(trainCB_path)
    
    image_name_CT = random.choice(trainCT_image_names)
    image_name_CB = random.choice(trainCB_image_names)
    
    mat_contents=scipy.io.loadmat(os.path.join(trainCT_path,image_name_CT))
    CT_b=mat_contents['CT_b']
    CT_b=np.array(CT_b)
    # CT_b = ((CT_b-np.min(CT_b))/((np.max(CT_b)-np.min(CT_b))*0.5))-1#Normalisation needs proper
    CT_b=np.expand_dims(CT_b, axis=-1)
    
    mat_contents1=scipy.io.loadmat(os.path.join(trainCB_path,image_name_CB))
    CB_b=mat_contents1['CB_b']
    CB_b=np.array(CB_b)
    # CT_b = 2.*(CT_b-np.min(CT_b))/(np.max(CT_b)-np.min(CT_b))-1
    # CT_b = ((CT_b-np.min(CT_b))/((np.max(CT_b)-np.min(CT_b))*0.5))-1
    CB_b=np.expand_dims(CB_b, axis=-1)
    return CT_b, CB_b
#%%

trainCT_path = os.path.join(mypath, 'trainCT')
trainCB_path = os.path.join(mypath, 'trainCB')

data_ds_seq=loadprintoutgen(trainCT_path,trainCB_path,batch_size=5,batch_set_size=10)
dataCT,dataCB=next(iter(data_ds_seq))

dataCT1,dataCB1=randdomblockchoose(trainCT_path,trainCB_path)
#%% SSIM metrics 3D

def tfssim(img1,img2,max_val,filter_size,filter_sigma):
    score = tf.image.ssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return score
def tfmsssim(img1,img2,max_val,filter_size,filter_sigma):
    score = tf.image.ssim_multiscale(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)
    return score

#%% Custom loss function
def custom_loss_2_beta3D(y_true, y_pred):# SSIM
    max_val1=tf.math.reduce_max(y_true)-tf.math.reduce_min(y_true)
    max_val2=tf.math.reduce_max(y_pred)-tf.math.reduce_min(y_pred)
    max_val =0.5*(max_val1+max_val2)
    filter_size=11
    filter_sigma=0.5
    batchsize=K.int_shape(y_pred)
    # batchsize=tf.gather(batchsize1,0)
    ssimscores=[]
    for batchelei in range(batchsize[0]):
        y_pred1=y_pred[batchelei,:,:,:,:]
        y_true1=y_true[batchelei,:,:,:,:]
        depthsize=K.int_shape(y_pred1)
        for depthelei in range(depthsize[0]):
            y_pred2=y_pred1[depthelei,:,:,:]
            y_true2=y_true1[depthelei,:,:,:]
            max_val1=tf.math.reduce_max(y_true2)-tf.math.reduce_min(y_true2)
            max_val2=tf.math.reduce_max(y_pred2)-tf.math.reduce_min(y_pred2)
            max_val =0.5*(max_val1+max_val2)
            ssimscoreele,_=ssTF.tfssim_custom(y_true2, y_pred2, max_val,filter_size,filter_sigma)
            ssimscores.append(ssimscoreele)
    ssimscore=tf.reduce_mean(ssimscores)
    # ssimscore=ssTF.tfssim(y_true, y_pred, max_val,filter_size,filter_sigma)
    # loss=tf.math.subtract(1, ssimscore)# Will cause error when train_on_batch: run_eagerly=False
    loss=1-ssimscore
    return loss

#%%

# score=tfssim(dataCT1,dataCB1,0.5*(tf.math.reduce_max(dataCT1)+tf.math.reduce_max(dataCB1)),filter_size=11,filter_sigma=0.5)

score=custom_loss_2_beta3D(dataCT, dataCB)