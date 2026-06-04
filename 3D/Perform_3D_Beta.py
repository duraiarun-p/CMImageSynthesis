#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:49:58 2022

@author: s1785969
"""

import time
import datetime
# import cv2
# import itk
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import random
import os
import sys, getopt
# import multiprocessing




# os.environ['CUDA_VISIBLE_DEVICES'] = "-1" # comment this line when running in eddie
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
# import tensorflow.python.keras.engine
from tensorflow.keras.layers import ZeroPadding3D
from tensorflow.keras.models import clone_model

import scipy.io
from scipy import ndimage as nd
from scipy.io import savemat

from tensorflow.keras.utils import Sequence

from CycleGAN_3DArchs_lib import CycleGAN_3D_Alpha, CycleGAN_3D_Beta, CycleGAN_3D_Gamma, CycleGAN_3D_Epsilon, dataload3D_2_predict


tf.keras.backend.clear_session()

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
print('Script started at')
print(st_0)
     
#%%

mypath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/'
Datapath='/home/s1785969/RDS/MATLAB/ImageDB/PrintoutDB/DB33/'

weightoutputpath1='/home/s1785969/RDS/PyWS/Pyoutputs/cycleganweights/CMImageSynthesis_Outputs/'
weightoutputpath=os.path.join(weightoutputpath1,'Perform_I2I_3D_Output')
if not os.path.exists(weightoutputpath):
    os.makedirs(weightoutputpath)


epochs=500
save_epoch_frequency=50
batch_size=5
# batch_set_size=100
imgshape=(256,256,1)
newshape=(256,256)
batch_set_size=100
saveweightflag=True
breakflag=False


cGAN=CycleGAN_3D_Alpha(mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)
# cGAN=CycleGAN_3D_Beta(mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)
# cGAN=CycleGAN_3D_Gamma(mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)
# cGAN=CycleGAN_3D_Epsilon(mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)


#%%

TestGenCT2CB=cGAN.build_generator3D()
TestGenCT2CB.load_weights('/home/s1785969/RDS/PyWS/CMImageSynthesis/3D/Perf_Comp_3D/Alpha_GenCT2CBWeights-550.h5')
# TestGenCT2CB.load_weights('/home/s1785969/RDS/PyWS/CMImageSynthesis/3D/Perf_Comp_3D/Beta_GenCT2CBWeights-1000.h5')
# # batch_CB_P=TestGenCT2CB.predict(batch_CT)

TestGenCB2CT=cGAN.build_generator3D()
TestGenCB2CT.load_weights('/home/s1785969/RDS/PyWS/CMImageSynthesis/3D/Perf_Comp_3D/Alpha_GenCB2CTWeights-550.h5')
# TestGenCB2CT.load_weights('/home/s1785969/RDS/PyWS/CMImageSynthesis/3D/Perf_Comp_3D/Beta_GenCB2CTWeights-1000.h5')
# batch_CT_P=TestGenCB2CT.predict(batch_CB)

CT,CBCT=dataload3D_2_predict(Datapath)
CT=CT[:,:,-32:]
# CBCT=CBCT[:,:,-88:]


#%%
#%%
overlap=1
border=5 # 10 is better

#%%
CTsiz1=CT.shape 
# CT_P=np.zeros_like(CT,dtype=float)
# CT1=np.zeros((CTsiz1[0],CTsiz1[1],CTsiz1[2]+(2*border)),dtype=float)
# for sli in range(border):CT1[:,:,sli]=CT[:,:,0]
# for sli in range(-border,0):CT1[:,:,sli]=CT[:,:,-1]

# CT1[:,:,border:-border]=CT

# CT=CT1

CT_P=np.zeros_like(CT,dtype=float)
#%%

CT_blks=[]
CT_blks_pred=[]

CTblkindex=[]
CTblkindex1=[]
# for zi in range(0,CTsiz1[2],(cGAN.depth_size//overlap)-(2*border)):
for zi in range(0,CTsiz1[2],(cGAN.depth_size//overlap)):    
    for j in  range(0,CTsiz1[0],(cGAN.patch_size//overlap)-(2*border)):
        for i in range(0,CTsiz1[1],(cGAN.patch_size//overlap)-(2*border)):          

            ele=[i,j,zi]
            
            CTblkindex.append(ele)
            
            currentBlk=CT[i:i+cGAN.patch_size,j:j+cGAN.patch_size,zi:zi+cGAN.depth_size]
            blksiz=currentBlk.shape

            if blksiz[1] != cGAN.patch_size or blksiz[0] != cGAN.patch_size:
                if blksiz[2] != cGAN.depth_size:
                    diff_zi=blksiz[2]-cGAN.depth_size
                    zi=zi+diff_zi
                    currentBlk=CT[i:i+cGAN.patch_size,j:j+cGAN.patch_size,zi:zi+cGAN.depth_size]
                
                if blksiz[1] != cGAN.patch_size:
                    diff_zj=blksiz[1]-cGAN.patch_size
                    j=j+diff_zj
                    currentBlk=CT[i:i+cGAN.patch_size,j:j+cGAN.patch_size,zi:zi+cGAN.depth_size]
                
                if blksiz[0] != cGAN.patch_size:
                    diff_i=blksiz[0]-cGAN.patch_size
                    i=i+diff_i
                    currentBlk=CT[i:i+cGAN.patch_size,j:j+cGAN.patch_size,zi:zi+cGAN.depth_size]
                
                dsfactor1=(cGAN.patch_size/blksiz[0],cGAN.patch_size/blksiz[1],cGAN.depth_size/blksiz[2])
                
                currentBlk_i=np.expand_dims(currentBlk, axis=-1)
                currentBlk_i=np.expand_dims(currentBlk_i, axis=0)
                currentBlk_t=tf.convert_to_tensor(currentBlk_i, dtype=tf.float32)
                # currentBlk_p=tf.where(currentBlk_t > threshold, 1, 0)
                # currentBlk_p=tf.where(currentBlk_t > tf.reduce_mean(currentBlk_t), 1, 0)
                currentBlk_p = TestGenCT2CB.predict(currentBlk_t)
                
                # currentBlk_p = currentBlk_t.numpy()/2
                currentBlk_p = np.squeeze(currentBlk_p,axis=-1)
                currentBlk_p = np.squeeze(currentBlk_p,axis=0)
                dsfactor=(blksiz[0]/cGAN.patch_size,blksiz[1]/cGAN.patch_size,blksiz[2]/cGAN.depth_size)
                
                # currentBlk_p = nd.interpolation.zoom(currentBlk_p, zoom=dsfactor)
                
                currentBlk_p = currentBlk_p[border:-border,border:-border,:]
                CT_P[i+border:i+cGAN.patch_size-border,j+border:j+cGAN.patch_size-border,zi:zi+cGAN.depth_size]=currentBlk_p
                
                # currentBlk_p = currentBlk_p[border:-border,border:-border,border:-border]
                # CT_P[i+border:i+cGAN.patch_size-border,j+border:j+cGAN.patch_size-border,zi+border:zi+cGAN.depth_size-border]=currentBlk_p
                
                CT_blks.append(currentBlk)
                CT_blks_pred.append(currentBlk_p)
                # ele1=[i+cGAN.patch_size,j+cGAN.patch_size,zi+cGAN.depth_size]
                
                ele1=[i+cGAN.patch_size-1,j+cGAN.patch_size,zi+cGAN.depth_size]
                # ele1=[i+cGAN.patch_size-overlap,j+cGAN.patch_size-overlap,zi+cGAN.depth_size-overlap]
                CTblkindex1.append(ele1)
                
            else:
                if blksiz[2] != cGAN.depth_size:
                    diff_zi=blksiz[2]-cGAN.depth_size
                    zi=zi+diff_zi
                    currentBlk=CT[i:i+cGAN.patch_size,j:j+cGAN.patch_size,zi:zi+cGAN.depth_size]
                currentBlk_i=np.expand_dims(currentBlk, axis=-1)
                currentBlk_i=np.expand_dims(currentBlk_i, axis=0)
                currentBlk_t=tf.convert_to_tensor(currentBlk_i, dtype=tf.float32)
                # currentBlk_p=tf.where(currentBlk_t > threshold, 1, 0)
                # currentBlk_p=tf.where(currentBlk_t > tf.reduce_mean(currentBlk_t), 1, 0)
                # currentBlk_p = currentBlk_t.numpy()/2
                currentBlk_p = TestGenCT2CB.predict(currentBlk_t)
                currentBlk_p = np.squeeze(currentBlk_p,axis=-1)
                currentBlk_p = np.squeeze(currentBlk_p,axis=0)
                # currentBlk_p = currentBlk_p[3:-3,3:-3,:]
                
                currentBlk_p = currentBlk_p[border:-border,border:-border,:]
                CT_P[i+border:i+cGAN.patch_size-border,j+border:j+cGAN.patch_size-border,zi:zi+cGAN.depth_size]=currentBlk_p
                
                # currentBlk_p = currentBlk_p[border:-border,border:-border,border:-border]
                # CT_P[i+border:i+cGAN.patch_size-border,j+border:j+cGAN.patch_size-border,zi+border:zi+cGAN.depth_size-border]=currentBlk_p
                                
                ele1=[i+cGAN.patch_size-1,j+cGAN.patch_size-1,zi+cGAN.depth_size-1]
                # ele1=[i+cGAN.patch_size-overlap,j+cGAN.patch_size-overlap,zi+cGAN.depth_size-overlap]
                CTblkindex1.append(ele1)
                
                CT_blks.append(currentBlk)
                CT_blks_pred.append(currentBlk_p)
                
#%%

# CT1=CT
# CT_P1=CT_P 

# CT=CT[:,:,border:-border-2]
# CT_P=CT_P[:,:,border:-border-2]

#%%
# CTsiz1=CT.shape 
# CTpixel=[]
# CTpixel1=[]
# for zi in range(CTsiz1[2]):
#     for j in  range(cGAN.patch_size//overlap,CTsiz1[0]-cGAN.patch_size//overlap,cGAN.patch_size//overlap):
#         for i in range(cGAN.patch_size//overlap,CTsiz1[1]-cGAN.patch_size//overlap,cGAN.patch_size//overlap):
#             prior=CT_P[i-1,j-1,zi]
#             posteri=CT_P[i+1,j+1,zi]
#             CTpixel.append(CT_P[i,j,zi])
#             CT_P[i,j,zi]=(prior+posteri)*0.5
#             CTpixel1.append(CT_P[i,j,zi])

# from scipy.io import savemat
# mdic = {"CT_P1":CT_P1,"CT1":CT1,"CT_P":CT_P,"CT":CT}
mdic = {"CT_P":CT_P,"CT":CT}
# savemat("Pred_volumes.mat",mdic)      
            
            
#%%
# slice_index=5
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(CT[:,:,0],cmap='gray')
plt.show()
plt.show()
plt.title('CT')
plt.subplot(1,2,2)
plt.imshow(CT_P[:,:,0],cmap='gray')
plt.show()
plt.show()
plt.title('pseudo CB')
#%%
CTsiz1=CT.shape
slice_index=np.random.choice(CTsiz1[2])
# slice_index=-2
plt.figure(2)
plt.subplot(1,2,1)
plt.imshow(CT[:,:,slice_index],cmap='gray')
plt.show()
plt.show()
plt.title('CT')
plt.subplot(1,2,2)
plt.imshow(CT_P[:,:,slice_index],cmap='gray')
plt.show()
plt.show()
plt.title('pseudo CB')
#%%
plt.figure(3)
plt.subplot(1,2,1)
plt.imshow(CT[:,:,-1],cmap='gray')
plt.show()
plt.show()
plt.title('CT')
plt.subplot(1,2,2)
plt.imshow(CT_P[:,:,-1],cmap='gray')
plt.show()
plt.show()
plt.title('pseudo CB')
#%%  
# plt.figure(4)
# plt.subplot(1,2,1)
# plt.imshow(CT_blks[100][:,:,15],cmap='gray')
# plt.show()
# plt.show()
# plt.title('CT')
# plt.subplot(1,2,2)
# plt.imshow(CT_blks_pred[100][:,:,15],cmap='gray')
# plt.show()
# plt.show()
# plt.title('pseudo CB')
#%%
CT_line1=CT_P[255,:,0]
CT_line2=CT_P[:,255,0]
plt.figure(5)
plt.subplot(2,1,1)
plt.plot(CT_line1)
plt.subplot(2,1,2)
plt.plot(CT_line2)
#%%  
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
# runtimeN0=(time.time()-start_time_0)
print('Script Total Time = %s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)
tf.keras.backend.clear_session()
