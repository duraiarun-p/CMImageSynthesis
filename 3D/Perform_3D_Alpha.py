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

from CycleGAN_3DArchs_lib import CycleGAN_3D_Alpha, dataload3D_2_predict, perf_metrics, I2I_CB2CT, I2I_CT2CB


tf.keras.backend.clear_session()

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
print('Script started at')
print(st_0)
#%% I2I synthesis

#%%

mypath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/'
Datapath='/home/s1785969/RDS/MATLAB/ImageDB/PrintoutDB/DB33/'

weightoutputpath1='/home/s1785969/RDS/PyWS/Pyoutputs/cycleganweights/CMImageSynthesis_Outputs/'
weightoutputpath=os.path.join(weightoutputpath1,'Perform_I2I_3D_Output')
if not os.path.exists(weightoutputpath):
    os.makedirs(weightoutputpath)
    
weightoutputpath2=os.path.join(weightoutputpath,'Alpha_3D_Output_All')
if not os.path.exists(weightoutputpath2):
    os.makedirs(weightoutputpath2)    


epochs=500
save_epoch_frequency=50
batch_size=5
# batch_set_size=100
imgshape=(256,256,1)
newshape=(256,256)
batch_set_size=100
saveweightflag=True
breakflag=False


cGAN=CycleGAN_3D_Alpha(Datapath,weightoutputpath2,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)
# cGAN=CycleGAN_3D_Beta(mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)
# cGAN=CycleGAN_3D_Gamma(mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)
# cGAN=CycleGAN_3D_Epsilon(mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)


#%%

TestGenCT2CB=cGAN.build_generator3D()
TestGenCT2CB.load_weights('/home/s1785969/RDS/PyWS/CMImageSynthesis/3D/Perf_Comp_3D/Alpha_GenCT2CBWeights-550.h5')

TestGenCB2CT=cGAN.build_generator3D()
TestGenCB2CT.load_weights('/home/s1785969/RDS/PyWS/CMImageSynthesis/3D/Perf_Comp_3D/Alpha_GenCB2CTWeights-550.h5')

CT,CBCT=dataload3D_2_predict(Datapath)
CT=CT[:,:,-32:]
# CBCT=CBCT[:,:,-88:]

#%%
overlap=1
border=10 # 10 is better

#%%
CB_P=I2I_CT2CB(cGAN,CT,overlap,border,TestGenCT2CB)
CT_P=I2I_CT2CB(cGAN,CB_P,overlap,border,TestGenCB2CT)
# CT_P=I2I(cGAN,CT,overlap,border)

#%%
CT1=tf.expand_dims(CT,0)
CT_P1=tf.expand_dims(CT_P,0)

SSIM_score,MAE_score,MSE_score,SNU_score,PSNR_score,NCC_score=perf_metrics(CT1,CT_P1)
Perf_score_CT_ele=[SSIM_score.numpy(),MAE_score.numpy(),MSE_score.numpy(),SNU_score.numpy(),PSNR_score.numpy(),NCC_score.numpy()]
                
#%%
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
