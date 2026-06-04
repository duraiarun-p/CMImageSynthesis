#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:46:35 2022

@author: arun
"""

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

from CycleGAN_3DArchs_lib import CycleGAN_3D_Alpha, dataload3D_2_predict, perf_metrics, I2I_CB2CT, I2I_CT2CB, CycleGAN_3D_Beta,CycleGAN_3D_Gamma,CycleGAN_3D_Epsilon


tf.keras.backend.clear_session()
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
print('Script started at')
print(st_0)
#%% I2I synthesis
def CT_I2I(cGAN,CT,TestGenCT2CB,TestGenCB2CT):
    overlap=1
    border=15 # 10 is better
    CB_P=I2I_CT2CB(cGAN,CT,overlap,border,TestGenCT2CB)
    CT_P=I2I_CT2CB(cGAN,CB_P,overlap,border,TestGenCB2CT)
    return CT_P

def CB_I2I(cGAN,CB,TestGenCT2CB,TestGenCB2CT):
    overlap=1
    border=15 #10 is better
    CT_P=I2I_CT2CB(cGAN,CB,overlap,border,TestGenCB2CT)
    CB_P=I2I_CT2CB(cGAN,CT_P,overlap,border,TestGenCT2CB)    
    return CB_P

#%%

mypath='/home/arun/Documents/PyWSPrecision/datasets/printoutblks/db4/'
Datapath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'

weightoutputpath1='/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/'
weightoutputpath=os.path.join(weightoutputpath1,'Perform_I2I_3D_Output')
if not os.path.exists(weightoutputpath):
    os.makedirs(weightoutputpath)
    
weightoutputpath2=os.path.join(weightoutputpath,'3D_Output_All/run_3/')
if not os.path.exists(weightoutputpath2):
    os.makedirs(weightoutputpath2)    


epochs=500
save_epoch_frequency=50
batch_size=10
# batch_set_size=100
imgshape=(256,256,1)
newshape=(256,256)
batch_set_size=100
saveweightflag=True
breakflag=False

#%%
cGAN=CycleGAN_3D_Alpha(Datapath,weightoutputpath2,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)
TestGenCT2CB=cGAN.build_generator3D()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Alpha_GenCT2CBWeights-550.h5')
TestGenCB2CT=cGAN.build_generator3D()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Alpha_GenCB2CTWeights-550.h5')
CT,CBCT=dataload3D_2_predict(Datapath)
CBsiz=CBCT.shape
CTsiz=CT.shape
# CT=CT[:,:,-32:]
# CB=CBCT[:,:,-32:]
CT=CT[:,:,CTsiz[2]//2:CTsiz[2]//2+32]
CB=CBCT[:,:,CBsiz[2]//2:CBsiz[2]//2+32]
CT_P=CT_I2I(cGAN,CT,TestGenCT2CB,TestGenCB2CT)
CB_P=CB_I2I(cGAN,CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_alpha.mat",mdic)   

del cGAN,TestGenCT2CB,TestGenCB2CT,CT_P,CB_P

#%%
cGAN1=CycleGAN_3D_Beta(mypath,weightoutputpath2,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)
TestGenCT2CB=cGAN1.build_generator3D()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Beta_GenCT2CBWeights-1000.h5')
TestGenCB2CT=cGAN1.build_generator3D()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Beta_GenCB2CTWeights-1000.h5')
# CT,CBCT=dataload3D_2_predict(Datapath)
# CT=CT[:,:,-32:]
# CB=CBCT[:,:,-32:]
CT_P=CT_I2I(cGAN1,CT,TestGenCT2CB,TestGenCB2CT)
CB_P=CB_I2I(cGAN1,CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_beta.mat",mdic) 

del cGAN1,TestGenCT2CB,TestGenCB2CT,CT_P,CB_P

#%%
cGAN2=CycleGAN_3D_Gamma(mypath,weightoutputpath2,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)
TestGenCT2CB=cGAN2.build_generator3D()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Gamma_GenCT2CBWeights-1000.h5')
TestGenCB2CT=cGAN2.build_generator3D()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Gamma_GenCB2CTWeights-1000.h5')
# CT,CBCT=dataload3D_2_predict(Datapath)
# CT=CT[:,:,-32:]
# CB=CBCT[:,:,-32:]
CT_P=CT_I2I(cGAN2,CT,TestGenCT2CB,TestGenCB2CT)
CB_P=CB_I2I(cGAN2,CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_gamma.mat",mdic)
del cGAN2,TestGenCT2CB,TestGenCB2CT,CT_P,CB_P
#%%
cGAN3=CycleGAN_3D_Epsilon(mypath,weightoutputpath2,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag)
TestGenCT2CB=cGAN3.build_generator3D()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Epsilon_GenCT2CBWeights-1000.h5')
TestGenCB2CT=cGAN3.build_generator3D()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Epsilon_GenCB2CTWeights-1000.h5')
# CT,CBCT=dataload3D_2_predict(Datapath)
# CT=CT[:,:,-32:]
# CB=CBCT[:,:,-32:]
CT_P=CT_I2I(cGAN3,CT,TestGenCT2CB,TestGenCB2CT)
CB_P=CB_I2I(cGAN3,CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_epsilon.mat",mdic)
del cGAN3,TestGenCT2CB,TestGenCB2CT,CT_P,CB_P
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