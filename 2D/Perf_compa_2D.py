#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:51:40 2022

@author: arun
"""

import time
import datetime
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import numpy as np

from matplotlib import pyplot as plt
from scipy.io import savemat

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)

from CycleGAN_Archs_lib import CycleGAN_Alpha, dataload, dataload_direct, perf_metrics

#%%


st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()


# mypath='/home/arun/Documents/PyWSPrecision/datasets/printoutslices'
datapath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
mypath='/home/arun/Documents/PyWSPrecision/datasets/printout2d_data'
# data same as printout2d folder-slices were not normalised but normalised during pre-processing training and prediction
# weightoutputpath1='/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/CMImageSynthesis_Outputs/Alpha_Output'
weightoutputpath1=os.getcwd()
weightoutputpath=os.path.join(weightoutputpath1, 'alpha_predicted_volume')
if not os.path.isdir(weightoutputpath):
    os.mkdir(weightoutputpath)
    
cGAN=CycleGAN_Alpha(mypath,weightoutputpath,epochs=40,save_epoch_frequency=2,batch_size=3,imgshape=(256,256,1),newshape=(256,256),batch_set_size=10,saveweightflag=True)

saved_weigth_path='/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp'
GenCB2CTweight='Alpha_GenCB2CTWeights-100.h5'
GenCT2CBweight='Alpha_GenCT2CBWeights-100.h5'

TestGenCB2CT_path=os.path.join(saved_weigth_path,GenCB2CTweight)
TestGenCB2CT=cGAN.build_generator()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights(TestGenCB2CT_path)

TestGenCT2CB_path=os.path.join(saved_weigth_path,GenCT2CBweight)
TestGenCT2CB=cGAN.build_generator()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights(TestGenCT2CB_path)

#%%

onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
onlyfiles.sort()
onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
onlyfiles = onlyfiles[0:-onlyfileslenrem]
matfiles=[join(datapath,f) for f in onlyfiles]

Patlen=len(onlyfiles)
Perf_score_CBCT=[]
Perf_score_CT=[]
Patlen=1
for pati in range(Patlen):
    
    CT,CBCTs=dataload_direct(matfiles[pati])
        
    saveweightoutputpath1=os.getcwd()
    saveweightoutputpath=os.path.join(saveweightoutputpath1, onlyfiles[pati])
    if not os.path.isdir(saveweightoutputpath):
        os.mkdir(saveweightoutputpath)
    
    CTsiz=CT.shape
    for zi in range(CTsiz[2]):
            image=CT[:,:,zi]
            image = (image-np.min(image))/(np.max(image)-np.min(image))
            CT[:,:,zi]=image
    batch_CT=tf.image.resize(CT,[256,256])
    batch_CT = tf.transpose(batch_CT,perm=[2,0,1])
    batch_CT = tf.expand_dims(batch_CT, -1)
    # CT modality
    batch_CB_P=TestGenCT2CB.predict(batch_CT)
    batch_CT_P=TestGenCB2CT.predict(batch_CB_P)
    SSIM_score,MAE_score,MSE_score,SNU_score,PSNR_score,NCC_score=perf_metrics(batch_CT,batch_CT_P)
    Perf_score_CT_ele=[SSIM_score.numpy(),MAE_score.numpy(),MSE_score.numpy(),SNU_score.numpy(),PSNR_score.numpy(),NCC_score.numpy()]
    Perf_score_CT.append(Perf_score_CT_ele)
    
    batch_CT=tf.image.resize(batch_CT,[512,512])
    # batch_CB=tf.image.resize(batch_CB,[512,512])
    batch_CT_P=tf.image.resize(batch_CT_P,[512,512])
    batch_CB_P=tf.image.resize(batch_CB_P,[512,512])
    batch_CB_P=np.squeeze(batch_CB_P,axis=-1)
    batch_CT=np.squeeze(batch_CT,axis=-1)
    batch_CT_P=np.squeeze(batch_CT_P,axis=-1)
    # batch_CB=np.squeeze(batch_CB,axis=-1)
    mdic = {"batch_CB_P":batch_CB_P,"batch_CT":batch_CT,"batch_CT_P":batch_CT_P}
    ct_vol_matfile1='CT-'+str(cbcti)+'.mat'
    ct_vol_matfile=os.path.join(saveweightoutputpath,ct_vol_matfile1)
    savemat(ct_vol_matfile,mdic)
    
    
    CBCTLen=len(CBCTs)
    
    
    for cbcti in range(CBCTLen):
        CBCT=CBCTs[cbcti]
        # Slice-wise data normalisation from N-net training script
        CBsiz=CBCT.shape  
        # print('Before Normalise std: %s'%(np.std(CBCT)))
        for zi in range(CBsiz[2]):
            image1=CBCT[:,:,zi]
            image1 = (image1-np.min(image1))/(np.max(image1)-np.min(image1))
            CBCT[:,:,zi]=image1
        # Data pre-processing i.e array to tensor conversion with dimensional expansion for Tensorflow
        # print('After Normalise std: %s'%(np.std(CBCT)))
        batch_CB = tf.image.resize(CBCT,[256,256])
        batch_CB = tf.transpose(batch_CB,perm=[2,0,1])
        batch_CB = tf.expand_dims(batch_CB, -1)
        # CBCT modality
        batch_CT_P=TestGenCB2CT.predict(batch_CB)
        batch_CB_P=TestGenCT2CB.predict(batch_CT_P)
        
        # #%%
        SSIM_score,MAE_score,MSE_score,SNU_score,PSNR_score,NCC_score=perf_metrics(batch_CB,batch_CB_P)
        # print('SSIM = %s MAE = %s MSE = %s SNU = %s PSNR = %s NCC = %s'%(SSIM_score.numpy(),MAE_score.numpy(),MSE_score.numpy(),SNU_score.numpy(),PSNR_score.numpy(),NCC_score.numpy()))
        Perf_score_CBCT_ele=[SSIM_score.numpy(),MAE_score.numpy(),MSE_score.numpy(),SNU_score.numpy(),PSNR_score.numpy(),NCC_score.numpy()]
        Perf_score_CBCT.append(Perf_score_CBCT_ele)
        
        # batch_CT=tf.image.resize(batch_CT,[512,512])
        batch_CB=tf.image.resize(batch_CB,[512,512])
        batch_CT_P=tf.image.resize(batch_CT_P,[512,512])
        batch_CB_P=tf.image.resize(batch_CB_P,[512,512])
        batch_CB_P=np.squeeze(batch_CB_P,axis=-1)
        # batch_CT=np.squeeze(batch_CT,axis=-1)
        batch_CT_P=np.squeeze(batch_CT_P,axis=-1)
        batch_CB=np.squeeze(batch_CB,axis=-1)
        mdic = {"batch_CB_P":batch_CB_P,"batch_CB":batch_CB,"batch_CT_P":batch_CT_P}
        cb_vol_matfile1='CBCT-'+str(cbcti)+'.mat'
        cb_vol_matfile=os.path.join(saveweightoutputpath,cb_vol_matfile1)
        savemat(cb_vol_matfile,mdic)
        
        # plt.figure(cbcti+1)
        # plt.subplot(1,3,1)
        # plt.imshow(batch_CB[0,:,:],cmap='gray')
        # plt.show()
        # plt.title('CB')
        # plt.subplot(1,3,2)
        # plt.imshow(batch_CB_P[0,:,:],cmap='gray')
        # plt.show()
        # plt.title('pseudo-CB')
        # plt.subplot(1,3,3)
        # plt.imshow(batch_CT_P[0,:,:],cmap='gray')
        # plt.show()
        # plt.title('pseudo-CT')
perf_dic = {"Perf_score_CBCT":Perf_score_CBCT,"Perf_score_CT":Perf_score_CT}
perf_file1 = 'Perf_score.mat'
perf_file=os.path.join(weightoutputpath,perf_file1)
savemat(perf_file,perf_dic)
