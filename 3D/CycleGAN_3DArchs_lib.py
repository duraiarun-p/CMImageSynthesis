#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:57:49 2022

@author: s1785969
"""
import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
import random
import os
import sys, getopt
# import multiprocessing

# os.environ['CUDA_VISIBLE_DEVICES'] = "" # comment this line when running in eddie
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import ZeroPadding3D


# import tensorflow_probability as tfp

import PIL
from PIL import Image
from tensorflow.keras.utils import Sequence

import scipy
from scipy import signal

import cycleganssimetriclib as ssTF
#%%
def I2I_CT2CB(cGAN,CT,overlap,border,TestGenCT2CB):
    CTsiz1=CT.shape 
    
    CT_P=np.zeros_like(CT,dtype=float)
    # CB_P=np.zeros_like(CT,dtype=float)
    #%%
    
    CT_blks=[]
    CT_blks_pred=[]
    
    CTblkindex=[]
    CTblkindex1=[]
    # for zi in range(0,CTsiz1[2],(cGAN.depth_size//overlap)-(2*border)):
    for zi in range(0,CTsiz1[2],(cGAN.depth_size//overlap)):    
        for j in  range(0,CTsiz1[0],(cGAN.patch_size//overlap)-(2*border)):
            for i in range(0,CTsiz1[1],(cGAN.patch_size//overlap)-(2*border)):   
                # border=border+5
    
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
                    # border=border+5
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
                    # border=border+5
                    currentBlk_p = currentBlk_p[border:-border,border:-border,:]
                    CT_P[i+border:i+cGAN.patch_size-border,j+border:j+cGAN.patch_size-border,zi:zi+cGAN.depth_size]=currentBlk_p
                    
                    # currentBlk_p = currentBlk_p[border:-border,border:-border,border:-border]
                    # CT_P[i+border:i+cGAN.patch_size-border,j+border:j+cGAN.patch_size-border,zi+border:zi+cGAN.depth_size-border]=currentBlk_p
                                    
                    ele1=[i+cGAN.patch_size-1,j+cGAN.patch_size-1,zi+cGAN.depth_size-1]
                    # ele1=[i+cGAN.patch_size-overlap,j+cGAN.patch_size-overlap,zi+cGAN.depth_size-overlap]
                    CTblkindex1.append(ele1)
                    
                    CT_blks.append(currentBlk)
                    CT_blks_pred.append(currentBlk_p)
    
    return CT_P

def I2I_CB2CT(cGAN,CT,overlap,border,TestGenCB2CT):
    CTsiz1=CT.shape 
    
    CT_P=np.zeros_like(CT,dtype=float)
    # CB_P=np.zeros_like(CT,dtype=float)
    #%%
    
    CT_blks=[]
    CT_blks_pred=[]
    
    CTblkindex=[]
    CTblkindex1=[]
    # for zi in range(0,CTsiz1[2],(cGAN.depth_size//overlap)-(2*border)):
    for zi in range(0,CTsiz1[2],(cGAN.depth_size//overlap)):    
        for j in  range(0,CTsiz1[0],(cGAN.patch_size//overlap)-(2*border)):
            for i in range(0,CTsiz1[1],(cGAN.patch_size//overlap)-(2*border)):        
                # border=border+5
    
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
                    currentBlk_p = TestGenCB2CT.predict(currentBlk_t)
                    
                    # currentBlk_p = currentBlk_t.numpy()/2
                    currentBlk_p = np.squeeze(currentBlk_p,axis=-1)
                    currentBlk_p = np.squeeze(currentBlk_p,axis=0)
                    dsfactor=(blksiz[0]/cGAN.patch_size,blksiz[1]/cGAN.patch_size,blksiz[2]/cGAN.depth_size)
                    
                    # currentBlk_p = nd.interpolation.zoom(currentBlk_p, zoom=dsfactor)
                    # border=border+5
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
                    currentBlk_p = TestGenCB2CT.predict(currentBlk_t)
                    currentBlk_p = np.squeeze(currentBlk_p,axis=-1)
                    currentBlk_p = np.squeeze(currentBlk_p,axis=0)
                    # currentBlk_p = currentBlk_p[3:-3,3:-3,:]
                    # border=border+5
                    currentBlk_p = currentBlk_p[border:-border,border:-border,:]
                    CT_P[i+border:i+cGAN.patch_size-border,j+border:j+cGAN.patch_size-border,zi:zi+cGAN.depth_size]=currentBlk_p
                    
                    # currentBlk_p = currentBlk_p[border:-border,border:-border,border:-border]
                    # CT_P[i+border:i+cGAN.patch_size-border,j+border:j+cGAN.patch_size-border,zi+border:zi+cGAN.depth_size-border]=currentBlk_p
                                    
                    ele1=[i+cGAN.patch_size-1,j+cGAN.patch_size-1,zi+cGAN.depth_size-1]
                    # ele1=[i+cGAN.patch_size-overlap,j+cGAN.patch_size-overlap,zi+cGAN.depth_size-overlap]
                    CTblkindex1.append(ele1)
                    
                    CT_blks.append(currentBlk)
                    CT_blks_pred.append(currentBlk_p)
    
    return CT_P
#%%
def perf_metrics(batch_CT,batch_CT_P):
    
    def custom_tf_ssim_score(y_true, y_pred):# SSIM
        max_val1=tf.math.reduce_max(y_true)-tf.math.reduce_min(y_true)
        max_val2=tf.math.reduce_max(y_pred)-tf.math.reduce_min(y_pred)
        max_val =0.5*(max_val1+max_val2)
        filter_size=11
        filter_sigma=0.5
        batchsize=K.int_shape(y_pred)
        # batchsize=tf.gather(batchsize1,0)
        ssimscores=[]
        for batchelei in range(batchsize[0]):
            y_pred1=y_pred[batchelei,:,:,:]
            y_true1=y_true[batchelei,:,:,:]
            ssimscoreele,_=ssTF.tfssim_custom(y_true1, y_pred1, max_val,filter_size,filter_sigma)
            ssimscores.append(ssimscoreele)
        ssimscore=tf.reduce_mean(ssimscores)
        # ssimscore=ssTF.tfssim(y_true, y_pred, max_val,filter_size,filter_sigma)
        # loss=tf.math.subtract(1, ssimscore)# Will cause error when train_on_batch: run_eagerly=False
        loss=1-ssimscore
        # return 
        return ssimscore
    
    SSIM_score=custom_tf_ssim_score(batch_CT, batch_CT_P)
    
    # Mean Absolute Error
    mae_fx = tf.keras.losses.MeanAbsoluteError()
    MAE_score=mae_fx(batch_CT, batch_CT_P)
    
    # Mean Squared Error
    mse_fx = tf.keras.losses.MeanSquaredError()
    MSE_score=mae_fx(batch_CT, batch_CT_P)
    
    # Spatial non-uniformity does not work better
    SNU_score=(tf.math.reduce_max(batch_CT)-tf.math.reduce_min(batch_CT_P))/(tf.math.reduce_max(batch_CT)+tf.math.reduce_min(batch_CT_P))
    
    #Peak Signal to noise ratio
    PSNR_score=10*(tf.math.log((tf.math.reduce_max(batch_CT)*tf.math.reduce_max(batch_CT_P))/MSE_score))/tf.math.log(tf.constant(10, dtype=MSE_score.dtype))
    
    # Cross correlation
    N=tf.math.multiply((batch_CT-tf.math.reduce_mean(batch_CT)),(batch_CT_P-tf.math.reduce_mean(batch_CT_P)))
    D=tf.math.multiply((tf.math.reduce_std(batch_CT)),(tf.math.reduce_std(batch_CT_P)))
    NCC_score=tf.math.reduce_mean(N/D)
    
    # corr = tfp.stats.correlation(batch_CT, batch_CT_P, sample_axis=0, event_axis=None)
    # NCC_score=tf.mat.reduce_max(corr)
    return SSIM_score,MAE_score,MSE_score,SNU_score,PSNR_score,NCC_score

def dataload3D_2_predict(DataPath):
        # self.batch_size=1
        # mypath=self.DataPath
        onlyfiles = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
        onlyfiles.sort()
        onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
        onlyfiles = onlyfiles[0:-onlyfileslenrem]
        matfiles=[join(DataPath,f) for f in onlyfiles]
        mat_fname_ind=np.random.choice(len(matfiles),replace=False)  
        mat_contents=h5py.File(matfiles[mat_fname_ind],'r')
        # mat_contents_list=list(mat_contents.keys())    
        PlanCTCellRef=mat_contents['CTInfoCell']
        CTLen=np.shape(PlanCTCellRef)
        CTsl=np.zeros([CTLen[1],1])
        for cti in range(CTLen[1]):
            CTmatsizref=mat_contents['CTInfoCell'][1,cti]
            CTLocR=mat_contents[CTmatsizref]
            CTLoc=CTLocR[()]
            CTsiz=np.shape(CTLoc)
            if CTsiz[1]>300:
                CTsl[cti]=1
            else:
                CTsl[cti]=0
        CTindex=np.where(CTsl==1)
        CTindex=CTindex[0]   
        CTindex=int(CTindex)
        PlanCTLocRef=mat_contents['CTInfoCell'][1, CTindex]
        PlanCTLocRef=mat_contents[PlanCTLocRef]
        # PlanCTLoc=PlanCTLocRef[()]
        PlanCTCellRef=mat_contents['CTInfoCell'][2, CTindex]
        PlanCTCellRef=mat_contents[PlanCTCellRef]
        CT=PlanCTCellRef[()]
        CT=np.transpose(CT,(2,1,0))#data volume
        # CT=(CT-np.min(CT))/np.ptp(CT)
        CT = (CT-np.min(CT))/(np.max(CT)-np.min(CT))
        CTsiz1=CT.shape  
        CBCTCellRef=mat_contents['CBCTInfocell']
        CBCLen=np.shape(CBCTCellRef)
        CBCTi=np.random.choice(CBCLen[1],replace=False)  
        CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
        CBCellRef=mat_contents[CBCellRef]
        CBCT=CBCellRef[()]
        CBCT=np.transpose(CBCT,(2,1,0))
        # CBCT=(CBCT-np.min(CBCT))/np.ptp(CBCT)
        CBCT=(CBCT-np.min(CBCT))/(np.max(CBCT)-np.min(CBCT))
        CBsiz=CBCT.shape
        # i=np.random.randint(CTsiz1[0]-self.patch_size*2)
        # j=np.random.randint(CTsiz1[1]-self.patch_size*2)
        # zi1=np.random.randint(CTsiz1[2]-self.depth_size)
        # zi2=np.random.randint(CBsiz[2]-self.depth_size)
        # CTblocks=CT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi1:zi1+self.depth_size]
        # CBblocks=CBCT[i:i+self.patch_size*2,j:j+self.patch_size*2,zi2:zi2+self.depth_size]
        # CTblocks=np.expand_dims(CTblocks, axis=0)
        # CBblocks=np.expand_dims(CBblocks, axis=0)
        return CT, CBCT

#%% Data loader for volume prediction
def dataload_direct(matfile):
    # mypath=self.DataPath
    # onlyfiles = [f for f in listdir(DataPath) if isfile(join(DataPath, f))]
    # onlyfiles.sort()
    # onlyfileslenrem=len(onlyfiles)-round(len(onlyfiles)*0.7)
    # onlyfiles = onlyfiles[0:-onlyfileslenrem]
    # matfiles=[join(DataPath,f) for f in onlyfiles]
    # mat_fname_ind=np.random.choice(len(matfiles),replace=False)  
    mat_contents=h5py.File(matfile,'r')
    # mat_contents_list=list(mat_contents.keys())    
    PlanCTCellRef=mat_contents['CTInfoCell']
    CTLen=np.shape(PlanCTCellRef)
    CTsl=np.zeros([CTLen[1],1])
    for cti in range(CTLen[1]):
        CTmatsizref=mat_contents['CTInfoCell'][1,cti]
        CTLocR=mat_contents[CTmatsizref]
        CTLoc=CTLocR[()]
        CTsiz=np.shape(CTLoc)
        if CTsiz[1]>300:
            CTsl[cti]=1
        else:
            CTsl[cti]=0
    CTindex=np.where(CTsl==1)
    CTindex=CTindex[0]  
    CTindex=int(CTindex)
    PlanCTLocRef=mat_contents['CTInfoCell'][1, CTindex]
    PlanCTLocRef=mat_contents[PlanCTLocRef]
    # PlanCTLoc=PlanCTLocRef[()]
    PlanCTCellRef=mat_contents['CTInfoCell'][2, CTindex]
    PlanCTCellRef=mat_contents[PlanCTCellRef]
    CT=PlanCTCellRef[()]
    CT=np.transpose(CT,(2,1,0))
    CTsiz1=CT.shape
    # CT_rand_index=np.random.choice(CTsiz1[2],size=batch_size,replace=False)
    # batch_CT_img=np.zeros((CTsiz1[0],CTsiz1[1],len(CT_rand_index)))
    # for ri in range(len(CT_rand_index)):
    #     batch_CT_img[:,:,ri]=CT[:,:,CT_rand_index[ri]]    
    CBCTCellRef=mat_contents['CBCTInfocell']
    CBCLen=np.shape(CBCTCellRef)
    # CBCTi=np.random.choice(CBCLen[1],size=1)
    # CBCellRef1=mat_contents['CBCTInfocell'][4, CBCTi]
    # # CBCellRef2=mat_contents[CBCellRef1]
    # CBCT=CBCellRef1[()]
    # CBCT=np.transpose(CBCT,(2,1,0))

       
    CBCTs=[]
    for CBCTi in range(CBCLen[1]):
        # print(CBCTi)
        CBCellRef=mat_contents['CBCTInfocell'][4, CBCTi]
        CBCellRef=mat_contents[CBCellRef]
        CBCT=CBCellRef[()]
        CBCT=np.transpose(CBCT,(2,1,0))
        CBCTs.append(CBCT)
        CBLocRef=mat_contents['CBCTInfocell'][1, CBCTi]
        CBLocRef=mat_contents[CBLocRef]
        CBCTLoc=CBLocRef[()]
    # CBsiz=CBCT.shape
    # batch_CB_img=np.zeros((CTsiz1[0],CTsiz1[1],batch_size))
    # for cbi in range(batch_size):
    #     CB_rand_sl_index=np.random.choice(CBsiz[2])
    #     CB_rand_pat_index=np.random.choice(CBCLen[1],replace=False)
    #     # print(CB_rand_pat_index)
    #     # print(CB_rand_sl_index)
    #     batch_CB_img[:,:,cbi]=CBCTs[CB_rand_pat_index][:,:,CB_rand_sl_index]
    # del mat_contents
    # batch_CT_img=tf.image.resize(batch_CT_img,[256,256])
    # batch_CB_img=tf.image.resize(batch_CB_img,[256,256])
    # batch_CT_img = tf.transpose(batch_CT_img,perm=[2,0,1])
    # batch_CB_img = tf.transpose(batch_CB_img,perm=[2,0,1])
    # batch_CT_img = tf.expand_dims(batch_CT_img, -1)
    # batch_CB_img = tf.expand_dims(batch_CB_img, -1)
   
    # return batch_CT_img, batch_CB_img
    return CT,CBCTs

def tfssim(img1,img2,max_val):
    score = tf.image.ssim(img1, img2, max_val, filter_size=2, filter_sigma=1.5, k1=0.01, k2=0.03)
    return score
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
        max_val1=tf.math.reduce_max(y_true1)-tf.math.reduce_min(y_true1)
        max_val2=tf.math.reduce_max(y_pred1)-tf.math.reduce_min(y_pred1)
        max_val =0.5*(max_val1+max_val2)
        for depthelei in range(depthsize[0]):
            y_pred2=y_pred1[depthelei,:,:,:]
            y_true2=y_true1[depthelei,:,:,:]
            # ssimscoreele,_=ssTF.tfssim_custom(y_true2, y_pred2, max_val,filter_size,filter_sigma)
            ssimscoreele=tfssim(y_true2, y_pred2, max_val)
            ssimscores.append(ssimscoreele)
    ssimscore=tf.reduce_mean(ssimscores)
    # ssimscore=ssTF.tfssim(y_true, y_pred, max_val,filter_size,filter_sigma)
    # loss=tf.math.subtract(1, ssimscore)# Will cause error when train_on_batch: run_eagerly=False
    loss=1-ssimscore
    return loss


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
    def __init__(self, trainA_path, trainB_path, image_list_A, image_list_B, batch_size):
        # self.newshape=newshape
        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []
        # self.batch_set_size=batch_set_size
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
    trainCT_image_names=random.sample(trainCT_image_names,batch_set_size)
    trainCB_image_names=random.sample(trainCB_image_names,batch_set_size)
    return data_sequence(trainCT_path, trainCB_path, trainCT_image_names, trainCB_image_names,batch_size=batch_size)

#%%
class ReflectionPadding3D(ZeroPadding3D):
    """Reflection-padding layer for 3D data (spatial or spatio-temporal).

    Args:
        padding (int, tuple): The pad-width to add in each dimension.
            If an int, the same symmetric padding is applied to height and
            width.
            If a tuple of 3 ints, interpreted as two different symmetric
            padding values for height and width:
            ``(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)``.
            If tuple of 3 tuples of 2 ints, interpreted as
            ``((left_dim1_pad, right_dim1_pad),
            (left_dim2_pad, right_dim2_pad),
            (left_dim3_pad, right_dim3_pad))``
        data_format (str): A string, one of ``channels_last`` (default)
            or ``channels_first``. The ordering of the dimensions in the
            inputs. ``channels_last`` corresponds to inputs with shape
            ``(batch, height, width, channels)`` while ``channels_first``
            corresponds to inputs with shape
            ``(batch, channels, height, width)``.
    """

    def call(self, inputs):
        d_pad, w_pad, h_pad = self.padding
        if self.data_format == 'channels_first':
            pattern = [[0, 0], [0, 0], [d_pad[0], d_pad[1]],
                       [w_pad[0], w_pad[1]], [h_pad[0], h_pad[1]]]
        else:
            pattern = [[0, 0], [d_pad[0], d_pad[1]],
                       [w_pad[0], w_pad[1]], [h_pad[0], h_pad[1]], [0, 0]]
        return tf.pad(inputs, pattern, mode='REFLECT')
    
#%%


class CycleGAN_3D_Alpha():
#%% 3D
    @staticmethod
    def convblk3d(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
            #                        center=True,
            #                        scale=True,
            #                        beta_initializer="random_uniform",
            #                        gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.LeakyReLU(alpha=0.2)(opL)
        return opL
   
    @staticmethod
    def convblk3d_ReLU(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
                                   # center=True,
                                   # scale=True,
                                   # beta_initializer="random_uniform",
                                   # gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.ReLU()(opL)
        return opL
    @staticmethod
    def convblk3d_valid(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='valid')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
            #                        center=True,
            #                        scale=True,
            #                        beta_initializer="random_uniform",
            #                        gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.ReLU()(opL)
        return opL
    @staticmethod
    def attentionblk3D(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=3)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
   
    @staticmethod
    def attentionblk3D_1(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=13)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
   
    @staticmethod
    def attentionblk3D_2(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=25)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
   
    @staticmethod
    def deconvblk3D(ipL,filters,kernel_size,strides):
        # opL=layers.UpSampling3D(size=2)(ipL)
        # opL=ReflectionPadding3D()(opL)
        # opL=layers.Conv3DTranspose(filters, kernel_size, strides=strides,padding='SAME')(ipL)
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        # opL=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(opL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.Activation('relu')(opL)
        return opL
    @staticmethod
    def upsample3D(ipL,filters,kernel_size,strides):
        # opL=layers.UpSampling3D(size=2)(ipL)
        # opL=ReflectionPadding3D()(opL)
        opL=layers.Conv3DTranspose(filters, kernel_size, strides=strides,padding='SAME')(ipL)
        # opL=layers.Conv3D(filters, kernel_size=1, strides=strides,padding='SAME')(ipL)
        # opL=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(opL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.Activation('relu')(opL)
        return opL
    @staticmethod
    def resblock(x,filters,kernelsize,stride):
        fx = layers.Conv3D(filters, kernelsize,strides=stride,padding='same')(x)
        # fx=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(fx)
        fx=layers.BatchNormalization()(fx)
        fx = layers.Activation('relu')(fx)
        fx = layers.Conv3D(filters, kernelsize,strides=stride, padding='same')(fx)
        # fx=tfa.layers.InstanceNormalization(axis=3,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(fx)
        fx=layers.BatchNormalization()(fx)
        out = layers.Add()([x,fx])
        return out
   
    def build_generator3D(self):
        ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
        opL = layers.Conv3D(self.genafilter, self.kernel_size_gen_1, padding='same')(ipL)
        # opL=ReflectionPadding3D()(opL)
        opL=self.deconvblk3D(opL,self.genafilter*2,self.kernel_size_gen_2,self.stride2)
        opL=self.deconvblk3D(opL,self.genafilter*4,self.kernel_size_gen_2,self.stride2)
        for _ in range(5):
            opL=self.resblock(opL,self.genafilter*4,self.kernel_size_gen_2,self.stride1)
        opL=self.upsample3D(opL,self.genafilter*2,self.kernel_size_gen_2,self.stride2)
        opL=self.upsample3D(opL,self.genafilter,self.kernel_size_gen_2,self.stride2)
        opL = layers.Conv3D(1, self.kernel_size_gen_1,self.stride1, padding='same')(opL)
        # opL = layers.Conv3D(1, self.kernel_size_gen_1,self.stride1, padding='same')(opL)
       
       
        return keras.Model(ipL,opL)
   
    def build_discriminator3D(self):
        ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
        opL1=self.convblk3d(ipL,self.discfilter,self.kernel_size_disc,self.stride2,normalization=False)    
        opL2=self.convblk3d(opL1,self.discfilter*2,self.kernel_size_disc,self.stride2,normalization=True)
        opL3=self.convblk3d(opL2,self.discfilter*4,self.kernel_size_disc,self.stride1,normalization=True)
        opL4=self.convblk3d(opL3,self.discfilter*8,self.kernel_size_disc,self.stride1,normalization=True)
        opL5 = layers.Conv3D(1, kernel_size=4, strides=1, padding='same',activation='sigmoid')(opL4)
        # opL6 = layers.Flatten()(opL5)
        # opL7 = layers.Dense(1, activation='Sigmoid')(opL6)
        return keras.Model(ipL,opL5)
   
   
    def __init__(self,mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag):
          self.DataPath=mypath
          self.WeightSavePath=weightoutputpath
          self.batch_size=batch_size
          self.img_shape=imgshape
          self.input_shape=tuple([batch_size,imgshape])
          self.genafilter = 32
          self.discfilter = 64
          self.epochs = epochs+1
          self.save_epoch_frequency=save_epoch_frequency
          self.batch_set_size=batch_set_size
          self.lambda_cycle = 10.0                    # Cycle-consistency loss
          self.lambda_id = 0.9 * self.lambda_cycle    # Identity loss
          self.saveweightflag=saveweightflag
          self.patch_size=32
          self.depth_size=32
          self.input_layer_shape_3D=tuple([self.patch_size,self.patch_size,self.depth_size,1])
          self.stride2=2
          self.stride1=1
          self.kernel_size = 3
          self.kernel_size_disc = 4
          self.kernel_size_gen_1=7
          self.kernel_size_gen_2=3
         
          os.chdir(self.WeightSavePath)
          self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))
          os.mkdir(self.folderlen)
          self.WeightSavePath = os.path.join(self.WeightSavePath, self.folderlen)
          os.chdir(self.WeightSavePath)
         
          newdir='arch'
         
          self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
          os.mkdir(self.WeightSavePathNew)
          os.chdir(self.WeightSavePathNew)
         
          self.Disc_lr=4e-6
          self.Gen_lr=0.001
         
          self.Disc_optimizer = keras.optimizers.Adam(self.Disc_lr, 0.5,0.999)
          self.Gen_optimizer = keras.optimizers.Adam(self.Gen_lr, 0.5,0.999)
         
          # optimizer = keras.optimizers.Adam(0.0002, 0.5)

          self.DiscCT=self.build_discriminator3D()
          self.DiscCT.compile(loss='mse', optimizer=self.Disc_optimizer, metrics=['accuracy'])
          self.DiscCT._name='Discriminator-CT'
          tf.keras.utils.plot_model(self.DiscCT, to_file='Discriminator-CT.png', show_shapes=True)
          # self.DiscCT.summary()
          with open('Disc.txt', 'w+') as f:
              self.DiscCT.summary(print_fn=lambda x: f.write(x + '\n'))
         
          self.DiscCB=self.build_discriminator3D()
          self.DiscCB.compile(loss='mse', optimizer=self.Disc_optimizer, metrics=['accuracy'])
          self.DiscCB._name='Discriminator-CB'
         
          self.DiscCT_static=self.DiscCT
          self.DiscCB_static=self.DiscCB
         
          layer_len=len(self.DiscCT.layers)
          layers_lis=self.DiscCT.layers
          labelshapearr=list(layers_lis[layer_len-1].output_shape)
          labelshapearr[0]=self.batch_size
          labelshape=tuple(labelshapearr)
          self.labelshape=labelshape
       
          self.GenCB2CT=self.build_generator3D()
          self.GenCB2CT.compile(loss='mse', optimizer=self.Gen_optimizer, metrics=['accuracy'])
          self.GenCB2CT._name='Generator-CB2CT'
          tf.keras.utils.plot_model(self.GenCB2CT, to_file='Generator-CB2CT.png', show_shapes=True)        
          # self.GenCB2CT.summary()
          with open('Gena.txt', 'w+') as f:
              self.GenCB2CT.summary(print_fn=lambda x: f.write(x + '\n'))
         
          self.GenCT2CB=self.build_generator3D()
          self.GenCT2CB.compile(loss='mse', optimizer=self.Gen_optimizer, metrics=['accuracy'])
          self.GenCT2CB._name='Generator-CT2CB'
         
          img_CT = keras.Input(shape=self.input_layer_shape_3D)
          img_CB = keras.Input(shape=self.input_layer_shape_3D)
         
          valid_CT = self.DiscCT(img_CT)
          valid_CB = self.DiscCB(img_CB)
         
          # self.DiscCT_static = Network(inputs=img_CT, outputs=guess_A, name='D_A_static_model')
          # self.DiscCB_static = Network(inputs=img_CB, outputs=guess_B, name='D_B_static_model')
          # self.DiscCT_static = clone_model(self.DiscCT)
          # self.DiscCB_static = clone_model(self.DiscCB)
          self.DiscCT_static.set_weights(self.DiscCT.get_weights())
          self.DiscCB_static.set_weights(self.DiscCB.get_weights())
          # self.DiscCT_static = clone_model(self.DiscCT)
          # self.DiscCB_static = clone_model(self.DiscCB)
       
        # For the combined model we will only train the generators
          # self.DiscCT_static.trainable = False
          # self.DiscCB_static.trainable = False
         
          # Input images from both domains

       
        # Translate images to the other domain
          fake_CB = self.GenCT2CB(img_CT)
          fake_CT = self.GenCB2CT(img_CB)
        # Translate images back to original domain
          reconstr_CT = self.GenCB2CT(fake_CB)
          reconstr_CB = self.GenCT2CB(fake_CT)
        # Identity mapping of images
          img_CT_id = self.GenCT2CB(img_CT)
          img_CB_id = self.GenCB2CT(img_CB)
         
        #   self.DiscCT_static = Network(inputs=img_CT, outputs=guess_A, name='D_A_static_model')
        #   self.DiscCB_static = Network(inputs=img_CB, outputs=guess_B, name='D_B_static_model')
       
        # # For the combined model we will only train the generators
        #   self.DiscCT_static.trainable = False
        #   self.DiscCB_static.trainable = False
       
        # Discriminators determines validity of translated images
          valid_CT = self.DiscCT_static(fake_CT)
          valid_CB = self.DiscCB_static(fake_CB)
       
        # Combined model trains generators to fool discriminators
          self.cycleGAN_Model = keras.Model(inputs=[img_CT, img_CB], outputs=[valid_CT, valid_CB, reconstr_CT, reconstr_CB, img_CT_id, img_CB_id])
          self.cycleGAN_Model.compile(loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
          loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id], optimizer=self.Gen_optimizer)
          self.cycleGAN_Model._name='CycleGAN'
         
          with open('cycleGAN.txt', 'w+') as f:
              self.cycleGAN_Model.summary(print_fn=lambda x: f.write(x + '\n'))
         
          trainCT_path = os.path.join(self.DataPath, 'trainCT')
          trainCB_path = os.path.join(self.DataPath, 'trainCB')
          # testCT_path = os.path.join(self.DataPath, 'validCT')
          # testCB_path = os.path.join(self.DataPath, 'validCB')
          # self.data_generator=loadprintoutgen(trainCT_path,trainCB_path,self.batch_size,self.batch_set_size)
         
          # for images in self.data_generator:
          #     batch_CT = images[0]
          #     batch_CB = images[1]
             
          #     print(batch_CT.shape)
          # print(self.data_generator.__len__())
          # os.system("nvidia-smi")
          # print('Init')
         
    def learningrate_log_scheduler(self):
        learning_rates=np.logspace(-8, 1,num=self.epochs)
        return learning_rates
   
    def traincgan(self):
        os.chdir(self.WeightSavePathNew)
         # self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))        
        newdir='weights'        
        self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
        os.mkdir(self.WeightSavePathNew)
        os.chdir(self.WeightSavePathNew)
       
        gen1fname="GenCT2CBWeights"
        gen2fname="GenCB2CTWeights"
        disc1fname="DiscCTWeights"
        disc2fname="DiscCBWeights"
        # D_losses = np.zeros((self.batch_set_size,2,self.epochs))
        # G_losses = np.zeros((self.batch_set_size,7,self.epochs))
        D_losses = []
        G_losses = []
       
        #Learning rate schedule
        learning_rates=self.learningrate_log_scheduler()
       
        def run_training_iteration(loop_index, epoch_iterations):
              valid = tf.ones((self.labelshape))
              fake = tf.zeros((self.labelshape))
                 # ----------------------
                 #  Train Discriminators
                 # ----------------------

                 # Translate images to opposite domain
              fake_CB = self.GenCT2CB.predict(batch_CT)
              fake_CT = self.GenCB2CT.predict(batch_CB)

                 # Train the discriminators (original images = real / translated = Fake)
              dCT_loss_real = self.DiscCT.train_on_batch(batch_CT, valid)
              dCT_loss_fake = self.DiscCT.train_on_batch(fake_CT, fake)
              dCT_loss = 0.5 * tf.math.add(dCT_loss_real, dCT_loss_fake)

              dCB_loss_real = self.DiscCB.train_on_batch(batch_CB, valid)
              dCB_loss_fake = self.DiscCB.train_on_batch(fake_CB, fake)
              dCB_loss = 0.5 * tf.math.add(dCB_loss_real, dCB_loss_fake)

                 # Total discriminator loss
              d_loss = 0.5 * tf.math.add(dCT_loss, dCB_loss)

                 # ------------------
                 #  Train Generators
                 # ------------------

                 # Train the generators
              g_loss = self.cycleGAN_Model.train_on_batch([batch_CT, batch_CB],
                                                       [valid, valid,
                                                        batch_CT, batch_CB,
                                                        batch_CT, batch_CB])
             
              return g_loss, d_loss.numpy()
             
        for epochi in range(self.epochs):
            # if self.use_data_generator:
                loop_index = 1
               
                # K.set_value(self.Gen_optimizer.learning_rate, learning_rates[epochi])
                # K.set_value(self.Disc_optimizer.learning_rate, learning_rates[epochi])
                # print(self.data_generator.__len__())
                for images in self.data_generator:
                    batch_CT = images[0]
                    batch_CB = images[1]
                    # Run all training steps
                    g_loss, d_loss=run_training_iteration(loop_index, self.data_generator.__len__())
                    # os.system("nvidia-smi")
                    #print('Trainingstep')
                   
                    if loop_index >= self.data_generator.__len__():
                        break
                    loop_index = loop_index+1
                   
                   
               
                D_losses.append(d_loss)
                G_losses.append(g_loss)
                   
                if epochi % self.save_epoch_frequency == 0 and self.saveweightflag==True: # Weights saved based on epoch intervals
                    gen1fname1=gen1fname+'-'+str(epochi)+'.h5'    
                    gen2fname1=gen2fname+'-'+str(epochi)+'.h5'
                    disc1fname1=disc1fname+'-'+str(epochi)+'.h5'
                    disc2fname1=disc2fname+'-'+str(epochi)+'.h5'
                    self.GenCT2CB.save_weights(gen1fname1)
                    self.GenCB2CT.save_weights(gen2fname1)
                    self.DiscCT.save_weights(disc1fname1)
                    self.DiscCB.save_weights(disc2fname1)
        # os.system("nvidia-smi")
                print('Epoch =%s'%epochi)
        return D_losses,G_losses


#%%
class CycleGAN_3D_Beta():   
  
#%% 2D     
    @staticmethod
    def conv2d(layer_input, filters, f_size=4,stride=2,normalization=True):
        """Discriminator layer"""
        d = layers.Conv2D(filters, kernel_size=f_size,strides=stride, padding='same',activation='relu')(layer_input)
        if normalization:
            d = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(d)
        d = layers.LeakyReLU(alpha=0.2)(d)
        return d
    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, stride=1, dropout_rate=0,skip=True):
          """Layers used during upsampling"""
          u = layers.UpSampling2D(size=2)(layer_input)
          u = layers.Conv2D(filters, kernel_size=f_size, strides=stride, padding='same', activation='tanh')(u)
          if dropout_rate:
              u = layers.Dropout(dropout_rate)(u)
          u = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(u)
          if skip:
              u = layers.Concatenate()([u, skip_input])
          return u
#%% 3D
    @staticmethod
    def convblk3d(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
            #                        center=True,
            #                        scale=True,
            #                        beta_initializer="random_uniform",
            #                        gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.LeakyReLU(alpha=0.2)(opL)
        return opL
   
    @staticmethod
    def convblk3d_ReLU(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
                                   # center=True,
                                   # scale=True,
                                   # beta_initializer="random_uniform",
                                   # gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.ReLU()(opL)
        return opL
    @staticmethod
    def convblk3d_valid(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='valid')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
            #                        center=True,
            #                        scale=True,
            #                        beta_initializer="random_uniform",
            #                        gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.ReLU()(opL)
        return opL
    
    @staticmethod
    def attentionblk3D(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=3)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def attentionblk3D_1(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=13)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def attentionblk3D_2(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=25)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def deconvblk3D(ipL,filters,kernel_size,strides):
        # opL=layers.UpSampling3D(size=2)(ipL)
        # opL=ReflectionPadding3D()(opL)
        # opL=layers.Conv3DTranspose(filters, kernel_size, strides=strides,padding='SAME')(ipL)
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        # opL=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(opL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.Activation('relu')(opL)
        return opL
    @staticmethod
    def upsample3D(ipL,filters,kernel_size,strides):
        # opL=layers.UpSampling3D(size=2)(ipL)
        # opL=ReflectionPadding3D()(opL)
        opL=layers.Conv3DTranspose(filters, kernel_size, strides=strides,padding='SAME')(ipL)
        # opL=layers.Conv3D(filters, kernel_size=1, strides=strides,padding='SAME')(ipL)
        # opL=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(opL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.Activation('relu')(opL)
        return opL
    @staticmethod
    def resblock(x,filters,kernelsize,stride):
        fx = layers.Conv3D(filters, kernelsize,strides=stride,padding='same')(x)
        # fx=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(fx)
        fx=layers.BatchNormalization()(fx)
        fx = layers.Activation('relu')(fx)
        fx = layers.Conv3D(filters, kernelsize,strides=stride, padding='same')(fx)
        # fx=tfa.layers.InstanceNormalization(axis=3,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(fx)
        fx=layers.BatchNormalization()(fx)
        out = layers.Add()([x,fx])
        return out
    
    def build_generator3D(self):
        ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
        opL = layers.Conv3D(self.genafilter, self.kernel_size_gen_1, padding='same')(ipL)
        # opL=ReflectionPadding3D()(opL)
        opL=self.deconvblk3D(opL,self.genafilter*2,self.kernel_size_gen_2,self.stride2)
        opL=self.deconvblk3D(opL,self.genafilter*4,self.kernel_size_gen_2,self.stride2)
        for _ in range(5):
            opL=self.resblock(opL,self.genafilter*4,self.kernel_size_gen_2,self.stride1)
        opL=self.upsample3D(opL,self.genafilter*2,self.kernel_size_gen_2,self.stride2)
        opL=self.upsample3D(opL,self.genafilter,self.kernel_size_gen_2,self.stride2)
        opL = layers.Conv3D(1, self.kernel_size_gen_1,self.stride1, padding='same')(opL)
        # opL = layers.Conv3D(1, self.kernel_size_gen_1,self.stride1, padding='same')(opL)
       
       
        return keras.Model(ipL,opL)
   
    def build_discriminator3D(self):
        ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
        opL1=self.convblk3d(ipL,self.discfilter,self.kernel_size_disc,self.stride2,normalization=False)    
        opL2=self.convblk3d(opL1,self.discfilter*2,self.kernel_size_disc,self.stride2,normalization=True)
        opL3=self.convblk3d(opL2,self.discfilter*4,self.kernel_size_disc,self.stride1,normalization=True)
        opL4=self.convblk3d(opL3,self.discfilter*8,self.kernel_size_disc,self.stride1,normalization=True)
        opL5 = layers.Conv3D(1, kernel_size=4, strides=1, padding='same',activation='sigmoid')(opL4)
        # opL6 = layers.Flatten()(opL5)
        # opL7 = layers.Dense(1, activation='Sigmoid')(opL6)
        return keras.Model(ipL,opL5)
    
    # def build_generator3D(self):
    #     ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
    #     opL1=self.convblk3d(ipL,self.genafilter,self.kernel_size,self.stride2)
    #     opL2=self.convblk3d(opL1,self.genafilter*2,self.kernel_size,self.stride2)
    #     opL3=self.convblk3d(opL2,self.genafilter*4,self.kernel_size,self.stride2)
    #     opL4=layers.Conv3D(filters=self.genafilter*8, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL3)
    #     # opL4=self.convblk3d(opL3,self.genafilter*8,self.kernel_size,self.stride2)
        
    #     opL5=self.attentionblk3D(opL3,opL4,filters=self.genafilter*8, kernel_size=self.kernel_size, strides=self.stride2)
        
    #     opL6=layers.Concatenate()([opL4,opL5])
    #     opL7=self.deconvblk3D(opL6,self.genafilter*4,self.kernel_size,self.stride1)
    #     opL8=self.deconvblk3D(opL7,self.genafilter*2,self.kernel_size,self.stride1)
    #     # opL9=self.convblk3d(opL8,self.genafilter*4,self.kernel_size,self.stride2)
    #     opL9=layers.Conv3D(filters=self.genafilter*4, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL8)
        
    #     opL10=self.attentionblk3D_1(opL1,opL9,filters=self.genafilter*4, kernel_size=self.kernel_size, strides=self.stride2)
        
    #     opL11=layers.Concatenate()([opL9,opL10])
    #     opL12=self.deconvblk3D(opL11,self.genafilter*2,self.kernel_size,self.stride1)
    #     opL13=self.deconvblk3D(opL12,self.genafilter*4,self.kernel_size,self.stride1)
    #     # opL14=self.convblk3d(opL13,self.genafilter*2,self.kernel_size,self.stride2)
    #     opL14=layers.Conv3D(filters=self.genafilter*2, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL13)
        
    #     opL15=self.attentionblk3D_2(ipL,opL14,filters=self.genafilter*2, kernel_size=self.kernel_size, strides=self.stride2)
        
    #     opL16=layers.Concatenate()([opL14,opL15])
    #     opL17=self.deconvblk3D(opL16,self.genafilter*1,self.kernel_size,self.stride1)
    #     opL18=self.deconvblk3D(opL17,self.genafilter*2,self.kernel_size,self.stride1)
    #     # opL19=self.convblk3d(opL18,self.genafilter*1,self.kernel_size,self.stride1)
    #     opL19=layers.Conv3D(filters=self.genafilter*1, kernel_size=self.kernel_size, strides=self.stride1,padding='SAME')(opL18)
        
    #     opL20=self.resblock(opL19, self.genafilter*1, self.kernel_size)
    #     opL21=self.resblock(opL20, self.genafilter*1, self.kernel_size)
    #     opL22=layers.Conv3D(filters=1, kernel_size=self.kernel_size, strides=self.stride1,padding='SAME')(opL21)
        
    #     return keras.Model(ipL,opL22)
    
    # def build_discriminator3D(self):
    #     ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
    #     opL1=self.convblk3d(ipL,self.discfilter,self.kernel_size_disc,self.stride2)
    #     opL2=self.convblk3d(opL1,self.discfilter*2,self.kernel_size_disc,self.stride2)
    #     opL3=self.convblk3d(opL2,self.discfilter*4,self.kernel_size_disc,self.stride2)
    #     opL4=self.convblk3d(opL3,self.discfilter*8,self.kernel_size_disc,self.stride1)
        
    #     # opL5=layers.Flatten()(opL4)
        
    #     # opL5=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL5)
    #     # opL5=layers.LeakyReLU()(opL5)
        
    #     # opL6=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*16)(opL5)
    #     # opL6=layers.LeakyReLU()(opL6)

    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL6)
    #     # opL8=layers.Activation('sigmoid')(opL7)

    #     # opL9=layers.Dense(1)(opL8)
    #     # opL=layers.Activation('sigmoid')(opL9)
        
        
    #     opL5=layers.Conv3D(1, kernel_size=self.kernel_size_disc, strides=self.stride1,padding='SAME')(opL4)
        
    #     opL7 = layers.Flatten()(opL5)
        
    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL5)
    #     # opL7=layers.LeakyReLU()(opL7)
        
    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*16)(opL7)
    #     # opL7=layers.LeakyReLU()(opL7)
        
    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*8))(opL7)
    #     # opL7=layers.LeakyReLU()(opL7)
        
    #     # opL7=layers.Conv3D(1, kernel_size=self.kernel_size_disc, strides=self.stride1,padding='SAME')(opL6)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.5))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.25))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.125))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.0625))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7 = layers.Dense(1)(opL7)
    #     opL = layers.Activation('relu')(opL7)
        
    #     return keras.Model(ipL,opL)
    
    def build_discriminator(self):
        d0 = keras.Input(shape=self.img_shape,name='Input')
        d1 = self.conv2d(d0, self.discfilter, stride=2, normalization=False)
        d2 = self.conv2d(d1, self.discfilter*2, stride=2, normalization=True)
        d3 = self.conv2d(d2, self.discfilter*4, stride=2, normalization=True)
        d4 = self.conv2d(d3, self.discfilter*8, stride=2, normalization=True)
        d5 = self.conv2d(d4, self.discfilter*8, stride=1, normalization=True)
        d6 = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)
        d7 = layers.Activation('relu')(d6)
        
        return keras.Model(d0,d7)
    
    def build_generator(self):
        d0 = keras.Input(shape=self.img_shape,name='Input')
        d1 = self.conv2d(d0, self.genafilter,stride=1,normalization=True)
        d2 = self.conv2d(d1, self.genafilter,stride=2,normalization=True)
        d3 = self.conv2d(d2, self.genafilter*2,stride=1,normalization=True)
        d4 = self.conv2d(d3, self.genafilter*2,stride=2,normalization=True)
        d5 = self.conv2d(d4, self.genafilter*4,stride=1,normalization=True)
        d6 = self.conv2d(d5, self.genafilter*4,stride=2,normalization=True)
        d7 = self.conv2d(d6, self.genafilter*8,stride=1,normalization=True)
        d8 = self.conv2d(d7, self.genafilter*8,stride=2,normalization=True)
        d9 = self.conv2d(d8, self.genafilter*16,stride=1,normalization=True)
        d10 = self.conv2d(d9, self.genafilter*16,stride=2,normalization=True)
        
        u10 = self.deconv2d(d10, d8, self.genafilter*8,stride=1)
        u9 = self.conv2d(u10, self.genafilter*8,stride=1,normalization=True)
        u8 = self.deconv2d(u9, d6, self.genafilter*4,stride=1)
        u7 = self.conv2d(u8, self.genafilter*4,stride=1,normalization=True)
        u6 = self.deconv2d(u7, d4, self.genafilter*2,stride=1)
        u5 = self.conv2d(u6, self.genafilter*2,stride=1,normalization=True)
        u4 = self.deconv2d(u5, d2, self.genafilter,stride=1)
        u3 = self.conv2d(u4, self.genafilter,stride=1,normalization=True)
        u2 = self.deconv2d(u3,d1, self.genafilter,stride=1,skip=False)
        u1 = layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(u2)
        u1 = layers.Activation('tanh')(u1)
        
        return keras.Model(d0,u1)
    
    def __init__(self,mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag):
          self.DataPath=mypath
          #self.breakflag=breakflag
          self.WeightSavePath=weightoutputpath
          self.batch_size=batch_size
          self.img_shape=imgshape
          self.input_shape=tuple([batch_size,imgshape])
          self.genafilter = 16
          self.discfilter = 16
          self.epochs = epochs+1
          self.save_epoch_frequency=save_epoch_frequency
          self.batch_set_size=batch_set_size
          self.lambda_cycle = 10.0                    # Cycle-consistency loss
          self.lambda_id = 0.9 * self.lambda_cycle    # Identity loss
          self.saveweightflag=saveweightflag
          self.patch_size=32
          self.depth_size=32
          self.input_layer_shape_3D=tuple([self.patch_size,self.patch_size,self.depth_size,1])
          self.stride2=2
          self.stride1=1
          self.kernel_size = 3
          self.kernel_size_disc = 4
          self.kernel_size_gen_1=7
          self.kernel_size_gen_2=3
         
          os.chdir(self.WeightSavePath)
          self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))
          os.mkdir(self.folderlen)
          self.WeightSavePath = os.path.join(self.WeightSavePath, self.folderlen)
          os.chdir(self.WeightSavePath)
         
          newdir='arch'
         
          self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
          os.mkdir(self.WeightSavePathNew)
          os.chdir(self.WeightSavePathNew)
          
          self.Disc_lr=0.0002
          self.Gen_lr=0.0002
          
          # self.Disc_lr=0.001
          # self.Gen_lr=0.01
         
          self.Disc_optimizer = keras.optimizers.Adam(self.Disc_lr, 0.5,0.999)
          self.Gen_optimizer = keras.optimizers.Adam(self.Gen_lr, 0.5,0.999)
         
          # optimizer = keras.optimizers.Adam(0.0002, 0.5)

          self.DiscCT=self.build_discriminator3D()
          self.DiscCT.compile(loss='mse', optimizer=self.Disc_optimizer, metrics=['accuracy'])
          self.DiscCT._name='Discriminator-CT'
          # self.DiscCT.summary()
          with open('Disc.txt', 'w+') as f:
              self.DiscCT.summary(print_fn=lambda x: f.write(x + '\n'))
         
          self.DiscCB=self.build_discriminator3D()
          self.DiscCB.compile(loss='mse', optimizer=self.Disc_optimizer, metrics=['accuracy'])
          self.DiscCB._name='Discriminator-CB'
         
          self.DiscCT_static=self.DiscCT
          self.DiscCB_static=self.DiscCB
          
          layer_len=len(self.DiscCT.layers)
          layers_lis=self.DiscCT.layers
          labelshapearr=list(layers_lis[layer_len-1].output_shape)
          labelshapearr[0]=self.batch_size
          labelshape=tuple(labelshapearr)
          self.labelshape=labelshape
        
          self.GenCB2CT=self.build_generator3D()
          self.GenCB2CT.compile(loss='mse', optimizer=self.Gen_optimizer, metrics=['accuracy'])
          self.GenCB2CT._name='Generator-CB2CT'
         
          
          # self.GenCB2CT.summary()
          with open('Gena.txt', 'w+') as f:
              self.GenCB2CT.summary(print_fn=lambda x: f.write(x + '\n'))
         
          self.GenCT2CB=self.build_generator3D()
          self.GenCT2CB.compile(loss='mse', optimizer=self.Gen_optimizer, metrics=['accuracy'])
          self.GenCT2CB._name='Generator-CT2CB'
          
          img_CT = keras.Input(shape=self.input_layer_shape_3D)
          img_CB = keras.Input(shape=self.input_layer_shape_3D)
          
          valid_CT = self.DiscCT(img_CT)
          valid_CB = self.DiscCB(img_CB)
          
          # self.DiscCT_static = Network(inputs=img_CT, outputs=guess_A, name='D_A_static_model')
          # self.DiscCB_static = Network(inputs=img_CB, outputs=guess_B, name='D_B_static_model')
          # self.DiscCT_static = clone_model(self.DiscCT)
          # self.DiscCB_static = clone_model(self.DiscCB)
          self.DiscCT_static.set_weights(self.DiscCT.get_weights())
          self.DiscCB_static.set_weights(self.DiscCB.get_weights())
          # self.DiscCT_static = clone_model(self.DiscCT)
          # self.DiscCB_static = clone_model(self.DiscCB)
        
        # For the combined model we will only train the generators
          # self.DiscCT_static.trainable = False
          # self.DiscCB_static.trainable = False
         
          # Input images from both domains

        
        # Translate images to the other domain
          fake_CB = self.GenCT2CB(img_CT)
          fake_CT = self.GenCB2CT(img_CB)
        # Translate images back to original domain
          reconstr_CT = self.GenCB2CT(fake_CB)
          reconstr_CB = self.GenCT2CB(fake_CT)
        # Identity mapping of images
          img_CT_id = self.GenCT2CB(img_CT)
          img_CB_id = self.GenCB2CT(img_CB)
          
        #   self.DiscCT_static = Network(inputs=img_CT, outputs=guess_A, name='D_A_static_model')
        #   self.DiscCB_static = Network(inputs=img_CB, outputs=guess_B, name='D_B_static_model')
        
        # # For the combined model we will only train the generators
        #   self.DiscCT_static.trainable = False
        #   self.DiscCB_static.trainable = False
        
        # Discriminators determines validity of translated images
          valid_CT = self.DiscCT_static(fake_CT)
          valid_CB = self.DiscCB_static(fake_CB)
        
        # Combined model trains generators to fool discriminators
          self.cycleGAN_Model = keras.Model(inputs=[img_CT, img_CB], outputs=[valid_CT, valid_CB, reconstr_CT, reconstr_CB, img_CT_id, img_CB_id,reconstr_CT, reconstr_CB])
          self.cycleGAN_Model.compile(loss=['mse', 'mse', 'mae', 'mae','mae', 'mae', custom_loss_2_beta3D,custom_loss_2_beta3D],
                                     loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id,10,10], 
                                     optimizer=self.Gen_optimizer)
          self.cycleGAN_Model._name='CycleGAN'
         
          with open('cycleGAN.txt', 'w+') as f:
              self.cycleGAN_Model.summary(print_fn=lambda x: f.write(x + '\n'))
          
          self.trainCT_path = os.path.join(self.DataPath, 'trainCT')
          self.trainCB_path = os.path.join(self.DataPath, 'trainCB')
          # testCT_path = os.path.join(self.DataPath, 'validCT')
          # testCB_path = os.path.join(self.DataPath, 'validCB')
          # self.data_generator=loadprintoutgen(trainCT_path,trainCB_path,self.batch_size,self.batch_set_size)
          
          # for images in self.data_generator:
          #     batch_CT = images[0]
          #     batch_CB = images[1]
              
          #     print(batch_CT.shape)
          # print(self.data_generator.__len__())
          os.system("nvidia-smi")
          print('Init')
          
    def learningrate_log_scheduler(self):
        learning_rates=np.logspace(-8, 1,num=self.epochs)
        return learning_rates
    
    def traincgan(self):
        os.chdir(self.WeightSavePathNew)
         # self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))         
        newdir='weights'        
        self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
        os.mkdir(self.WeightSavePathNew)
        os.chdir(self.WeightSavePathNew)
        
        gen1fname="GenCT2CBWeights"
        gen2fname="GenCB2CTWeights"
        disc1fname="DiscCTWeights"
        disc2fname="DiscCBWeights"
        # D_losses = np.zeros((self.batch_set_size,2,self.epochs))
        # G_losses = np.zeros((self.batch_set_size,7,self.epochs))
        D_losses = []
        G_losses = []
        
        #Learning rate schedule
        learning_rates=self.learningrate_log_scheduler()
        
        epoch_start=0
        if self.breakflag:
            lastweightpath=self.lastweightpath
            lst=os.listdir(lastweightpath)
            weightfilename=lst[-1]
            x=weightfilename.split(".")
            x=x[0].split("-")
            epochi=int(x[-1])
            gen1fname1_break="GenCT2CBWeights"+'-'+str(epochi)+'.h5'
            gen2fname1_break="GenCB2CTWeights"+'-'+str(epochi)+'.h5'
            disc1fname1_break="DiscCTWeights"+'-'+str(epochi)+'.h5'
            disc2fname1_break="DiscCBWeights"+'-'+str(epochi)+'.h5'
            self.GenCT2CB.load_weights(os.path.join(lastweightpath,gen1fname1_break))
            self.GenCB2CT.load_weights(os.path.join(lastweightpath,gen2fname1_break))
            self.DiscCT.load_weights(os.path.join(lastweightpath,disc1fname1_break))
            self.DiscCB.load_weights(os.path.join(lastweightpath,disc2fname1_break))
            epoch_start=epochi
        
        def run_training_iteration(loop_index, epoch_iterations):
              valid = tf.ones((self.labelshape))
              fake = tf.zeros((self.labelshape))
                 # ----------------------
                 #  Train Discriminators
                 # ----------------------

                 # Translate images to opposite domain
              fake_CB = self.GenCT2CB.predict(batch_CT)
              fake_CT = self.GenCB2CT.predict(batch_CB)

                 # Train the discriminators (original images = real / translated = Fake)
              dCT_loss_real = self.DiscCT.train_on_batch(batch_CT, valid)
              dCT_loss_fake = self.DiscCT.train_on_batch(fake_CT, fake)
              dCT_loss = 0.5 * tf.math.add(dCT_loss_real, dCT_loss_fake)

              dCB_loss_real = self.DiscCB.train_on_batch(batch_CB, valid)
              dCB_loss_fake = self.DiscCB.train_on_batch(fake_CB, fake)
              dCB_loss = 0.5 * tf.math.add(dCB_loss_real, dCB_loss_fake)

                 # Total discriminator loss
              d_loss = 0.5 * tf.math.add(dCT_loss, dCB_loss)

                 # ------------------
                 #  Train Generators
                 # ------------------

                 # Train the generators
              g_loss = self.cycleGAN_Model.train_on_batch([batch_CT, batch_CB],
                                                       [valid, valid,
                                                        batch_CT, batch_CB,
                                                        batch_CT, batch_CB,
                                                        batch_CT, batch_CB])
              
              return g_loss, d_loss.numpy()
              
        for epochi in range(self.epochs):
            # if self.use_data_generator:
                loop_index = 1
                
                # K.set_value(self.Gen_optimizer.learning_rate, learning_rates[epochi])
                # K.set_value(self.Disc_optimizer.learning_rate, learning_rates[epochi])
                self.data_generator=loadprintoutgen(self.trainCT_path,self.trainCB_path,self.batch_size,self.batch_set_size)
                print(self.data_generator.__len__())
                
                for images in self.data_generator:
                    batch_CT = images[0]
                    batch_CB = images[1]
                    # Run all training steps
                    g_loss, d_loss=run_training_iteration(loop_index, self.data_generator.__len__())
                    # os.system("nvidia-smi")
                    #print('Trainingstep')
                    
                    if loop_index >= self.data_generator.__len__():
                        break
                    loop_index = loop_index+1
                    
                    
                
                D_losses.append(d_loss)
                G_losses.append(g_loss)
                    
                if epochi % self.save_epoch_frequency == 0 and self.saveweightflag==True: # Weights saved based on epoch intervals
                    gen1fname1=gen1fname+'-'+str(epochi)+'.h5'    
                    gen2fname1=gen2fname+'-'+str(epochi)+'.h5'
                    disc1fname1=disc1fname+'-'+str(epochi)+'.h5'
                    disc2fname1=disc2fname+'-'+str(epochi)+'.h5'
                    self.GenCT2CB.save_weights(gen1fname1)
                    self.GenCB2CT.save_weights(gen2fname1)
                    self.DiscCT.save_weights(disc1fname1)
                    self.DiscCB.save_weights(disc2fname1)
        # os.system("nvidia-smi")
                print('Epoch =%s'%epochi)
        return D_losses,G_losses


def tfmsssim(img1,img2,max_val):
    score = tf.image.ssim_multiscale(img1, img2, max_val, filter_size=2, filter_sigma=1.5, k1=0.01, k2=0.03)
    return score
def custom_loss_3_gamma3D(y_true, y_pred):# MS-SSIM
    _MSSSIM_WEIGHTS=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    max_val1=tf.math.reduce_max(y_true)-tf.math.reduce_min(y_true)
    max_val2=tf.math.reduce_max(y_pred)-tf.math.reduce_min(y_pred)
    max_val =0.5*(max_val1+max_val2)
    filter_size=2
    filter_sigma=0.5
    batchsize=K.int_shape(y_pred)
    ssimscores=[]
    for batchelei in range(batchsize[0]):
        y_pred1=y_pred[batchelei,:,:,:,:]
        y_true1=y_true[batchelei,:,:,:,:]
        depthsize=K.int_shape(y_pred1)
        max_val1=tf.math.reduce_max(y_true1)-tf.math.reduce_min(y_true1)
        max_val2=tf.math.reduce_max(y_pred1)-tf.math.reduce_min(y_pred1)
        max_val =0.5*(max_val1+max_val2)
        for depthelei in range(depthsize[0]):
            y_pred2=y_pred1[depthelei,:,:,:]
            y_true2=y_true1[depthelei,:,:,:]
            # ssimscoreele=tfmsssim(y_true2, y_pred2, max_val,filter_size,filter_sigma)
            # ssimscoreele=tfssim(y_true2, y_pred2, max_val)
            # ssimscoreele=tfmsssim(y_true2, y_pred2, max_val)
            ssimscoreele=ssTF.tfmssim_custom(y_true2, y_pred2, max_val,_MSSSIM_WEIGHTS,filter_size,filter_sigma)
            # ssimscoreele=tf.image.ssim_multiscale(y_pred2, y_true2, max_val, filter_size=2, filter_sigma=0.5, k1=0.01, k2=0.03)
            ssimscores.append(ssimscoreele)
    ssimscore=tf.reduce_mean(ssimscores)
    # ssimscore=ssTF.tfmsssim(y_true, y_pred, max_val,filter_size,filter_sigma)
    # ssimscore=ssTF.tfmssim_custom(y_true, y_pred, max_val,_MSSSIM_WEIGHTS,filter_size,filter_sigma)
    loss=1-ssimscore
    return loss

class CycleGAN_3D_Gamma():  
#%% 2D     
    @staticmethod
    def conv2d(layer_input, filters, f_size=4,stride=2,normalization=True):
        """Discriminator layer"""
        d = layers.Conv2D(filters, kernel_size=f_size,strides=stride, padding='same',activation='relu')(layer_input)
        if normalization:
            d = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(d)
        d = layers.LeakyReLU(alpha=0.2)(d)
        return d
    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, stride=1, dropout_rate=0,skip=True):
          """Layers used during upsampling"""
          u = layers.UpSampling2D(size=2)(layer_input)
          u = layers.Conv2D(filters, kernel_size=f_size, strides=stride, padding='same', activation='tanh')(u)
          if dropout_rate:
              u = layers.Dropout(dropout_rate)(u)
          u = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(u)
          if skip:
              u = layers.Concatenate()([u, skip_input])
          return u
#%% 3D
    @staticmethod
    def convblk3d(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
            #                        center=True,
            #                        scale=True,
            #                        beta_initializer="random_uniform",
            #                        gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.LeakyReLU(alpha=0.2)(opL)
        return opL
   
    @staticmethod
    def convblk3d_ReLU(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
                                   # center=True,
                                   # scale=True,
                                   # beta_initializer="random_uniform",
                                   # gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.ReLU()(opL)
        return opL
    @staticmethod
    def convblk3d_valid(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='valid')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
            #                        center=True,
            #                        scale=True,
            #                        beta_initializer="random_uniform",
            #                        gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.ReLU()(opL)
        return opL
    
    @staticmethod
    def attentionblk3D(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=3)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def attentionblk3D_1(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=13)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def attentionblk3D_2(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=25)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def deconvblk3D(ipL,filters,kernel_size,strides):
        # opL=layers.UpSampling3D(size=2)(ipL)
        # opL=ReflectionPadding3D()(opL)
        # opL=layers.Conv3DTranspose(filters, kernel_size, strides=strides,padding='SAME')(ipL)
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        # opL=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(opL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.Activation('relu')(opL)
        return opL
    @staticmethod
    def upsample3D(ipL,filters,kernel_size,strides):
        # opL=layers.UpSampling3D(size=2)(ipL)
        # opL=ReflectionPadding3D()(opL)
        opL=layers.Conv3DTranspose(filters, kernel_size, strides=strides,padding='SAME')(ipL)
        # opL=layers.Conv3D(filters, kernel_size=1, strides=strides,padding='SAME')(ipL)
        # opL=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(opL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.Activation('relu')(opL)
        return opL
    @staticmethod
    def resblock(x,filters,kernelsize,stride):
        fx = layers.Conv3D(filters, kernelsize,strides=stride,padding='same')(x)
        # fx=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(fx)
        fx=layers.BatchNormalization()(fx)
        fx = layers.Activation('relu')(fx)
        fx = layers.Conv3D(filters, kernelsize,strides=stride, padding='same')(fx)
        # fx=tfa.layers.InstanceNormalization(axis=3,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(fx)
        fx=layers.BatchNormalization()(fx)
        out = layers.Add()([x,fx])
        return out
    
    def build_generator3D(self):
        ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
        opL = layers.Conv3D(self.genafilter, self.kernel_size_gen_1, padding='same')(ipL)
        # opL=ReflectionPadding3D()(opL)
        opL=self.deconvblk3D(opL,self.genafilter*2,self.kernel_size_gen_2,self.stride2)
        opL=self.deconvblk3D(opL,self.genafilter*4,self.kernel_size_gen_2,self.stride2)
        for _ in range(5):
            opL=self.resblock(opL,self.genafilter*4,self.kernel_size_gen_2,self.stride1)
        opL=self.upsample3D(opL,self.genafilter*2,self.kernel_size_gen_2,self.stride2)
        opL=self.upsample3D(opL,self.genafilter,self.kernel_size_gen_2,self.stride2)
        opL = layers.Conv3D(1, self.kernel_size_gen_1,self.stride1, padding='same')(opL)
        # opL = layers.Conv3D(1, self.kernel_size_gen_1,self.stride1, padding='same')(opL)
       
       
        return keras.Model(ipL,opL)
   
    def build_discriminator3D(self):
        ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
        opL1=self.convblk3d(ipL,self.discfilter,self.kernel_size_disc,self.stride2,normalization=False)    
        opL2=self.convblk3d(opL1,self.discfilter*2,self.kernel_size_disc,self.stride2,normalization=True)
        opL3=self.convblk3d(opL2,self.discfilter*4,self.kernel_size_disc,self.stride1,normalization=True)
        opL4=self.convblk3d(opL3,self.discfilter*8,self.kernel_size_disc,self.stride1,normalization=True)
        opL5 = layers.Conv3D(1, kernel_size=4, strides=1, padding='same',activation='sigmoid')(opL4)
        # opL6 = layers.Flatten()(opL5)
        # opL7 = layers.Dense(1, activation='Sigmoid')(opL6)
        return keras.Model(ipL,opL5)
    
    # def build_generator3D(self):
    #     ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
    #     opL1=self.convblk3d(ipL,self.genafilter,self.kernel_size,self.stride2)
    #     opL2=self.convblk3d(opL1,self.genafilter*2,self.kernel_size,self.stride2)
    #     opL3=self.convblk3d(opL2,self.genafilter*4,self.kernel_size,self.stride2)
    #     opL4=layers.Conv3D(filters=self.genafilter*8, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL3)
    #     # opL4=self.convblk3d(opL3,self.genafilter*8,self.kernel_size,self.stride2)
        
    #     opL5=self.attentionblk3D(opL3,opL4,filters=self.genafilter*8, kernel_size=self.kernel_size, strides=self.stride2)
        
    #     opL6=layers.Concatenate()([opL4,opL5])
    #     opL7=self.deconvblk3D(opL6,self.genafilter*4,self.kernel_size,self.stride1)
    #     opL8=self.deconvblk3D(opL7,self.genafilter*2,self.kernel_size,self.stride1)
    #     # opL9=self.convblk3d(opL8,self.genafilter*4,self.kernel_size,self.stride2)
    #     opL9=layers.Conv3D(filters=self.genafilter*4, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL8)
        
    #     opL10=self.attentionblk3D_1(opL1,opL9,filters=self.genafilter*4, kernel_size=self.kernel_size, strides=self.stride2)
        
    #     opL11=layers.Concatenate()([opL9,opL10])
    #     opL12=self.deconvblk3D(opL11,self.genafilter*2,self.kernel_size,self.stride1)
    #     opL13=self.deconvblk3D(opL12,self.genafilter*4,self.kernel_size,self.stride1)
    #     # opL14=self.convblk3d(opL13,self.genafilter*2,self.kernel_size,self.stride2)
    #     opL14=layers.Conv3D(filters=self.genafilter*2, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL13)
        
    #     opL15=self.attentionblk3D_2(ipL,opL14,filters=self.genafilter*2, kernel_size=self.kernel_size, strides=self.stride2)
        
    #     opL16=layers.Concatenate()([opL14,opL15])
    #     opL17=self.deconvblk3D(opL16,self.genafilter*1,self.kernel_size,self.stride1)
    #     opL18=self.deconvblk3D(opL17,self.genafilter*2,self.kernel_size,self.stride1)
    #     # opL19=self.convblk3d(opL18,self.genafilter*1,self.kernel_size,self.stride1)
    #     opL19=layers.Conv3D(filters=self.genafilter*1, kernel_size=self.kernel_size, strides=self.stride1,padding='SAME')(opL18)
        
    #     opL20=self.resblock(opL19, self.genafilter*1, self.kernel_size)
    #     opL21=self.resblock(opL20, self.genafilter*1, self.kernel_size)
    #     opL22=layers.Conv3D(filters=1, kernel_size=self.kernel_size, strides=self.stride1,padding='SAME')(opL21)
        
    #     return keras.Model(ipL,opL22)
    
    # def build_discriminator3D(self):
    #     ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
    #     opL1=self.convblk3d(ipL,self.discfilter,self.kernel_size_disc,self.stride2)
    #     opL2=self.convblk3d(opL1,self.discfilter*2,self.kernel_size_disc,self.stride2)
    #     opL3=self.convblk3d(opL2,self.discfilter*4,self.kernel_size_disc,self.stride2)
    #     opL4=self.convblk3d(opL3,self.discfilter*8,self.kernel_size_disc,self.stride1)
        
    #     # opL5=layers.Flatten()(opL4)
        
    #     # opL5=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL5)
    #     # opL5=layers.LeakyReLU()(opL5)
        
    #     # opL6=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*16)(opL5)
    #     # opL6=layers.LeakyReLU()(opL6)

    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL6)
    #     # opL8=layers.Activation('sigmoid')(opL7)

    #     # opL9=layers.Dense(1)(opL8)
    #     # opL=layers.Activation('sigmoid')(opL9)
        
        
    #     opL5=layers.Conv3D(1, kernel_size=self.kernel_size_disc, strides=self.stride1,padding='SAME')(opL4)
        
    #     opL7 = layers.Flatten()(opL5)
        
    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL5)
    #     # opL7=layers.LeakyReLU()(opL7)
        
    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*16)(opL7)
    #     # opL7=layers.LeakyReLU()(opL7)
        
    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*8))(opL7)
    #     # opL7=layers.LeakyReLU()(opL7)
        
    #     # opL7=layers.Conv3D(1, kernel_size=self.kernel_size_disc, strides=self.stride1,padding='SAME')(opL6)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.5))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.25))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.125))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.0625))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7 = layers.Dense(1)(opL7)
    #     opL = layers.Activation('relu')(opL7)
        
    #     return keras.Model(ipL,opL)
    
    def build_discriminator(self):
        d0 = keras.Input(shape=self.img_shape,name='Input')
        d1 = self.conv2d(d0, self.discfilter, stride=2, normalization=False)
        d2 = self.conv2d(d1, self.discfilter*2, stride=2, normalization=True)
        d3 = self.conv2d(d2, self.discfilter*4, stride=2, normalization=True)
        d4 = self.conv2d(d3, self.discfilter*8, stride=2, normalization=True)
        d5 = self.conv2d(d4, self.discfilter*8, stride=1, normalization=True)
        d6 = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)
        d7 = layers.Activation('relu')(d6)
        
        return keras.Model(d0,d7)
    
    def build_generator(self):
        d0 = keras.Input(shape=self.img_shape,name='Input')
        d1 = self.conv2d(d0, self.genafilter,stride=1,normalization=True)
        d2 = self.conv2d(d1, self.genafilter,stride=2,normalization=True)
        d3 = self.conv2d(d2, self.genafilter*2,stride=1,normalization=True)
        d4 = self.conv2d(d3, self.genafilter*2,stride=2,normalization=True)
        d5 = self.conv2d(d4, self.genafilter*4,stride=1,normalization=True)
        d6 = self.conv2d(d5, self.genafilter*4,stride=2,normalization=True)
        d7 = self.conv2d(d6, self.genafilter*8,stride=1,normalization=True)
        d8 = self.conv2d(d7, self.genafilter*8,stride=2,normalization=True)
        d9 = self.conv2d(d8, self.genafilter*16,stride=1,normalization=True)
        d10 = self.conv2d(d9, self.genafilter*16,stride=2,normalization=True)
        
        u10 = self.deconv2d(d10, d8, self.genafilter*8,stride=1)
        u9 = self.conv2d(u10, self.genafilter*8,stride=1,normalization=True)
        u8 = self.deconv2d(u9, d6, self.genafilter*4,stride=1)
        u7 = self.conv2d(u8, self.genafilter*4,stride=1,normalization=True)
        u6 = self.deconv2d(u7, d4, self.genafilter*2,stride=1)
        u5 = self.conv2d(u6, self.genafilter*2,stride=1,normalization=True)
        u4 = self.deconv2d(u5, d2, self.genafilter,stride=1)
        u3 = self.conv2d(u4, self.genafilter,stride=1,normalization=True)
        u2 = self.deconv2d(u3,d1, self.genafilter,stride=1,skip=False)
        u1 = layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(u2)
        u1 = layers.Activation('tanh')(u1)
        
        return keras.Model(d0,u1)
    
    def __init__(self,mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag):
          self.DataPath=mypath
          #self.breakflag=breakflag
          self.WeightSavePath=weightoutputpath
          self.batch_size=batch_size
          self.img_shape=imgshape
          self.input_shape=tuple([batch_size,imgshape])
          self.genafilter = 16
          self.discfilter = 16
          self.epochs = epochs+1
          self.save_epoch_frequency=save_epoch_frequency
          self.batch_set_size=batch_set_size
          self.lambda_cycle = 10.0                    # Cycle-consistency loss
          self.lambda_id = 0.9 * self.lambda_cycle    # Identity loss
          self.saveweightflag=saveweightflag
          self.patch_size=32
          self.depth_size=32
          self.input_layer_shape_3D=tuple([self.patch_size,self.patch_size,self.depth_size,1])
          self.stride2=2
          self.stride1=1
          self.kernel_size = 3
          self.kernel_size_disc = 4
          self.kernel_size_gen_1=7
          self.kernel_size_gen_2=3
         
          os.chdir(self.WeightSavePath)
          self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))
          os.mkdir(self.folderlen)
          self.WeightSavePath = os.path.join(self.WeightSavePath, self.folderlen)
          os.chdir(self.WeightSavePath)
         
          newdir='arch'
         
          self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
          os.mkdir(self.WeightSavePathNew)
          os.chdir(self.WeightSavePathNew)
          
          self.Disc_lr=0.0002
          self.Gen_lr=0.0001
          
          # self.Disc_lr=0.001
          # self.Gen_lr=0.01
         
          self.Disc_optimizer = keras.optimizers.Adam(self.Disc_lr, 0.5,0.999)
          self.Gen_optimizer = keras.optimizers.Adam(self.Gen_lr, 0.5,0.999)
         
          # optimizer = keras.optimizers.Adam(0.0002, 0.5)

          self.DiscCT=self.build_discriminator3D()
          self.DiscCT.compile(loss='mse', optimizer=self.Disc_optimizer, metrics=['accuracy'])
          self.DiscCT._name='Discriminator-CT'
          # self.DiscCT.summary()
          with open('Disc.txt', 'w+') as f:
              self.DiscCT.summary(print_fn=lambda x: f.write(x + '\n'))
         
          self.DiscCB=self.build_discriminator3D()
          self.DiscCB.compile(loss='mse', optimizer=self.Disc_optimizer, metrics=['accuracy'])
          self.DiscCB._name='Discriminator-CB'
         
          self.DiscCT_static=self.DiscCT
          self.DiscCB_static=self.DiscCB
          
          layer_len=len(self.DiscCT.layers)
          layers_lis=self.DiscCT.layers
          labelshapearr=list(layers_lis[layer_len-1].output_shape)
          labelshapearr[0]=self.batch_size
          labelshape=tuple(labelshapearr)
          self.labelshape=labelshape
        
          self.GenCB2CT=self.build_generator3D()
          self.GenCB2CT.compile(loss='mse', optimizer=self.Gen_optimizer, metrics=['accuracy'])
          self.GenCB2CT._name='Generator-CB2CT'
         
          
          # self.GenCB2CT.summary()
          with open('Gena.txt', 'w+') as f:
              self.GenCB2CT.summary(print_fn=lambda x: f.write(x + '\n'))
         
          self.GenCT2CB=self.build_generator3D()
          self.GenCT2CB.compile(loss='mse', optimizer=self.Gen_optimizer, metrics=['accuracy'])
          self.GenCT2CB._name='Generator-CT2CB'
          
          img_CT = keras.Input(shape=self.input_layer_shape_3D)
          img_CB = keras.Input(shape=self.input_layer_shape_3D)
          
          valid_CT = self.DiscCT(img_CT)
          valid_CB = self.DiscCB(img_CB)
          
          # self.DiscCT_static = Network(inputs=img_CT, outputs=guess_A, name='D_A_static_model')
          # self.DiscCB_static = Network(inputs=img_CB, outputs=guess_B, name='D_B_static_model')
          # self.DiscCT_static = clone_model(self.DiscCT)
          # self.DiscCB_static = clone_model(self.DiscCB)
          self.DiscCT_static.set_weights(self.DiscCT.get_weights())
          self.DiscCB_static.set_weights(self.DiscCB.get_weights())
          # self.DiscCT_static = clone_model(self.DiscCT)
          # self.DiscCB_static = clone_model(self.DiscCB)
        
        # For the combined model we will only train the generators
          # self.DiscCT_static.trainable = False
          # self.DiscCB_static.trainable = False
         
          # Input images from both domains

        
        # Translate images to the other domain
          fake_CB = self.GenCT2CB(img_CT)
          fake_CT = self.GenCB2CT(img_CB)
        # Translate images back to original domain
          reconstr_CT = self.GenCB2CT(fake_CB)
          reconstr_CB = self.GenCT2CB(fake_CT)
        # Identity mapping of images
          img_CT_id = self.GenCT2CB(img_CT)
          img_CB_id = self.GenCB2CT(img_CB)
          
        #   self.DiscCT_static = Network(inputs=img_CT, outputs=guess_A, name='D_A_static_model')
        #   self.DiscCB_static = Network(inputs=img_CB, outputs=guess_B, name='D_B_static_model')
        
        # # For the combined model we will only train the generators
        #   self.DiscCT_static.trainable = False
        #   self.DiscCB_static.trainable = False
        
        # Discriminators determines validity of translated images
          valid_CT = self.DiscCT_static(fake_CT)
          valid_CB = self.DiscCB_static(fake_CB)
        
        # Combined model trains generators to fool discriminators
          self.cycleGAN_Model = keras.Model(inputs=[img_CT, img_CB], outputs=[valid_CT, valid_CB, reconstr_CT, reconstr_CB, img_CT_id, img_CB_id,reconstr_CT, reconstr_CB])
          self.cycleGAN_Model.compile(loss=['mse', 'mse', 'mae', 'mae','mae', 'mae', custom_loss_3_gamma3D,custom_loss_3_gamma3D],
                                     loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id,10,10], 
                                     optimizer=self.Gen_optimizer)
          self.cycleGAN_Model._name='CycleGAN'
         
          with open('cycleGAN.txt', 'w+') as f:
              self.cycleGAN_Model.summary(print_fn=lambda x: f.write(x + '\n'))
          
          self.trainCT_path = os.path.join(self.DataPath, 'trainCT')
          self.trainCB_path = os.path.join(self.DataPath, 'trainCB')
          # testCT_path = os.path.join(self.DataPath, 'validCT')
          # testCB_path = os.path.join(self.DataPath, 'validCB')
          # self.data_generator=loadprintoutgen(trainCT_path,trainCB_path,self.batch_size,self.batch_set_size)
          
          # for images in self.data_generator:
          #     batch_CT = images[0]
          #     batch_CB = images[1]
              
          #     print(batch_CT.shape)
          # print(self.data_generator.__len__())
          os.system("nvidia-smi")
          print('Init')
          
    def learningrate_log_scheduler(self):
        learning_rates=np.logspace(-8, 1,num=self.epochs)
        return learning_rates
    
    def traincgan(self):
        os.chdir(self.WeightSavePathNew)
         # self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))         
        newdir='weights'        
        self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
        os.mkdir(self.WeightSavePathNew)
        os.chdir(self.WeightSavePathNew)
        
        gen1fname="GenCT2CBWeights"
        gen2fname="GenCB2CTWeights"
        disc1fname="DiscCTWeights"
        disc2fname="DiscCBWeights"
        # D_losses = np.zeros((self.batch_set_size,2,self.epochs))
        # G_losses = np.zeros((self.batch_set_size,7,self.epochs))
        D_losses = []
        G_losses = []
        
        #Learning rate schedule
        # learning_rates=self.learningrate_log_scheduler()
        epoch_start=0
        if self.breakflag:
            lastweightpath=self.lastweightpath
            lst=os.listdir(lastweightpath)
            weightfilename=lst[-1]
            x=weightfilename.split(".")
            x=x[0].split("-")
            epochi=int(x[-1])
            gen1fname1_break="GenCT2CBWeights"+'-'+str(epochi)+'.h5'
            gen2fname1_break="GenCB2CTWeights"+'-'+str(epochi)+'.h5'
            disc1fname1_break="DiscCTWeights"+'-'+str(epochi)+'.h5'
            disc2fname1_break="DiscCBWeights"+'-'+str(epochi)+'.h5'
            self.GenCT2CB.load_weights(os.path.join(lastweightpath,gen1fname1_break))
            self.GenCB2CT.load_weights(os.path.join(lastweightpath,gen2fname1_break))
            self.DiscCT.load_weights(os.path.join(lastweightpath,disc1fname1_break))
            self.DiscCB.load_weights(os.path.join(lastweightpath,disc2fname1_break))
            epoch_start=epochi
        
        def run_training_iteration(loop_index, epoch_iterations):
              valid = tf.ones((self.labelshape))
              fake = tf.zeros((self.labelshape))
                 # ----------------------
                 #  Train Discriminators
                 # ----------------------

                 # Translate images to opposite domain
              fake_CB = self.GenCT2CB.predict(batch_CT)
              fake_CT = self.GenCB2CT.predict(batch_CB)

                 # Train the discriminators (original images = real / translated = Fake)
              dCT_loss_real = self.DiscCT.train_on_batch(batch_CT, valid)
              dCT_loss_fake = self.DiscCT.train_on_batch(fake_CT, fake)
              dCT_loss = 0.5 * tf.math.add(dCT_loss_real, dCT_loss_fake)

              dCB_loss_real = self.DiscCB.train_on_batch(batch_CB, valid)
              dCB_loss_fake = self.DiscCB.train_on_batch(fake_CB, fake)
              dCB_loss = 0.5 * tf.math.add(dCB_loss_real, dCB_loss_fake)

                 # Total discriminator loss
              d_loss = 0.5 * tf.math.add(dCT_loss, dCB_loss)

                 # ------------------
                 #  Train Generators
                 # ------------------

                 # Train the generators
              g_loss = self.cycleGAN_Model.train_on_batch([batch_CT, batch_CB],
                                                       [valid, valid,
                                                        batch_CT, batch_CB,
                                                        batch_CT, batch_CB,
                                                        batch_CT, batch_CB])
              
              return g_loss, d_loss.numpy()
              
        for epochi in range(self.epochs):
            # if self.use_data_generator:
                loop_index = 1
                
                # K.set_value(self.Gen_optimizer.learning_rate, learning_rates[epochi])
                # K.set_value(self.Disc_optimizer.learning_rate, learning_rates[epochi])
                self.data_generator=loadprintoutgen(self.trainCT_path,self.trainCB_path,self.batch_size,self.batch_set_size)
                print(self.data_generator.__len__())
                
                for images in self.data_generator:
                    batch_CT = images[0]
                    batch_CB = images[1]
                    # Run all training steps
                    g_loss, d_loss=run_training_iteration(loop_index, self.data_generator.__len__())
                    # os.system("nvidia-smi")
                    #print('Trainingstep')
                    
                    if loop_index >= self.data_generator.__len__():
                        break
                    loop_index = loop_index+1
                    
                    
                
                D_losses.append(d_loss)
                G_losses.append(g_loss)
                    
                if epochi % self.save_epoch_frequency == 0 and self.saveweightflag==True: # Weights saved based on epoch intervals
                    gen1fname1=gen1fname+'-'+str(epochi)+'.h5'    
                    gen2fname1=gen2fname+'-'+str(epochi)+'.h5'
                    disc1fname1=disc1fname+'-'+str(epochi)+'.h5'
                    disc2fname1=disc2fname+'-'+str(epochi)+'.h5'
                    self.GenCT2CB.save_weights(gen1fname1)
                    self.GenCB2CT.save_weights(gen2fname1)
                    self.DiscCT.save_weights(disc1fname1)
                    self.DiscCB.save_weights(disc2fname1)
        # os.system("nvidia-smi")
                print('Epoch =%s'%epochi)
        return D_losses,G_losses

def custom_loss_5_epsilon3D(y_true, y_pred):# MS-SSIM
    _MSSSIM_WEIGHTS=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    max_val1=tf.math.reduce_max(y_true)-tf.math.reduce_min(y_true)
    max_val2=tf.math.reduce_max(y_pred)-tf.math.reduce_min(y_pred)
    max_val =0.5*(max_val1+max_val2)
    filter_size=2
    filter_sigma=0.5
    batchsize=K.int_shape(y_pred)
    ssimscores=[]
    for batchelei in range(batchsize[0]):
        y_pred1=y_pred[batchelei,:,:,:,:]
        y_true1=y_true[batchelei,:,:,:,:]
        depthsize=K.int_shape(y_pred1)
        max_val1=tf.math.reduce_max(y_true1)-tf.math.reduce_min(y_true1)
        max_val2=tf.math.reduce_max(y_pred1)-tf.math.reduce_min(y_pred1)
        max_val =0.5*(max_val1+max_val2)
        for depthelei in range(depthsize[0]):
            y_pred2=y_pred1[depthelei,:,:,:]
            y_true2=y_true1[depthelei,:,:,:]
            y_pred2=tf.expand_dims(y_pred2,axis=0)
            y_true2=tf.expand_dims(y_true2,axis=0)
            # ssimscoreele=tfmsssim(y_true2, y_pred2, max_val,filter_size,filter_sigma)
            # ssimscoreele=tfssim(y_true2, y_pred2, max_val)
            ssimscoreele,_=ssTF.tfssim4cg(y_true2, y_pred2, max_val,filter_size,filter_sigma)
            # ssimscoreele=ssTF.tfmssim_custom(y_true, y_pred, max_val,_MSSSIM_WEIGHTS,filter_size,filter_sigma)
            # ssimscoreele=tf.image.ssim_multiscale(y_pred2, y_true2, max_val, filter_size=2, filter_sigma=0.5, k1=0.01, k2=0.03)
            ssimscores.append(ssimscoreele)
    ssimscore=tf.reduce_mean(ssimscores)
    # ssimscore=ssTF.tfmsssim(y_true, y_pred, max_val,filter_size,filter_sigma)
    # ssimscore=ssTF.tfmssim_custom(y_true, y_pred, max_val,_MSSSIM_WEIGHTS,filter_size,filter_sigma)
    loss=1-ssimscore
    return loss
#%%

class CycleGAN_3D_Epsilon():
     
  
#%% 2D     
    @staticmethod
    def conv2d(layer_input, filters, f_size=4,stride=2,normalization=True):
        """Discriminator layer"""
        d = layers.Conv2D(filters, kernel_size=f_size,strides=stride, padding='same',activation='relu')(layer_input)
        if normalization:
            d = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(d)
        d = layers.LeakyReLU(alpha=0.2)(d)
        return d
    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, stride=1, dropout_rate=0,skip=True):
          """Layers used during upsampling"""
          u = layers.UpSampling2D(size=2)(layer_input)
          u = layers.Conv2D(filters, kernel_size=f_size, strides=stride, padding='same', activation='tanh')(u)
          if dropout_rate:
              u = layers.Dropout(dropout_rate)(u)
          u = tfa.layers.InstanceNormalization(axis=3, 
                                   center=True, 
                                   scale=True,
                                   beta_initializer="random_uniform",
                                   gamma_initializer="random_uniform")(u)
          if skip:
              u = layers.Concatenate()([u, skip_input])
          return u
#%% 3D
    @staticmethod
    def convblk3d(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
            #                        center=True,
            #                        scale=True,
            #                        beta_initializer="random_uniform",
            #                        gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.LeakyReLU(alpha=0.2)(opL)
        return opL
   
    @staticmethod
    def convblk3d_ReLU(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
                                   # center=True,
                                   # scale=True,
                                   # beta_initializer="random_uniform",
                                   # gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.ReLU()(opL)
        return opL
    @staticmethod
    def convblk3d_valid(ipL,filters,kernel_size,strides,normalization):
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='valid')(ipL)
        if normalization:
            # opL=tfa.layers.InstanceNormalization(axis=3,
            #                        center=True,
            #                        scale=True,
            #                        beta_initializer="random_uniform",
            #                        gamma_initializer="random_uniform")(opL)
            opL=layers.BatchNormalization()(opL)
        opL=layers.ReLU()(opL)
        return opL
    
    @staticmethod
    def attentionblk3D(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=3)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def attentionblk3D_1(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=13)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def attentionblk3D_2(x,gating,filters,kernel_size,strides):
        gating_op=layers.Conv3D(filters, kernel_size=1)(gating)
        x_op=layers.Conv3D(filters, kernel_size=25)(x)
        net=layers.add([x_op,gating_op])
        net=layers.Activation('relu')(net)
        net=layers.Conv3D(filters, kernel_size=1)(net)
        net=layers.Activation('sigmoid')(net)
        # net=layers.UpSampling3D(size=2)(net)
        net=layers.multiply([net,gating])
        return net
    
    @staticmethod
    def deconvblk3D(ipL,filters,kernel_size,strides):
        # opL=layers.UpSampling3D(size=2)(ipL)
        # opL=ReflectionPadding3D()(opL)
        # opL=layers.Conv3DTranspose(filters, kernel_size, strides=strides,padding='SAME')(ipL)
        opL=layers.Conv3D(filters, kernel_size=kernel_size, strides=strides,padding='SAME')(ipL)
        # opL=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(opL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.Activation('relu')(opL)
        return opL
    @staticmethod
    def upsample3D(ipL,filters,kernel_size,strides):
        # opL=layers.UpSampling3D(size=2)(ipL)
        # opL=ReflectionPadding3D()(opL)
        opL=layers.Conv3DTranspose(filters, kernel_size, strides=strides,padding='SAME')(ipL)
        # opL=layers.Conv3D(filters, kernel_size=1, strides=strides,padding='SAME')(ipL)
        # opL=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(opL)
        opL=layers.BatchNormalization()(opL)
        opL=layers.Activation('relu')(opL)
        return opL
    @staticmethod
    def resblock(x,filters,kernelsize,stride):
        fx = layers.Conv3D(filters, kernelsize,strides=stride,padding='same')(x)
        # fx=tfa.layers.InstanceNormalization(axis=-1,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(fx)
        fx=layers.BatchNormalization()(fx)
        fx = layers.Activation('relu')(fx)
        fx = layers.Conv3D(filters, kernelsize,strides=stride, padding='same')(fx)
        # fx=tfa.layers.InstanceNormalization(axis=3,
        #                            center=True,
        #                            scale=True,
        #                            beta_initializer="random_uniform",
        #                            gamma_initializer="random_uniform")(fx)
        fx=layers.BatchNormalization()(fx)
        out = layers.Add()([x,fx])
        return out
    # def build_generator3D(self):
    #     ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
    #     opL1=self.convblk3d(ipL,self.genafilter,self.kernel_size,self.stride2)
    #     opL2=self.convblk3d(opL1,self.genafilter*2,self.kernel_size,self.stride2)
    #     opL3=self.convblk3d(opL2,self.genafilter*4,self.kernel_size,self.stride2)
    #     opL4=layers.Conv3D(filters=self.genafilter*8, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL3)
    #     # opL4=self.convblk3d(opL3,self.genafilter*8,self.kernel_size,self.stride2)
        
    #     opL5=self.attentionblk3D(opL3,opL4,filters=self.genafilter*8, kernel_size=self.kernel_size, strides=self.stride2)
        
    #     opL6=layers.Concatenate()([opL4,opL5])
    #     opL7=self.deconvblk3D(opL6,self.genafilter*4,self.kernel_size,self.stride1)
    #     opL8=self.deconvblk3D(opL7,self.genafilter*2,self.kernel_size,self.stride1)
    #     # opL9=self.convblk3d(opL8,self.genafilter*4,self.kernel_size,self.stride2)
    #     opL9=layers.Conv3D(filters=self.genafilter*4, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL8)
        
    #     opL10=self.attentionblk3D_1(opL1,opL9,filters=self.genafilter*4, kernel_size=self.kernel_size, strides=self.stride2)
        
    #     opL11=layers.Concatenate()([opL9,opL10])
    #     opL12=self.deconvblk3D(opL11,self.genafilter*2,self.kernel_size,self.stride1)
    #     opL13=self.deconvblk3D(opL12,self.genafilter*4,self.kernel_size,self.stride1)
    #     # opL14=self.convblk3d(opL13,self.genafilter*2,self.kernel_size,self.stride2)
    #     opL14=layers.Conv3D(filters=self.genafilter*2, kernel_size=self.kernel_size, strides=self.stride2,padding='SAME')(opL13)
        
    #     opL15=self.attentionblk3D_2(ipL,opL14,filters=self.genafilter*2, kernel_size=self.kernel_size, strides=self.stride2)
        
    #     opL16=layers.Concatenate()([opL14,opL15])
    #     opL17=self.deconvblk3D(opL16,self.genafilter*1,self.kernel_size,self.stride1)
    #     opL18=self.deconvblk3D(opL17,self.genafilter*2,self.kernel_size,self.stride1)
    #     # opL19=self.convblk3d(opL18,self.genafilter*1,self.kernel_size,self.stride1)
    #     opL19=layers.Conv3D(filters=self.genafilter*1, kernel_size=self.kernel_size, strides=self.stride1,padding='SAME')(opL18)
        
    #     opL20=self.resblock(opL19, self.genafilter*1, self.kernel_size)
    #     opL21=self.resblock(opL20, self.genafilter*1, self.kernel_size)
    #     opL22=layers.Conv3D(filters=1, kernel_size=self.kernel_size, strides=self.stride1,padding='SAME')(opL21)
        
    #     return keras.Model(ipL,opL22)
    
    # def build_discriminator3D(self):
    #     ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
    #     opL1=self.convblk3d(ipL,self.discfilter,self.kernel_size_disc,self.stride2)
    #     opL2=self.convblk3d(opL1,self.discfilter*2,self.kernel_size_disc,self.stride2)
    #     opL3=self.convblk3d(opL2,self.discfilter*4,self.kernel_size_disc,self.stride2)
    #     opL4=self.convblk3d(opL3,self.discfilter*8,self.kernel_size_disc,self.stride1)
        
    #     # opL5=layers.Flatten()(opL4)
        
    #     # opL5=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL5)
    #     # opL5=layers.LeakyReLU()(opL5)
        
    #     # opL6=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*16)(opL5)
    #     # opL6=layers.LeakyReLU()(opL6)

    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL6)
    #     # opL8=layers.Activation('sigmoid')(opL7)

    #     # opL9=layers.Dense(1)(opL8)
    #     # opL=layers.Activation('sigmoid')(opL9)
        
        
    #     opL5=layers.Conv3D(1, kernel_size=self.kernel_size_disc, strides=self.stride1,padding='SAME')(opL4)
        
    #     opL7 = layers.Flatten()(opL5)
        
    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*8)(opL5)
    #     # opL7=layers.LeakyReLU()(opL7)
        
    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*self.discfilter*16)(opL7)
    #     # opL7=layers.LeakyReLU()(opL7)
        
    #     # opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*8))(opL7)
    #     # opL7=layers.LeakyReLU()(opL7)
        
    #     # opL7=layers.Conv3D(1, kernel_size=self.kernel_size_disc, strides=self.stride1,padding='SAME')(opL6)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.5))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.25))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.125))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7=layers.Dense(self.kernel_size_disc*self.kernel_size_disc*self.kernel_size_disc*(self.discfilter*0.0625))(opL7)
    #     opL7=layers.LeakyReLU()(opL7)
        
    #     opL7 = layers.Dense(1)(opL7)
    #     opL = layers.Activation('relu')(opL7)
        
    #     return keras.Model(ipL,opL)
    
    def build_generator3D(self):
        ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
        opL = layers.Conv3D(self.genafilter, self.kernel_size_gen_1, padding='same')(ipL)
        # opL=ReflectionPadding3D()(opL)
        opL=self.deconvblk3D(opL,self.genafilter*2,self.kernel_size_gen_2,self.stride2)
        opL=self.deconvblk3D(opL,self.genafilter*4,self.kernel_size_gen_2,self.stride2)
        for _ in range(5):
            opL=self.resblock(opL,self.genafilter*4,self.kernel_size_gen_2,self.stride1)
        opL=self.upsample3D(opL,self.genafilter*2,self.kernel_size_gen_2,self.stride2)
        opL=self.upsample3D(opL,self.genafilter,self.kernel_size_gen_2,self.stride2)
        opL = layers.Conv3D(1, self.kernel_size_gen_1,self.stride1, padding='same')(opL)
        # opL = layers.Conv3D(1, self.kernel_size_gen_1,self.stride1, padding='same')(opL)
       
       
        return keras.Model(ipL,opL)
   
    def build_discriminator3D(self):
        ipL=keras.Input(shape=self.input_layer_shape_3D,name='Input')
        opL1=self.convblk3d(ipL,self.discfilter,self.kernel_size_disc,self.stride2,normalization=False)    
        opL2=self.convblk3d(opL1,self.discfilter*2,self.kernel_size_disc,self.stride2,normalization=True)
        opL3=self.convblk3d(opL2,self.discfilter*4,self.kernel_size_disc,self.stride1,normalization=True)
        opL4=self.convblk3d(opL3,self.discfilter*8,self.kernel_size_disc,self.stride1,normalization=True)
        opL5 = layers.Conv3D(1, kernel_size=4, strides=1, padding='same',activation='sigmoid')(opL4)
        # opL6 = layers.Flatten()(opL5)
        # opL7 = layers.Dense(1, activation='Sigmoid')(opL6)
        return keras.Model(ipL,opL5)
    
    def build_discriminator(self):
        d0 = keras.Input(shape=self.img_shape,name='Input')
        d1 = self.conv2d(d0, self.discfilter, stride=2, normalization=False)
        d2 = self.conv2d(d1, self.discfilter*2, stride=2, normalization=True)
        d3 = self.conv2d(d2, self.discfilter*4, stride=2, normalization=True)
        d4 = self.conv2d(d3, self.discfilter*8, stride=2, normalization=True)
        d5 = self.conv2d(d4, self.discfilter*8, stride=1, normalization=True)
        d6 = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d5)
        d7 = layers.Activation('relu')(d6)
        
        return keras.Model(d0,d7)
    
    def build_generator(self):
        d0 = keras.Input(shape=self.img_shape,name='Input')
        d1 = self.conv2d(d0, self.genafilter,stride=1,normalization=True)
        d2 = self.conv2d(d1, self.genafilter,stride=2,normalization=True)
        d3 = self.conv2d(d2, self.genafilter*2,stride=1,normalization=True)
        d4 = self.conv2d(d3, self.genafilter*2,stride=2,normalization=True)
        d5 = self.conv2d(d4, self.genafilter*4,stride=1,normalization=True)
        d6 = self.conv2d(d5, self.genafilter*4,stride=2,normalization=True)
        d7 = self.conv2d(d6, self.genafilter*8,stride=1,normalization=True)
        d8 = self.conv2d(d7, self.genafilter*8,stride=2,normalization=True)
        d9 = self.conv2d(d8, self.genafilter*16,stride=1,normalization=True)
        d10 = self.conv2d(d9, self.genafilter*16,stride=2,normalization=True)
        
        u10 = self.deconv2d(d10, d8, self.genafilter*8,stride=1)
        u9 = self.conv2d(u10, self.genafilter*8,stride=1,normalization=True)
        u8 = self.deconv2d(u9, d6, self.genafilter*4,stride=1)
        u7 = self.conv2d(u8, self.genafilter*4,stride=1,normalization=True)
        u6 = self.deconv2d(u7, d4, self.genafilter*2,stride=1)
        u5 = self.conv2d(u6, self.genafilter*2,stride=1,normalization=True)
        u4 = self.deconv2d(u5, d2, self.genafilter,stride=1)
        u3 = self.conv2d(u4, self.genafilter,stride=1,normalization=True)
        u2 = self.deconv2d(u3,d1, self.genafilter,stride=1,skip=False)
        u1 = layers.Conv2D(1, kernel_size=1, strides=1, padding='same')(u2)
        u1 = layers.Activation('tanh')(u1)
        
        return keras.Model(d0,u1)
    
    def __init__(self,mypath,weightoutputpath,epochs,save_epoch_frequency,batch_size,imgshape,batch_set_size,saveweightflag):
          self.DataPath=mypath
          # self.breakflag=breakflag
          self.WeightSavePath=weightoutputpath
          # self.lastweightpath=lastweightpath
          self.batch_size=batch_size
          self.img_shape=imgshape
          self.input_shape=tuple([batch_size,imgshape])
          self.genafilter = 16
          self.discfilter = 16
          self.epochs = epochs+1
          self.save_epoch_frequency=save_epoch_frequency
          self.batch_set_size=batch_set_size
          self.lambda_cycle = 10.0                    # Cycle-consistency loss
          self.lambda_id = 0.9 * self.lambda_cycle    # Identity loss
          self.saveweightflag=saveweightflag
          self.patch_size=32
          self.depth_size=32
          self.input_layer_shape_3D=tuple([self.patch_size,self.patch_size,self.depth_size,1])
          self.stride2=2
          self.stride1=1
          self.kernel_size = 3
          self.kernel_size_disc = 4
          self.kernel_size_gen_1=7
          self.kernel_size_gen_2=3
         
          os.chdir(self.WeightSavePath)
          self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))
          os.mkdir(self.folderlen)
          self.WeightSavePath = os.path.join(self.WeightSavePath, self.folderlen)
          os.chdir(self.WeightSavePath)
         
          newdir='arch'
         
          self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
          os.mkdir(self.WeightSavePathNew)
          os.chdir(self.WeightSavePathNew)
          
          self.Disc_lr=0.0002
          self.Gen_lr=0.0002
          
          # self.Disc_lr=0.001
          # self.Gen_lr=0.01
         
          self.Disc_optimizer = keras.optimizers.Adam(self.Disc_lr, 0.5,0.999)
          self.Gen_optimizer = keras.optimizers.Adam(self.Gen_lr, 0.5,0.999)
         
          # optimizer = keras.optimizers.Adam(0.0002, 0.5)

          self.DiscCT=self.build_discriminator3D()
          self.DiscCT.compile(loss='mse', optimizer=self.Disc_optimizer, metrics=['accuracy'])
          self.DiscCT._name='Discriminator-CT'
          # self.DiscCT.summary()
          with open('Disc.txt', 'w+') as f:
              self.DiscCT.summary(print_fn=lambda x: f.write(x + '\n'))
         
          self.DiscCB=self.build_discriminator3D()
          self.DiscCB.compile(loss='mse', optimizer=self.Disc_optimizer, metrics=['accuracy'])
          self.DiscCB._name='Discriminator-CB'
         
          self.DiscCT_static=self.DiscCT
          self.DiscCB_static=self.DiscCB
          
          layer_len=len(self.DiscCT.layers)
          layers_lis=self.DiscCT.layers
          labelshapearr=list(layers_lis[layer_len-1].output_shape)
          labelshapearr[0]=self.batch_size
          labelshape=tuple(labelshapearr)
          self.labelshape=labelshape
        
          self.GenCB2CT=self.build_generator3D()
          self.GenCB2CT.compile(loss='mse', optimizer=self.Gen_optimizer, metrics=['accuracy'])
          self.GenCB2CT._name='Generator-CB2CT'
         
          
          # self.GenCB2CT.summary()
          with open('Gena.txt', 'w+') as f:
              self.GenCB2CT.summary(print_fn=lambda x: f.write(x + '\n'))
         
          self.GenCT2CB=self.build_generator3D()
          self.GenCT2CB.compile(loss='mse', optimizer=self.Gen_optimizer, metrics=['accuracy'])
          self.GenCT2CB._name='Generator-CT2CB'
          
          img_CT = keras.Input(shape=self.input_layer_shape_3D)
          img_CB = keras.Input(shape=self.input_layer_shape_3D)
          
          valid_CT = self.DiscCT(img_CT)
          valid_CB = self.DiscCB(img_CB)
          
          # self.DiscCT_static = Network(inputs=img_CT, outputs=guess_A, name='D_A_static_model')
          # self.DiscCB_static = Network(inputs=img_CB, outputs=guess_B, name='D_B_static_model')
          # self.DiscCT_static = clone_model(self.DiscCT)
          # self.DiscCB_static = clone_model(self.DiscCB)
          self.DiscCT_static.set_weights(self.DiscCT.get_weights())
          self.DiscCB_static.set_weights(self.DiscCB.get_weights())
          # self.DiscCT_static = clone_model(self.DiscCT)
          # self.DiscCB_static = clone_model(self.DiscCB)
        
        # For the combined model we will only train the generators
          # self.DiscCT_static.trainable = False
          # self.DiscCB_static.trainable = False
         
          # Input images from both domains

        
        # Translate images to the other domain
          fake_CB = self.GenCT2CB(img_CT)
          fake_CT = self.GenCB2CT(img_CB)
        # Translate images back to original domain
          reconstr_CT = self.GenCB2CT(fake_CB)
          reconstr_CB = self.GenCT2CB(fake_CT)
        # Identity mapping of images
          img_CT_id = self.GenCT2CB(img_CT)
          img_CB_id = self.GenCB2CT(img_CB)
          
        #   self.DiscCT_static = Network(inputs=img_CT, outputs=guess_A, name='D_A_static_model')
        #   self.DiscCB_static = Network(inputs=img_CB, outputs=guess_B, name='D_B_static_model')
        
        # # For the combined model we will only train the generators
        #   self.DiscCT_static.trainable = False
        #   self.DiscCB_static.trainable = False
        
        # Discriminators determines validity of translated images
          valid_CT = self.DiscCT_static(fake_CT)
          valid_CB = self.DiscCB_static(fake_CB)
        
        # Combined model trains generators to fool discriminators
          self.cycleGAN_Model = keras.Model(inputs=[img_CT, img_CB], outputs=[valid_CT, valid_CB, reconstr_CT, reconstr_CB, img_CT_id, img_CB_id,reconstr_CT, reconstr_CB])
          self.cycleGAN_Model.compile(loss=['mse', 'mse', 'mae', 'mae','mae', 'mae', custom_loss_5_epsilon3D,custom_loss_5_epsilon3D],
                                     loss_weights=[1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id,10,10], 
                                     optimizer=self.Gen_optimizer)
          self.cycleGAN_Model._name='CycleGAN'
         
          with open('cycleGAN.txt', 'w+') as f:
              self.cycleGAN_Model.summary(print_fn=lambda x: f.write(x + '\n'))
          
          self.trainCT_path = os.path.join(self.DataPath, 'trainCT')
          self.trainCB_path = os.path.join(self.DataPath, 'trainCB')
          # testCT_path = os.path.join(self.DataPath, 'validCT')
          # testCB_path = os.path.join(self.DataPath, 'validCB')
          # self.data_generator=loadprintoutgen(trainCT_path,trainCB_path,self.batch_size,self.batch_set_size)
          
          # for images in self.data_generator:
          #     batch_CT = images[0]
          #     batch_CB = images[1]
              
          #     print(batch_CT.shape)
          # print(self.data_generator.__len__())
          os.system("nvidia-smi")
          print('Init')
          
    def learningrate_log_scheduler(self):
        learning_rates=np.logspace(-8, 1,num=self.epochs)
        return learning_rates
    
    def traincgan(self):
        os.chdir(self.WeightSavePathNew)
         # self.folderlen='run'+str(len(next(os.walk(self.WeightSavePath))[1]))         
        newdir='weights'        
        self.WeightSavePathNew = os.path.join(self.WeightSavePath, newdir)
        os.mkdir(self.WeightSavePathNew)
        os.chdir(self.WeightSavePathNew)
        
        gen1fname="GenCT2CBWeights"
        gen2fname="GenCB2CTWeights"
        disc1fname="DiscCTWeights"
        disc2fname="DiscCBWeights"
        # D_losses = np.zeros((self.batch_set_size,2,self.epochs))
        # G_losses = np.zeros((self.batch_set_size,7,self.epochs))
        D_losses = []
        G_losses = []
        
        #Learning rate schedule
        learning_rates=self.learningrate_log_scheduler()
        
        epoch_start=0
        if self.breakflag:
            lastweightpath=self.lastweightpath
            lst=os.listdir(lastweightpath)
            lst=sorted(lst,key=lambda x: os.path.getmtime(os.path.join(lastweightpath,x)))
            weightfilename=lst[-1]
            x=weightfilename.split(".")
            x=x[0].split("-")
            epochi=int(x[-1])
            gen1fname1_break="GenCT2CBWeights"+'-'+str(epochi)+'.h5'
            gen2fname1_break="GenCB2CTWeights"+'-'+str(epochi)+'.h5'
            disc1fname1_break="DiscCTWeights"+'-'+str(epochi)+'.h5'
            disc2fname1_break="DiscCBWeights"+'-'+str(epochi)+'.h5'
            self.GenCT2CB.load_weights(os.path.join(lastweightpath,gen1fname1_break))
            self.GenCB2CT.load_weights(os.path.join(lastweightpath,gen2fname1_break))
            self.DiscCT.load_weights(os.path.join(lastweightpath,disc1fname1_break))
            self.DiscCB.load_weights(os.path.join(lastweightpath,disc2fname1_break))
            epoch_start=epochi
            print("left from the epoch %s"%epochi)
        
        def run_training_iteration(loop_index, epoch_iterations):
              valid = tf.ones((self.labelshape))
              fake = tf.zeros((self.labelshape))
                 # ----------------------
                 #  Train Discriminators
                 # ----------------------

                 # Translate images to opposite domain
              fake_CB = self.GenCT2CB.predict(batch_CT)
              fake_CT = self.GenCB2CT.predict(batch_CB)

                 # Train the discriminators (original images = real / translated = Fake)
              dCT_loss_real = self.DiscCT.train_on_batch(batch_CT, valid)
              dCT_loss_fake = self.DiscCT.train_on_batch(fake_CT, fake)
              dCT_loss = 0.5 * tf.math.add(dCT_loss_real, dCT_loss_fake)

              dCB_loss_real = self.DiscCB.train_on_batch(batch_CB, valid)
              dCB_loss_fake = self.DiscCB.train_on_batch(fake_CB, fake)
              dCB_loss = 0.5 * tf.math.add(dCB_loss_real, dCB_loss_fake)

                 # Total discriminator loss
              d_loss = 0.5 * tf.math.add(dCT_loss, dCB_loss)

                 # ------------------
                 #  Train Generators
                 # ------------------

                 # Train the generators
              g_loss = self.cycleGAN_Model.train_on_batch([batch_CT, batch_CB],
                                                       [valid, valid,
                                                        batch_CT, batch_CB,
                                                        batch_CT, batch_CB,
                                                        batch_CT, batch_CB])
              
              return g_loss, d_loss.numpy()
              
        for epochi in range(epoch_start,self.epochs):
            # if self.use_data_generator:
                loop_index = 1
                
                # K.set_value(self.Gen_optimizer.learning_rate, learning_rates[epochi])
                # K.set_value(self.Disc_optimizer.learning_rate, learning_rates[epochi])
                self.data_generator=loadprintoutgen(self.trainCT_path,self.trainCB_path,self.batch_size,self.batch_set_size)
                print(self.data_generator.__len__())
                
                for images in self.data_generator:
                    batch_CT = images[0]
                    batch_CB = images[1]
                    # Run all training steps
                    g_loss, d_loss=run_training_iteration(loop_index, self.data_generator.__len__())
                    # os.system("nvidia-smi")
                    #print('Trainingstep')
                    
                    if loop_index >= self.data_generator.__len__():
                        break
                    loop_index = loop_index+1
                    
                    
                
                D_losses.append(d_loss)
                G_losses.append(g_loss)
                    
                if epochi % self.save_epoch_frequency == 0 and self.saveweightflag==True: # Weights saved based on epoch intervals
                    gen1fname1=gen1fname+'-'+str(epochi)+'.h5'    
                    gen2fname1=gen2fname+'-'+str(epochi)+'.h5'
                    disc1fname1=disc1fname+'-'+str(epochi)+'.h5'
                    disc2fname1=disc2fname+'-'+str(epochi)+'.h5'
                    self.GenCT2CB.save_weights(gen1fname1)
                    self.GenCB2CT.save_weights(gen2fname1)
                    self.DiscCT.save_weights(disc1fname1)
                    self.DiscCB.save_weights(disc2fname1)
        # os.system("nvidia-smi")
                print('Epoch =%s'%epochi)
        return D_losses,G_losses

