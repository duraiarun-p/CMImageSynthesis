#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:07:14 2022

@author: arun
"""

import time
import datetime
import tensorflow as tf
# from tensorflow.keras.callbacks import TensorBoard
# from tensorboard import program
import os
# from os import listdir
# from os.path import isfile, join
# import numpy as np

# from matplotlib import pyplot as plt
# from scipy.io import savemat

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)

from CycleGAN_Archs_lib import CycleGAN_Beta, dataload, dataload_direct, perf_metrics

#%%


st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()


# mypath='/home/arun/Documents/PyWSPrecision/datasets/printoutslices'
datapath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'
mypath='/home/arun/Documents/PyWSPrecision/datasets/printout2d_data'

weightoutputpath1='/home/arun/Documents/PyWSPrecision/Pyoutputs/cycleganweights/CMImageSynthesis_Outputs/'

weightoutputpath=os.path.join(weightoutputpath1, 'Vis_Output')
if not os.path.isdir(weightoutputpath):
    os.mkdir(weightoutputpath)
    
logdirpath=os.path.join(weightoutputpath, 'log_dir')
if not os.path.isdir(logdirpath):
    os.mkdir(logdirpath)
    
cGAN=CycleGAN_Beta(mypath,weightoutputpath,epochs=40,save_epoch_frequency=2,batch_size=3,imgshape=(256,256,1),newshape=(256,256),batch_set_size=10,saveweightflag=True)

archoutputpath1=os.getcwd()
# saved_weigth_path='/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp'
# GenCB2CTweight='Beta_GenCB2CTWeights-500.h5'
# GenCT2CBweight='Beta_GenCT2CBWeights-500.h5'

# TestGenCB2CT_path=os.path.join(saved_weigth_path,GenCB2CTweight)
TestGenCB2CT=cGAN.build_generator()
TestGenCB2CT.trainable=False
# TestGenCB2CT.save(os.path.join(weightoutputpath1, 'TestGenCB2CT'))
TestGenCB2CT.save('TestGenCB2CT.h5')
# TestGenCB2CT.load_weights(TestGenCB2CT_path)

# TestGenCT2CB_path=os.path.join(saved_weigth_path,GenCT2CBweight)
TestGenCT2CB=cGAN.build_generator()
TestGenCT2CB.trainable=False
TestGenCT2CB.save('TestGenCT2CB.h5')
# TestGenCT2CB.load_weights(TestGenCT2CB_path)

TestDisCB2CT=cGAN.build_discriminator()
TestDisCB2CT.trainable=False
TestDisCB2CT.save('TestDisCB2CT.h5')

TestDisCT2CB=cGAN.build_discriminator()
TestDisCT2CB.trainable=False
TestDisCT2CB.save('TestDisCT2CB.h5')

#%%
# TC = tf.keras.callbacks.TensorBoard(logdirpath)
# TC.set_model(model=TestGenCB2CT)
# tb = program.TensorBoard()
# tb.configure(argv=[None, '--logdir', logdirpath])
# print("Launching Tensorboard ")
# url = tb.launch()
# print(url)

#%%
dot_img_file = 'Generator2D.png'
modelimgfilepath=os.path.join(archoutputpath1, dot_img_file)
tf.keras.utils.plot_model(TestGenCT2CB, to_file=modelimgfilepath, show_shapes=True)