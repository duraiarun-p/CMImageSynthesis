#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 11:25:34 2022

@author: arun
"""

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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

cfg = tf.compat.v1.ConfigProto() 
cfg.gpu_options.allow_growth = True
sess= tf.compat.v1.Session(config=cfg)


from CycleGAN_Archs_lib import CycleGAN_Eta,CycleGAN_Zeta,CycleGAN_Epsilon,CycleGAN_Delta,CycleGAN_Gamma,CycleGAN_Beta,CycleGAN_Alpha, dataload, dataload_direct, perf_metrics, dataload3D_2_predict,normalise_img_volume, I2I_2D_CT,I2I_2D_CB
#%%


#%%


st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()


mypath='/home/arun/Documents/PyWSPrecision/datasets/printoutslices'
Datapath='/home/arun/Documents/MATLAB/ImageDB/PrintoutDB/DB33/'

weightoutputpath1='/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/'
weightoutputpath=os.path.join(weightoutputpath1,'Perform_I2I_2D_Output')
if not os.path.exists(weightoutputpath):
    os.makedirs(weightoutputpath)
    
weightoutputpath2=os.path.join(weightoutputpath,'run')
if not os.path.exists(weightoutputpath2):
    os.makedirs(weightoutputpath2)   
    
saved_weigth_path='/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp'
lastweightpath=weightoutputpath2

CT,CBCT=dataload3D_2_predict(Datapath)
CBsiz=CBCT.shape
CTsiz=CT.shape

# CT=CT[:,:,-32:]
# CB=CBCT[:,:,-32:]

CT=CT[:,:,CTsiz[2]//2:CTsiz[2]//2+32]
CB=CBCT[:,:,CBsiz[2]//2:CBsiz[2]//2+32]

CT=normalise_img_volume(CT)
CB=normalise_img_volume(CB)

#%%    
cGAN=CycleGAN_Alpha(mypath,weightoutputpath2,epochs=40,save_epoch_frequency=2,batch_size=3,imgshape=(256,256,1),newshape=(256,256),batch_set_size=10,saveweightflag=True)


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

CT_P=I2I_2D_CT(CT,TestGenCT2CB,TestGenCB2CT)
CB_P=I2I_2D_CB(CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_alpha.mat",mdic) 
del cGAN,TestGenCB2CT,TestGenCT2CB_path,CT_P,CB_P
#%%
cGAN1=CycleGAN_Beta(mypath,weightoutputpath2,epochs=40,save_epoch_frequency=2,batch_size=3,imgshape=(256,256,1),newshape=(256,256),batch_set_size=10,saveweightflag=True)


GenCB2CTweight='Beta_GenCB2CTWeights-500.h5'
GenCT2CBweight='Beta_GenCT2CBWeights-500.h5'

TestGenCB2CT_path=os.path.join(saved_weigth_path,GenCB2CTweight)
TestGenCB2CT=cGAN1.build_generator()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights(TestGenCB2CT_path)

TestGenCT2CB_path=os.path.join(saved_weigth_path,GenCT2CBweight)
TestGenCT2CB=cGAN1.build_generator()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights(TestGenCT2CB_path)

CT_P=I2I_2D_CT(CT,TestGenCT2CB,TestGenCB2CT)
CB_P=I2I_2D_CB(CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_beta.mat",mdic) 
del cGAN1,TestGenCB2CT,TestGenCT2CB_path,CT_P,CB_P
#%%
cGAN2=CycleGAN_Gamma(mypath,weightoutputpath2,lastweightpath,epochs=40,save_epoch_frequency=2,batch_size=3,imgshape=(256,256,1),newshape=(256,256),batch_set_size=10,saveweightflag=True,breakflag=False)


GenCB2CTweight='Gamma_GenCB2CTWeights-500.h5'
GenCT2CBweight='Gamma_GenCT2CBWeights-500.h5'

TestGenCB2CT_path=os.path.join(saved_weigth_path,GenCB2CTweight)
TestGenCB2CT=cGAN2.build_generator()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights(TestGenCB2CT_path)

TestGenCT2CB_path=os.path.join(saved_weigth_path,GenCT2CBweight)
TestGenCT2CB=cGAN2.build_generator()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights(TestGenCT2CB_path)

CT_P=I2I_2D_CT(CT,TestGenCT2CB,TestGenCB2CT)
CB_P=I2I_2D_CB(CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_gamma.mat",mdic) 
del cGAN2,TestGenCB2CT,TestGenCT2CB_path,CT_P,CB_P
#%%
cGAN3=CycleGAN_Delta(mypath,weightoutputpath2,lastweightpath,epochs=40,save_epoch_frequency=2,batch_size=3,imgshape=(256,256,1),newshape=(256,256),batch_set_size=10,saveweightflag=True,breakflag=False)


GenCB2CTweight='Delta_GenCB2CTWeights-500.h5'
GenCT2CBweight='Delta_GenCT2CBWeights-500.h5'

TestGenCB2CT_path=os.path.join(saved_weigth_path,GenCB2CTweight)
TestGenCB2CT=cGAN3.build_generator()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights(TestGenCB2CT_path)

TestGenCT2CB_path=os.path.join(saved_weigth_path,GenCT2CBweight)
TestGenCT2CB=cGAN3.build_generator()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights(TestGenCT2CB_path)

CT_P=I2I_2D_CT(CT,TestGenCT2CB,TestGenCB2CT)
CB_P=I2I_2D_CB(CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_delta.mat",mdic) 
del cGAN3,TestGenCB2CT,TestGenCT2CB_path,CT_P,CB_P
#%%
cGAN4=CycleGAN_Epsilon(mypath,weightoutputpath2,lastweightpath,epochs=40,save_epoch_frequency=2,batch_size=3,imgshape=(256,256,1),newshape=(256,256),batch_set_size=10,saveweightflag=True,breakflag=False)


GenCB2CTweight='Epsilon_GenCB2CTWeights-500.h5'
GenCT2CBweight='Epsilon_GenCT2CBWeights-500.h5'

TestGenCB2CT_path=os.path.join(saved_weigth_path,GenCB2CTweight)
TestGenCB2CT=cGAN4.build_generator()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights(TestGenCB2CT_path)

TestGenCT2CB_path=os.path.join(saved_weigth_path,GenCT2CBweight)
TestGenCT2CB=cGAN4.build_generator()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights(TestGenCT2CB_path)

CT_P=I2I_2D_CT(CT,TestGenCT2CB,TestGenCB2CT)
CB_P=I2I_2D_CB(CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_epsilon.mat",mdic) 
del cGAN4,TestGenCB2CT,TestGenCT2CB_path,CT_P,CB_P
#%%
cGAN5=CycleGAN_Zeta(mypath,weightoutputpath2,lastweightpath,epochs=40,save_epoch_frequency=2,batch_size=3,imgshape=(256,256,1),newshape=(256,256),batch_set_size=10,saveweightflag=True,breakflag=False)


GenCB2CTweight='Zeta_GenCB2CTWeights-500.h5'
GenCT2CBweight='Zeta_GenCT2CBWeights-500.h5'

TestGenCB2CT_path=os.path.join(saved_weigth_path,GenCB2CTweight)
TestGenCB2CT=cGAN5.build_generator()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights(TestGenCB2CT_path)

TestGenCT2CB_path=os.path.join(saved_weigth_path,GenCT2CBweight)
TestGenCT2CB=cGAN5.build_generator()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights(TestGenCT2CB_path)

CT_P=I2I_2D_CT(CT,TestGenCT2CB,TestGenCB2CT)
CB_P=I2I_2D_CB(CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_zeta.mat",mdic) 
del cGAN5,TestGenCB2CT,TestGenCT2CB_path,CT_P,CB_P
#%%
cGAN6=CycleGAN_Eta(mypath,weightoutputpath2,lastweightpath,epochs=40,save_epoch_frequency=2,batch_size=3,imgshape=(256,256,1),newshape=(256,256),batch_set_size=10,saveweightflag=True,breakflag=False)


GenCB2CTweight='Eta_GenCB2CTWeights-500.h5'
GenCT2CBweight='Eta_GenCT2CBWeights-500.h5'

TestGenCB2CT_path=os.path.join(saved_weigth_path,GenCB2CTweight)
TestGenCB2CT=cGAN6.build_generator()
TestGenCB2CT.trainable=False
TestGenCB2CT.load_weights(TestGenCB2CT_path)

TestGenCT2CB_path=os.path.join(saved_weigth_path,GenCT2CBweight)
TestGenCT2CB=cGAN6.build_generator()
TestGenCT2CB.trainable=False
TestGenCT2CB.load_weights(TestGenCT2CB_path)

CT_P=I2I_2D_CT(CT,TestGenCT2CB,TestGenCB2CT)
CB_P=I2I_2D_CB(CB,TestGenCT2CB,TestGenCB2CT)
mdic = {"CT_P":CT_P,"CT":CT,"CB_P":CB_P,"CB":CB}
savemat("Pred_volumes_eta.mat",mdic) 
del cGAN6,TestGenCB2CT,TestGenCT2CB_path,CT_P,CB_P