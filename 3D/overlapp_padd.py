#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 15:52:08 2022

@author: arun
"""

import numpy as np


#%%

z=np.arange(0, 50)

depth_size=32
overlap=1
border=5

z=np.pad(z,depth_size//2,mode='edge')

z_p=np.zeros_like(z,dtype=float)
zl=len(z)
#%%
for zi in range(0,zl,depth_size//overlap):
    
    # print(zi)
    # print(zi+depth_size)
    # print(zi+border)
    # print(zi+depth_size-border)
    currentBlk=z[zi:zi+depth_size]
    blksiz=currentBlk.shape
    # if blksiz[0] != depth_size:
    #     diff_zi=blksiz[0]-depth_size
    #     zi=zi+diff_zi
    #     currentBlk=z[zi:zi+depth_size]
        # print(zi)
        # print(zi+depth_size)
    currentBlk_p = currentBlk[border:-border]
    z_p[zi+border:zi+depth_size-border]=currentBlk_p
        # print(zi+border)
        # print(zi+depth_size-border)