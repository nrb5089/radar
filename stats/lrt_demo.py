# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 07:43:23 2023

@author: nrb50
"""

import numpy as np

def pdf(x,mu = 0, sigma= 1): return 1/np.sqrt(2*np.pi * sigma**2) * np.exp(-(x-mu)**2/2/sigma**2)

mu = 1
sigma = 1

L = 2*