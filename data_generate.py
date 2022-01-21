#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:33:40 2021

@author: zhanruicai
"""


from scipy.stats import norm
import numpy as np
import scipy.stats as sss
from scipy import stats
import matplotlib


def get_test_stat(thx, thy):
    
    x, y = map(np.asarray, (thx, thy))
    n1 = len(x)
    n2 = len(y)
   
    alldata = np.concatenate((x, y))
    ranked = sss.rankdata(alldata)
    x = ranked[:n1]
    s1 = np.sum(x, axis=0) - n1*(n1+1)/2

    return 1 - s1/(n1*n2)


def get_sim_model_1(n1, d, aa):
               
    x = np.random.normal(0, 1, (n1, d))
    y = np.random.normal(0, 1, (n1, d))
                  
    ep = np.random.normal(0, 1, (n1))
    
    y[:,1] = aa*x[:,1] + ep

    x1 = np.concatenate((x[1:n1,], x[0:1,]), axis = 0)
        
    xall = np.concatenate((x, x1), axis = 0)
    yall = np.concatenate((y, y), axis = 0)
    
    datall = np.concatenate((xall, yall), axis = 1)
            
        
    label1 = np.ones(shape = n1)
    label0 = np.zeros(shape = n1)
    label = np.concatenate((label1, label0), axis = 0)
            
    return datall, label


def get_sim_model_2(n1, d, aa):
        
    x = np.random.negative_binomial(n = 1, p = 0.2, size = (n1, d))  
    #matplotlib.pyplot.hist(x)
    
    
    y = np.random.negative_binomial(0, 1, (n1, d))
                  
    ep = np.random.negative_binomial(0, 1, (n1))
    
    y[:,1] = x[:,1] + ep

        
    x1 = np.concatenate((x[1:n1,], x[0:1,]), axis = 0)
        
    xall = np.concatenate((x, x1), axis = 0)
    yall = np.concatenate((y, y), axis = 0)
    
    datall = np.concatenate((xall, yall), axis = 1)
            
        
    label1 = np.ones(shape = n1)
    label0 = np.zeros(shape = n1)
    label = np.concatenate((label1, label0), axis = 0)
            
    return datall, label


def get_sim_model_0(n1, d, aa):
        
    x = np.random.normal(0, 1, (n1, d))
    y = np.random.normal(0, 1, (n1, d))
                  
    ep = np.random.normal(0, 1, (n1))
    
    y[:,1] = x[:,1]*x[:,1] + ep

        
    x1 = np.concatenate((x[1:n1,], x[0:1,]), axis = 0)
        
    xall = np.concatenate((x, x1), axis = 0)
    yall = np.concatenate((y, y), axis = 0)
    
    datall = np.concatenate((xall, yall), axis = 1)
            
        
    label1 = np.ones(shape = n1)
    label0 = np.zeros(shape = n1)
    label = np.concatenate((label1, label0), axis = 0)
            
    return datall, label

def get_sim_model_3(n1, d, aa):
        
    x = np.random.normal(0, 1, (n1, d))
    y = np.random.normal(0, 1, (n1, d))
                  
    ep = np.random.standard_t(df = 1, size = (n1))
    
    y[:,1] = aa*x[:,1] + ep

        
    x1 = np.concatenate((x[1:n1,], x[0:1,]), axis = 0)
        
    xall = np.concatenate((x, x1), axis = 0)
    yall = np.concatenate((y, y), axis = 0)
    
    datall = np.concatenate((xall, yall), axis = 1)
            
        
    label1 = np.ones(shape = n1)
    label0 = np.zeros(shape = n1)
    label = np.concatenate((label1, label0), axis = 0)
    
    datall -= np.mean(datall, axis=0)
    
    return datall, label

def get_sim_model_4(n1, d, aa):
        
    x = np.random.normal(0, 1, (n1, d))
    y = np.random.normal(0, 1, (n1, d))
                  
    ep = np.random.standard_t(df = 1, size = (n1))
    
    y[:,1] = x[:,1] + ep

        
    x1 = np.concatenate((x[1:n1,], x[0:1,]), axis = 0)
        
    xall = np.concatenate((x, x1), axis = 0)
    yall = np.concatenate((y, y), axis = 0)
    
    datall = np.concatenate((xall, yall), axis = 1)
            
        
    label1 = np.ones(shape = n1)
    label0 = np.zeros(shape = n1)
    label = np.concatenate((label1, label0), axis = 0)
    
    datall /= np.std(datall, axis=0)
    
    return datall, label


def get_permute_data(datall):
       
    (n1, d) = datall.shape
    
    n1 = n1//2
    d = d//2
    
    x = datall[0:n1, 0:d]
    y = datall[(n1):(2*n1), 0:d]
    
    x = np.take(x,np.random.permutation(x.shape[0]),axis=0,out=x)    
    
    x1 = np.concatenate((x[1:n1,], x[0:1,]), axis = 0)

    xall = np.concatenate((x, x1), axis = 0)
    yall = np.concatenate((y, y), axis = 0)

    newdatall = np.concatenate((xall, yall), axis = 1)

    return newdatall
