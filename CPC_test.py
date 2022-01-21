#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:30:12 2021

@author: zhanruicai
"""
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import time
import sys
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from statistics import mean
from scipy.stats import norm
import data_generate as dg
import tensorflow as tf


simutime = 100
sample_size = 1000
train_data_size = sample_size//2
val_data_size = sample_size//2
d = 100
batchsize = 30

epochs = 20
a_seq = np.linspace(1, 0, num = 5)

bei = 40
SGD_rate = 0.1
 

p_all = np.ones(shape = (sample_size, simutime*len(a_seq)))
pvalue_all = np.ones((simutime,len(a_seq)))

for jj in range(len(a_seq)):

    sys.stdout.write("Signal: %.2f \n" %   (a_seq[jj]) )

    start = time.time()

    for ii in range(simutime):
        
        
        train_x, train_ground_truth_class = dg.get_sim_model_1(train_data_size, d, a_seq[jj])
        val_x, val_ground_truth_class = dg.get_sim_model_1(val_data_size, d, a_seq[jj])
        
        model = tf.keras.models.Sequential([
          tf.keras.layers.Dense(d*bei, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.001)),
         # tf.keras.layers.Dense(d*bei, activation='relu'),
          tf.keras.layers.Dropout(0.3),
          tf.keras.layers.Dense(1, activation='sigmoid')
        ])        
        
        # Create an optimizer with the desired parameters.
        myopt = tf.keras.optimizers.SGD(learning_rate = SGD_rate)

        model.compile(optimizer=myopt, loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit(train_x, train_ground_truth_class,  epochs=epochs, batch_size=batchsize, verbose = 0)


        # Evaluate the model on the test data using `evaluate`
        #print("Evaluate on test data")
        #results = model.evaluate(val_x, val_ground_truth_class)
        #print("test loss, test acc:", results)


        prediction2 = model.predict(val_x)

        thx = prediction2[0: val_data_size]
        thy = prediction2[val_data_size : sample_size]

        test_stat = dg.get_test_stat(thx, thy)

        thx = thx.reshape(val_data_size,)
        thy = thy.reshape(val_data_size,)
        fn1 = ECDF(thx)
        fn2 = ECDF(thy)
        a1 = fn1(thy) - 0.5
        a2 = 0.5 - fn1(thx)
        sda = 2*(mean(a1*a2) + mean(np.concatenate( (a2[1:val_data_size], a2[0:1]), axis = 0) *a1) )
        
        pvalue = norm.cdf(test_stat, 0.5, np.sqrt((1/6+sda)/val_data_size) )
           
        pvalue_all[ii, jj] = pvalue
        
        p_all[0:sample_size, ii] = prediction2.reshape(sample_size,)
           
        sys.stdout.write("Signal: %.2f, Simutime : %s \n" %   (a_seq[jj], ii) )
        sys.stdout.write("test statistic is: %.4f, sda is: %.4f, p-value is: %.4f  \n" % ( test_stat, sda, pvalue) )
        if ii>=1:
            sys.stdout.write("Power now is : %.3f  \n" % (pvalue_all[0:ii, jj]<=0.05).mean() )
        sys.stdout.flush() 
        
    sys.stdout.write("When signal is: %.2f, power is: %.3f. \n" % ( a_seq[jj], (pvalue_all[0:simutime,jj]<0.05).mean() ) )

    end = time.time()
    timelap = end - start
    
    
    sys.stdout.write("Time passed for when signal is  %.2f: %.3f \n"  % (a_seq[jj], timelap))


nam1 = "pvalues.csv"
nam2 = "pred.csv"

np.savetxt(nam1, pvalue_all, delimiter=",")
np.savetxt(nam2, p_all, delimiter=",")


        
        





