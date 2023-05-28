# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:30:53 2023

@author: yejiw
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm



# Generate training set (beta, I) by solving SIR using 
# Explicit Euler with fixed beta
# We need u(t), x and G(u(x)) as a training set.
# which is equivalent to sensor_values,
# u(t) = sensor_values, where the value of u calcuated at sensor positions 
# x = points where output functions will be calculated
 


# G(u(x)) = num_train x 1

np.random.seed(0)


# Global parameter


def gen_data_const(m,num_train,T):
    
    
    gamma = 0.1 # Recovery rate. 1/(recovery days)
    
    # Generate random R_0 values
    R_0_vals = np.random.uniform(low=0.7, high=2.5, size=num_train) 
    
    # time list, 0 to 29days
    dt = 1
    t = np.arange(0,T,dt)

    I_0 = 0.01 # Initial rate of infected individuals
    
    
    # list of x where I will be evaluated.
    x_list = np.random.randint(0,29,num_train)
    
    # list of I value evaluated at x 
    I_eval = np.array([])

    for k in range(num_train):
        
        R_0 = R_0_vals[k]
        
        # Get the kth value of x in the x_list
        x = x_list[k]
          
        I = np.zeros(len(t))

        I[0] = I_0
        
        # Solve the ODE using Explicit Euler. Get the full list of I
        for j in range(len(t)-1): # -1 because the value is 
        
            I[j+1] = I[j] + dt*(R_0 * gamma * (1 - I[j]) * I[j]- gamma * I[j])
            
        # Append I value evaluated at x     
        I_eval = np.append(I_eval,I[x])
    
    # Sensor values. Since R_0 is constant, just duplicate the first column
    # m times.
    sensor_val = np.tile(np.array([R_0_vals]).transpose(), (1, m))
    
    # Resize the x_list so that it has shape for column.
    x_list = np.resize(x_list, (len(x_list),1))
    
    # Get X_training set
    X_train = [sensor_val, x_list]
    
    
    # Get y_training set
    y_train = np.resize(I_eval, (len(I_eval),1))
    
    return X_train, y_train
    

    
def gen_data_t(m,num_train,T):
    '''
    m: number of sensors
    num_train : number of trainig set
    dt: time interval
    T: length of period
    '''
    
    gamma = 0.1 # Recovery rate. 1/(recovery days)
    
    
    m_dt = (T+1)/m # Interval of sensors
        
    # Generate sensor values
    sensors = np.arange(0, T, m_dt, dtype=int)
    
    # t_list
    dt = 1
    t = np.arange(0,T,dt)
    
    I_0 = 0.01 
    
    #  at t
    sensor_val_t = np.array([])
    
    # Build R_0 function

    R_0_block = np.zeros((num_train, len(t)))
            
    
    for i in range(num_train):
        
        # Generate random values of R_0 for all t
        R_0_list = np.random.uniform(low=0.7, high=2.5, size=len(t)) 
        
        # Replace each row with random R_0 list
        R_0_block[i,:] = R_0_list
        
        # Append 
        for k in sensors:
            sensor_val_t = np.append(sensor_val_t, R_0_list[k])
            
    # transform array to a matrix                    
    sensor_val_t = np.resize(sensor_val_t,(num_train, m))
    
    # Create random values of x
    x_list = np.random.randint(0,29,num_train)
    
    I_eval_t = np.array([])
    
    for k in range(num_train):
        
        R_0 = R_0_block[k,:]
        
        x = x_list[k]
        
        I = np.zeros(len(t))

        I[0] = I_0

        for j in range(len(t)-1): # -1 because the value is 
        
            I[j+1] = I[j] + dt*(R_0[j] * gamma * (1 - I[j]) * I[j]- gamma * I[j])
            
        I_eval_t = np.append(I_eval_t,I[x])   
        
    
    x_list = np.resize(x_list, (len(x_list),1))
    
    I_eval_t = np.resize(I_eval_t, (len(I_eval_t),1))
    
    # Get X_training set
    X_train = [sensor_val_t, x_list]
    
    
    # Get y_training set
    y_train = np.resize(I_eval_t, (len(I_eval_t),1))
    
    return X_train, y_train
    
num_train = 100

num_test = 100
    
m = 10

X_train, y_train = gen_data_t(m, num_train, 29)

X_test, y_test = gen_data_t(m, num_test, 29)
    
    
    
    

    
    
    






      
