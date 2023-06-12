# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:30:53 2023

@author: yejiw
"""

import numpy as np
import matplotlib.pyplot as plt
from system import ODESystem 
from scipy.integrate import solve_ivp


# This document generates training and test datset to feed the neural network
# The training and test set will be directly imported from this script.

#**************Parameters******************

T = 29 # Max time step
gamma = 0.1 # Recovery rate
m = 50 # Number of sensors
num_train = 1000 # Number of training set
num_test = 1000 # Number of test set

#...............................................


def gen_data(T,m,case):
    
    '''
    Generate data for training set by solving equation.
    '''
        
    def dI_dt(t, I, R_0_list, gamma):
        '''
        Nonlinear ode equation
        
        dI/dt = R_0 * gamma * (1 - I) * I - gamma * I 
        
        '''
        
        gamma = 0.1 # Recovery rate, 1/(time to recover)
        
        # Get the corresponding R_0 value based on the time index
        R_0 = R_0_list[np.clip(int(round(t)), 0, len(R_0_list) - 1)]  
        
        return R_0 * gamma * (1 - I) * I - gamma * I
    
    # Define the initial condition of I
    I0 = 0.1
    
    # Define the time span
    t_span = [0, T]  # Start and end time
    
    # Define the time points at which to evaluate the solution
    t_eval = np.linspace(t_span[0], t_span[1], T+1)
 
    
    if case =='const':
        # When R_0 is constant over time
        list_one = np.ones(T+1)
        R_0_const = np.random.uniform(low=1, high=2.5)
        R_0_list= list_one*R_0_const
        
    else:
        # When R_0 is time dependent
        # Generate the list of R_0 values at each t
        R_0_list = np.random.uniform(low=1, high=2.5, size=T+1)   # Example values for R_0 at different time points
        
    
    # Solve the ODE using the RK4 method
    sol = solve_ivp(dI_dt, t_span, [I0], method='RK45', t_eval=t_eval, args=(R_0_list, gamma))
    
    # Extract the solution
    t = sol.t
    I = sol.y[0]
    
    # Define specific times of interest
    x = np.random.randint(0,29)
    
    # Get the values of I at the specific times of interest
    I_x = I[x]
    
    
    sensors = np.arange(0, T+1, (T+1)/m, dtype = int)
    
    R_0_vals = R_0_list[sensors]
    
    #print(R_0_list)
    
    '''
    plt.figure()
    plt.plot(t,I)
    plt.title('%s' %R_0_list[0])
    '''
    
    return R_0_vals, x, I_x


def train_data(T,m,case,num_train):
    
    '''
    Using data from gen_data, build a data set in the format that 
    the trainig model code requires
    '''
    
    sensor_values = np.empty((0,m))
    x_list = np.array([])
    I_x_list = np.array([])
    
    num = 0
    
    while num < num_train:
        
        R_0_vals,x,I_x = gen_data(T,m,'const')
        
        sensor_values = np.vstack((sensor_values,R_0_vals))
        x_list = np.append(x_list,x)
        I_x_list = np.append(I_x_list,I_x)
        
        num = num+1
        
    # Resize the list 
    x_list = np.resize(x_list, (num_train,1))
    I_x_list = np.resize(I_x_list, (num_train,1))
    
    # Final training data set 
    X_train = [sensor_values,x_list]
    y_train = I_x_list
    
    return  X_train, y_train

def gen_test(T):
    
    '''
    Generate solution of ODE to compare with the prediction from model 
    '''
    
    
    def rhs(t, I, R_0_list, gamma):
        # Get the corresponding R_0 value based on the time index
        R_0 = R_0_list[int(t)]  
        return R_0 * gamma * (1 - I) * I - gamma * I
    
    # Initial value of I(t)
    I0 = 0.1

    # Define the time span
    t_span = [0, T]  # Start and end time
    
    # Define the time points at which to evaluate the solution
    t_eval = np.linspace(t_span[0], t_span[1], 100)


    # Define the initial condition
    I0 = 0.1
    
    # Define the time span
    t_span = [0, T]  # Start and end time
    
    # Define the time points at which to evaluate the solution
    t_eval = np.linspace(t_span[0], t_span[1], 100)
    
    # Generate the list of  random R_0 values
    list_one = np.ones(T+1)
    R_0_const = np.random.uniform(low=1, high=2.5)
    R_0_list= list_one*R_0_const
    
    # Solve the ODE using the RK4 method
    sol = solve_ivp(rhs, t_span, [I0], method='RK45', t_eval=t_eval, args=(R_0_list, gamma))
    
    # Extract the solution
    t = sol.t
    I = sol.y[0]
    
    return t, I, R_0_const


#************************Output

X_train, y_train = train_data(T,m,'const',num_train)
X_test, y_test = train_data(T,m,'const',num_train)


# True solutions to compare with predicted solution
t1,I1,R_0_const1 = gen_test(T)
t2,I2,R_0_const2 = gen_test(T)
t3,I3,R_0_const3 = gen_test(T)
