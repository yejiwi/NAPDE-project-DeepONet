# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:30:53 2023

@author: yejiw

This code generates the training data set when R_0 is constant over time.

"""

import numpy as np
import matplotlib.pyplot as plt
from system import ODESystem 
from scipy.integrate import solve_ivp, odeint
import random



num_train = 20000
num_test = 40000
m = 100 # number of sensors
T = 29  # time span

np.random.seed(1)

def ode_system(T):
    """Our ODE function"""
    # beta = R_0 * gamma
    #     
    # s : Output function, I(t)
    # u : Input function, R_0
    
    
    # Our case: dI/dt = beta * S * I - gamma * I 
    #                 = R_0 * gamma * (1 - I) * I - gamma * I 
    # S + I = 1
    # R_0 is u. I is s.
    
    def g(s, u, x):

        gamma = 0.1 # Recovery rate
        
        
        return u * gamma * (1-s) * s - gamma * s
   
    s0 = [0.1] # Initial condition
    
    
    return ODESystem(g, s0, T)

system = ode_system(T)



def ode_fn(I, t, R_0):
    "Our ODE function to be solved"
    
    gamma = 0.1
    
    dI_dt = R_0 * gamma * (1 - I) * I - gamma * I
    
    return dI_dt



def solve_ODE(R_0,T,m):
    '''
    Solve ODE and return solution
    '''

    I0 = 0.1

    t_span = np.linspace(0, T,m)

    
    I = odeint(ode_fn, I0, t_span, args=(R_0,))

        
    return t_span, I



# List of random R_0 value
R_0 = np.random.uniform(low=1, high=2.5, size = 2) 

R_01 = R_0[0]

R_02 = R_0[1]

# When R_0 is time dependent
R_0_list = np.random.uniform(low=1, high=2.5, size=T+1) 


t1, I1 = solve_ODE(R_01,T, m)
t2, I2 = solve_ODE(R_02,T, m)


plt.plot(t1,I1)
plt.plot(t2,I2)


plt.title('proportion of infections when R_0 is constant')
plt.xlabel('Time [day]')
plt.ylabel('Proportion of infected individuals')





def r_0_list(x):
    
    '''
    R_0 value function returning the value of R_0 at x, which is t
    '''
    new = np.ones_like(x) * R_0_list
    
    return new[x]






def gen_data_const(m,num_train,T):
    '''
    Generate trainig data set when R_0 is constant.
    
    Parameters
    ----------
    m : number of sensors
    num_train : number of training set
    T : length of period

    Returns
    -------
    X_train : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.

    '''
    
    # Generate random R_0 values
    R_0_vals = np.linspace(1,2.5,num_train)
    #R_0_vals = np.random.uniform(low=1, high=2.5, size=num_train) 
    R_0_vals = np.resize(R_0_vals, (len(R_0_vals),1))

    t_list = np.zeros(0)
    I_list = np.zeros(0)

    
    for R in R_0_vals:
        
        t_sol, I_sol = solve_ODE(R,T,m)

        # Value of I at specific time t
        specific_t = random.uniform(0,T) 
        
        t_list = np.append(t_list, specific_t)
        
        index_t = np.abs(t_sol - specific_t).argmin()
        
        I_t = I_sol[index_t]
        
        I_list = np.append(I_list, I_t)
        
    
    sensor_values = np.tile(R_0_vals,(1,m))
    
    t_list = np.resize(t_list, (len(t_list),1))
    I_list = np.resize(I_list, (len(I_list),1))
    
    X_train = [sensor_values, t_list]
    y_train = I_list
    


    return X_train, y_train
    

# 
X_train, y_train = gen_data_const(m,num_train,T)
X_test, y_test = gen_data_const(m,num_test,T)

    