# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 23:00:51 2023

@author: Yeji Wi


"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

T = 29 # Time span
m = 100 # Number of sensors

np.random.seed(1)


num_tot = 60000 # Total number of data set
num_train = 20000 # Number of training set
num_test = int(num_tot - num_train) # Number of test set

sensor_values = [] # List to store sensor values

x_list = [] # List of time points where I(t) will be evaluated

y_list = [] # List of I(t) values evaluated at time points in x_list



def R0(T):
    '''
    Generate time dependent R_0 value where R_0 values of each day is a step function.

    Parameters
    ----------
    T : Max time interval

    Returns
    -------
    t_interp : time values
    R_0_interp : corresponding R_0 value list

    '''
    # Time points
    t_natural = np.arange(0, T+1)  # From 0 to 29
    
    # Generate random values for each interval
    random_values = np.random.uniform(1, 2.5, len(t_natural) - 1)
    
    # Create the step function
    t_step = []
    R_0_values = []
    for i, t_n in enumerate(t_natural[:-1]):
        t_step.extend([t_n - 0.5, t_n + 0.5])
        R_0_values.extend([random_values[i], random_values[i]])
    
    # Interpolate the step function
    t_interp = np.linspace(0, 29, 1000)  # More points for finer grid
    R_0_interp = np.interp(t_interp, t_step, R_0_values)
    
    return t_interp, R_0_interp


def get_R0_t(t_specific, t_interp, R_0_interp):
    '''
    Get R_0 values at specific time desired.

    Parameters
    ----------
    t_specific : List of time to evaluate R_0 value
    t_interp : full time list
    R_0_interp : corresponding R_0 values at t_interp

    Returns
    -------
    R_0_specific : R_0 values evaluated at t_specific

    '''

    R_0_specific = np.interp(t_specific, t_interp, R_0_interp)
    return R_0_specific


def model(I, t, t_interp, R_0_interp):
    
    gamma = 0.1
    
    R_0 = np.interp(t, t_interp, R_0_interp)  # Interpolate R_0 at the given time t
    dIdt = R_0 * gamma * (1 - I) * I - gamma * I
    
    return dIdt
    
    

for i in range(num_train+num_test):
    
    # Initial condition of infected population
    I0 = 0.1

    t_interp, R_0_interp = R0(T)
    
    # Plot the step function and interpolated values
    # plt.step(t_step, R_0_values, where='mid', label='Step Function')
    # plt.plot(t_interp, R_0_interp)
    # plt.xlabel('Time')
    # plt.ylabel('R_0')
    # plt.title('Step Function R_0 over Time')
    # plt.xlim(0,29)
    # plt.grid(True, linewidth=0.5)  # Adjust the linewidth parameter for a finer grid
    # plt.minorticks_on()  # Show minor ticks for even finer grid lines
    # plt.grid(which='minor', linewidth=0.25, linestyle='--', alpha=0.7)  # Customize minor grid
    # plt.show()
    
    
    # Solve the equation using odeint
    I_solution = odeint(model, I0, t_interp, args=(t_interp,R_0_interp))
    
    #Plot the solution
    # plt.figure()
    # plt.plot(t_interp, I_solution, color='r')
    # plt.xlabel('Time')
    # plt.ylabel('I(t)')
    # plt.title('Solution of dI/dt = R_0(t) * gamma * (1 - I) * I - gamma * I')
    # plt.grid()
    # plt.show()
    
        
    # Specific time at which you want to get the value of I_solution
    specific_time = np.random.uniform(t_interp[0], t_interp[-1])
    
    # Get the index corresponding to the specific time in t_interp array
    index = np.argmin(np.abs(t_interp - specific_time))
    
    # Get the corresponding value of I_solution at the specified time
    specific_I = I_solution[index] 
    
    
    y_list = np.append(y_list, specific_I)
    x_list = np.append(x_list, specific_time)

    
    # List of specific times
    time_list = np.linspace(0,T,m)


    # Interpolate R_0_interp values at the specific times
    sensor_val_list = np.interp(time_list, t_interp, R_0_interp)                                               
    
    sensor_values.append(sensor_val_list)

    

    
    
    
# Resize the list to match the format
x_list = np.resize(x_list, (len(x_list),1))
y_list = np.resize(y_list, (len(y_list),1))


num = num_train


# Training set
sensor_values_1 = sensor_values[:num]
x_list_1 = x_list[:num]
y_list_1 = y_list[:num]

X_train = [sensor_values_1, x_list_1]
y_train = y_list_1


# Test set
sensor_values_2 = sensor_values[num:]
x_list_2 = x_list[num:]
y_list_2 = y_list[num:]


X_test = [sensor_values_2, x_list_2]
y_test = y_list_2


# Generate exact solution to compare with prediction of the network

t_test1, R_test1 = R0(T)
t_test2, R_test2 = R0(T)

I_test1 = odeint(model, I0, t_test1, args=(t_test1,R_test1))
I_test2 = odeint(model, I0, t_test2, args=(t_test2,R_test2))

def get_R_01(t):
    "Function to test the network"
    return np.interp(t, t_test1, R_test1) 

def get_R_02(t):
    "Another function to test the network"
    return np.interp(t, t_test2, R_test2) 

t1 = np.linspace(0,29,1000)

sol1 = get_R_01(t1)
sol2 = get_R_02(t1)

plt.figure()

plt.plot(t1,sol1)
plt.title('R_0(t)')
plt.xlabel('Time [day]')
plt.ylabel('R_0')

plt.figure()
plt.plot(t1,sol2)
plt.title('R_0(t)')
plt.xlabel('Time [day]')
plt.ylabel('R_0')
