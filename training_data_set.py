
import numpy as np
import matplotlib.pyplot as plt
from system import ODESystem 
from scipy.integrate import solve_ivp



# This document generates training and test datset to feed the neural network
# The training and test set will be directly imported from this script.

#**************Parameters******************

T = 29 # Max time step
gamma = 0.1 # Recovery rate
m = 10 # Number of sensors
num_train = 5000 # Number of training set
num_test = 5000 # Number of test set

#...............................................



def gen_data(T,m,case):
    
    '''
    Generate data for training set by solving equation using RK4.
    '''
        
    def solve_ode(R_0, gamma, t_range):
       """
       Solve the ordinary differential equation (ODE) dI/dt = R_0 * gamma * (1 - I) * I - gamma * I
       over the range of `x` values specified by `x_range`.
       """
       # Define the initial condition of I
      
       I_0 = 0.1
       
       def ode_func(t, I):
           return R_0 * gamma * (1 - I) * I - gamma * I

       solution = solve_ivp(ode_func, t_range, [I_0], dense_output=True)

       def I(x):
           if t < t_range[0] or t > t_range[1]:
               raise ValueError("The value of x is outside the specified range.")
           return solution.sol(t)[0]

       return I

    if case == 'const':
        # R_0(t) when R_0 is constant over time
        list_one = np.ones(T)
        R_0_const = round(np.random.uniform(low=0.4, high=2.5),1)
        R_0_list= list_one*R_0_const
        
    else:
        # When R_0 is time dependent
        # Generate the list of R_0 values at each t
        R_0_list = round(np.random.uniform(low=0.4, high=2.5, size=T+1),1)   # Example values for R_0 at different time points
        
    
    
    gamma = 0.1
    
    # Define the time span
    t_range = [0, T]  # Start and end time
    
    I_solution = solve_ode(R_0_const, gamma, t_range)
    
    # Define specific times of interest, x

    t_vals = np.random.uniform(low=0, high=29, size = 100)
    
    
    
    # List to store I(x) values
    I_t = []

    # Evaluate I(x) at each x value and store the results
    for t in t_vals:
        I = I_solution(t)
        I_t.append(I)
        
        
    # Sensor points
    sensors = np.arange(0, T+1, (T+1)/m, dtype = int)
    
    # Get R_0 values at sensor points
    
    R_0_vals = R_0_list[sensors]
    
    # Matrix of R_0 values at sensors repeated len(x) times
    # Size = len(x) x (num of sensors)
    R_0_mat = np.tile(R_0_vals,(len(t_vals),1))
        

    return R_0_mat, t_vals, I_t



def train_data(T,m,case,num_train):
    
    '''
    Using data from gen_data, build a data set in the format that 
    the trainig model code requires
    '''

    # Empty array to add values of R_0 at sensor point
    sensor_values = np.empty((0,m))
    
    # List of x points where I will be evaluated
    t_list = np.array([])
    
    # List of I values at points x
    I_t_list = np.array([])
    
    num = 0
    
    while num < num_train:
        
        R_0_mat,t_vals,I_t = gen_data(T,m,case)
        
        # Append the each R_0_mat 
        sensor_values = np.concatenate((sensor_values,R_0_mat))
        
        # Append
        t_list = np.append(t_list, t_vals)
        I_t_list = np.append(I_t_list, I_t)
        
        num = num+1
        
    # Resize the list 
    t_list = np.resize(t_list,(len(t_list),1))
    I_t_list = np.resize(I_t_list,(len(I_t_list),1))
    
    # Final training data set 
    X_train = [sensor_values,t_list]
    y_train = I_t_list
    
    return  X_train, y_train

def gen_test(T):
    
    '''
    Generate solution of ODE to compare with the prediction from model 
    '''
    
    
    def rhs(t, I, R_0_list):
        gamma = 0.1
        # Get the corresponding R_0 value based on the time index
        R_0 = R_0_list[int(t)]  
        return R_0 * gamma * (1 - I) * I - gamma * I
    
    # Initial value of I(t)
    I0 = 0.1

    # Define the time span
    t_span = [0, T]  # Start and end time
    
    # Define the time points at which to evaluate the solution
    t_eval = np.linspace(t_span[0], t_span[1], 30)


    # Define the initial condition
    I0 = 0.1
    
    # Define the time span
    t_span = [0, T]  # Start and end time
    
    # Define the time points at which to evaluate the solution
    t_eval = np.linspace(t_span[0], t_span[1], 30)
    
    # Generate the list of random R_0 values
    list_one = np.ones(T+1)
    R_0_const = np.round(np.random.uniform(low=0.4, high=2.5),1)
    R_0_list= list_one*R_0_const
    
    # Solve the ODE using the RK4 method
    sol = solve_ivp(rhs, t_span, [I0], method='RK45', t_eval=t_eval, args=(R_0_list,))
    
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

