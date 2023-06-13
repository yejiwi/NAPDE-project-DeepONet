from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import deepxde as dde
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test

import scipy

'''
Tutorial:
    
    Test DeepONet with nonlinear ode examples

'''

def fun(x, s, u):
    '''
    Example nonlinear ode to test 

    '''
    return -s**2 + u(x)


def fun2(x, s, R):
    
    '''
    Our ODE, dI/dt
    '''
    
    gamma = 0.1
    return R(x) * gamma * (1-s) * s - gamma * s




# Define the initial condition
s0 = 0

# R_0 value when it is constant 
R_const = 1.5


# Define the input function u(x)
def u(x):
    return x**2 + x

def R(x):
    return R_const



# Define the length of period
T = 1


#............................Solve the ODE using numerical method

# Define the range of x values
x_span = [0, T]

# Define the desired time step size
dt = 0.01

# Generate the time points with the time step sizem dt
t_eval = np.arange(x_span[0], x_span[1], dt)

# Solve the ODE with the specified time points
sol = scipy.integrate.solve_ivp(fun, x_span, [s0], method = 'LSODA', args=(u,), t_eval=t_eval)


# Access the solution
x = sol.t
s = sol.y[0]

# Plot the solution
plt.plot(x,s,'.')
plt.xlabel('x')
plt.ylabel('s(x)')

# Print the solution
for i in range(len(x)):
    print(f"x = {x[i]}, s = {s[i]}")

#...................................................................


def test_u_ode(nn, system, T, m, model, data, u, fname, num=100):
    """Test ODE"""
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values = u(sensors)
    x = np.linspace(0, T, num=num)[:, None]
    X_test = [np.tile(sensor_values.T, (num, 1)), x]
    y_test = system.eval_s_func(u, x)
    if nn != "opnn":
        X_test = merge_values(X_test)
    y_pred = model.predict(data.transform_inputs(X_test))
    np.savetxt(fname, np.hstack((x, y_test, y_pred)))
    print("L2relative error:", dde.metrics.l2_relative_error(y_test, y_pred))
    
    return X_test[1], y_pred


#......................................Different ODE system 
'''

def ode_system(T):
    """Antiderivative ODE system"""

    def g(s, u, x):
        # Antiderivative
        return u

    s0 = [0]

    return ODESystem(g, s0, T)

'''




def ode_system(T):
    """nonlinear ODE system"""

    def g(s, u, x):

        return -s**2 + u

    s0 = [0]
    

    return ODESystem(g, s0, T)

'''


def ode_system(T):
    """Our ODE system, dI/dt"""

    def g(s, u, x):
                  
        gamma = 0.1
        return u * gamma * (1-s) * s - gamma * s
        
    s0 = [0.1] # Initial value of I
    
    return ODESystem(g, s0, T)

'''

#........................................................................

def run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test):
    # space_test = GRF(1, length_scale=0.1, N=1000, interp="cubic")

    X_train, y_train = system.gen_operator_data(space, m, num_train)
    X_test, y_test = system.gen_operator_data(space, m, num_test)
    if nn != "opnn":
        X_train = merge_values(X_train)
        X_test = merge_values(X_test)


    X_test_trim = trim_to_65535(X_test)[0]
    y_test_trim = trim_to_65535(y_test)[0]
    if nn == "opnn":
        data = dde.data.OpDataSet(
            X_train=X_train, y_train=y_train, X_test=X_test_trim, y_test=y_test_trim
        )
    else:
        data = dde.data.DataSet(
            X_train=X_train, y_train=y_train, X_test=X_test_trim, y_test=y_test_trim
        )

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=[mean_squared_error_outlier])
    checker = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", save_better_only=True, period=10
    )
    losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])
    print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
    safe_test(model, data, X_test, y_test)
    
    
    # Function to test the trained model
    tests = [
        (lambda x: x**2 + x, "x.dat"),
        #(lambda x: np.ones_like(x)*R_const, 'x.dat'),

    ]
    
    for u, fname in tests:

        if problem == "ode":
            
            # Predicted solution of trained model
            x_pred, y_pred = test_u_ode(nn, system, T, m, model, data, u, fname)
            
            plt.plot(x_pred,y_pred, '.',label = 'Predicted')
            plt.plot(x,s,'.', label = 'True')
            plt.xlabel('x')
            plt.ylabel('s(x)')
            #plt.xlabel('time [day]')
            #plt.ylabel('I(t)')
            #plt.title('R_0(t) = %s' %R_const)

            plt.legend()
            

def main():

    problem = "ode"
    

    if problem == "ode":
        system = ode_system(T)
    

    # Function space
    # space = FinitePowerSeries(N=100, M=1)
    # space = FiniteChebyshev(N=20, M=1)
    # space = GRF(2, length_scale=0.2, N=2000, interp="cubic")  # "lt"
    space = GRF(T, length_scale=0.1, N=3000, interp="cubic")
    # space = GRF(T, length_scale=0.2, N=1000 * T, interp="cubic")

    # Hyperparameters
    
    # Number of sensors
    m = 100
    
    # Number of training set
    num_train = 1000
    
    # Number of test set
    num_test = 2000
    
    # Learning rate 
    lr = 0.01
    
    # Number of epochs
    epochs = 100

    # Network
    nn = "opnn"
    activation = "relu"
    initializer = "Glorot normal"  # "He normal" or "Glorot normal"
    dim_x = 1 
    if nn == "opnn":
        net = dde.maps.OpNN(
            [m, 40, 40],
            [dim_x, 40, 40],
            activation,
            initializer,
            use_bias=False,
            stacked=False,
        )
    elif nn == "fnn":
        net = dde.maps.FNN([m + dim_x] + [100] * 2 + [1], activation, initializer)
    elif nn == "resnet":
        net = dde.maps.ResNet(m + dim_x, 1, 128, 2, activation, initializer)

    run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test)


if __name__ == "__main__":
    main()
