from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import tensorflow as tf
from tensorflow import keras

from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test

import deepxde as dde

import config

import matplotlib.pyplot as plt

from training_data_set import X_train, y_train, X_test, y_test, m, num_train, num_test,T

from training_data_set import t1,t2,t3, I1,I2,I3, R_0_const1,R_0_const2, R_0_const3


'''
In this script, using the training and test set generated in
training_data_set.py, we train the model and compare its prediction
with true solution calculated using numerical method.
'''


def test_u_ode(nn, system, T, m, model, data, u, fname, num=m):
    """Test ODE with trained network"""

    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values = u(sensors)
    
    sensor_values = np.array(sensor_values) 
    
    
    x = np.linspace(0, T, num=num)[:, None]
    ####x = np.arange(0, T+1, 1, dtype=int)[:, None]
    
    X_Test = [np.tile(sensor_values.T, (num, 1)), x]
    y_Test = system.eval_s_func(u, x)
    if nn != "opnn":
        X_Test = merge_values(X_Test)
    y_pred = model.predict(data.transform_inputs(X_Test))

    
    np.savetxt(fname, np.hstack((x, y_Test, y_pred)))
    print("L2 relative error:", dde.metrics.l2_relative_error(y_Test, y_pred))
    
    return X_Test[1], y_pred


def ode_system(T):
    """ODE system we want to find solution"""

    def g(s, u, x):
        
    # beta = R_0 * gamma
    
    # u : Input function, R_0
    # s : Output function, I(t), infected

    
    
    # Our case: dI/dt = beta * S * I - gamma * I 
    #                 = R_0 * gamma * (1 - I) * I - gamma * I 
    # S + I = 1
    # R_0 is u. I is s.
     
        gamma = 0.1 # Recovery rate
        
        
        return u * gamma * (1-s) * s - gamma * s
   
    # Initial value of I(t)
    s0 = [0.1]
    
    
    return ODESystem(g, s0, T)


def run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test):
    
    # space_test = GRF(1, length_scale=0.1, N=1000, interp="cubic")      
    # X_train: R_0(t)
    # y_train: I(t)    
    
    # Import the data sets 
    global X_train,y_train,X_test,y_test
    
    ##X_train, y_train = system.gen_operator_data(space, m, num_train)
    ##X_test, y_test = system.gen_operator_data(space, m, num_test)
    
    
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
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
    
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=[mean_squared_error_outlier])
    checker = dde.callbacks.ModelCheckpoint("model/model.ckpt", save_better_only=True, period=10)



    losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])
    print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
    
    safe_test(model, data, X_test, y_test)


    # Inputs to test and evaluate trained model
    tests = [
        (lambda x: R_0_const1*np.ones_like(x), "x1.dat", R_0_const1,t1,I1),
        (lambda x: R_0_const2*np.ones_like(x), "x2.dat", R_0_const2,t2,I2),
        (lambda x: R_0_const3*np.ones_like(x), "x3.dat", R_0_const3,t3,I3),
        
    ]
    
    for u, fname,R,t,I in tests:
        
        x_pred, y_pred = test_u_ode(nn, system, T, m, model, data, u, fname)
 
        plt.figure()
        plt.plot(x_pred,y_pred, '.', label = 'predict')
        plt.plot(t,I,'.', label='real')
        plt.legend()
        plt.title('%s' %R)
    

    
    
    
def main():
    # Problem:

    # - "ode": Antiderivative, Nonlinear ODE, Gravity pendulum
    
    problem = "ode"
    
    #T = 59 # Final time value
    
    system = ode_system(T)
    
    
    # Function space

    space = GRF(T, length_scale=0.1, N=3000, interp="cubic")
    

    lr = 0.01 # learning rate
    epochs = 100
    


    # Network
    nn = "opnn"
    activation = "relu"
    initializer = "Glorot normal"  # "He normal" or "Glorot normal"
    dim_x = 1 if problem in ["ode", "lt"] else 2
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
    

