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

from training_data_set import X_train, y_train, X_test, y_test, m, num_train, num_test





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


def ode_system(T):
    """ODE"""

    def g(s, u, x):
        
    # beta = R_0 * gamma
    #     
    # s : Output function, I(t)
    # u : Input function, R_0
    
    
    # Our case: dI/dt = beta * S * I - gamma * I 
    #                 = R_0 * gamma * (1 - I) * I - gamma * I 
    # S + I = 1
    # R_0 is u. I is s.
     
        gamma = 0.1 # Recovery rate
        
        
        return u * gamma * (1-s) * s - gamma * s
   
    # Initial value of Infected
    s0 = [0.1]
    
    
    return ODESystem(g, s0, T)


def run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test):
    # space_test = GRF(1, length_scale=0.1, N=1000, interp="cubic")      
    # X_train: beta(t)
    # y_train: I(t)       
    global X_train, y_train, X_test, y_test
    
    ## X_train, y_train = system.gen_operator_data(space, m, num_train)


    
    ## X_test, y_test = system.gen_operator_data(space, m, num_test)
    
    
    if nn != "opnn":
        #X_train = merge_values(X_train)
        X_test = merge_values(X_test)

    # np.savez_compressed("train.npz", X_train0=X_train[0], X_train1=X_train[1], y_train=y_train)
    # np.savez_compressed("test.npz", X_test0=X_test[0], X_test1=X_test[1], y_test=y_test)
    # return

    # d = np.load("train.npz")
    # X_train, y_train = (d["X_train0"], d["X_train1"]), d["y_train"]
    # d = np.load("test.npz")
    # X_test, y_test = (d["X_test0"], d["X_test1"]), d["y_test"]

    X_test_trim = trim_to_65535(X_test)[0]
    y_test_trim = trim_to_65535(y_test)[0]
    

    
    plt.show()
    plt.scatter(X_test_trim[1],y_test_trim)
    
    
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
    checker = dde.callbacks.ModelCheckpoint("model/model.ckpt", save_better_only=True, period=10)
    #....................


    losshistory, train_state = model.train(epochs=epochs, callbacks=[checker])
    print("# Parameters:", np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]))
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    model.restore("model/model.ckpt-" + str(train_state.best_step), verbose=1)
    
    #Safe TEST........................................
    
    safe_test(model, data, X_test, y_test)

    tests = [
        (lambda x: x, "x.dat"),
        (lambda x: np.sin(np.pi * x), "sinx.dat"),
        (lambda x: np.sin(2 * np.pi * x), "sin2x.dat"),
        (lambda x: x * np.sin(2 * np.pi * x), "xsin2x.dat"),
    ]
    for u, fname in tests:
        if problem == "ode":
            test_u_ode(nn, system, T, m, model, data, u, fname)
       


def main():
    # Problem:

    # - "ode": Antiderivative, Nonlinear ODE, Gravity pendulum
    
    
    problem = "ode"
    
    T = 29 # Final time value
    
    system = ode_system(T)
    
    
    # Function space
    # space = FinitePowerSeries(N=100, M=1)
    # space = FiniteChebyshev(N=20, M=1)
    # space = GRF(2, length_scale=0.2, N=2000, interp="cubic")  # "lt"
    space = GRF(T, length_scale=0.2, N=1000, interp="cubic")
    # space = GRF(T, length_scale=0.2, N=1000 * T, interp="cubic")
    
    # Hyperparameters
    #m = 10 # numer of sensors
    
    
    #num_train = 1000 # number of training set
    
    #num_test = 1000 # number of test set
    lr = 0.005 # learning rate
    epochs = 200
    


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
            use_bias=True,
            stacked=False,
        )
    elif nn == "fnn":
        net = dde.maps.FNN([m + dim_x] + [100] * 2 + [1], activation, initializer)
    elif nn == "resnet":
        net = dde.maps.ResNet(m + dim_x, 1, 128, 2, activation, initializer)

    run(problem, system, space, T, m, nn, net, lr, epochs, num_train, num_test)
    




if __name__ == "__main__":
    main()

