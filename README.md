# DeepONet for epidemic modelling

DeepONet is an artificial neural network framework to solve partial differential equations (PDEs). Deep learning techniques are used to approximate the solutions of PDE by directly learning a mapping between input variables and the corresponding PDE solutions. We adapt DeepONet to learn an operator mapping basic reproduction number $R_0$ to the solution of compartment model, $I(t)$ in SIS model.


# How the code works
First, install deepxde https://github.com/lululxvi/deepxde

Then download all the scripts in this repository.

First, run trainig_data_set.py to generate training and test data set.

Then run deeponet_pde.py to train and see the result.


# Warm-up: Test the code with nonlinear ODE
Code: deeponet_pde_ex.py


Instead of jumping right into the problem, we test the DeepONet on nonlinear ODE to get familiar with it. With DeepONet, we will solve the ODE $\frac{dS(x)}{dx} = -s(x)^2 + u(x)$ given $u(x) = x^2 + x$ with an initial condition $s(0) = 0$.

We compare the solution using numerical method with the solution that DeepONet predicted after trainig. 

Number of sensors: 100

Number of traning set: 1000

Number of test set: 2000

Learning rate: 0.01

Epochs: 100

![Alt text](/Example.png)

L2 relative error: 0.10766981541574315

# Case1: $R_0(t)$ is constant

![Alt text](/R=1.05.png)

![Alt text](/R=2.1.png)

![Alt text](/R=2.3.png)

# Case2: $R_0(t)$ is time dependent
