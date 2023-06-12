# DeepONet for epidemic modelling

DeepONet is an artificial neural network framework to solve partial differential equations (PDEs). Deep learning techniques are used to approximate the solutions of PDE by directly learning a mapping between input variables and the corresponding PDE solutions. We adapt DeepONet to learn an operator mapping basic reproduction number $R_0$ to the solution of compartment model, $I(t)$ in SIS model.

# Warm-up: Test the code with nonlinear ODE

Instead of jumping right into the problem, we test the DeepONet on nonlinear ODE to get familiar with it. With DeepONet, we will solve the ODE $\frac{dS(x)}{dx} = -s(x)^2 + u(x)$ given $u(x) = x^2 + x$ with an initial condition $s(0) = 0$.

We compare the solution using numerical method with the solution that DeepONet guessed after trainig. 





# Case1: $R_0(t)$ is constant

# Case2: $R_0(t)$ is time dependent
