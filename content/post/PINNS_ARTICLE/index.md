---
title: PINNs, Not the Spiky Kind.
date: 2023-07-28
math: true
summary: An investigation into Physics Informed Neural Networks and their
  ability to solve forward/inverse problems with PDEs.
authors:
  - alexander-cheetham
commentable: false
image:
  placement: 2
  caption: "Image credit: [**Lexica Aperture
    v2**](https://lexica.art/prompt/79d0a820-627a-48c7-88eb-05ae0a63eba2)"
---
## Physics Informed Neural Networks (PINNs)

Physics-informed neural networks (PINNs) are a deep learning framework for solving partial differential equations (PDEs). PINNs work by adding the PDE to the loss function of a neural network and using automatic differentiation to train the network. This allows PINNs to solve PDEs that are difficult or impossible to solve with traditional numerical methods, such as finite element methods (FEM).

 PINNs are particularly well-suited for solving problems where the local consistency of FEM is broken. This can happen in situations where the PDE is sensitive to initial conditions, such as in chaotic differential equations or in integro-differential equations. In these situations, PINNs can often achieve much higher accuracy than FEM. Invented in 2017 by M. Raisse et al. in two papers [^PINNPAPER] for the forward/inverse problems, which are defined as follows:
- Foward Problem: the underlying equation and the boundary conditions are known, and the goal is to find the solution to the equation.
- Inverse Problem:  The underlying equation and the boundary conditions are known, but the values of some of the parameters in the equation are unknown. The goal is to find the values of these parameters, given experimental data.

In both forward and inverse problems, PINNs are trained on a dataset of known solutions to the underlying equation. This dataset can be generated using numerical methods, such as the finite element method, finite difference method or from physical experiments. This article explores PINNS by generating a solution for the Sprott attractor (forward problem) and the solution to an inverse problem involving the Navier-Stokes equations.

## Deep Neural Networks
In PINNS, the neural network(NN) is placed in the position of the solution to the PDE.The PDE is defined on the domain $\Omega \in \mathbb{R}^d$ with a solution $u(\mathbf{x})$ approximated by the neural network $f$ parameterised by $\lambda$. Where the input vector as $\mathbf{x}=(x\_{1},...,x\_{d})^T$.
{{< math >}}
$$
f(\mathbf{x};\frac{\delta u}{\delta x\\_{1}},...,\frac{\delta u}{\delta x\\_{d}};\frac{\delta^2 u}{\delta x\\_1^2},...,\frac{\delta^2 u}{\delta x\\_1\delta x\\_d},...;\lambda).

$${{< /math >}}

Alongside suitable boundary conditions $\mathcal{B}(u,\mathbf{x})=0 \text{ on } \delta \Omega$. Although many NN architectures exist, the multi-layer perceptron is suitable for solving most PDEs and as such for the remainder of the article the architecture will be assumed to be a fully connected feed-forward network.
## Adapted Loss Function
It was stated in the first paragraph that PINNs operate by $\textcolor{red}{\text{incorporating the PDE into the loss function}}$. So in the usual manner the model, loss and training dataset  are denoted as $f$, $V$ and $\mathcal{T}$ respectively. Then $\mathbf{\theta}$ is defined as the weights and biases in the NN and $\mathcal{T}=\mathcal{T}\_f + \mathcal{T}\_b$ are two training sets where $\mathcal{T}\_f \in \Omega$ are the points in the domain and $\mathcal{T}\_b \in \Omega$ are the problem boundary conditions. Then the loss function can be setup as
{{< math >}}
$$
\ell(\mathcal{T},f(x;\mathbf{\theta}),R) =   \omega\\_f \textcolor{red}{R(f(\mathbf{x};\mathbf{\theta}),\mathcal{T}\\_f)} + \omega\\_b  V(f(\mathbf{x};\mathbf{\theta}),\mathcal{T}\\_b)
$$
{{< /math >}}
where for PINNs
{{< math >}}
$$
\textcolor{red}{R(f(\mathbf{x};\mathbf{\theta}),\mathcal{T}\\_f)}=\frac{1}{|\mathcal{T}\\_f|} \underset{x \in \mathcal{T}_f}{\sum} \lVert f(\mathbf{x};\frac{\delta u}{\delta x\\_{1}},...,\frac{\delta u}{\delta x\\_{d}};\frac{\delta^2 u}{\delta x\\_1^2},...,\frac{\delta^2 u}{\delta x\\_1\delta x\\_d},...;\lambda) \rVert\\_2^2
$$
{{< /math >}}
{{< math >}}
$$
V(f(\mathbf{x};\mathbf{\theta}),\mathcal{T}\\_b)=\frac{1}{|\mathcal{T}\\_b|} \underset{x\\_i \in \mathcal{T}_b}{\sum} \lVert f(x\\_i;\mathbf{\theta}) - \mathcal{B}(x\\_i) \rVert\\_2^2
$$
{{< /math >}}
So there are two terms for the loss the loss induced on the boundary conditions and the loss on the domain that are simultaneously backpropogated to train the model. Observed or experimental data can be added into the boundary condition loss set as $(x\_i,\mathcal{B}(x\_i))$ pairs. Building upon the classic misty hill analogy for gradient descent[^mistyhill], the $\textcolor{red}{R(f(\mathbf{x};\mathbf{\theta}),\mathcal{T}\_f)}$ (the loss induced by the governing PDE) provides a set of steps down the hill that act as a good heuristic to improve the chances of getting to the bottom (global minima and solution to the PDE). For example if the aim is to approximate a solution to the diffusion equation $\frac{\delta^2 f}{\delta x^2}= \frac{1}{\kappa} \frac{\delta f}{\delta t}$ then the corresponding PDE loss would be $\underset{f}{argmin}\\,\frac{\delta^2 f}{\delta x^2}- \frac{1}{\kappa} \frac{\delta f}{\delta t}$ as the second expression is the corresponding minimisation task that corresponds to the NN obeying the PDE.
{{< figure src="./PINNS/grad_desc.png" caption="Gradient descent as steps on a misty hill" numbered="true" >}}
## Automatic Differentiation
During the backpropogation process the traditional aim is to minimise the derivatives $\frac{\delta \ell(f)}{\delta x\_1},...,\frac{\delta \ell(f) }{\delta x\_d}$ i.e the loss of the model with respect to the inputs. In the process of computing these derivatives we also end up computing the loss with respect to each model parameter and can then utilise a weight update algorithm such as ADAM[^adam], L-BFGS[^L-BFGS] or the traditional $w\_{t+1}=w\_t + \eta \frac{\delta \ell(f)}{\delta w}$. In PINNS **we now have the conditions of the PDE to enforce on the NN as well through the addition of the PDE error to the loss function**. Furthermore, the advantage is notable over FEM as NNs only require one pass of the data through the approximating function however FEM require the computation of both $\forall i \\, f(x\_1,...x\_i,...,x\_d)$ and $f(x\_1,...x\_i+ \Delta x\_i,...,x\_d) $. Meaning for large $dim(\mathbf{x})$ automatic differentiation is much more efficient.


## The PINN
We now formalise the PINN algorithm and NN structure, copying from Lu Lu and DEEPXDE[^deepxde].
1. Construct a Neural Network with accompanying parameters,
2. Create the two training datasets $\mathcal{T}\_f,\mathcal{T}\_b$,
3. Specify the loss function and weights associated to $\omega\_f, \omega\_b$,
4. Train the Neural Network to find find the best paramets that minimise the loss function.

{{< figure src="./PINNS/PINN_schematic.png" caption="Schematic of a PINN for solving the diffusion equation credit to Lu Lu et al. and DEEPXDE" numbered="true" >}}

## Forward Problem: Sprott Atractor
The Sprott Attractor[^sprott] is a chaotic dynamical system proposed in a 2014 paper that exhibits sensitivity to initial conditions and exhibits complex structure. It will be treated as a forward problem and therefore know in advance the parameters $\alpha, \beta$ and select $t \in [0,20]$ as the domain to learn on.
{{< math >}}
$$
 f(t)=\begin{cases} 
      \frac{dx}{dt} = y +\alpha xy + xz \\
      \frac{dy}{dt} = 1 - \beta x^2 + y*z  \\
      \frac{dz}{dt} = x - x^2 - y^2
   \end{cases},\quad \alpha=2.07, \beta = 1.79

$$
{{< /math >}}
### Data/PDE Setup
```python
import deepxde as dde
import numpy as np

geom = dde.geometry.TimeDomain(0, 20)
dde.config.set_default_float("float64")
a = 2.07
b = 1.79


def ode_system(t, Y):
    # https://www.dynamicmath.xyz/strange-attractors/ sprot attractor

    X = Y[:, 0:1]
    y = Y[:, 1:2]
    z = Y[:, 2:3]
    dX_t = dde.grad.jacobian(Y, t, i=0)
    dy_t = dde.grad.jacobian(Y, t, i=1)
    dz_t = dde.grad.jacobian(Y, t, i=2)
    return [
        dX_t - (y + (X * a * y) + (X * z)),
        dy_t - 1 + b * X * X - y * z,
        dz_t - X + X * X + y * y,
    ]
```
### Neural Network Setup
```python
n = 10
x_true, y_true, z_true = gen_truedata(n)

t = np.linspace(0, 20, n).reshape(n, 1)
observe_x = dde.icbc.PointSetBC(t, x_true, component=0)
observe_y = dde.icbc.PointSetBC(t, y_true, component=1)
observe_z = dde.icbc.PointSetBC(t, z_true, component=2)
data = dde.data.PDE(geom,ode_system,[observe_x,observe_y,observe_z],40,10,anchors=t,num_test=100)
layer_size = [1] + [20] *4 + [3]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)

def input_transform(t):
    return tf.concat(
        (
            t,
            tf.sin(t),
            tf.sin(2 * t),
            tf.sin(3 * t),
            tf.sin(4 * t),
            tf.cos(5 * t),
            tf.sin(6 * t),
        ),
        axis=1,
    )


net.apply_feature_transform(input_transform)
```
### Train
```python
model = dde.Model(data, net)
model.compile("adam", lr=1e-3,)
losshistory, train_state =model.train(iterations=200,display_every=1,disregard_previous_best=False, ) 
model.save(save_path="./strange_attract-10t/model")
```
### Results

{{< figure style='width:20%' src="./PINNS/training.gif" caption="Training animation for Sprott Attractor" numbered="true" >}}
{{< figure style='width:20%' src="./PINNS/solution.gif" caption="Solution Plot" numbered="true" >}}



## Inverse Problem: Inlet/Outlet Cavity
The problem considered is one created by John Burkardt for the paper 'Centroidal voronoi tesselation-based reduced-order modelling of complex systems'[^inlet] as the problem considered in the paper has an accompanying dataset[^dataset] of $(u,v)$ pairs for the domain that can be used for $\mathcal{T}_b$.
The spatial domain is defined as $\Omega = (0,1)\times(0,1)$ and time domain $(0,T)$ and use the 2d Navier-Stokes Equations with the following boundary conditions:
$$
 \begin{align}
 \frac{\delta \mathbf{u}}{\delta t} - \nu \Delta \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u} + \nabla p = 0 \quad \text{in} \quad (0,T)\times\Omega \\\
  \nabla \cdot \mathbf{u} = 0 \quad \text{in} \quad (0,T)\times\Omega \\\
  \mathbf{u}(0,x)=\mathbf{u}\_0(x) \quad \text{in} \quad \Omega \\\
  \\\
  \mathbf{u}=5*(1-(y-0.1)^2/0.01) \quad \text{on} \quad  (0,T)\times\Gamma\_i \\\
  (\mathbf{n}\cdot\nabla)\mathbf{u}=0 \quad \text{on} \quad (0,T)\times\Gamma\_o \\\
  \mathbf{u}=0\quad \text{on} \quad (0,T)\times\Gamma\_d
 \end{align}
$$
{{< figure style='width:20%' src="./PINNS/cavity.jpg" caption="Spatial Domain $\Omega$" numbered="true" >}}
### Data/PDE Setup
```python
import deepxde as dde
import numpy as np
from create_datasets import create_dataset
# Parameters to be identified
C1 = dde.Variable(0.8)
C2 = dde.Variable(1 / 300)

def Navier_Stokes_Equation(x, y):
    """
        System of PDEs to be minimized: incompressible Navier-Stokes equation in the
        continuum-mechanics based forC2lations.

        """
    psi, p, u_real, v_real = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4],

    p_x = dde.grad.jacobian(p, x, i=0,j=0)
    p_y = dde.grad.jacobian(p, x, i=0,j=1)

    u = dde.grad.jacobian(psi, x, i=0, j=1)
    v = - dde.grad.jacobian(psi, x, i=0, j=0)

    u_x = dde.grad.jacobian(u, x, i=0, j=0)
    u_y = dde.grad.jacobian(u, x, i=0, j=1)
    u_t = dde.grad.jacobian(u, x, i=0, j=2)

    v_x = dde.grad.jacobian(v, x, i=0, j=0)
    v_y = dde.grad.jacobian(v, x, i=0, j=1)
    v_t = dde.grad.jacobian(v, x, i=0, j=2)

    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    dv_xx = dde.grad.hessian(v, x, i=0, j=0)
    dv_yy = dde.grad.hessian(v, x, i=1, j=1)
    continuity = u_x + v_y
    x_momentum = u_t + C1 * (u * u_x + v * u_y) + p_x - C2 * (du_xx + du_yy)
    y_momentum = v_t + C1 * (u * v_x + v * v_y) + p_y - C2 * (dv_xx + dv_yy)
    u_diff = u_real - u
    v_diff = v_real - v
    return continuity, x_momentum, y_momentum, u_diff, v_diff

space_domain = dde.geometry.Rectangle([0, 0], [1, 1])
time_domain = dde.geometry.TimeDomain(0, 6)
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)
```
### Neural Network Setup
```python
[ob_x, ob_y, ob_t, ob_u, ob_v] = create_dataset('data_full.csv',5000)
ob_xyt = np.hstack((ob_x, ob_y, ob_t))
observe_u = dde.icbc.PointSetBC(ob_xyt, ob_u, component=2)
observe_v = dde.icbc.PointSetBC(ob_xyt, ob_v, component=3)

data = dde.data.TimePDE(
    geomtime,
    Navier_Stokes_Equation,
    [observe_u, observe_v],
    num_domain=700,
    num_boundary=200,
    num_initial=100,
    anchors=ob_xyt,
    num_test=3000
)

layer_size = [3] + [50] * 6 + [4]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
model = dde.Model(data, net)
```
### Train
```python
model.compile("adam", lr=1e-3, external_trainable_variables=[C1, C2])
loss_history, train_state = model.train(
    iterations=10000, callbacks=[variable], display_every=10, disregard_previous_best=False,)
model.save(save_path="./psi_pinn_nt/model")
```
### Results
{{< figure style='width:20%' src="./PINNS/ns.gif" caption="Solution Plot" numbered="true" >}}


[^PINNPAPER]:M. Raissi, P. Perdikaris, G.E. Karniadakis,
[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://doi.org/10.1016/j.jcp.2018.10.045)

[^mistyhill]:[What can we learn from gradient descent?](https://ayush-98282.medium.com/what-can-we-learn-from-gradient-descent-8ef0827902e1)
[^adam]:[Kingma, D. P., & Ba, J. L. (2015). Adam: A method for stochastic optimization.](https://arxiv.org/abs/1412.6980)
[^L-BFGS]:[Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method for large scale optimization.](https://link.springer.com/article/10.1007/bf01589116)
[^deepxde]:[LU LU, XUHUI MENG, ZHIPING MAO, AND GEORGE EM KARNIADAKIS, DEEPXDE: A DEEP LEARNING LIBRARY FOR SOLVING DIFFERENTIAL EQUATIONS](https://arxiv.org/pdf/1907.04502.pdf)
[^sprott]:[J.C. Sprott, A dynamical system with a strange attractor and invariant tori](https://sprott.physics.wisc.edu/pubs/paper423.pdf)
[^inlet]:[JOHN BURKARDT, MAX GUNZBURGER, AND HYUNG-CHUN LEE, CENTROIDAL VORONOI TESSELLATION-BASEDREDUCED-ORDER MODELING OF COMPLEX SYSTEMS](https://people.sc.fsu.edu/~jburkardt/publications/MR2231716.pdf)
[^dataset]:[INOUT_FLOW2 INOUT Flow Problem Solution Datasets](https://people.sc.fsu.edu/~jburkardt/datasets/inout_flow2/inout_flow2.html)