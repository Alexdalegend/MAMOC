---
title: PINNs - An Introductory Explanation
date: 2023-07-28
math: true
summary: An investigation into Physics Informed Neural Networks and their
  ability to solve forward/inverse problems with PDEs.
authors:
  - alexander-cheetham
image:
  placement: 2
  caption: "Image credit: [**Lexica Aperture
    v2**](https://lexica.art/prompt/79d0a820-627a-48c7-88eb-05ae0a63eba2)"
---
## Physics Informed Neural Networks (PINNs)

PINNs are a machine learning model that can be used to solve forward/inverse problems involving partial differential equations(PDEs). In particular utilising regularisation through physics laws to reduce overfitting and improve performance on unseen or noisy data. Invented in 2017 by M. Raisse et al. in two papers[^1] on the forward/inverse problems respectively

## Breaking down the definition

### Regularisation

Regularisation is a technique to improve both training speed and model performance of a Neural Network (NN) by adding a penalty term to the loss function to enforce a characteristic on the model. In the usual manner we denote our model as $f$ and $V$ as the loss function that measures the cost of a prediction {{< math >}}  $f(x_{*i})$* {{< /math >}}  and a label $y$. Then denote the regulariser function as $ R(f) $ . Finally we can define our objective function as:

{{< math >}}
$$
﻿\underset{f}{argmin} \sum\\_{i=1}^{N}V(f(x\\_i),y\\_i) + \lambda R(f)
$﻿$

{{< /math >}}

tteee{{< math >}}$\nabla F(\mathbf{x}_{n})${{< /math >}}sss
For example overfitting is prevented by penalising large weights in the model or ensuring model predictions obey the laws of physics in the case of PINNs, explained in depth later. Simply, building on the misty hill gradient descent analogy[^2]  regularisation acts as a path on the misty hill and we penalise any step in a direction that deviates from the path as that would be silly.
{{< figure src="./PINNS/grad_desc.png" caption="Gradient Descent as steps on a misty hill" numbered="true" >}}

### Forward PDE problem

### Inverse PDE problem

\[﻿^1]:M. Raissi, P. Perdikaris, G.E. Karniadakis,
[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://doi.org/10.1016/j.jcp.2018.10.045)

\[﻿^2]:[What can we learn from gradient descent?](https://ayush-98282.medium.com/what-can-we-learn-from-gradient-descent-8ef0827902e1)