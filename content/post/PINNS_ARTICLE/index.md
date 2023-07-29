---
title: PINNs - An Introductory Explanation
date: 2023-07-28
math: true
image:
  placement: 2
  caption: 'Image credit: [**Lexica Aperture v2**](https://lexica.art/prompt/79d0a820-627a-48c7-88eb-05ae0a63eba2)'
authors:
- alexander-cheetham
summary: An investigation into Physics Informed Neural Networks and their ability to solve forward/inverse problems with PDEs.
---

## Physics Informed Neural Networks (PINNs)
PINNs are a machine learning model that can be used to solve forward/inverse problems involving partial differential equations(PDEs). In particular utilising regularisation through physics laws to reduce overfitting and improve performance on unseen or noisy data. Invented in 2017 by M. Raisse et al. in two papers on the forward/inverse problems respectively.

## Breaking down the definition

### Regularisation
Regularisation is a technique to improve both training speed and model performance of a Neural Network (NN) by adding a penalty term to the loss function to enforce a characteristic on the model. For example overfitting is prevented by penalising large weights in the model or ensuring model predictions obey the laws of physics in the case of PINNs, explained in depth later. Simply, building on the misty hill gradient descent analogy  regularisation acts as a path on the misty hill and we penalise any step in a direction that deviates from the path as that would be silly.
{{< figure src="./PINNS/grad_desc.png" caption="Gradient Descent as steps on a misty hill" numbered="true" >}}


### Forward PDE problem
### Inverse PDE problem
