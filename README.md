# Optimal loss functions for distributionally robust optimization (DRO) of neural networks

## Project overview

The code in this repository is used to conduct empirical studies for the Part C Computer Science project at the University of Oxford.

As neural networks are increasingly widely applied in safety-critical systems (e.g. autonomous vehicles), it is essential to ensure safety of systems involving neural networks. 
It has been discovered that despite their stellar generalization perfoemance, neural networks are surprisingly vulnerable to so-called adversarial perturbations in computer vision; i.e. small and oftentimes imperceptible perturbations to an input image that can trick the neural networks into misclassification of the image. 
One promising approach to improving robutness of neural networks to adversarial perturbations is adversarial training, whereby neural networks are trained using not only the original training data but also adversarial examples that can be generated from the training data. 

In this project, I investigate the relationship between (i) loss functions used in training feedforward neural networks and (ii) the robustness of neural networks that are trained by distributionally robust optimization (DRO), which is a variant of adversarial traning. 

I specifically consider the following DRO algorithms:
1. WRM developed by Sinha et al. ([paper](https://arxiv.org/abs/1710.10571))
2. FWDRO developed by Staib and Jegelka ([paper](https://machine-learning-and-security.github.io/papers/mlsec17_paper_30.pdf))
3. Distributional projected gradient descent (PGD). 

The loss functions examined in this project come from the paper by Carlini and Wagner ([paper](https://arxiv.org/abs/1608.04644)).
