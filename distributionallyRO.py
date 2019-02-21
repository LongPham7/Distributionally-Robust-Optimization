import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_MNIST import randomStart
from adversarial_training import AdversarialTraining

class ProjetcedDRO(AdversarialTraining):
    """
    Execute distributionally robust optimization (DRO) using the Euclidean
    projection in the adversarial attack. This class is applicable only when
    the underlying distance is the L2-norm and the distributional distance is
    the 2-Wasserstein distance (i.e. W2). 
    """

    def __init__(self, model, loss_criterion):
        super().__init__(model, loss_criterion)

    def attack(self, budget, data, steps=15):
        lr = 0.001
        images, labels = data
        images_adv = images.clone().detach()
        images_adv.requires_grad_(True)
        images_adv = images_adv.to(self.device)

        desirable_distance = budget * math.sqrt(images.size()[0])

        # Choose a random strating point where the constraint for perturbations
        # is tight. Without randomly choosing a starting point, the adversarial
        # attack fails most of the time because the loss function is flat near
        # the training input, which was used in training the neural network.  
        randomStart(images_adv, budget)
        for i in range(steps):
            if images_adv.grad is not None:
                images_adv.grad.data.zero_()
            outputs = self.model(images_adv)
            loss = self.loss_criterion(outputs, labels)
            loss.backward()
            images_adv.data.add_(lr * images_adv.grad)
            diff_tensor = images.detach() - images_adv.detach()
            diff_tensor = diff_tensor.to(self.device)
            distance = torch.norm(diff_tensor, p=2).item()

            # Inside this conditional statement, we can be certain that
            # distance > 0, provided that budget > 0. 
            # Hence, there is no risk of division by 0. 
            if distance > desirable_distance:
                images_adv.data.add_((1 - (desirable_distance / distance)) * diff_tensor)
            images_adv.data.clamp_(0, 1)     
        return images_adv, labels

class LagrangianDRO(AdversarialTraining):
    """
    Execute DRO using the Lagrangian relaxation of the original theoretical
    formulation of DRO. This approach is developed by Sinha, Namkoong, and
    Duchi (2018). 
    """

    def __init__(self, model, loss_criterion, cost_function):
        """
        Initialize instance variables

        Arguments:
            cost_function: underlying distance metric for the instance space
        """

        super().__init__(model, loss_criterion)
        self.cost_function = cost_function

    def attack(self, budget, data, steps=15):
        """
        Launch an adversarial attack using the Lagrangian relaxation.

        Arguments:
            budget: gamma in the original paper. Note that this parameter is
                different from the budget parameter in other DRO classes. 
        """
        
        images, labels = data
        images_adv = images.clone().detach()
        images_adv.requires_grad_(True)
        images_adv = images_adv.to(self.device)
        
        for i in range(steps):
            if images_adv.grad is not None:
                images_adv.grad.data.zero_()
            outputs = self.model(images_adv)
            loss = self.loss_criterion(outputs, labels) - budget * self.cost_function(images, images_adv)
            loss.backward()
            images_adv.data.add_(1 / math.sqrt(i+1) * images_adv.grad)
            images_adv.data.clamp_(0, 1)
        return images_adv, labels

class FrankWolfeDRO(AdversarialTraining):
    """
    Execute DRO using the Frank-Wolfe method together with the stochastic
    block coordinate descent (BCD). This approach is developed by Staib and
    Jegelka (2017). 
    """

    def __init__(self, model, loss_criterion, p, q):
        """
        Initialize instance variables.

        Arguments:
            p: distributional distance will be Wp
            q: underlying distance for the instance space will be Lq
        """

        super().__init__(model, loss_criterion)
        assert p > 1 and q > 1
        self.p = p
        self.q = q

    def attack(self, budget, data, steps=15):
        """
        Launch an adversarial attack using the Frank-Wolfe method.
        The algorithm is taken from 'Convex Optimization: Algorithms and
        Complexity' by Bubeck. 
        """

        images, labels = data
        images_adv = images.clone().detach()
        images_adv.requires_grad_(True)
        images_adv = images_adv.to(self.device)
        
        for i in range(steps):
            if images_adv.grad is not None:
                images_adv.grad.zero_()
            outputs = self.model(images_adv)
            loss = self.loss_criterion(outputs, labels)
            loss.backward()

            # desitnation corresponds to y_t in the paper by Bubeck. 
            destination = images_adv.data + self.getOptimalDirection(budget=budget, data=images_adv.grad)
            destination = destination.to(self.device)
            gamma = 2 / (i + 2)
            images_adv.data = (1 - gamma) * images_adv.data + gamma * destination
        return images_adv, labels

    def getOptimalDirection(self, budget, data):
        """
        Calculate the minimizer of a linear subproblem in the Frank-Wolfe
        method. The objective function is linear, and the constraint is
        a mixed-norm ball.

        Instead of calculating a local constraint, I use the same budget
        parameter in every iteration. 

        Arguments:
            budget: epsilon in the paper by Staib et al.
            data: gradient of the total loss with respect to the current
                batch of adversarial examples. This corresponds to C in
                Appendix B of the paper by Staib et al. 
        
        Returns:
            X in Appendix B of Staib et al.'s paper 
        """

        # The number of samples
        batch_size = data.size()[0]

        # 'directions' corresponds to v's in Staib et al.'s paper. 
        directions = data.clone().detach().view((batch_size, -1))
        directions = directions.to(self.device)

        if self.q == float('inf'):
            normalize_dim = float('inf')
            directions = directions.sign()
        elif self.q > 1:
            normalize_dim = 1 / (self.q - 1)
            directions.pow_(normalize_dim)
            directions = F.normalize(directions, p=self.q, dim=1)
        else:
            raise ValueError("The value of q must be larger than 1.")      
        
        # This corresponds to a's in the original paper. 
        products = []
        for i, direction in enumerate(directions):
            sample = data[i].view(-1)
            products.append(torch.dot(direction, sample))
        products = torch.stack(products)
        products = products.to(self.device)

        # This corresponds to epsilons in the original paper. 
        size_factors = products.clone().detach()
        size_factors = size_factors.to(self.device)
        if self.p == float('inf'):
            normalize_dim = float('inf')
            size_factors = size_factors.sign()
        elif self.p > 1:
            normalize_dim = 1 / (self.p - 1)
            size_factors.pow_(normalize_dim)
            distance = torch.norm(size_factors, p=self.p).item()
            size_factors = size_factors / distance # This is now normalized. 
        else:
            raise ValueError("The value of p must be larger than 1.")  
        
        outputs = []
        for i, size_factor in enumerate(size_factors):
            outputs.append(directions[i] * size_factor * budget)
        result = torch.stack(outputs).view(data.size())
        return result.to(self.device)
