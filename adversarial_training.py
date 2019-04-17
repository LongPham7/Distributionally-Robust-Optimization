import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from util_MNIST import retrieveMNISTTrainingData
from adversarial_attack import FGSM, PGD


class AdversarialTraining:
    """
    Base class for adversarial training.
    This class does not add any perturbation for adversarial attacks.
    Hence, this class is equivalent to empirical risk minimization (ERM). 
    """

    def __init__(self, model, loss_criterion):
        """
        Initialize instance variables.

        Arguments:
            model: neural network to be trained
            loss_criterion: loss function
        """

        self.model = model
        self.loss_criterion = loss_criterion

        # Use GPU for computation if it is available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        # Load a model on an appropriate device
        self.model.to(self.device)
        print("The neural network is now loaded on {}.".format(self.device))

    def attack(self, budget, data, steps=15):
        """
        Launch an adversarial attack. 
        This is equivalent to solving the inner maximization problem in the
        formulation of RO or DRO. This specific method serves as an abstract
        method and hence does not launch an adversarial attack. In a derived
        class, this method needs to be overridden. 

        Arguments:
            budget: limit on the size of adversarial perturbations.
                This normally corresponds to epsilon in Staib and Jegedlka's
                paper, but in the DRO developed by Sinha et al., the budget
                parameter refers to gamma in their paper. 
            steps: number of iterations in the adversarial attack

        Returns:
            images_adv: adversarially perturbed images (in batch)
            labels: labels of the adversarially perturbed images
        """
        return data

    def train(self, budget, batch_size=128, epochs=25, steps_adv=15):
        """
        Train a neural network (using an adversarial attack if it is defined).
        For optimization, Adam is used. 

        Arguments:
            budget: limit on the size of adversarial perturbations
            batch_size: batch size for training
            epochs: number of epochs in training
            steps_adv: number of iterations inside adversarial attacks

        Returns:
            None
        """

        data_loader = retrieveMNISTTrainingData(batch_size, shuffle=True)
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(epochs):
            for i, data in enumerate(data_loader, 0):
                images, labels = data
                # Input images and labels are loaded by this method.
                # Hence, they do not need to be loaded by the attack method.
                images, labels = images.to(self.device), labels.to(self.device)
                data = (images, labels)

                # However, the attack method should load images_adv on GPU
                # before returning the output.
                images_adv, labels = self.attack(budget, data, steps=steps_adv)

                optimizer.zero_grad()
                outputs = self.model(images_adv)
                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if i % 100 == 99:
                    print("Epoch: {}, iteration: {}".format(epoch, i))


class ProjectedGradientTraining(AdversarialTraining):
    """
    Execute adversarial training using projected gradient descent (PGD).
    The PGD-based attack subsumes FGSM and IFGSM.
    If the number of iterations in an adversarial attack is 1, FGSM is used to
    generate adversarial examples; otherwise, IFGSM/PGD is used. 
    Note that random restart are not used here, unlike in the PGD descirbed by
    Madry et al. 
    """

    def __init__(self, model, loss_criterion, q=2):
        """
        Initialize instance variables.

        Arguments:
            q: underlying distance metric in the instance space
        """

        super().__init__(model, loss_criterion)
        assert q == 1 or q == 2 or q == np.inf
        self.q = q
        self.pgd = PGD(model, loss_criterion, norm=self.q, batch_size=128)
        self.fgsm = FGSM(model, loss_criterion, norm=self.q)

    def attack(self, budget, data, steps=15):
        """
        Launch an FGSM or IGFSM attack. These two are distinguished based on
        the number of iterations for each minibatch. 

        Arguments:
            steps: number of iterations in the adversarial attack.
                    If steps = 1, then this attack is equivalent to FGSM;
                    otherwise, this is IFGSM.
        """

        _, labels = data
        # The output of generatePerturbation should already be loaded on GPU.
        if steps == 1:
            images_adv = self.fgsm.generatePerturbation(
                data, budget, minimal=False)
        else:
            images_adv = self.pgd.generatePerturbation(
                data, budget, max_iter=steps)
        return images_adv, labels
