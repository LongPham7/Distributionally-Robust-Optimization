import numpy as np
import torch
from torch import optim, nn
import torch.nn.functional as F
from util_MNIST import retrieveMNISTTestData
from util_model import SimpleNeuralNet, loadModel, wrapModel

from art.attacks import FastGradientMethod, ProjectedGradientDescent

img_rows, img_cols = 28, 28

class FGSM:
    """
    Class for the fast gradient sign method (FGSM).
    This class delegates the implementation of the attack to the ART library
    developed by IBM. 
    """

    def __init__(self, model, loss_criterion, norm, batch_size=128):
        self.pytorch_model = wrapModel(model, loss_criterion)
        self.norm = norm
        self.batch_size = batch_size
        self.attack = FastGradientMethod(self.pytorch_model, batch_size=batch_size)
        
        # Use GPU for computation if it is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generatePerturbation(self, data, budget, minimal=False):
        """
        Generate adversarial examples from a given batch of images. 
        The input data should have already been loaded on an appropriate
        device. 

        Arguments:
            data: pairs of a batch of images and a batch of labels. The batch
                of images should be a numpy array. The batch of labels should
                be a numpy array of integers. 
            budget: the maximal size of perturbation allowed. This parameter
                is not used if minimal = True. 
            minimal: whether the minimal adversarial perturbation is computed.
                If yes, the maximal size of perturbation is 1.0. Consequently,
                the budget parameter is overridden. 
        """

        images, _ = data
        images_adv = self.attack.generate(x=images.cpu().numpy(), norm=self.norm, eps=budget, minimal=minimal, eps_step=0.02, eps_max=1.0, batch_size=self.batch_size)
        images_adv = torch.from_numpy(images_adv)

        # The output to be returned should be loaded on an appropriate device. 
        return images_adv.to(self.device)

class FGSMNative:
    """
    Class for manually implemented FGSM, unlike the above FGSM class in this
    module. For some unknown reason, this class produces a different
    performance in adversarial attacks than the FGSM class. The performance of
    FGSMNative is better than that of FGSM only in some cases. 
    """

    def __init__(self, model, loss_criterion, norm=np.inf, batch_size=128):
        self.model = model
        self.loss_criterion = loss_criterion
        self.norm = norm
        self.batch_size = batch_size
        
        # Use GPU for computation if it is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generatePerturbation(self, data, budget, minimal=False):
        """
        Generate adversarial examples from a given batch of images. 
        The input data should have already been loaded on an appropriate
        device. 

        Note that unlike the FGSM class, in this FGSMNative class, the
        computation of minimal perturbations is not supported. 

        Arguments:
            data: pairs of a batch of images and a batch of labels. The batch
                of images should be a numpy array. The batch of labels should
                be a numpy array of integers. 
            budget: the maximal size of perturbation allowed. This parameter
                is not used if minimal = True. 
            minimal: whether the minimal adversarial perturbation is computed.
                If yes, the maximal size of perturbation is 1.0. Consequently,
                the budget parameter is overridden. 
        """

        images, labels = data
        images_adv = images.clone().detach().to(self.device)
        # We will never need to compute a gradient with respect to images_adv. 
        images_adv.requires_grad_(False)

        images.requires_grad_(True)
        output = self.model(images)
        loss = self.loss_criterion(output, labels)
        loss.backward()
        images.requires_grad_(False)

        if self.norm == np.inf:
            direction = images.grad.data.sign()
        elif self.norm == 2:
            flattened_images = images_adv.view(-1, img_rows * img_cols)
            direction = F.normalize(flattened_images, p=2, dim=1).view(images.size())
        else:
            raise ValueError("The norm is not valid.")
        
        if minimal:
            iterations = 50
            incremental_size = budget / iterations
            minimal_perturbations = torch.zeros(images.size())
            for i in range(iterations):
                outputs = self.model((images_adv + minimal_perturbations).clamp(0, 1))
                _, predicted = torch.max(outputs.data, 1)
                for j in range(labels.size()[0]):
                    # If the current adversarial exampels are correctly
                    # classified, increase the size of the perturbations. 
                    if predicted[j] == labels[j]:
                        minimal_perturbations[j].add_(incremental_size * direction[j])
            images_adv.add_(minimal_perturbations)
        else:
            images_adv.add_(budget * direction)
        
        images_adv.clamp_(0,1)

        # The output to be returned should be loaded on an appropriate device. 
        return images_adv

class PGD:
    """
    Module for adversarial attacks based on projected gradient descent (PGD).
    The implementation of PGD in ART executes projection on a feasible region
    after each iteration. However, random restrating is not used in this
    implementation. Not using radom restarting is the difference between the
    PGD implemented in ART and the one described by Madry et al. 

    This adversarial attack subsumes the iterative FGSM. 
    """

    def __init__(self, model, loss_criterion, norm=np.inf, batch_size=128):
        self.pytorch_model = wrapModel(model, loss_criterion)
        self.norm = norm
        self.batch_size = batch_size
        self.attack = ProjectedGradientDescent(self.pytorch_model, norm=norm, batch_size=batch_size)

        # Use GPU for computation if it is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generatePerturbation(self, data, budget, max_iter=15):
        images, _ = data
        
        # eps_step is not allowed to be larger than budget according to the 
        # documentation of ART. 
        eps_step = budget / 5
        images_adv = self.attack.generate(x=images.cpu().numpy(), norm=self.norm, eps=budget, eps_step=eps_step, max_iter=max_iter, batch_size=self.batch_size)
        images_adv = torch.from_numpy(images_adv)

        # The output to be returned should be loaded on an appropriate device. 
        return images_adv.to(self.device)

if __name__ == "__main__":
    # Load a simple neural network
    model = SimpleNeuralNet()
    loadModel(model, "./ERM_models/SimpleModel.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) # Load the neural network on GPU if it is available
    print("The neural network is now loaded on {}.".format(device))

    # Create an object for PGD
    criterion = nn.CrossEntropyLoss()
    batch_size = 128
    pgd = PGD(model, criterion, batch_size=batch_size)
    pytorch_model = pgd.pytorch_model

    # Read MNIST dataset
    test_loader = retrieveMNISTTestData(batch_size=1024)

    # Craft adversarial examples with PGD
    epsilon = 0.1  # Maximum perturbation
    total, correct = 0, 0
    for i, data in enumerate(test_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        # images_adv is already loaded on GPU by generatePerturbation
        images_adv = pgd.generatePerturbation(data, epsilon)
        with torch.no_grad():
            outputs =  model(images_adv)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        acc = (predicted == labels).sum().item() / labels.size(0)
        print("Iteration: {}; test accuracy on adversarial sample: {}".format(i+1, acc))
    print("Overall accuracy on adversarial exampels: {}.".format(correct / total))
