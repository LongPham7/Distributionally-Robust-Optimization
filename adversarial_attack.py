import numpy as np
import torch
from torch import optim, nn
from util_MNIST import retrieveMNISTTestData
from util_model import SimpleNeuralNet, loadModel, wrapModel

from art.attacks import FastGradientMethod, ProjectedGradientDescent

img_rows, img_cols = 28, 28

class FGSM:

    def __init__(self, model, loss_criterion, norm=np.inf, batch_size=128):
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
        images_adv = self.attack.generate(x=images.numpy(), norm=self.norm, eps=budget, minimal=minimal, eps_steps=0.005, eps_max=1.0, batch_size=self.batch_size)
        images_adv = torch.from_numpy(images_adv)

        # The output to be returned should be loaded on an appropriate device. 
        return images_adv.to(self.device)

class PGD:
    """
    Module for adversarial attacks based on projected gradient descent (PGD).
    The implementation of PGD in ART executes projection on a feasible region
    after each iteration. However, random restrating is not used in this
    implementation. Not using radom restarting is the difference between the
    PGD implemented in ART and the one described by Madry et al. 
    """

    def __init__(self, model, loss_criterion, norm=np.inf, max_iter=15, batch_size=128):
        self.pytorch_model = wrapModel(model, loss_criterion)
        self.norm = norm
        self.batch_size = batch_size
        self.attack = ProjectedGradientDescent(self.pytorch_model, norm=norm, max_iter=max_iter, batch_size=batch_size)

        # Use GPU for computation if it is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generatePerturbation(self, data, budget, max_iter=20):
        images, _ = data
        
        # eps_step is not allowed to be larger than budget according to the 
        # documentation of ART. 
        eps_step = budget / 8
        images_adv = self.attack.generate(x=images.numpy(), norm=self.norm, eps=budget, eps_step=eps_step, max_iter=max_iter, batch_size=self.batch_size)
        images_adv = torch.from_numpy(images_adv)

        # The output to be returned should be loaded on an appropriate device. 
        return images_adv.to(self.device)

if __name__ == "__main__":
    # Load a simple neural network
    model = SimpleNeuralNet()
    loadModel(model, "C:\\Users\\famth\\Desktop\\DRO\\models\\SimpleModel.pt")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) # Load the neural network on GPU if it is available
    print("The neural network is now loaded on {}.".format(device))

    # Create an object for FGSM
    criterion = nn.CrossEntropyLoss()
    batch_size = 128
    pgd = PGD(model, criterion, batch_size=batch_size)
    pytorch_model = pgd.pytorch_model

    # Read MNIST dataset
    test_loader = retrieveMNISTTestData(batch_size=1024)

    # Craft adversarial examples with FGSM
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
        print("{}-th iteration: the test accuracy on adversarial sample: {}%".format(i+1, acc * 100))
    print("The overall accuracy on adversarial exampels is {}%.".format(correct / total * 100))
