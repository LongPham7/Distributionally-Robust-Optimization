import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util_MNIST import retrieveMNISTTestData
from util_model import loadModel, evaluateModel, SimpleNeuralNet
from adversarial_attack import FGSM, PGD

class Analysis:

    def __init__(self, model, filepath):
        self.model = loadModel(model, filepath)
        
        # Use GPU for computation if it is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("The model is now loaded on {}.".format(self.device))

    def testAccuracy(self):
        return evaluateModel(self.model)

    def adversarialAccuracy(self, adversarial_type, budget, norm):
        batch_size = 128
        test_loader = retrieveMNISTTestData(batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        if adversarial_type == "FGSM":
            adversarial_module = FGSM(self.model, criterion, norm=norm, batch_size=batch_size)
        elif adversarial_type == 'PGD':
            adversarial_module = PGD(self.model, criterion, norm=norm, max_iter=15, batch_size=batch_size)
        else:
            raise ValueError("The type of adversarial attack is not valid.")

        # Craft adversarial examples
        total, correct = 0, 0
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            # images_adv is already loaded on GPU by generatePerturbation.
            # Also, if FGSM is used, we have minimal=False by default. 
            images_adv = adversarial_module.generatePerturbation(data, budget)
            with torch.no_grad():
                outputs =  self.model(images_adv)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct / total

    def fgsmDistribution(self, norm):
        batch_size = 128
        test_loader = retrieveMNISTTestData(batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        fgsm = FGSM(self.model, criterion, norm=norm, batch_size=batch_size)

        # Craft "minimal" adversarial examples with FGSM
        minimal_perturbations = [] # List of minimal perturbations
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            # images_adv is already loaded on GPU by generatePerturbation.
            images_adv = fgsm.generatePerturbation(data, budget=1.0, minimal=True)
            
            for j in range(images.size(0)):
                minimal_perturbations.append(torch.dist(images[j], images_adv[j], p=norm).item())
        
        plt.hist(minimal_perturbations, bins=50, range=(0,1))
        plt.ylabel("Frequency")
        plt.ylabel("Minimal perturbation")
        plt.show()

if __name__ == '__main__':
    model = SimpleNeuralNet()
    filepath = r"C:\Users\famth\Desktop\DRO\ERM_models\SimpleModel.pt"
    analysis = Analysis(model, filepath)

    test_accuracy = analysis.testAccuracy()
    adversarial_accuracy = analysis.adversarialAccuracy("FGSM", 0.1, norm=np.inf)
    print("The test accuracy is {}.".format(test_accuracy))
    print("The adversarial accuracy for FGSM is {}.".format(adversarial_accuracy))

    analysis.fgsmDistribution(norm=np.inf)
