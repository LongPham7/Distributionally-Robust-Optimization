import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util_MNIST import retrieveMNISTTestData
from util_model import MNISTClassifier, SimpleNeuralNet, loadModel, evaluateModel
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
        return minimal_perturbations

    def displayHistogram(self, minimal_perturbations, title=None):
        plt.hist(minimal_perturbations, bins=50, range=(0,1))
        plt.ylabel("Frequency")
        plt.xlabel("Minimal perturbation")
        if title is not None:
            plt.title(title)
        plt.show()

class ERMAnalysis:

    def __init__(self):
        model_relu = MNISTClassifier(activation='relu')
        model_elu = MNISTClassifier(activation='elu')
        filepath_relu = r".\ERM_models\MNISTClassifier_relu.pt"
        filepath_elu = r".\ERM_models\MNISTClassifier_elu.pt"
        self.analyzer_relu = Analysis(model_relu, filepath_relu)
        self.analyzer_elu = Analysis(model_elu, filepath_elu)

    def analysisResult(self, analyzer):
        test_accuracy = analyzer.testAccuracy()
        print("Test accuracy: {}.".format(test_accuracy))
        adversarial_accuracy = analyzer.adversarialAccuracy('FGSM', budget=0.1, norm=2)
        print("Adversarial accuracy with respect to FGSM-2: {}".format(adversarial_accuracy))
        adversarial_accuracy = analyzer.adversarialAccuracy('FGSM', budget=0.1, norm=np.inf)
        print("The adversarial accuracy with respect to FGSM-inf: {}".format(adversarial_accuracy))
        adversarial_accuracy = analyzer.adversariaAccuracy('PGD', budget=0.1, norm=2)
        print("The adversarial accuracy with respect to PGD-2: {}".format(adversarial_accuracy))

        minimal_perturbations_two = analyzer.fgsmDistribution(norm=2)
        minimal_perturbations_inf = analyzer.fgsmDistribution(norm=np.inf)

        bins = 50
        range = (0,1)
        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.hist(minimal_perturbations_two, bins=bins, range=range)
        ax0.set_xlabel("Minimal perturbation")
        ax0.set_ylabel("Frequency")
        ax0.set_title("FGSM-2")

        ax1.hist(minimal_perturbations_inf, bins=bins, range=range)
        ax1.set_xlabel("Minimal perturbation")
        ax1.set_ylabel("Frequency")
        ax1.set_title("FGSM-inf")

        plt.show()

    def analyzeERM(self):
        print("The analysis result of the MNIST classifier with the relu activation function is as follows.")
        self.analysisResult(self.analyzer_relu)

        print("The analysis result of the MNIST classifier with the elu activation function is as follows.")
        self.analysisResult(self.analyzer_elu)

if __name__ == '__main__':
    erm_analysis = ERMAnalysis()
    erm_analysis.analyzeERM()   
