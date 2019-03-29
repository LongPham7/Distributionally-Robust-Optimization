import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util_MNIST import retrieveMNISTTestData
from util_model import MNISTClassifier, SimpleNeuralNet, loadModel, evaluateModel, evaluateModelSingleInput
from adversarial_attack import FGSM, PGD, FGSMNative

class Analysis:

    """
    This class conducts robustness analysis of neural networks.
    """

    def __init__(self, model, filepath):
        self.model = loadModel(model, filepath)
        
        # Use GPU for computation if it is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("The model is now loaded on {}.".format(self.device))

    def testAccuracy(self):
        """
        Evaluate the accuracy of a neural network on the MNIST test data.
        """

        return evaluateModel(self.model)

    def adversarialAccuracy(self, adversarial_type, budget, norm):
        """
        Evaluate the accuracy of a neural network on a set of adversarial
        examples. 
        """

        batch_size = 128
        max_iter = 15
        test_loader = retrieveMNISTTestData(batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        if adversarial_type == "FGSM":
            adversarial_module = FGSM(self.model, criterion, norm=norm, batch_size=batch_size)
            #adversarial_module = FGSMNative(self.model, criterion, norm=norm, batch_size=batch_size)
        elif adversarial_type == 'PGD':
            adversarial_module = PGD(self.model, criterion, norm=norm, batch_size=batch_size)
        else:
            raise ValueError("The type of adversarial attack is not valid.")

        # Craft adversarial examples
        total, correct = 0, 0
        period = 100
        for i, data in enumerate(test_loader):
            if i == period:
                break
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            data = (images, labels)

            # images_adv is already loaded on GPU by generatePerturbation.
            # Also, if FGSM is used, we have minimal=False by default. 
            if adversarial_type == "FGSM":
                images_adv = adversarial_module.generatePerturbation(data, budget)
            else:
                images_adv = adversarial_module.generatePerturbation(data, budget, max_iter=max_iter)
            with torch.no_grad():
                softmax = nn.Softmax(dim=1)
                outputs =  softmax(self.model(images_adv))
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct, total

    def fgsmPerturbationDistribution(self, budget, norm):
        batch_size = 128
        test_loader = retrieveMNISTTestData(batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        fgsm = FGSM(self.model, criterion, norm=norm, batch_size=batch_size)
        #fgsm = FGSMNative(self.model, criterion, norm=norm, batch_size=batch_size)

        # Craft "minimal" adversarial examples with FGSM
        minimal_perturbations = [] # List of minimal perturbations
        period = 100
        for i, data in enumerate(test_loader):
            if i == period:
                break
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            # images_adv is already loaded on GPU by generatePerturbation.
            images_adv = fgsm.generatePerturbation(data, budget=budget, minimal=True)
            
            for j in range(images.size(0)):
                minimal_perturbation = torch.dist(images[j], images_adv[j], p=norm).item()
                minimal_perturbations.append(minimal_perturbation)
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
        model_sgd_relu = MNISTClassifier(activation='relu')
        model_sgd_elu = MNISTClassifier(activation='elu')
        
        # These file paths only work on UNIX. 
        filepath_relu = "./ERM_models/MNISTClassifier_relu.pt"
        filepath_elu = "./ERM_models/MNISTClassifier_elu.pt"
        filepath_sgd_relu = "./ERM_models/MNISTClassifier_SGD_relu.pt"
        filepath_sgd_elu = "./ERM_models/MNISTClassifier_SGD_elu.pt"
        
        self.analyzer_relu = Analysis(model_relu, filepath_relu)
        self.analyzer_elu = Analysis(model_elu, filepath_elu)
        self.analyzer_sgd_relu = Analysis(model_sgd_relu, filepath_sgd_relu)
        self.analyzer_sgd_elu = Analysis(model_sgd_elu, filepath_sgd_elu)

    def printBasicResult(self, analyzer, budget_two, budget_inf):     
        correct, total = analyzer.testAccuracy()
        print("Test accuracy: {} / {} = {}".format(correct, total, correct / total))
        
        correct, total = analyzer.adversarialAccuracy('FGSM', budget=budget_two, norm=2)
        print("Adversarial accuracy with respect to FGSM-2: {} / {} = {}".format(correct, total, correct / total))
        correct, total = analyzer.adversarialAccuracy('FGSM', budget=budget_inf, norm=np.inf)
        print("Adversarial accuracy with respect to FGSM-inf: {} / {} = {}".format(correct, total, correct / total))
        
        correct, total = analyzer.adversarialAccuracy('PGD', budget=budget_two, norm=2)
        print("Adversarial accuracy with respect to PGD-2: {} / {} = {}".format(correct, total, correct / total))
        correct, total = analyzer.adversarialAccuracy('PGD', budget=budget_inf, norm=np.inf)
        print("Adversarial accuracy with respect to PGD-inf: {} / {} = {}".format(correct, total, correct / total))

    def producePerturbationHistorgram(self, analyzer, filename):
        budget_two = 3.0
        budget_inf = 1.0

        #TODO: It may be neecessary to prune out those adversarial examples that are corectly classified. 
        minimal_perturbations_two = analyzer.fgsmPerturbationDistribution(budget=budget_two, norm=2)
        minimal_perturbations_inf = analyzer.fgsmPerturbationDistribution(budget=budget_inf, norm=np.inf)

        bins = 20
        fig, (ax0, ax1) = plt.subplots(1, 2)
        n, _, _ = ax0.hist(minimal_perturbations_two, bins=bins, range=(0, budget_two))
        ax0.set_xlabel("Minimal perturbation")
        ax0.set_ylabel("Frequency")
        ax0.set_title("FGSM-2")
        print("Distribution in FGSM-2: {}".format(n))

        n, _, _ = ax1.hist(minimal_perturbations_inf, bins=bins, range=(0, budget_inf))
        ax1.set_xlabel("Minimal perturbation")
        ax1.set_ylabel("Frequency")
        ax1.set_title("FGSM-inf")
        print("Distribution in FGSM-inf: {}".format(n))

        plt.tight_layout()
        
        #plt.show()
        filepath = "./images/" + filename
        plt.savefig(filepath)
        print("A histogram is now saved at {}.".format(filepath))
        plt.close()

    def plotPerturbationLineGraph(self, ax, adversarial_type, budget, norm, bins):
        analyzers = [self.analyzer_relu, self.analyzer_elu, self.analyzer_sgd_relu, self.analyzer_sgd_elu]
        results = [[], [], [], []]
        increment_size = budget / bins
        perturbations = [i * increment_size for i in range(bins+1)]
        labels = ['ReLU Adam', 'ELU Adam', 'ReLU SGD', 'ELU SGD']
        colours = ['blue', 'green', 'red', 'cyan']

        # Evaluate the test accuracy; i.e. robustness against adverarial
        # attacks with the budget of 0. 
        for j in range(4):
            analyzer = analyzers[j]
            correct, total = analyzer.testAccuracy()
            results[j].append(1 - correct / total)
            print("Test accuracy: {}".format(correct / total))

        # Evaluate the robustness against adversarial attacks with non-zero
        # budget. 
        for i in range(bins):
            for j in range(4):
                analyzer = analyzers[j]
                correct, total = analyzer.adversarialAccuracy(adversarial_type, increment_size * (i+1), norm)
                results[j].append(1 - correct / total)
            print("Adversarial attack correct prediction: {}".format(correct / total))

        for i in range(4):
            ax.plot(perturbations, results[i], color=colours[i], linestyle='-', label=labels[i])
        ax.legend()
        ax.set_xlabel("Perturbation size")
        ax.set_ylabel("Adversarial attack success rate")
        ax.set_xlim(0, budget)
        #ax.set_ylim(0, 1) # This is only valid for the linear y-axis scale. 
        ax.set_yscale('log')

    def producePerturbationLineGraph(self, budget, norm, bins, filename):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        self.plotPerturbationLineGraph(ax1, "FGSM", budget, norm, bins)
        self.plotPerturbationLineGraph(ax2, "PGD", budget, norm, bins)

        ax1.set_title("FGSM")
        ax2.set_title("PGD")

        #plt.show()
        filepath = "./images/" + filename
        plt.savefig(filepath)
        print("A graph is now saved at {}.".format(filepath))
        plt.close()

if __name__ == '__main__':
    erm_analysis = ERMAnalysis()

    budget_two = 4.0
    budget_inf = 0.4
    bins = 20
    
    erm_analysis.producePerturbationLineGraph(budget=budget_two, norm=2, bins=bins, filename='L2-norm.png')
    erm_analysis.producePerturbationLineGraph(budget=budget_inf, norm=np.inf, bins=bins, filename='Linf-norm.png')
