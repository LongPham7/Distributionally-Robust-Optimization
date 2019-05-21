import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util_MNIST import retrieveMNISTTestData
from util_model import loadModel, evaluateModelAccuracy
from util_adversarial_attack import FGSM, PGD, FGSMNative, DistributionalPGD

"""
This module contains two base classes for analysis of the robustness of neural
networks. The first class, Analysis, wraps a single neural network, and the
second class, AnalysisMulitpleModels, supports analysis on a list of
neural networks. 
"""

class Analysis:

    """
    Class for the robustness analysis on a single neural network.
    """

    def __init__(self, skeleton_model, filepath):
        self.model = loadModel(skeleton_model, filepath)

        # Use GPU for computation if it is available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("The model is now loaded on {}.".format(self.device))

        self.filepath = filepath

    def testAccuracy(self):
        """
        Evaluate the accuracy of a neural network on the MNIST test data.
        """

        return evaluateModelAccuracy(self.model)

    def adversarialAccuracy(self, adversarial_type, budget, norm):
        """
        Evaluate the accuracy of a neural network on a set of adversarial
        examples. 
        """

        batch_size = 512 if adversarial_type == "distributional_PGD" else 128
        
        # Numbers of iterations for pointwise and distributional PGD attacks
        max_iter_point, max_iter_dist = 15, 40
        test_loader = retrieveMNISTTestData(batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        if adversarial_type == "FGSM":
            adversarial_module = FGSM(
                self.model, criterion, norm=norm, batch_size=batch_size)
        elif adversarial_type == 'PGD':
            adversarial_module = PGD(
                self.model, criterion, norm=norm, batch_size=batch_size)
        elif adversarial_type == "distributional_PGD":
            adversarial_module = DistributionalPGD(self.model, criterion)
        else:
            raise ValueError("The type of adversarial attack is not valid.")

        # Craft adversarial examples
        total, correct = 0, 0
        for i, data in enumerate(test_loader):
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            data = (images, labels)

            # images_adv is already loaded on GPU by generatePerturbation.
            # Also, if FGSM is used, we have minimal=False by default.
            if adversarial_type == "FGSM":
                images_adv = adversarial_module.generatePerturbation(
                    data, budget)
            elif adversarial_type == "PGD":
                images_adv = adversarial_module.generatePerturbation(
                    data, budget, max_iter=max_iter_point)
            else:
                # For distributional PGD attacks
                images_adv = adversarial_module.generatePerturbation(
                    data, budget, max_iter=max_iter_dist)
            with torch.no_grad():
                softmax = nn.Softmax(dim=1)
                outputs = softmax(self.model(images_adv))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct, total


class AnalysisMulitpleModels:

    """
    Base class for the robustness analysis on multiple neural networks.
    """

    def __init__(self):
        pass

    def printBasicResult(self, analyzer, budget_two, budget_inf):
        """
        Print out (i) the accuracy of a neural network on MNIST and 
        (ii) its robustness to FGSM and PGD.
        """

        correct, total = analyzer.testAccuracy()
        print("Test accuracy: {} / {} = {}".format(correct, total, correct / total))

        correct, total = analyzer.adversarialAccuracy(
            'FGSM', budget=budget_two, norm=2)
        print("Adversarial accuracy with respect to FGSM-2: {} / {} = {}".format(correct,
                                                                                 total, correct / total))
        correct, total = analyzer.adversarialAccuracy(
            'FGSM', budget=budget_inf, norm=np.inf)
        print("Adversarial accuracy with respect to FGSM-inf: {} / {} = {}".format(
            correct, total, correct / total))

        correct, total = analyzer.adversarialAccuracy(
            'PGD', budget=budget_two, norm=2)
        print("Adversarial accuracy with respect to PGD-2: {} / {} = {}".format(correct,
                                                                                total, correct / total))
        correct, total = analyzer.adversarialAccuracy(
            'PGD', budget=budget_inf, norm=np.inf)
        print("Adversarial accuracy with respect to PGD-inf: {} / {} = {}".format(
            correct, total, correct / total))

    def plotPerturbationLineGraph(self, ax, analyzers, labels, adversarial_type, budget, norm, bins, record_file):
        """
        Plot a line graph of the adversarial attack success rates with various
        budgets for an adversarial attack. 

        Arguments:
            ax: Axes object (in pyplot) where a plot a drawn
            analyzers: list of Analysis objects
            labels: list of labels of the Analysis objects in the input list
            bins: the number of different budgets to examine
            record_file: file object to be used to record the adversarial
                attack success rates
        """

        length = len(analyzers)
        results = [[] for i in range(length)]
        increment_size = budget / bins if bins != 0 else None
        perturbations = [i * increment_size for i in range(bins+1)]
        assert length <= 10
        # Colours of lines in a graph; this colour map only has ten colours.
        cmap = plt.get_cmap("tab10")

        # Evaluate the test accuracy; i.e. robustness against adverarial
        # attacks with the adversarial budget of 0.
        for j in range(length):
            analyzer = analyzers[j]
            correct, total = analyzer.testAccuracy()
            results[j].append(1 - correct / total)
        print("0-th iteration complete")

        # Evaluate the robustness against adversarial attacks with non-zero
        # budget.
        for i in range(bins):
            for j in range(length):
                analyzer = analyzers[j]
                correct, total = analyzer.adversarialAccuracy(
                    adversarial_type, increment_size * (i+1), norm)
                results[j].append(1 - correct / total)
            print("{}-th iteration complete".format(i+1))

        # Record the results in a log if required
        if record_file is not None:
            for i in range(length):
                analyzer = analyzers[i]
                record_file.write(
                    "Adversarial attack on {}\n".format(analyzer.filepath))
                record_file.write(
                    "Attack type: {}; Norm: {}\n".format(adversarial_type, norm))
                record_file.write(
                    "Budget: {}; Bins: {}\n".format(budget, bins))
                zipped_reuslt = list(zip(perturbations, results[i]))
                record_file.write(str(zipped_reuslt) + "\n\n")

        for i in range(length):
            ax.plot(perturbations, results[i], color=cmap(
                i), linestyle='-', label=labels[i])
        ax.legend()
        ax.set_xlabel("Perturbation size")
        ax.set_ylabel("Adversarial attack success rate")
        ax.set_xlim(0, budget)
        ax.set_yscale('log')
