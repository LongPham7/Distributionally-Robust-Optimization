import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util_MNIST import retrieveMNISTTestData
from util_model import loadModel, evaluateModel
from adversarial_attack import FGSM, PGD, FGSMNative


class Analysis:

    """
    Class for the robustness analysis on a single neural network.
    """

    def __init__(self, model, filepath):
        self.model = loadModel(model, filepath)
        #self.model = loadModel(model=None, filepath=filepath)

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
            adversarial_module = FGSM(
                self.model, criterion, norm=norm, batch_size=batch_size)
            #adversarial_module = FGSMNative(self.model, criterion, norm=norm, batch_size=batch_size)
        elif adversarial_type == 'PGD':
            adversarial_module = PGD(
                self.model, criterion, norm=norm, batch_size=batch_size)
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
                images_adv = adversarial_module.generatePerturbation(
                    data, budget)
            else:
                images_adv = adversarial_module.generatePerturbation(
                    data, budget, max_iter=max_iter)
            with torch.no_grad():
                softmax = nn.Softmax(dim=1)
                outputs = softmax(self.model(images_adv))

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        return correct, total

    def fgsmPerturbationDistribution(self, budget, norm):
        """
        Compute the distribution of the minimal size of perturbations that are
        required to trick the neural network into misclassification. 

        This utilises the functionality of computing minimal perturbations 
        provided by the ART library. 
        """

        batch_size = 128
        test_loader = retrieveMNISTTestData(batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        fgsm = FGSM(self.model, criterion, norm=norm, batch_size=batch_size)
        #fgsm = FGSMNative(self.model, criterion, norm=norm, batch_size=batch_size)

        # Craft "minimal" adversarial examples with FGSM
        minimal_perturbations = []  # List of minimal perturbations
        period = 100
        for i, data in enumerate(test_loader):
            if i == period:
                break
            images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)

            # images_adv is already loaded on GPU by generatePerturbation.
            images_adv = fgsm.generatePerturbation(
                data, budget=budget, minimal=True)

            for j in range(images.size(0)):
                minimal_perturbation = torch.dist(
                    images[j], images_adv[j], p=norm).item()
                minimal_perturbations.append(minimal_perturbation)
        return minimal_perturbations

    def displayHistogram(self, minimal_perturbations, title=None):
        """
        Display a histogram (for minimal perturbation size in FGSM).
        """

        plt.hist(minimal_perturbations, bins=50, range=(0, 1))
        plt.ylabel("Frequency")
        plt.xlabel("Minimal perturbation")
        if title is not None:
            plt.title(title)
        plt.show()


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
            bins: the number of different budgets to examine
            record_file: file object to be used to record the adversarial
                attack success rates
        """

        length = len(analyzers)
        results = [[] for i in range(length)]
        increment_size = budget / bins if bins != 0 else None
        perturbations = [i * increment_size for i in range(bins+1)]
        assert length <= 10
        # Colours of lines in a graph; we have ten colours only.
        cmap = plt.get_cmap("tab10")

        # Evaluate the test accuracy; i.e. robustness against adverarial
        # attacks with the budget of 0.
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

        # Record the results in a file if it is given
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
        # ax.set_ylim(0, 1) # This is only valid for the linear y-axis scale.
        ax.set_yscale('log')
