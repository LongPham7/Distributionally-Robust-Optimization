import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from util_MNIST import retrieveMNISTTestData
from util_model import MNISTClassifier, SimpleNeuralNet, loadModel, evaluateModel, evaluateModelSingleInput
from adversarial_attack import FGSM, PGD, FGSMNative

class Analysis:

    """
    Class for the robustness analysis on a single neural network.
    """

    def __init__(self, model, filepath):
        self.model = loadModel(model, filepath)
        #self.model = loadModel(model=None, filepath=filepath)
        
        # Use GPU for computation if it is available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        """
        Display a histogram (for minimal perturbation size in FGSM).
        """

        plt.hist(minimal_perturbations, bins=50, range=(0,1))
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
        
        correct, total = analyzer.adversarialAccuracy('FGSM', budget=budget_two, norm=2)
        print("Adversarial accuracy with respect to FGSM-2: {} / {} = {}".format(correct, total, correct / total))
        correct, total = analyzer.adversarialAccuracy('FGSM', budget=budget_inf, norm=np.inf)
        print("Adversarial accuracy with respect to FGSM-inf: {} / {} = {}".format(correct, total, correct / total))
        
        correct, total = analyzer.adversarialAccuracy('PGD', budget=budget_two, norm=2)
        print("Adversarial accuracy with respect to PGD-2: {} / {} = {}".format(correct, total, correct / total))
        correct, total = analyzer.adversarialAccuracy('PGD', budget=budget_inf, norm=np.inf)
        print("Adversarial accuracy with respect to PGD-inf: {} / {} = {}".format(correct, total, correct / total))

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
        cmap = plt.get_cmap("tab10") # Colours of lines in a graph; we have ten colours only. 

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
                correct, total = analyzer.adversarialAccuracy(adversarial_type, increment_size * (i+1), norm)
                results[j].append(1 - correct / total)
            print("{}-th iteration complete".format(i+1))

        # Record the results in a file if it is given
        if record_file is not None:
            for i in range(length):
                analyzer = analyzers[i]
                record_file.write("Adversarial attack on {}\n".format(analyzer.filepath))
                record_file.write("Attack type: {}; Norm: {}\n".format(adversarial_type, norm))
                record_file.write("Budget: {}; Bins: {}\n".format(budget, bins))
                zipped_reuslt = list(zip(perturbations, results[i]))
                record_file.write(str(zipped_reuslt) + "\n\n")

        for i in range(length):
            ax.plot(perturbations, results[i], color=cmap(i), linestyle='-', label=labels[i])
        ax.legend()
        ax.set_xlabel("Perturbation size")
        ax.set_ylabel("Adversarial attack success rate")
        ax.set_xlim(0, budget)
        #ax.set_ylim(0, 1) # This is only valid for the linear y-axis scale. 
        ax.set_yscale('log')

class ERMAnalysis(AnalysisMulitpleModels):

    """
    Class for the robustness analysis on neural networks trained by ERM.
    """

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

    def producePerturbationHistorgram(self, analyzer, filename):
        """
        Produce a histogram of the minimal size of perturbations. 
        """

        budget_two = 3.0
        budget_inf = 1.0

        # Note that those adversarial examples that are corectly classified
        # are not pruned out. 
        # Also, note that they do not necesarily have the minimal perturbation
        # size that is euqal to budget; i.e. the minimal perturbation size is
        # not maximal. This is bizarre. 
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
        print("Histogram now saved at {}".format(filepath))
        plt.close()

    def plotERMModels(self, budget, norm, bins):
        """
        Produce a line graph of adversarial attack success rates for various
        budgets. 
        """

        analyzers = [self.analyzer_relu, self.analyzer_elu, self.analyzer_sgd_relu, self.analyzer_sgd_elu]
        labels = ['ReLU Adam', 'ELU Adam', 'ReLU SGD', 'ELU SGD']

        fig, (ax1, ax2) = plt.subplots(1, 2)

        record_filepath = "./records/ERM_analysis_norm={}.txt".format("L2" if norm == 2 else "Linf")
        try:
            record_file = open(record_filepath, mode='w')
            self.plotPerturbationLineGraph(ax1, analyzers, labels, "FGSM", budget, norm, bins, record_file)
            self.plotPerturbationLineGraph(ax2, analyzers, labels, "PGD", budget, norm, bins, record_file)
        finally:
            record_file.close()         

        ax1.set_title("FGSM")
        ax2.set_title("PGD")
        plt.tight_layout()

        width, height = fig.get_size_inches()
        fig.set_size_inches(width * 1.8, height)

        #plt.show()
        filepath = "./images/ERM_norm={}.png".format("L2" if norm == 2 else "Linf")
        plt.savefig(filepath)
        print("Graph now saved {}".format(filepath))
        plt.close()

class DROAnalysis(AnalysisMulitpleModels):

    """
    Class for the robustness analysis on the neural networks trained by DRO.
    """

    def __init__(self):
        self.gammas = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
        self.Lag_relu_analyzers, self.Lag_elu_analyzers = self.initializeLagAnalyzers()
        self.FW_relu_analyzer, self.FW_elu_analyzer = self.initializeAnalyzers(dro_type='FW')
        self.PGD_relu_analyzer, self.PGD_elu_analyzer = self.initializeAnalyzers(dro_type='PGD')

    def initializeLagAnalyzers(self):
        """
        Initialize Analysis objects for neural networks trained by the DRO
        algorithm proposed by Sinha et al.
        """

        folderpath = "./DRO_models/"
        Lag_relu_analyzers = []
        Lag_elu_analyzers = []
        length = len(self.gammas)
        for i in range(length):
            gamma = self.gammas[i]
            filepath_relu = folderpath + "{}_DRO_activation={}_epsilon={}.pt".format("Lag", "relu", gamma)
            filepath_elu = folderpath + "{}_DRO_activation={}_epsilon={}.pt".format("Lag", "elu", gamma)
            model_relu = MNISTClassifier(activation='relu')
            model_elu = MNISTClassifier(activation='elu')
            Lag_relu_analyzers.append(Analysis(model_relu, filepath_relu))
            Lag_elu_analyzers.append(Analysis(model_elu, filepath_elu))
        return Lag_relu_analyzers, Lag_elu_analyzers

    def initializeAnalyzers(self, dro_type):
        """
        Initialize Analysis objects for neural networks trained by the
        Frank-Wolfe method and PGD
        """

        folderpath = "./DRO_models/"
        filepath_relu = folderpath + "{}_DRO_activation={}_epsilon={}.pt".format(dro_type, "relu", 0.1)
        filepath_elu = folderpath + "{}_DRO_activation={}_epsilon={}.pt".format(dro_type, "elu", 0.1)
        model_relu = MNISTClassifier(activation='relu')
        model_elu = MNISTClassifier(activation='elu')
        analyzer_relu = Analysis(model_relu, filepath_relu)
        analyzer_elu = Analysis(model_elu, filepath_elu)
        return analyzer_relu, analyzer_elu

    def plotLagDROModels(self, adversarial_type, budget, norm, bins):
        """
        Produce line graphs of adversarial attack success rates on neural
        networks trained by WRM.
        """

        # Pyplot supports LaTex syntax.
        labels = [r"$\gamma = {}$".format(gamma) for gamma in self.gammas]

        fig, (ax1, ax2) = plt.subplots(1, 2)

        record_filepath = "./records/DRO_analysis_{}_norm={}.txt".format(adversarial_type, "L2" if norm == 2 else "Linf")
        try:
            record_file = open(record_filepath, mode='w')
        finally:
            self.plotPerturbationLineGraph(ax1, self.Lag_relu_analyzers, labels, adversarial_type, budget, norm, bins, record_file)
            self.plotPerturbationLineGraph(ax2, self.Lag_elu_analyzers, labels, adversarial_type, budget, norm, bins, record_file)
            print("Record stored at {}".format(record_filepath))
            record_file.close()

        ax1.set_title("ReLU")
        ax2.set_title("ELU")
        plt.tight_layout()

        width, height = fig.get_size_inches()
        fig.set_size_inches(width * 1.8, height)

        #plt.show()
        filepath = "./images/Lag_{}_norm={}.png".format(adversarial_type, "L2" if norm == 2 else "Linf")
        plt.savefig(filepath)
        print("Graph now saved at {}".format(filepath))
        plt.close()

    def compareLagDROModels(self, budget_two, budget_inf, bins):
        """
        Compare the robustness of those neural networks trained by WRM with
        different values of gamma by using four types of adversarial attacks.
        """

        self.plotLagDROModels("FGSM", budget_inf, np.inf, bins)
        self.plotLagDROModels("FGSM", budget_two, 2, bins)

        self.plotLagDROModels("PGD", budget_inf, np.inf, bins)
        self.plotLagDROModels("PGD", budget_two, 2, bins)

    def plotDROModels(self, budget, norm, bins):      
        """
        Compare the robustness of neural networks trained by all three DRO
        algorithms: WRM, the Frank-Wolfe method, and PGD. 
        """

        # The optimal gamma for both ReLu and ELU has been determined to be 1.0. 
        optimal_gamma = 1.0
        index_optimal_gamma = self.gammas.index(optimal_gamma)
        LagAnalyzers = [self.Lag_relu_analyzers[index_optimal_gamma], self.Lag_elu_analyzers[index_optimal_gamma]]
        FWandPGDanalyzers = [self.FW_relu_analyzer, self.FW_elu_analyzer, self.PGD_relu_analyzer, self.PGD_elu_analyzer]
        analyzers = LagAnalyzers + FWandPGDanalyzers
        labels = ["Lag ReLU", "Lag ELU", "FW ReLU", "FW ELU", "PGD ReLU", "PGD ELU"]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.plotPerturbationLineGraph(ax1, analyzers, labels, "FGSM", budget, norm, bins, record_file=None)
        self.plotPerturbationLineGraph(ax2, analyzers, labels, "PGD", budget, norm, bins, record_file=None)

        ax1.set_title("FGSM")
        ax2.set_title("PGD")
        plt.tight_layout()

        width, height = fig.get_size_inches()
        fig.set_size_inches(width * 1.8, height)

        #plt.show()
        filepath = "./images/DRO_norm={}.png".format("L2" if norm == 2 else "Linf")
        plt.savefig(filepath)
        print("Graph now saved {}".format(filepath))
        plt.close()

if __name__ == '__main__':
    budget_two = 4.0
    budget_inf = 0.4
    bins = 20

    """
    erm_analysis = ERMAnalysis()   
    erm_analysis.plotERMModels(budget=budget_two, norm=2, bins=bins)
    erm_analysis.plotERMModels(budget=budget_inf, norm=np.inf, bins=bins)
    """

    dro_analysis = DROAnalysis()
    #dro_analysis.compareLagDROModels(budget_two=budget_two, budget_inf=budget_inf, bins=bins)
    dro_analysis.plotDROModels(budget=budget_two, norm=2, bins=bins)
    dro_analysis.plotDROModels(budget=budget_inf, norm=np.inf, bins=bins)
