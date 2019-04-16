import numpy as np
import matplotlib.pyplot as plt
from adversarial_attack import FGSM, PGD, FGSMNative
from util_model import MNISTClassifier
from util_analysis import Analysis, AnalysisMulitpleModels

"""
This module contains classes for robustness analysis of neural networks. 
"""

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
        plt.savefig(filepath, dpi=300)
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
        with open(record_filepath, mode='w') as f:
            self.plotPerturbationLineGraph(ax1, analyzers, labels, "FGSM", budget, norm, bins, f)
            self.plotPerturbationLineGraph(ax2, analyzers, labels, "PGD", budget, norm, bins, f)

        ax1.set_title("FGSM")
        ax2.set_title("PGD")
        plt.tight_layout()

        width, height = fig.get_size_inches()
        fig.set_size_inches(width * 1.8, height)

        #plt.show()
        filepath = "./images/ERM_norm={}.png".format("L2" if norm == 2 else "Linf")
        plt.savefig(filepath, dpi=300)
        print("Graph now saved at {}".format(filepath))
        plt.close()

class DROAnalysis(AnalysisMulitpleModels):

    """
    Class for the robustness analysis on the neural networks trained by DRO.
    """

    def __init__(self):
        self.gammas = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

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

        self.Lag_relu_analyzers, self.Lag_elu_analyzers = self.initializeLagAnalyzers()
        self.FW_relu_analyzer, self.FW_elu_analyzer = self.initializeAnalyzers(dro_type='FW')
        self.PGD_relu_analyzer, self.PGD_elu_analyzer = self.initializeAnalyzers(dro_type='PGD')

    def plotLagDROModels(self, adversarial_type, budget, norm, bins):
        """
        Produce line graphs of adversarial attack success rates on neural
        networks trained by WRM.
        """

        # Pyplot supports LaTex syntax.
        labels = [r"$\gamma = {}$".format(gamma) for gamma in self.gammas]

        fig, (ax1, ax2) = plt.subplots(1, 2)

        record_filepath = "./records/DRO_analysis_{}_norm={}.txt".format(adversarial_type, "L2" if norm == 2 else "Linf")
        with open(record_filepath, mode='w') as f:
            self.plotPerturbationLineGraph(ax1, self.Lag_relu_analyzers, labels, adversarial_type, budget, norm, bins, f)
            self.plotPerturbationLineGraph(ax2, self.Lag_elu_analyzers, labels, adversarial_type, budget, norm, bins, f)
            print("Record stored at {}".format(record_filepath))

        ax1.set_title("ReLU")
        ax2.set_title("ELU")
        plt.tight_layout()

        width, height = fig.get_size_inches()
        fig.set_size_inches(width * 1.8, height)

        #plt.show()
        filepath = "./images/Lag_{}_norm={}.png".format(adversarial_type, "L2" if norm == 2 else "Linf")
        plt.savefig(filepath, dpi=300)
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
        plt.savefig(filepath, dpi=300)
        print("Graph now saved at {}".format(filepath))
        plt.close()

class LossAnalysis(AnalysisMulitpleModels):

    """
    Class for the robustness analysis various loss functions
    """

    def __init__(self):

        def initializeAnalyzers(dro_type, activation, budget):
            analyzers = []
            filepath = folderpath = "./Loss_models/"
            for i in range(1, 8):
                filepath = folderpath + "{}_DRO_activation={}_epsilon={}_loss={}.pt".format(dro_type, activation, budget, "f_{}".format(i))
                model = MNISTClassifier(activation=activation)
                analyzers.append(Analysis(model, filepath))
            return analyzers
        
        epsilon = 0.1
        optimal_gamma = 1.0
        self.FWAnalyzers = initializeAnalyzers("FW", activation='relu', budget=epsilon)
        self.PGDAnalyzers = initializeAnalyzers("PGD", activation = 'elu', budget=epsilon)
        self.LagAnalyzers = initializeAnalyzers("Lag", activation='relu', budget=optimal_gamma)

    def plotModelsWithLosses(self, adversarial_type, budget, norm, bins):
        """
        Plot adversarial atack success rates of the following two types of
        neural networks with the seven loss functions given in Carlini &
        Wagner:
        - activation: ReLU; training procedure: the Frank-Wolfe method
        - actiivatoin: ELU; training procedure: PGD.
        """

        labels = [r"$f_{}$".format(i) for i in range(1, 8)]

        fig, (ax1, ax2) = plt.subplots(1, 2)

        record_filepath = "./records/Loss_analysis_{}_norm={}.txt".format(adversarial_type, "L2" if norm == 2 else "Linf")
        with open(record_filepath, mode='w') as f:
            self.plotPerturbationLineGraph(ax1, self.FWAnalyzers, labels, adversarial_type, budget, norm, bins, f)
            self.plotPerturbationLineGraph(ax2, self.PGDAnalyzers, labels, adversarial_type, budget, norm, bins, f)
            print("Record stored at {}".format(record_filepath))

        ax1.set_title("FW")
        ax2.set_title("PGD")
        plt.tight_layout()

        width, height = fig.get_size_inches()
        fig.set_size_inches(width * 1.8, height)

        #plt.show()
        filepath = "./images/Loss_adversarial_type={}norm={}.png".format(adversarial_type, "L2" if norm == 2 else "Linf")
        plt.savefig(filepath, dpi=300)
        print("Graph now saved at {}".format(filepath))
        plt.close()

    def compareLosses(self, budget_two, budget_inf, bins):
        """
        Compare the seven loss functions in terms of robustness of the
        resulting neural networks.
        """

        self.plotModelsWithLosses("FGSM", budget_inf, np.inf, bins)
        self.plotModelsWithLosses("FGSM", budget_two, 2, bins)

        self.plotModelsWithLosses("PGD", budget_inf, np.inf, bins)
        self.plotModelsWithLosses("PGD", budget_two, 2, bins)

    def pltoRobustnessLagModels(self, budget, norm, bins):
        """
        Plot the adversarial success rates of neural networks trained by WRM
        with gamma being 1.0 and the seven loss listed in Carlini & Wagner. 
        """

        labels = [r"$f_{}$".format(i) for i in range(1, 8)]

        fig, (ax1, ax2) = plt.subplots(1, 2)

        record_filepath = "./records/Loss_Lag_analysis_norm={}budget={}.txt".format("L2" if norm == 2 else "Linf", budget)
        with open(record_filepath, "w") as f:
            self.plotPerturbationLineGraph(ax1, self.LagAnalyzers, labels, "FGSM", budget, norm, bins, f)
            self.plotPerturbationLineGraph(ax2, self.LagAnalyzers, labels, "PGD", budget, norm, bins, f)
            print("Record stored at {}".format(record_filepath))

        ax1.set_title("FGSM")
        ax2.set_title("PGD")
        plt.tight_layout()

        width, height = fig.get_size_inches()
        fig.set_size_inches(width * 1.8, height)

        #plt.show()
        filepath = "./images/Loss_Lag_norm={}.png".format("L2" if norm == 2 else "Linf")
        plt.savefig(filepath, dpi=300)
        print("Graph now saved at {}".format(filepath))
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

    """
    dro_analysis = DROAnalysis()
    dro_analysis.compareLagDROModels(budget_two=budget_two, budget_inf=budget_inf, bins=bins)
    dro_analysis.plotDROModels(budget=budget_two, norm=2, bins=bins)
    dro_analysis.plotDROModels(budget=budget_inf, norm=np.inf, bins=bins)
    """

    """
    loss_analysis = LossAnalysis()
    loss_analysis.compareLosses(budget_two=budget_two, budget_inf=budget_inf, bins=bins)
    loss_analysis.pltoRobustnessLagModels(budget=budget_inf, norm=np.inf, bins=bins)
    loss_analysis.pltoRobustnessLagModels(budget=budget_two, norm=2, bins=bins)
    """
