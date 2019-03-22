import torch
import torch.nn as nn
from util_model import SimpleNeuralNet, MNISTClassifier
from util_MNIST import retrieveMNISTTrainingData
from distributionallyRO import ProjetcedDRO, LagrangianDRO, FrankWolfeDRO
from adversarial_training import ProjectedGradientTraining

def trainDROModel(dro_type, epochs, steps_adv, budget, activation, batch_size, loss_criterion, cost_function=None):
    model = MNISTClassifier(activation=activation)
    if dro_type == 'PGD':
        train_module = ProjetcedDRO(model, loss_criterion)
    elif dro_type == 'Lag':
        assert cost_function is not None
        train_module = LagrangianDRO(model, loss_criterion, cost_function)
    elif dro_type == 'FW':
        train_module = FrankWolfeDRO(model, loss_criterion, p=2, q=2)
    else:
        raise ValueError("The type of DRO is not valid.")

    train_module.train(budget=budget, batch_size=batch_size, epochs=epochs, steps_adv=steps_adv)
    filepath = "./DRO_models/{}_DRO_activation={}_epsilon={}.pt".format(dro_type, activation, budget)
    torch.save(model, filepath)
    print("A neural network adversarially trained using {} is now saved at {}.".format(dro_type, filepath))

if __name__ == "__main__":
    epochs = 25
    steps_adv = 15
    epsilon = 0.1
    gammas = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    batch_size = 128
    loss_criterion = nn.CrossEntropyLoss()
    cost_function = lambda x, y: torch.dist(x, y, p=2) ** 2

    #trainDROModel('PGD', epochs, steps_adv, epsilon, 'relu', batch_size, loss_criterion, cost_function=None)
    trainDROModel('FW', epochs, steps_adv, epsilon, 'relu', batch_size, loss_criterion, cost_function=None)

    #trainDROModel('PGD', epochs, steps_adv, epsilon, 'elu', batch_size, loss_criterion, cost_function=None)
    trainDROModel('FW', epochs, steps_adv, epsilon, 'elu', batch_size, loss_criterion, cost_function=None)

    """
    for gamma in gammas:
        trainDROModel('Lag', epochs, steps_adv, gamma, 'relu', batch_size, loss_criterion, cost_function=cost_function)
        trainDROModel('Lag', epochs, steps_adv, gamma, 'elu', batch_size, loss_criterion, cost_function=cost_function)
    """
