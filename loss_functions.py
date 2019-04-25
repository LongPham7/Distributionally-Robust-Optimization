import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_MNIST import retrieveMNISTTrainingData
from util_model import MNISTClassifier, trainModel
from adversarial_attack_DRO import ProjetcedDRO, LagrangianDRO, FrankWolfeDRO

"""
This module contains the seven loss functions listed in Carlini and Wagneer
(2017).
"""


def f_1(outputs, labels):
    return F.cross_entropy(outputs, labels)


def f_2(outputs, labels):
    outputs = F.softmax(outputs, dim=1)
    return f_6(outputs, labels)


def f_3(outputs, labels):
    outputs = F.softmax(outputs, dim=1)
    return f_7(outputs, labels)


def f_4(outputs, labels):
    outputs = F.softmax(outputs, dim=1)
    reference_outputs = torch.gather(
        outputs, 1, labels.view(-1, 1).long()).view(-1)
    return torch.mean(torch.clamp(0.5 - reference_outputs, min=0))


def f_5(outputs, labels):
    # Note that in the original version, the base of e is used instead of 2. 

    outputs = F.softmax(outputs, dim=1)
    reference_outputs = torch.gather(
        outputs, 1, labels.view(-1, 1).long()).view(-1)
    return torch.mean(torch.log2(2.125 - 2 * reference_outputs))


def f_6(outputs, labels):
    max_outputs, _ = torch.max(outputs, dim=1)
    reference_outputs = torch.gather(
        outputs, 1, labels.view(-1, 1).long()).view(-1)
    return torch.mean(max_outputs - reference_outputs)


def f_7(outputs, labels, nb_classes=10):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cuda:0"

    batch_size = labels.size()[0]
    indexes_row = torch.arange(0, nb_classes).to(device)
    indexes = indexes_row.repeat(batch_size, 1)
    labels_cloned = labels.view(-1, 1).repeat(1, nb_classes)

    new_outputs = outputs[indexes != labels_cloned].view(
        batch_size, nb_classes-1)
    reference_outputs = torch.gather(
        outputs, 1, labels.view(-1, 1).long()).view(-1)
    difference = torch.max(new_outputs, dim=1)[0] - reference_outputs
    return torch.mean(F.softplus(difference))


def trainModelLoss(dro_type, epochs, steps_adv, budget, activation, batch_size, loss_criterion, cost_function=None):
    """
    Train a neural network with a specified loss function.
    """

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

    train_module.train(budget=budget, batch_size=batch_size,
                       epochs=epochs, steps_adv=steps_adv)
    folderpath = "./Loss_models/"
    filepath = folderpath + "{}_DRO_activation={}_epsilon={}_loss={}.pt".format(
        dro_type, activation, budget, loss_criterion.__name__)
    torch.save(model.state_dict(), filepath)
    print("A neural network adversarially trained using {} now saved at: {}".format(
        dro_type, filepath))


if __name__ == "__main__":
    epochs = 25
    steps_adv = 15
    epsilon = 0.1
    optimal_gamma = 1.0
    batch_size = 128
    loss_criterions = [f_1, f_2, f_3, f_4, f_5, f_6, f_7]
    #loss_criterions = [f_1, f_2, f_3]
    #loss_criterions = [f_4, f_5]
    #loss_criterions = [f_6, f_7]

    def cost_function(x, y): return torch.dist(x, y, p=2) ** 2

    for loss_criterion in loss_criterions:
        trainModelLoss("FW", epochs, steps_adv, epsilon, "relu", batch_size, loss_criterion)
        trainModelLoss("PGD", epochs, steps_adv, epsilon, "elu", batch_size, loss_criterion)
        trainModelLoss("Lag", epochs, steps_adv, optimal_gamma, "relu",
                       batch_size, loss_criterion, cost_function=cost_function)
  
    """
    data_loader = retrieveMNISTTrainingData(batch_size=1, shuffle=True)
    iterator = iter(data_loader)
    images, labels = iterator.next()

    print("Shape of image: {}; shape of labels: {}".format(images.size(), labels.size()))
    #print("images: {}".format(images))
    print("labels: {}".format(labels))

    from util_model import loadModel

    dro_type = "PGD"
    activation = "elu"
    budget = 0.1
    loss_criterion = f_5
    folderpath = "./Loss_models/"
    filepath = folderpath + "{}_DRO_activation={}_epsilon={}_loss={}.pt".format(dro_type, activation, budget, loss_criterion.__name__)
    model_skeleton = MNISTClassifier(activation=activation)
    model = loadModel(model_skeleton, filepath)
    raw_outputs = model(images)
    outputs = F.softmax(raw_outputs, dim=1)
    print("Raw output: {}".format(raw_outputs))
    print("Softmax outputs: {}".format(outputs))
    print("Loss: {}".format(loss_criterion(raw_outputs, labels)))
    """
